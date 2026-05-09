"""Base LLM Judge with pluggable provider support via LiteLLM.

Provides a unified interface for calling LLMs (Ollama, OpenAI, Anthropic)
with structured JSON output parsing, retry logic, and multi-judge consensus.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    """Structured result from an LLM judge evaluation."""

    score: float = Field(..., description="Numeric score from the rubric")
    reasoning: str = Field(default="", description="Judge's reasoning for the score")
    raw_response: str = Field(default="", description="Raw LLM response text")
    latency_ms: float = Field(default=0.0, description="LLM call latency in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PairwiseResult(BaseModel):
    """Structured result from a pairwise comparison."""

    winner: str = Field(..., description="'A', 'B', or 'tie'")
    score_a: float = Field(..., description="Score for response A")
    score_b: float = Field(..., description="Score for response B")
    reasoning: str = Field(default="", description="Comparison reasoning")
    criteria_breakdown: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Per-criterion scores for A and B"
    )
    raw_response: str = Field(default="", description="Raw LLM response text")
    latency_ms: float = Field(default=0.0)


class ReferenceResult(BaseModel):
    """Structured result from a reference-based grading."""

    overall_score: float = Field(..., description="Overall match score (1-5)")
    factual_accuracy: float = Field(default=0.0)
    completeness: float = Field(default=0.0)
    missing_information: List[str] = Field(default_factory=list)
    extra_information: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    raw_response: str = Field(default="")
    latency_ms: float = Field(default=0.0)


def _get_project_root() -> Path:
    """Return the project root directory (where config/ lives)."""
    return Path(__file__).resolve().parent.parent.parent


def load_eval_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the main evaluation config from config/eval_config.yaml.

    Resolution order for config path:
    1. Explicit ``config_path`` argument.
    2. ``RAG_EVAL_CONFIG`` environment variable.
    3. Default: ``<project_root>/config/eval_config.yaml``.

    Args:
        config_path: Optional explicit path to eval_config.yaml.

    Returns:
        Parsed config dict.
    """
    if config_path is None:
        config_path = os.environ.get("RAG_EVAL_CONFIG", None)
    if config_path is None:
        config_path = str(_get_project_root() / "config" / "eval_config.yaml")
    resolved = Path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_rubric(rubric_name: str, rubrics_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load a rubric YAML file from config/rubrics/.

    Resolution order for rubrics directory:
    1. Explicit ``rubrics_dir`` argument.
    2. ``RAG_EVAL_RUBRICS_DIR`` environment variable.
    3. Default: ``<project_root>/config/rubrics/``.

    Args:
        rubric_name: Name of the rubric (without .yaml extension).
        rubrics_dir: Optional explicit path to rubrics directory.

    Returns:
        Parsed rubric dict.
    """
    if rubrics_dir is None:
        rubrics_dir = os.environ.get("RAG_EVAL_RUBRICS_DIR", None)
    if rubrics_dir is None:
        rubrics_dir = str(_get_project_root() / "config" / "rubrics")
    rubric_path = Path(rubrics_dir) / f"{rubric_name}.yaml"
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric not found: {rubric_path}")
    with open(rubric_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_env_vars(value: str) -> str:
    """Resolve ${ENV_VAR} references in a string value.

    Args:
        value: String potentially containing ${VAR} references.

    Returns:
        String with env vars resolved.
    """
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{(\w+)\}")
    def replacer(match):
        env_var = match.group(1)
        return os.environ.get(env_var, match.group(0))
    return pattern.sub(replacer, value)


class LLMJudge:
    """Pluggable LLM judge using LiteLLM for unified provider access.

    Supports Ollama (local), OpenAI, and Anthropic as LLM backends.
    Uses Jinja2 templates for prompt construction and JSON mode for
    structured output parsing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_eval_config()

        judge_cfg = config.get("judge", {})
        self._provider = judge_cfg.get("provider", "ollama")
        self._model = judge_cfg.get("model", "qwen2.5:7b")
        self._temperature = judge_cfg.get("temperature", 0.0)
        self._max_tokens = judge_cfg.get("max_tokens", 2048)
        self._structured_output = judge_cfg.get("structured_output", True)
        self._num_judges = judge_cfg.get("num_judges", 1)
        self._timeout = judge_cfg.get("timeout", 120)

        self._litellm_model = self._build_litellm_model(judge_cfg)

        prompts_dir = Path(__file__).resolve().parent / "prompts"
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _build_litellm_model(self, judge_cfg: Dict[str, Any]) -> str:
        """Build the LiteLLM model string based on provider config.

        Args:
            judge_cfg: Judge configuration dict.

        Returns:
            LiteLLM-compatible model string (e.g., 'ollama/qwen2.5:7b').
        """
        provider = self._provider
        model = self._model

        if provider == "ollama":
            base_url = judge_cfg.get("ollama_base_url", "http://localhost:11434")
            os.environ["OLLAMA_API_BASE"] = base_url
            return f"ollama/{model}"
        elif provider == "openai":
            api_key = _resolve_env_vars(judge_cfg.get("openai_api_key", ""))
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            return model
        elif provider == "anthropic":
            api_key = _resolve_env_vars(judge_cfg.get("anthropic_api_key", ""))
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            return f"anthropic/{model}"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _call_llm(self, prompt: str) -> tuple[str, float]:
        """Call the LLM via LiteLLM and return (response_text, latency_ms).

        Args:
            prompt: The full prompt string.

        Returns:
            Tuple of (response text, latency in milliseconds).
        """
        import litellm

        start = time.perf_counter()

        kwargs = {
            "model": self._litellm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
        }

        if self._structured_output and self._provider != "ollama":
            kwargs["response_format"] = {"type": "json_object"}

        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content.strip()
        latency_ms = (time.perf_counter() - start) * 1000

        return text, latency_ms

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from an LLM response.

        Handles cases where the JSON is wrapped in markdown code blocks.

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed JSON dict.
        """
        cleaned = text.strip()
        # Strip markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {"error": "Failed to parse JSON", "raw": text}

    def _render_template(self, template_name: str, **kwargs) -> str:
        """Render a Jinja2 prompt template.

        Args:
            template_name: Template filename (e.g., 'pointwise_score.j2').
            **kwargs: Template variables.

        Returns:
            Rendered prompt string.
        """
        template = self._jinja_env.get_template(template_name)
        return template.render(**kwargs)

    def _call_with_consensus(self, prompt: str) -> tuple[str, float]:
        """Call LLM multiple times and return the majority/average result.

        If num_judges == 1, equivalent to a single call.

        Args:
            prompt: The full prompt string.

        Returns:
            Tuple of (best response text, average latency).
        """
        if self._num_judges <= 1:
            return self._call_llm(prompt)

        responses = []
        total_latency = 0.0
        for _ in range(self._num_judges):
            text, latency = self._call_llm(prompt)
            responses.append(text)
            total_latency += latency

        avg_latency = total_latency / len(responses)

        # Parse all responses and average the scores
        parsed = [self._parse_json_response(r) for r in responses]
        scores = [p.get("score", 0) for p in parsed if "score" in p]
        if scores:
            avg_score = sum(scores) / len(scores)
            # Return the response whose score is closest to average
            best_idx = min(range(len(scores)), key=lambda i: abs(scores[i] - avg_score))
            return responses[best_idx], avg_latency

        return responses[0], avg_latency
