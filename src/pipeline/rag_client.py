"""RAG client: interface to the rag_2 RAG pipeline.

Provides a clean API for querying the RAG system, abstracting away
the import path and configuration details.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.datasets.schema import RAGResponse
from src.llm_judge.judge_base import load_eval_config


class RAGClient:
    """Client interface to the rag_2 RAG pipeline.

    Supports direct Python import (same machine) by adding the
    rag_2 project to sys.path and instantiating RAGPipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_eval_config()

        rag_cfg = config.get("rag", {})
        self._project_path = Path(rag_cfg.get("project_path", "../rag_2")).resolve()
        self._config_path = rag_cfg.get("config_path", None)
        self._pipeline = None
        self._rag_config = None

    def _ensure_pipeline(self, config_override: Optional[Dict[str, Any]] = None) -> None:
        """Lazily initialize the RAG pipeline.

        Args:
            config_override: Optional config values to override in the RAG system.
        """
        if self._pipeline is not None and config_override is None:
            return

        # Add rag_2 project to Python path
        project_str = str(self._project_path)
        if project_str not in sys.path:
            sys.path.insert(0, project_str)

        from src.config import load_config as rag_load_config
        from src.pipeline import RAGPipeline

        self._rag_config = rag_load_config()

        if config_override:
            self._rag_config = _deep_merge(self._rag_config, config_override)

        self._pipeline = RAGPipeline(self._rag_config)

    def query(self, question: str) -> RAGResponse:
        """Query the RAG system and return a structured response.

        Args:
            question: The user question.

        Returns:
            RAGResponse with answer, retrieved contexts, and metadata.
        """
        import time

        self._ensure_pipeline()
        start = time.perf_counter()

        try:
            result = self._pipeline.query(question)
            latency_ms = (time.perf_counter() - start) * 1000

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            contexts = []
            context_ids = []
            for src in sources:
                if isinstance(src, dict):
                    contexts.append(src.get("content", str(src)))
                    context_ids.append(src.get("id", src.get("source", "")))
                else:
                    contexts.append(str(src))

            return RAGResponse(
                answer=answer,
                retrieved_contexts=contexts,
                retrieved_ids=context_ids,
                latency_ms=latency_ms,
                metadata={"raw_result": result},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return RAGResponse(
                answer=f"[Error] {e}",
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    def query_with_config(
        self,
        question: str,
        config_override: Dict[str, Any],
    ) -> RAGResponse:
        """Query the RAG system with a temporary config override (for A/B testing).

        Creates a new pipeline instance with the override applied.

        Args:
            question: The user question.
            config_override: Config values to override (e.g., {"retrieval.hybrid_mode": true}).

        Returns:
            RAGResponse from the overridden configuration.
        """
        # Parse dot-notation keys into nested dict
        nested_override = _unflatten_dict(config_override)

        # Force re-init with override
        old_pipeline = self._pipeline
        self._pipeline = None
        try:
            self._ensure_pipeline(config_override=nested_override)
            return self.query(question)
        finally:
            self._pipeline = old_pipeline

    def get_rag_config(self) -> Dict[str, Any]:
        """Return the current RAG system configuration.

        Returns:
            RAG config dict.
        """
        self._ensure_pipeline()
        return self._rag_config or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts. Override values take precedence.

    Args:
        base: Base dict.
        override: Override dict.

    Returns:
        Merged dict.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _unflatten_dict(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dot-notation keys to nested dict.

    Example: {"retrieval.hybrid_mode": true} → {"retrieval": {"hybrid_mode": true}}

    Args:
        flat: Dict with potentially dot-notation keys.

    Returns:
        Nested dict.
    """
    result: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result
