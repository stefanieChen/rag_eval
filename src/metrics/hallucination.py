"""Hallucination detection via claim-level decomposition and grounding checks.

Pipeline:
1. Decompose the answer into individual claims (LLM call).
2. For each claim, check whether it is grounded in the retrieved context (LLM call).
3. Aggregate grounding scores to produce a hallucination rate.
"""

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.llm_judge.judge_base import LLMJudge, load_eval_config


class ClaimResult(BaseModel):
    """Grounding result for a single claim."""

    claim: str = Field(..., description="The extracted claim text")
    grounding_score: float = Field(
        ..., description="1.0 = grounded, 0.5 = partial, 0.0 = hallucinated"
    )
    reasoning: str = Field(default="", description="Why this score was assigned")


class HallucinationResult(BaseModel):
    """Full hallucination evaluation result."""

    claims: List[ClaimResult] = Field(default_factory=list)
    num_claims: int = Field(default=0)
    num_grounded: int = Field(default=0)
    num_hallucinated: int = Field(default=0)
    hallucination_rate: float = Field(
        default=0.0, description="Fraction of claims that are not grounded"
    )
    grounding_score: float = Field(
        default=0.0, description="Average grounding score across all claims"
    )
    latency_ms: float = Field(default=0.0)


_DECOMPOSE_PROMPT = """Extract all factual claims from the following answer. Each claim should be a single, atomic, verifiable statement.

**Answer:** {answer}

Return a JSON object with exactly this format:
```json
{{"claims": ["claim 1", "claim 2", "claim 3"]}}
```

Return ONLY the JSON object, no other text."""


_GROUNDING_PROMPT = """Determine whether the following claim is supported by the provided context.

**Claim:** {claim}

**Context:**
{context}

Score the claim's grounding:
- 1.0 = The claim is explicitly stated in or directly inferable from the context.
- 0.5 = The claim is loosely related to the context but adds unsupported details.
- 0.0 = The claim has no support in the context (hallucinated).

Return a JSON object:
```json
{{"grounding_score": <0.0 or 0.5 or 1.0>, "reasoning": "<brief explanation>"}}
```

Return ONLY the JSON object, no other text."""


class HallucinationDetector(LLMJudge):
    """Detect hallucinations by decomposing answers into claims and checking grounding.

    Two-stage process:
    1. Extract atomic claims from the answer.
    2. Verify each claim against the retrieved context.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_eval_config()
        super().__init__(config)
        eval_cfg = config.get("evaluation", {}).get("hallucination", {})
        self._claim_threshold = eval_cfg.get("claim_threshold", 0.5)

    def _decompose_claims(self, answer: str) -> List[str]:
        """Decompose an answer into atomic claims via LLM.

        Args:
            answer: The answer text to decompose.

        Returns:
            List of claim strings.
        """
        prompt = _DECOMPOSE_PROMPT.format(answer=answer)
        raw, _ = self._call_llm(prompt)
        parsed = self._parse_json_response(raw)
        claims = parsed.get("claims", [])
        if isinstance(claims, list):
            return [str(c) for c in claims]
        return []

    def _check_grounding(self, claim: str, contexts: List[str]) -> ClaimResult:
        """Check if a single claim is grounded in the context.

        Args:
            claim: The claim to verify.
            contexts: Retrieved context chunks.

        Returns:
            ClaimResult with grounding score and reasoning.
        """
        context_text = "\n".join(
            f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
        )
        prompt = _GROUNDING_PROMPT.format(claim=claim, context=context_text)
        raw, _ = self._call_llm(prompt)
        parsed = self._parse_json_response(raw)

        return ClaimResult(
            claim=claim,
            grounding_score=float(parsed.get("grounding_score", 0.0)),
            reasoning=parsed.get("reasoning", ""),
        )

    def evaluate(
        self,
        answer: str,
        contexts: List[str],
    ) -> HallucinationResult:
        """Run full hallucination detection: decompose → check → aggregate.

        Args:
            answer: The RAG system's answer.
            contexts: Retrieved context chunks.

        Returns:
            HallucinationResult with per-claim and aggregate scores.
        """
        import time
        start = time.perf_counter()

        claims_text = self._decompose_claims(answer)
        if not claims_text:
            return HallucinationResult(latency_ms=0.0)

        claim_results = []
        for claim in claims_text:
            result = self._check_grounding(claim, contexts)
            claim_results.append(result)

        num_grounded = sum(
            1 for c in claim_results if c.grounding_score >= self._claim_threshold
        )
        num_hallucinated = len(claim_results) - num_grounded
        avg_grounding = (
            sum(c.grounding_score for c in claim_results) / len(claim_results)
            if claim_results else 0.0
        )
        hallucination_rate = num_hallucinated / len(claim_results) if claim_results else 0.0

        latency_ms = (time.perf_counter() - start) * 1000

        return HallucinationResult(
            claims=claim_results,
            num_claims=len(claim_results),
            num_grounded=num_grounded,
            num_hallucinated=num_hallucinated,
            hallucination_rate=hallucination_rate,
            grounding_score=avg_grounding,
            latency_ms=latency_ms,
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate hallucination for a batch of test cases.

        Args:
            test_cases: List of dicts with keys: answer, contexts.

        Returns:
            Dict with 'per_case' results and 'aggregated' scores.
        """
        per_case = []
        for case in test_cases:
            result = self.evaluate(
                answer=case.get("answer", ""),
                contexts=case.get("contexts", []),
            )
            per_case.append(result)

        avg_hallucination_rate = (
            sum(r.hallucination_rate for r in per_case) / len(per_case)
            if per_case else 0.0
        )
        avg_grounding = (
            sum(r.grounding_score for r in per_case) / len(per_case)
            if per_case else 0.0
        )

        return {
            "per_case": per_case,
            "aggregated": {
                "hallucination_rate": avg_hallucination_rate,
                "grounding_score": avg_grounding,
            },
            "num_cases": len(test_cases),
        }
