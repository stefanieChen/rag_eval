"""Reference-based grading: compare a RAG response against a ground truth answer.

Produces structured grading with factual accuracy, completeness,
missing/extra information, and contradictions.
"""

from typing import Any, Dict, List, Optional

from src.llm_judge.judge_base import LLMJudge, ReferenceResult


class ReferenceJudge(LLMJudge):
    """Grade a response by comparing it to a reference (ground truth) answer.

    Provides detailed breakdown including factual accuracy, completeness,
    missing information, extra information, and contradictions.
    """

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> ReferenceResult:
        """Grade a response against the ground truth.

        Args:
            question: The user question.
            answer: The RAG system's answer.
            contexts: Retrieved context chunks.
            ground_truth: Reference answer to compare against.

        Returns:
            ReferenceResult with detailed grading.
        """
        prompt = self._render_template(
            "reference_grade.j2",
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        raw_response, latency_ms = self._call_with_consensus(prompt)
        parsed = self._parse_json_response(raw_response)

        return ReferenceResult(
            overall_score=float(parsed.get("overall_score", 0)),
            factual_accuracy=float(parsed.get("factual_accuracy", 0)),
            completeness=float(parsed.get("completeness", 0)),
            missing_information=parsed.get("missing_information", []),
            extra_information=parsed.get("extra_information", []),
            contradictions=parsed.get("contradictions", []),
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw_response,
            latency_ms=latency_ms,
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[ReferenceResult]:
        """Grade a batch of test cases against their ground truths.

        Args:
            test_cases: List of dicts with keys: question, answer, contexts, ground_truth.

        Returns:
            List of ReferenceResult, one per test case.
        """
        results = []
        for case in test_cases:
            result = self.evaluate(
                question=case["question"],
                answer=case.get("answer", ""),
                contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth", ""),
            )
            results.append(result)
        return results
