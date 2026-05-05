"""Pairwise evaluation: compare two RAG responses (A vs B) for A/B testing.

Uses a structured comparison prompt to determine which response is better
across multiple criteria, producing a winner and per-criterion breakdown.
"""

from typing import Any, Dict, List, Optional

from src.llm_judge.judge_base import LLMJudge, PairwiseResult


class PairwiseJudge(LLMJudge):
    """Compare two responses to the same question and determine which is better.

    Designed for A/B testing different RAG configurations. Supports
    position debiasing by optionally swapping A/B and averaging.
    """

    def evaluate(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        debias_position: bool = True,
    ) -> PairwiseResult:
        """Compare two responses and return structured comparison result.

        Args:
            question: The user question.
            answer_a: Response from configuration A.
            answer_b: Response from configuration B.
            contexts: Retrieved context chunks (shared).
            ground_truth: Optional reference answer.
            debias_position: If True, run comparison in both orderings
                and average scores to reduce position bias.

        Returns:
            PairwiseResult with winner, scores, and breakdown.
        """
        result_ab = self._compare_once(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        if not debias_position:
            return result_ab

        # Run again with swapped positions to debias
        result_ba = self._compare_once(
            question=question,
            answer_a=answer_b,
            answer_b=answer_a,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        return self._merge_debiased(result_ab, result_ba)

    def _compare_once(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        contexts: List[str],
        ground_truth: Optional[str],
    ) -> PairwiseResult:
        """Run a single pairwise comparison.

        Args:
            question: The user question.
            answer_a: First response.
            answer_b: Second response.
            contexts: Retrieved context chunks.
            ground_truth: Optional reference answer.

        Returns:
            PairwiseResult from this single comparison.
        """
        prompt = self._render_template(
            "pairwise_compare.j2",
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        raw_response, latency_ms = self._call_with_consensus(prompt)
        parsed = self._parse_json_response(raw_response)

        return PairwiseResult(
            winner=parsed.get("winner", "tie"),
            score_a=float(parsed.get("score_a", 0)),
            score_b=float(parsed.get("score_b", 0)),
            reasoning=parsed.get("reasoning", ""),
            criteria_breakdown=parsed.get("criteria_breakdown", {}),
            raw_response=raw_response,
            latency_ms=latency_ms,
        )

    def _merge_debiased(
        self,
        result_ab: PairwiseResult,
        result_ba: PairwiseResult,
    ) -> PairwiseResult:
        """Merge two comparison results (normal + swapped) to debias position.

        In result_ba, A and B are swapped, so we flip scores back before averaging.

        Args:
            result_ab: Result from normal ordering (A first, B second).
            result_ba: Result from swapped ordering (B first, A second).

        Returns:
            Merged PairwiseResult with debiased scores.
        """
        # In result_ba, "A" was actually original B and "B" was original A
        avg_score_a = (result_ab.score_a + result_ba.score_b) / 2
        avg_score_b = (result_ab.score_b + result_ba.score_a) / 2

        if avg_score_a > avg_score_b + 0.1:
            winner = "A"
        elif avg_score_b > avg_score_a + 0.1:
            winner = "B"
        else:
            winner = "tie"

        # Merge criteria breakdown
        merged_breakdown: Dict[str, Dict[str, float]] = {}
        for criterion in result_ab.criteria_breakdown:
            ab_scores = result_ab.criteria_breakdown.get(criterion, {})
            ba_scores = result_ba.criteria_breakdown.get(criterion, {})
            merged_breakdown[criterion] = {
                "a": (ab_scores.get("a", 0) + ba_scores.get("b", 0)) / 2,
                "b": (ab_scores.get("b", 0) + ba_scores.get("a", 0)) / 2,
            }

        avg_latency = (result_ab.latency_ms + result_ba.latency_ms) / 2

        return PairwiseResult(
            winner=winner,
            score_a=avg_score_a,
            score_b=avg_score_b,
            reasoning=f"[Debiased] AB: {result_ab.reasoning} | BA: {result_ba.reasoning}",
            criteria_breakdown=merged_breakdown,
            raw_response=f"AB: {result_ab.raw_response}\n---\nBA: {result_ba.raw_response}",
            latency_ms=avg_latency,
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        debias_position: bool = True,
    ) -> List[PairwiseResult]:
        """Compare responses for a batch of test cases.

        Args:
            test_cases: List of dicts with keys: question, answer_a, answer_b,
                contexts, ground_truth.
            debias_position: Whether to debias position.

        Returns:
            List of PairwiseResult, one per test case.
        """
        results = []
        for case in test_cases:
            result = self.evaluate(
                question=case["question"],
                answer_a=case["answer_a"],
                answer_b=case["answer_b"],
                contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth"),
                debias_position=debias_position,
            )
            results.append(result)
        return results
