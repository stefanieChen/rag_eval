"""Pointwise evaluation: score a single RAG response on one or more criteria.

Uses rubric-guided prompts to produce structured (score, reasoning) pairs
for each evaluation criterion.
"""

from typing import Any, Dict, List, Optional

from src.llm_judge.judge_base import (
    JudgeResult,
    LLMJudge,
    load_eval_config,
    load_rubric,
)


class PointwiseJudge(LLMJudge):
    """Score a single response against rubric-defined criteria.

    For each criterion (e.g., faithfulness, relevancy), renders the
    pointwise_score.j2 template with the rubric and calls the LLM judge.
    """

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        criteria: Optional[List[str]] = None,
    ) -> Dict[str, JudgeResult]:
        """Evaluate a response on multiple criteria using pointwise scoring.

        Args:
            question: The user question.
            answer: The RAG system's answer.
            contexts: Retrieved context chunks.
            ground_truth: Optional reference answer.
            criteria: List of rubric names to evaluate. Defaults to config defaults.

        Returns:
            Dict mapping criterion name to JudgeResult.
        """
        if criteria is None:
            config = load_eval_config()
            eval_cfg = config.get("evaluation", {})
            criteria = eval_cfg.get("default_metrics", ["faithfulness", "relevancy"])
            # Filter to only criteria that have rubrics
            criteria = [c for c in criteria if c in (
                "faithfulness", "relevancy", "completeness"
            )]

        results: Dict[str, JudgeResult] = {}
        for criterion in criteria:
            result = self._evaluate_single(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                criterion=criterion,
            )
            results[criterion] = result

        return results

    def _evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str],
        criterion: str,
    ) -> JudgeResult:
        """Evaluate a single criterion.

        Args:
            question: The user question.
            answer: The RAG system's answer.
            contexts: Retrieved context chunks.
            ground_truth: Optional reference answer.
            criterion: Rubric name (e.g., 'faithfulness').

        Returns:
            JudgeResult with score and reasoning.
        """
        rubric = load_rubric(criterion)
        prompt = self._render_template(
            "pointwise_score.j2",
            criterion_name=rubric.get("name", criterion),
            rubric_description=rubric.get("description", ""),
            rubric_levels=rubric.get("criteria", []),
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        raw_response, latency_ms = self._call_with_consensus(prompt)
        parsed = self._parse_json_response(raw_response)

        return JudgeResult(
            score=float(parsed.get("score", 0)),
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw_response,
            latency_ms=latency_ms,
            metadata={"criterion": criterion},
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None,
    ) -> List[Dict[str, JudgeResult]]:
        """Evaluate a batch of test cases.

        Args:
            test_cases: List of dicts with keys: question, answer, contexts, ground_truth.
            criteria: Criteria to evaluate on.

        Returns:
            List of result dicts, one per test case.
        """
        results = []
        for case in test_cases:
            result = self.evaluate(
                question=case["question"],
                answer=case.get("answer", ""),
                contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth"),
                criteria=criteria,
            )
            results.append(result)
        return results
