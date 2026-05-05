"""Generation quality metrics using LLM-as-a-Judge.

Wraps the pointwise judge to provide convenient metric-level functions
for faithfulness, relevancy, completeness, and conciseness evaluation.
"""

from typing import Any, Dict, List, Optional

from src.llm_judge.judge_base import JudgeResult, load_eval_config
from src.llm_judge.pointwise import PointwiseJudge


_DEFAULT_CRITERIA = ["faithfulness", "relevancy", "completeness"]


def evaluate_generation(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    criteria: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, JudgeResult]:
    """Evaluate generation quality on specified criteria using LLM judge.

    Args:
        question: User question.
        answer: RAG system answer.
        contexts: Retrieved context chunks.
        ground_truth: Optional reference answer.
        criteria: Rubric names to evaluate. Defaults to faithfulness, relevancy, completeness.
        config: Optional eval config override.

    Returns:
        Dict mapping criterion name to JudgeResult.
    """
    if criteria is None:
        criteria = _DEFAULT_CRITERIA

    judge = PointwiseJudge(config=config)
    return judge.evaluate(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        criteria=criteria,
    )


def evaluate_generation_batch(
    test_cases: List[Dict[str, Any]],
    criteria: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate generation quality for a batch of test cases.

    Args:
        test_cases: List of dicts with keys: question, answer, contexts, ground_truth.
        criteria: Rubric names to evaluate.
        config: Optional eval config override.

    Returns:
        Dict with 'per_case' results and 'aggregated' mean scores.
    """
    if criteria is None:
        criteria = _DEFAULT_CRITERIA

    judge = PointwiseJudge(config=config)
    per_case = judge.evaluate_batch(test_cases, criteria=criteria)

    # Aggregate scores per criterion
    aggregated: Dict[str, float] = {}
    for criterion in criteria:
        scores = [
            case_results[criterion].score
            for case_results in per_case
            if criterion in case_results
        ]
        aggregated[criterion] = sum(scores) / len(scores) if scores else 0.0

    return {
        "per_case": per_case,
        "aggregated": aggregated,
        "num_cases": len(test_cases),
    }
