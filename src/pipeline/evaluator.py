"""Evaluator: orchestrates dataset loading → RAG querying → metric scoring → result collection.

Central module that ties together all evaluation components into a cohesive pipeline.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.datasets.loader import load_test_set
from src.datasets.schema import EvalResult, EvalRunSummary, EvalTestCase, RAGResponse, TestCase
from src.llm_judge.judge_base import load_eval_config
from src.metrics.composite import compute_composite_score


class Evaluator:
    """Run evaluation over a test set, collecting results from all metric layers.

    Supports two modes:
    - **static**: Uses pre-defined contexts from the test set (no live RAG queries).
    - **pipeline**: Queries the live RAG system and evaluates real responses.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_eval_config()
        self._config = config
        self._eval_cfg = config.get("evaluation", {})

    def run(
        self,
        test_set_path: str,
        mode: str = "static",
        metrics: Optional[List[str]] = None,
    ) -> EvalRunSummary:
        """Run a full evaluation on a test set.

        Args:
            test_set_path: Path to the test set file (JSON/JSONL/CSV).
            mode: 'static' (use test set contexts) or 'pipeline' (query live RAG).
            metrics: List of metric names to compute. None = use config defaults.

        Returns:
            EvalRunSummary with per-case results and aggregated scores.
        """
        run_id = uuid4().hex[:12]
        start_time = time.perf_counter()

        test_cases = load_test_set(test_set_path)
        if metrics is None:
            metrics = self._eval_cfg.get("default_metrics", [
                "retrieval", "faithfulness", "relevancy",
                "completeness", "hallucination", "semantic_similarity",
            ])

        # Build eval test cases with RAG responses
        eval_cases = self._build_eval_cases(test_cases, mode)

        # Evaluate each case
        per_case_results = []
        for eval_case in eval_cases:
            result = self._evaluate_case(eval_case, metrics)
            per_case_results.append(result)

        # Aggregate scores
        aggregated = self._aggregate_results(per_case_results, metrics)

        total_ms = (time.perf_counter() - start_time) * 1000

        return EvalRunSummary(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            num_cases=len(test_cases),
            config_snapshot={
                "mode": mode,
                "metrics": metrics,
                "judge": self._config.get("judge", {}),
            },
            aggregated_scores=aggregated,
            per_case_results=per_case_results,
            total_latency_ms=total_ms,
        )

    def _build_eval_cases(
        self,
        test_cases: List[TestCase],
        mode: str,
    ) -> List[EvalTestCase]:
        """Build evaluation test cases, optionally querying the RAG system.

        Args:
            test_cases: Loaded test cases.
            mode: 'static' or 'pipeline'.

        Returns:
            List of EvalTestCase with RAG responses populated.
        """
        eval_cases = []
        rag_client = None

        if mode == "pipeline":
            from src.pipeline.rag_client import RAGClient
            rag_client = RAGClient(config=self._config)

        for tc in test_cases:
            if mode == "pipeline" and rag_client is not None:
                rag_response = rag_client.query(tc.question)
            else:
                # Static mode: use test set contexts as "retrieved" and leave answer empty
                rag_response = RAGResponse(
                    answer="",
                    retrieved_contexts=tc.contexts,
                )

            eval_cases.append(EvalTestCase(test_case=tc, rag_response=rag_response))

        return eval_cases

    def _evaluate_case(
        self,
        eval_case: EvalTestCase,
        metrics: List[str],
    ) -> EvalResult:
        """Evaluate a single test case on all requested metrics.

        Args:
            eval_case: The test case with RAG response.
            metrics: Metrics to compute.

        Returns:
            EvalResult with all scores.
        """
        result = EvalResult(
            question=eval_case.question,
            answer=eval_case.answer,
            ground_truth=eval_case.ground_truth,
            retrieved_contexts=eval_case.retrieved_contexts,
        )

        contexts = eval_case.retrieved_contexts
        answer = eval_case.answer
        gt = eval_case.ground_truth

        # Retrieval metrics (need ground-truth contexts)
        if "retrieval" in metrics and eval_case.gt_contexts:
            from src.metrics.retrieval import compute_all_retrieval_metrics
            k_values = self._eval_cfg.get("retrieval", {}).get("k_values", [1, 3, 5, 10])
            relevant_set = set(eval_case.gt_contexts)
            result.retrieval = compute_all_retrieval_metrics(
                retrieved=contexts, relevant=relevant_set, k_values=k_values,
            )

        # Generation metrics (LLM judge)
        generation_criteria = [m for m in metrics if m in ("faithfulness", "relevancy", "completeness")]
        if generation_criteria and answer:
            from src.metrics.generation import evaluate_generation
            gen_results = evaluate_generation(
                question=eval_case.question,
                answer=answer,
                contexts=contexts,
                ground_truth=gt if gt else None,
                criteria=generation_criteria,
                config=self._config,
            )
            for criterion, judge_result in gen_results.items():
                result.scores[criterion] = judge_result.score
                result.judge_reasoning[criterion] = judge_result.reasoning

        # Hallucination detection
        if "hallucination" in metrics and answer:
            from src.metrics.hallucination import HallucinationDetector
            detector = HallucinationDetector(config=self._config)
            hal_result = detector.evaluate(answer=answer, contexts=contexts)
            result.hallucination = {
                "hallucination_rate": hal_result.hallucination_rate,
                "grounding_score": hal_result.grounding_score,
                "num_claims": hal_result.num_claims,
                "num_hallucinated": hal_result.num_hallucinated,
            }
            result.scores["hallucination_score"] = hal_result.grounding_score

        # Semantic similarity
        if "semantic_similarity" in metrics and answer and gt:
            from src.metrics.semantic import compute_semantic_metrics
            sem_model = self._eval_cfg.get("semantic", {}).get(
                "embedding_model", "all-MiniLM-L6-v2"
            )
            result.semantic = compute_semantic_metrics(
                answer=answer, ground_truth=gt, model_name=sem_model,
            )
            result.scores["semantic_similarity"] = result.semantic.get(
                "embedding_similarity", 0.0
            )

        # Composite score
        if result.scores:
            composite = compute_composite_score(result.scores)
            result.composite_score = composite["composite_score"]

        return result

    def _aggregate_results(
        self,
        per_case: List[EvalResult],
        metrics: List[str],
    ) -> Dict[str, float]:
        """Aggregate per-case scores into dataset-level means.

        Args:
            per_case: List of per-case evaluation results.
            metrics: Metrics that were computed.

        Returns:
            Dict of metric_name → mean score.
        """
        if not per_case:
            return {}

        # Collect all score keys
        all_keys = set()
        for result in per_case:
            all_keys.update(result.scores.keys())

        aggregated: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [r.scores[key] for r in per_case if key in r.scores]
            if values:
                aggregated[key] = sum(values) / len(values)

        # Aggregate retrieval metrics if present
        retrieval_results = [r.retrieval for r in per_case if r.retrieval]
        if retrieval_results:
            from src.metrics.retrieval import aggregate_retrieval_metrics
            aggregated.update(aggregate_retrieval_metrics(retrieval_results))

        # Aggregate composite score
        composites = [r.composite_score for r in per_case if r.composite_score is not None]
        if composites:
            aggregated["composite"] = sum(composites) / len(composites)

        return aggregated
