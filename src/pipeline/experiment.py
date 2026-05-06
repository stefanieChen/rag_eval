"""A/B experiment runner: compare RAG configurations with statistical significance.

Runs the same test set against two different RAG configurations,
collects pairwise LLM judge comparisons and metric-level scores,
then applies statistical tests to determine if differences are significant.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from src.datasets.loader import load_test_set
from src.datasets.schema import TestCase
from src.llm_judge.judge_base import load_eval_config
from src.llm_judge.pairwise import PairwiseJudge, PairwiseResult
from src.pipeline.rag_client import RAGClient
from src.logging import get_logger


class ExperimentResult(BaseModel):
    """Result of an A/B experiment comparing two RAG configurations."""

    experiment_id: str = ""
    timestamp: str = ""
    config_a: Dict[str, Any] = Field(default_factory=dict)
    config_b: Dict[str, Any] = Field(default_factory=dict)
    num_cases: int = 0
    pairwise_results: List[Dict[str, Any]] = Field(default_factory=list)
    scores_a: Dict[str, float] = Field(default_factory=dict)
    scores_b: Dict[str, float] = Field(default_factory=dict)
    statistical_tests: Dict[str, Any] = Field(default_factory=dict)
    winner: str = Field(default="", description="'A', 'B', or 'tie'")
    total_latency_ms: float = 0.0


class ExperimentRunner:
    """Run A/B experiments comparing two RAG configurations.

    For each test case:
    1. Query both configurations.
    2. Run pairwise LLM judge comparison (with position debiasing).
    3. Run individual metrics for each configuration.
    4. Aggregate results and apply statistical significance tests.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_eval_config()
        self._config = config
        self._exp_cfg = config.get("experiment", {})
        self._logger = get_logger("pipeline.experiment_runner")

    def run(
        self,
        test_set_path: str,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        label_a: str = "Config A",
        label_b: str = "Config B",
    ) -> ExperimentResult:
        """Run a full A/B experiment.

        Args:
            test_set_path: Path to the test set file.
            config_a: RAG config overrides for variant A.
            config_b: RAG config overrides for variant B.
            label_a: Human-readable label for config A.
            label_b: Human-readable label for config B.

        Returns:
            ExperimentResult with all comparison data and statistical tests.
        """
        experiment_id = uuid4().hex[:12]
        start_time = time.perf_counter()

        test_cases = load_test_set(test_set_path)
        rag_client = RAGClient(config=self._config)
        pairwise_judge = PairwiseJudge(config=self._config)

        pairwise_results = []
        scores_a_all: List[float] = []
        scores_b_all: List[float] = []

        self._logger.info(
            "Experiment started",
            extra={
                "experiment_id": experiment_id,
                "test_set": test_set_path,
                "num_cases": len(test_cases),
                "config_a": config_a,
                "config_b": config_b,
            },
        )

        for tc in test_cases:
            # Query both configs
            try:
                response_a = rag_client.query_with_config(tc.question, config_a)
                response_b = rag_client.query_with_config(tc.question, config_b)
            except Exception as exc:
                self._logger.exception(
                    "Query failed", extra={"question": tc.question, "error": str(exc)}
                )
                raise

            # Pairwise comparison
            pairwise = pairwise_judge.evaluate(
                question=tc.question,
                answer_a=response_a.answer,
                answer_b=response_b.answer,
                contexts=response_a.retrieved_contexts,
                ground_truth=tc.ground_truth if tc.ground_truth else None,
                debias_position=True,
            )

            scores_a_all.append(pairwise.score_a)
            scores_b_all.append(pairwise.score_b)

            pairwise_results.append({
                "question": tc.question,
                "answer_a": response_a.answer,
                "answer_b": response_b.answer,
                "winner": pairwise.winner,
                "score_a": pairwise.score_a,
                "score_b": pairwise.score_b,
                "reasoning": pairwise.reasoning,
                "criteria_breakdown": pairwise.criteria_breakdown,
                "latency_a_ms": response_a.latency_ms,
                "latency_b_ms": response_b.latency_ms,
            })

        # Aggregate scores
        mean_a = float(np.mean(scores_a_all)) if scores_a_all else 0.0
        mean_b = float(np.mean(scores_b_all)) if scores_b_all else 0.0

        # Win rates
        wins_a = sum(1 for r in pairwise_results if r["winner"] == "A")
        wins_b = sum(1 for r in pairwise_results if r["winner"] == "B")
        ties = sum(1 for r in pairwise_results if r["winner"] == "tie")

        # Statistical tests
        stat_tests = self._run_statistical_tests(scores_a_all, scores_b_all)

        # Determine overall winner
        sig_level = self._exp_cfg.get("significance_level", 0.05)
        p_value = stat_tests.get("p_value", 1.0)
        if p_value < sig_level:
            winner = "A" if mean_a > mean_b else "B"
        else:
            winner = "tie"

        total_ms = (time.perf_counter() - start_time) * 1000

        self._logger.info(
            "Experiment completed",
            extra={
                "experiment_id": experiment_id,
                "winner": winner,
                "avg_latency_ms": total_ms / max(len(test_cases), 1),
                "total_latency_ms": total_ms,
                "p_value": p_value,
            },
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            config_a={**config_a, "_label": label_a},
            config_b={**config_b, "_label": label_b},
            num_cases=len(test_cases),
            pairwise_results=pairwise_results,
            scores_a={
                "mean": round(mean_a, 4),
                "std": round(float(np.std(scores_a_all)), 4) if scores_a_all else 0.0,
                "win_count": wins_a,
                "win_rate": round(wins_a / len(test_cases), 4) if test_cases else 0.0,
            },
            scores_b={
                "mean": round(mean_b, 4),
                "std": round(float(np.std(scores_b_all)), 4) if scores_b_all else 0.0,
                "win_count": wins_b,
                "win_rate": round(wins_b / len(test_cases), 4) if test_cases else 0.0,
            },
            statistical_tests={
                **stat_tests,
                "ties": ties,
                "tie_rate": round(ties / len(test_cases), 4) if test_cases else 0.0,
            },
            winner=winner,
            total_latency_ms=total_ms,
        )

    def _run_statistical_tests(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> Dict[str, Any]:
        """Run statistical significance tests on paired scores.

        Args:
            scores_a: Scores for config A (one per test case).
            scores_b: Scores for config B (one per test case).

        Returns:
            Dict with test name, p-value, statistic, and significance flag.
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {"test": "none", "reason": "insufficient_samples", "p_value": 1.0}

        test_type = self._exp_cfg.get("statistical_test", "wilcoxon")
        sig_level = self._exp_cfg.get("significance_level", 0.05)

        try:
            from scipy import stats

            if test_type == "wilcoxon":
                # Wilcoxon signed-rank test (non-parametric, paired)
                diffs = [a - b for a, b in zip(scores_a, scores_b)]
                if all(d == 0 for d in diffs):
                    return {"test": "wilcoxon", "p_value": 1.0, "statistic": 0.0, "significant": False}
                stat, p_value = stats.wilcoxon(scores_a, scores_b)
                stat = float(stat)
                p_value = float(p_value)
                return {
                    "test": "wilcoxon",
                    "statistic": round(stat, 4),
                    "p_value": round(p_value, 6),
                    "significant": bool(p_value < sig_level),
                }

            elif test_type == "paired_t":
                # Paired t-test (parametric)
                stat, p_value = stats.ttest_rel(scores_a, scores_b)
                stat = float(stat)
                p_value = float(p_value)
                return {
                    "test": "paired_t",
                    "statistic": round(stat, 4),
                    "p_value": round(p_value, 6),
                    "significant": bool(p_value < sig_level),
                }

            elif test_type == "bootstrap":
                return self._bootstrap_test(scores_a, scores_b)

            else:
                return {"test": "unknown", "reason": f"unknown test type: {test_type}", "p_value": 1.0}

        except ImportError:
            return {"test": "none", "reason": "scipy_not_installed", "p_value": 1.0}

    def _bootstrap_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> Dict[str, Any]:
        """Paired bootstrap significance test.

        Args:
            scores_a: Scores for config A.
            scores_b: Scores for config B.

        Returns:
            Dict with bootstrap test results.
        """
        n_iters = self._exp_cfg.get("bootstrap_iterations", 1000)
        sig_level = self._exp_cfg.get("significance_level", 0.05)
        n = len(scores_a)

        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        count_more_extreme = 0

        rng = np.random.default_rng(seed=42)
        for _ in range(n_iters):
            indices = rng.integers(0, n, size=n)
            boot_a = [scores_a[i] for i in indices]
            boot_b = [scores_b[i] for i in indices]
            boot_diff = np.mean(boot_a) - np.mean(boot_b)
            if abs(boot_diff) >= abs(observed_diff):
                count_more_extreme += 1

        p_value = count_more_extreme / n_iters

        return {
            "test": "bootstrap",
            "observed_diff": round(float(observed_diff), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < sig_level,
            "n_iterations": n_iters,
        }
