"""Unit tests for the evaluation pipeline module.

Tests evaluator orchestration logic and schema validation.
No live RAG queries — tests use static mode with mock data.
"""

import pytest

from src.datasets.schema import EvalResult, EvalRunSummary, EvalTestCase, RAGResponse, TestCase


class TestEvalResult:
    """Tests for EvalResult schema."""

    def test_basic_eval_result(self):
        result = EvalResult(
            question="What is RAG?",
            answer="RAG is...",
            ground_truth="RAG means...",
            scores={"faithfulness": 4.0, "relevancy": 3.5},
            composite_score=3.8,
        )
        assert result.question == "What is RAG?"
        assert result.scores["faithfulness"] == 4.0
        assert result.composite_score == 3.8

    def test_eval_result_defaults(self):
        result = EvalResult(question="Q")
        assert result.answer == ""
        assert result.scores == {}
        assert result.composite_score is None

    def test_eval_run_summary(self):
        summary = EvalRunSummary(
            run_id="abc123",
            num_cases=5,
            aggregated_scores={"faithfulness": 4.2},
        )
        assert summary.run_id == "abc123"
        assert summary.num_cases == 5


class TestEvalTestCase:
    """Tests for evaluation test case schema."""

    def test_properties(self):
        tc = TestCase(
            question="Q",
            ground_truth="GT",
            contexts=["c1", "c2"],
        )
        resp = RAGResponse(
            answer="A",
            retrieved_contexts=["r1"],
        )
        etc = EvalTestCase(test_case=tc, rag_response=resp)

        assert etc.question == "Q"
        assert etc.ground_truth == "GT"
        assert etc.gt_contexts == ["c1", "c2"]
        assert etc.answer == "A"
        assert etc.retrieved_contexts == ["r1"]


class TestEvaluatorConfig:
    """Tests for evaluator configuration loading."""

    def test_evaluator_init(self):
        from src.llm_judge.judge_base import load_eval_config
        config = load_eval_config()
        from src.pipeline.evaluator import Evaluator
        evaluator = Evaluator(config=config)
        assert evaluator._config is not None

    def test_default_metrics_from_config(self):
        from src.llm_judge.judge_base import load_eval_config
        config = load_eval_config()
        default_metrics = config.get("evaluation", {}).get("default_metrics", [])
        assert len(default_metrics) > 0
        assert "faithfulness" in default_metrics


class TestRAGClientConfig:
    """Tests for RAG client configuration."""

    def test_unflatten_dict(self):
        from src.pipeline.rag_client import _unflatten_dict
        flat = {"retrieval.hybrid_mode": True, "retrieval.top_k": 10}
        nested = _unflatten_dict(flat)
        assert nested["retrieval"]["hybrid_mode"] is True
        assert nested["retrieval"]["top_k"] == 10

    def test_deep_merge(self):
        from src.pipeline.rag_client import _deep_merge
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 99}, "e": 4}
        result = _deep_merge(base, override)
        assert result["a"]["b"] == 99
        assert result["a"]["c"] == 2
        assert result["d"] == 3
        assert result["e"] == 4
