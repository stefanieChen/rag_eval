"""Unit tests for metric modules.

Tests retrieval IR metrics (pure math), composite scoring, and dataset loading.
No LLM calls required.
"""

import math

import pytest

from src.metrics.retrieval import (
    aggregate_retrieval_metrics,
    average_precision,
    compute_all_retrieval_metrics,
    map_score,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.metrics.composite import (
    WEIGHT_PRESETS,
    compute_composite_batch,
    compute_composite_score,
)


class TestRetrievalMetrics:
    """Tests for classical IR metrics."""

    def test_recall_at_k_perfect(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_recall_at_k_partial(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c", "e"}
        assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)

    def test_recall_at_k_none(self):
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_recall_at_k_empty_relevant(self):
        assert recall_at_k(["a"], set(), 1) == 0.0

    def test_precision_at_k_perfect(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 2) == 1.0

    def test_precision_at_k_half(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_precision_at_k_zero(self):
        assert precision_at_k(["a"], set(), 0) == 0.0

    def test_mrr_first_hit(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert mrr(retrieved, relevant) == 1.0

    def test_mrr_second_hit(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert mrr(retrieved, relevant) == 0.5

    def test_mrr_no_hit(self):
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_ndcg_at_k_perfect(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        score = ndcg_at_k(retrieved, relevant, 2)
        assert score == pytest.approx(1.0)

    def test_ndcg_at_k_reversed(self):
        retrieved = ["x", "a"]
        relevant = {"a"}
        score = ndcg_at_k(retrieved, relevant, 2)
        # Ideal: a at position 1, actual: a at position 2
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert score == pytest.approx(expected)

    def test_average_precision(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        # AP = (1/1 + 2/3 + 3/5) / 3
        expected = (1.0 + 2/3 + 3/5) / 3
        assert average_precision(retrieved, relevant) == pytest.approx(expected)

    def test_average_precision_empty_relevant(self):
        assert average_precision(["a"], set()) == 0.0

    def test_map_score(self):
        q1_ret = ["a", "b"]
        q1_rel = {"a"}
        q2_ret = ["x", "a"]
        q2_rel = {"a"}
        score = map_score([q1_ret, q2_ret], [q1_rel, q2_rel])
        # AP1 = 1.0, AP2 = 0.5 → MAP = 0.75
        assert score == pytest.approx(0.75)

    def test_compute_all_retrieval_metrics(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        result = compute_all_retrieval_metrics(retrieved, relevant, k_values=[1, 3, 5])
        assert "recall@1" in result
        assert "precision@3" in result
        assert "ndcg@5" in result
        assert "mrr" in result

    def test_aggregate_retrieval_metrics(self):
        metrics = [
            {"recall@5": 1.0, "mrr": 1.0},
            {"recall@5": 0.5, "mrr": 0.5},
        ]
        agg = aggregate_retrieval_metrics(metrics)
        assert agg["recall@5"] == pytest.approx(0.75)
        assert agg["mrr"] == pytest.approx(0.75)


class TestCompositeMetrics:
    """Tests for composite scoring."""

    def test_balanced_preset(self):
        scores = {
            "faithfulness": 4.0,
            "relevancy": 3.0,
            "completeness": 4.0,
            "hallucination_score": 0.8,
            "semantic_similarity": 0.7,
        }
        result = compute_composite_score(scores, preset="balanced")
        assert "composite_score" in result
        assert 0 < result["composite_score"] <= 5.0

    def test_custom_weights(self):
        scores = {"faithfulness": 5.0, "relevancy": 1.0}
        weights = {"faithfulness": 0.9, "relevancy": 0.1}
        result = compute_composite_score(scores, weights=weights)
        # Should be heavily weighted toward faithfulness
        assert result["composite_score"] > 4.0

    def test_empty_scores(self):
        result = compute_composite_score({})
        assert result["composite_score"] == 0.0

    def test_presets_exist(self):
        assert "balanced" in WEIGHT_PRESETS
        assert "faithfulness_focused" in WEIGHT_PRESETS
        assert "user_experience" in WEIGHT_PRESETS

    def test_composite_batch(self):
        cases = [
            {"faithfulness": 4.0, "relevancy": 3.0},
            {"faithfulness": 5.0, "relevancy": 5.0},
        ]
        result = compute_composite_batch(cases)
        assert result["num_cases"] == 2
        assert result["mean_composite"] > 0


class TestDatasetLoader:
    """Tests for dataset loading and schema validation."""

    def test_load_sample_test_set(self):
        from src.datasets.loader import load_test_set
        cases = load_test_set("data/test_sets/sample.json")
        assert len(cases) > 0
        assert cases[0].question != ""
        assert cases[0].ground_truth != ""

    def test_test_case_schema(self):
        from src.datasets.schema import TestCase
        tc = TestCase(question="Q", ground_truth="A", contexts=["C1"])
        assert tc.question == "Q"
        assert len(tc.contexts) == 1

    def test_eval_test_case_schema(self):
        from src.datasets.schema import EvalTestCase, RAGResponse, TestCase
        tc = TestCase(question="Q", ground_truth="A")
        resp = RAGResponse(answer="ans", retrieved_contexts=["c1"])
        etc = EvalTestCase(test_case=tc, rag_response=resp)
        assert etc.question == "Q"
        assert etc.answer == "ans"

    def test_load_nonexistent_file(self):
        from src.datasets.loader import load_test_set
        with pytest.raises(FileNotFoundError):
            load_test_set("/nonexistent/path.json")

    def test_load_unsupported_format(self):
        from src.datasets.loader import load_test_set
        with pytest.raises(ValueError, match="Unsupported"):
            load_test_set("test.xyz")
