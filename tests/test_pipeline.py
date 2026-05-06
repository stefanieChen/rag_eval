"""Unit tests for the evaluation pipeline module.

Tests evaluator orchestration logic and schema validation.
No live RAG queries — tests use static mode with mock data.
"""

import pytest

from src.datasets.schema import EvalResult, EvalRunSummary, EvalTestCase, RAGResponse, TestCase
from src.pipeline.client_base import RAGClientBase


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


class TestRAGClientBase:
    """Tests for the abstract RAG client interface and concrete implementations."""

    def test_base_is_abstract(self):
        """RAGClientBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RAGClientBase()

    def test_custom_client_implements_interface(self):
        """A minimal custom client can implement the interface."""

        class MockRAGClient(RAGClientBase):
            def query(self, question: str) -> RAGResponse:
                return RAGResponse(answer=f"Mock: {question}")

        client = MockRAGClient()
        resp = client.query("hello")
        assert resp.answer == "Mock: hello"
        assert client.supports_config_override is False
        assert client.get_config() == {}

    def test_query_with_config_default_fallback(self):
        """Default query_with_config falls back to query."""

        class SimpleClient(RAGClientBase):
            def query(self, question: str) -> RAGResponse:
                return RAGResponse(answer="simple")

        client = SimpleClient()
        resp = client.query_with_config("q", {"override": True})
        assert resp.answer == "simple"

    def test_rag2_client_alias(self):
        """RAGClient alias still points to Rag2Client."""
        from src.pipeline.rag_client import RAGClient, Rag2Client
        assert RAGClient is Rag2Client

    def test_rag2_client_is_subclass(self):
        """Rag2Client inherits from RAGClientBase."""
        from src.pipeline.rag_client import Rag2Client
        assert issubclass(Rag2Client, RAGClientBase)

    def test_http_client_is_subclass(self):
        """HttpRAGClient inherits from RAGClientBase."""
        from src.pipeline.http_client import HttpRAGClient
        assert issubclass(HttpRAGClient, RAGClientBase)


class TestHttpRAGClient:
    """Tests for the HTTP RAG client."""

    def test_extract_field(self):
        from src.pipeline.http_client import HttpRAGClient
        data = {"a": {"b": {"c": 42}}}
        assert HttpRAGClient._extract_field(data, "a.b.c") == 42
        assert HttpRAGClient._extract_field(data, "a.b") == {"c": 42}
        assert HttpRAGClient._extract_field(data, "x.y") is None

    def test_extract_list_field_simple(self):
        from src.pipeline.http_client import HttpRAGClient
        data = {"items": ["a", "b", "c"]}
        assert HttpRAGClient._extract_list_field(data, "items") == ["a", "b", "c"]

    def test_extract_list_field_bracket_notation(self):
        from src.pipeline.http_client import HttpRAGClient
        data = {"sources": [
            {"content": "doc1", "id": "1"},
            {"content": "doc2", "id": "2"},
        ]}
        assert HttpRAGClient._extract_list_field(data, "sources[].content") == ["doc1", "doc2"]
        assert HttpRAGClient._extract_list_field(data, "sources[].id") == ["1", "2"]

    def test_extract_list_field_missing(self):
        from src.pipeline.http_client import HttpRAGClient
        data = {"other": 1}
        assert HttpRAGClient._extract_list_field(data, "missing") == []
        assert HttpRAGClient._extract_list_field(data, "missing[].x") == []

    def test_build_request_body(self):
        from src.pipeline.http_client import HttpRAGClient
        config = {"rag": {"http": {
            "request_body_template": {"query": "{question}", "top_k": 5},
        }}}
        client = HttpRAGClient(config=config)
        body = client._build_request_body("What is RAG?")
        assert body["query"] == "What is RAG?"
        assert body["top_k"] == 5

    def test_parse_response(self):
        from src.pipeline.http_client import HttpRAGClient
        config = {"rag": {"http": {
            "response_mapping": {
                "answer": "result.text",
                "contexts": "result.docs[].content",
                "context_ids": "result.docs[].id",
            },
        }}}
        client = HttpRAGClient(config=config)
        data = {
            "result": {
                "text": "RAG is retrieval-augmented generation",
                "docs": [
                    {"content": "doc1 content", "id": "d1"},
                    {"content": "doc2 content", "id": "d2"},
                ],
            }
        }
        resp = client._parse_response(data, latency_ms=100.0)
        assert resp.answer == "RAG is retrieval-augmented generation"
        assert resp.retrieved_contexts == ["doc1 content", "doc2 content"]
        assert resp.retrieved_ids == ["d1", "d2"]
        assert resp.latency_ms == 100.0


class TestClientFactory:
    """Tests for the client factory."""

    def test_factory_returns_rag2_by_default(self):
        from src.pipeline.client_factory import create_rag_client
        from src.pipeline.rag_client import Rag2Client
        client = create_rag_client()
        assert isinstance(client, Rag2Client)

    def test_factory_returns_http_client(self):
        from src.pipeline.client_factory import create_rag_client
        from src.pipeline.http_client import HttpRAGClient
        client = create_rag_client(client_type="http")
        assert isinstance(client, HttpRAGClient)

    def test_factory_unknown_type_raises(self):
        from src.pipeline.client_factory import create_rag_client
        with pytest.raises(ValueError, match="Unknown RAG client type"):
            create_rag_client(client_type="nonexistent")

    def test_factory_respects_config(self):
        from src.pipeline.client_factory import create_rag_client
        from src.pipeline.http_client import HttpRAGClient
        config = {"rag": {"client_type": "http", "http": {"base_url": "http://test:9999"}}}
        client = create_rag_client(config=config)
        assert isinstance(client, HttpRAGClient)
        assert client._base_url == "http://test:9999"

    def test_factory_explicit_type_overrides_config(self):
        from src.pipeline.client_factory import create_rag_client
        from src.pipeline.rag_client import Rag2Client
        config = {"rag": {"client_type": "http"}}
        client = create_rag_client(config=config, client_type="rag2")
        assert isinstance(client, Rag2Client)

    def test_evaluator_accepts_custom_client(self):
        """Evaluator can be initialized with a custom RAGClientBase."""

        class StubClient(RAGClientBase):
            def query(self, question: str) -> RAGResponse:
                return RAGResponse(answer="stub answer", retrieved_contexts=["ctx"])

        from src.pipeline.evaluator import Evaluator
        evaluator = Evaluator(rag_client=StubClient())
        assert evaluator._rag_client is not None
