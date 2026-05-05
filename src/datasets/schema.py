"""Pydantic schemas for evaluation test cases and results."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single evaluation test case with question, ground truth, and contexts."""

    question: str = Field(..., description="The user question")
    ground_truth: str = Field(default="", description="Expected/reference answer")
    contexts: List[str] = Field(
        default_factory=list,
        description="Ground-truth relevant context chunks (for static evaluation)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tags, difficulty, category, etc.)",
    )


class RAGResponse(BaseModel):
    """Response from a RAG system query."""

    answer: str = Field(default="", description="Generated answer text")
    retrieved_contexts: List[str] = Field(
        default_factory=list, description="Context chunks retrieved by the system"
    )
    retrieved_ids: List[str] = Field(
        default_factory=list, description="IDs/keys of retrieved documents"
    )
    latency_ms: float = Field(default=0.0, description="Query latency in ms")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )


class EvalTestCase(BaseModel):
    """Combined test case with both ground truth and RAG response for evaluation."""

    test_case: TestCase
    rag_response: RAGResponse = Field(default_factory=RAGResponse)

    @property
    def question(self) -> str:
        return self.test_case.question

    @property
    def ground_truth(self) -> str:
        return self.test_case.ground_truth

    @property
    def gt_contexts(self) -> List[str]:
        return self.test_case.contexts

    @property
    def answer(self) -> str:
        return self.rag_response.answer

    @property
    def retrieved_contexts(self) -> List[str]:
        return self.rag_response.retrieved_contexts


class EvalResult(BaseModel):
    """Full evaluation result for a single test case."""

    question: str
    answer: str = ""
    ground_truth: str = ""
    retrieved_contexts: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    judge_reasoning: Dict[str, str] = Field(default_factory=dict)
    hallucination: Optional[Dict[str, Any]] = None
    semantic: Optional[Dict[str, float]] = None
    retrieval: Optional[Dict[str, float]] = None
    composite_score: Optional[float] = None
    latency_ms: float = 0.0


class EvalRunSummary(BaseModel):
    """Summary of an evaluation run across all test cases."""

    run_id: str = ""
    timestamp: str = ""
    num_cases: int = 0
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    aggregated_scores: Dict[str, float] = Field(default_factory=dict)
    per_case_results: List[EvalResult] = Field(default_factory=list)
    total_latency_ms: float = 0.0
