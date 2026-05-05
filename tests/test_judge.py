"""Unit tests for LLM Judge module.

Tests prompt rendering, JSON parsing, rubric loading, and result schemas.
All tests run without LLM calls (mocked).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm_judge.judge_base import (
    JudgeResult,
    LLMJudge,
    PairwiseResult,
    ReferenceResult,
    _get_project_root,
    load_eval_config,
    load_rubric,
)


class TestJudgeBase:
    """Tests for judge_base.py utilities."""

    def test_project_root_exists(self):
        root = _get_project_root()
        assert root.exists()
        assert (root / "config").exists()

    def test_load_eval_config(self):
        config = load_eval_config()
        assert "judge" in config
        assert "provider" in config["judge"]

    def test_load_rubric_faithfulness(self):
        rubric = load_rubric("faithfulness")
        assert rubric["name"] == "faithfulness"
        assert "criteria" in rubric
        assert len(rubric["criteria"]) > 0

    def test_load_rubric_relevancy(self):
        rubric = load_rubric("relevancy")
        assert rubric["name"] == "relevancy"

    def test_load_rubric_completeness(self):
        rubric = load_rubric("completeness")
        assert rubric["name"] == "completeness"

    def test_load_rubric_hallucination(self):
        rubric = load_rubric("hallucination")
        assert rubric["name"] == "hallucination"

    def test_load_rubric_missing(self):
        with pytest.raises(FileNotFoundError):
            load_rubric("nonexistent_rubric")

    def test_judge_result_schema(self):
        result = JudgeResult(score=4.0, reasoning="Good answer")
        assert result.score == 4.0
        assert result.reasoning == "Good answer"

    def test_pairwise_result_schema(self):
        result = PairwiseResult(winner="A", score_a=4.0, score_b=3.0)
        assert result.winner == "A"
        assert result.score_a == 4.0

    def test_reference_result_schema(self):
        result = ReferenceResult(
            overall_score=4.0,
            factual_accuracy=5.0,
            completeness=3.0,
            missing_information=["detail X"],
        )
        assert result.overall_score == 4.0
        assert len(result.missing_information) == 1


class TestJudgeJsonParsing:
    """Tests for JSON response parsing."""

    def setup_method(self):
        """Create a judge instance with mocked LLM."""
        with patch("src.llm_judge.judge_base.LLMJudge._build_litellm_model", return_value="test"):
            self.judge = LLMJudge()

    def test_parse_clean_json(self):
        text = '{"score": 4, "reasoning": "test"}'
        result = self.judge._parse_json_response(text)
        assert result["score"] == 4

    def test_parse_json_in_code_block(self):
        text = '```json\n{"score": 3, "reasoning": "ok"}\n```'
        result = self.judge._parse_json_response(text)
        assert result["score"] == 3

    def test_parse_json_with_text_around(self):
        text = 'Here is my assessment:\n{"score": 5, "reasoning": "great"}\nEnd.'
        result = self.judge._parse_json_response(text)
        assert result["score"] == 5

    def test_parse_invalid_json(self):
        text = "This is not JSON at all"
        result = self.judge._parse_json_response(text)
        assert "error" in result


class TestPromptRendering:
    """Tests for Jinja2 template rendering."""

    def setup_method(self):
        with patch("src.llm_judge.judge_base.LLMJudge._build_litellm_model", return_value="test"):
            self.judge = LLMJudge()

    def test_render_pointwise_template(self):
        prompt = self.judge._render_template(
            "pointwise_score.j2",
            criterion_name="faithfulness",
            rubric_description="Test rubric",
            rubric_levels=[{"score": 5, "label": "Good", "description": "Very good"}],
            question="What is RAG?",
            answer="RAG is...",
            contexts=["Context 1", "Context 2"],
            ground_truth=None,
        )
        assert "faithfulness" in prompt
        assert "What is RAG?" in prompt
        assert "Context 1" in prompt

    def test_render_pairwise_template(self):
        prompt = self.judge._render_template(
            "pairwise_compare.j2",
            question="What is RAG?",
            answer_a="Answer A",
            answer_b="Answer B",
            contexts=["Context 1"],
            ground_truth="Reference",
        )
        assert "Answer A" in prompt
        assert "Answer B" in prompt

    def test_render_reference_template(self):
        prompt = self.judge._render_template(
            "reference_grade.j2",
            question="What is RAG?",
            answer="RAG is...",
            contexts=["Context 1"],
            ground_truth="RAG means...",
        )
        assert "Reference Answer" in prompt
        assert "RAG means..." in prompt
