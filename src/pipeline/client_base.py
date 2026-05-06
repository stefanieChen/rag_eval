"""Abstract base class for RAG system clients.

Any RAG system can be evaluated by implementing this interface.
The evaluation framework (Evaluator, ExperimentRunner) depends only
on RAGClientBase, not on any concrete RAG implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.datasets.schema import RAGResponse


class RAGClientBase(ABC):
    """Abstract interface that any RAG system client must implement."""

    @abstractmethod
    def query(self, question: str) -> RAGResponse:
        """Query the RAG system and return a structured response.

        Args:
            question: The user question.

        Returns:
            RAGResponse with answer, retrieved contexts, and metadata.
        """
        ...

    def query_with_config(
        self,
        question: str,
        config_override: Dict[str, Any],
    ) -> RAGResponse:
        """Query with a temporary config override (for A/B testing).

        Default implementation ignores the override and falls back to
        a normal query.  Subclasses that support runtime config changes
        should override this method.

        Args:
            question: The user question.
            config_override: Config values to override.

        Returns:
            RAGResponse from the (possibly overridden) configuration.
        """
        return self.query(question)

    def get_config(self) -> Dict[str, Any]:
        """Return the current RAG system configuration.

        Returns:
            RAG config dict. Empty dict by default.
        """
        return {}

    @property
    def supports_config_override(self) -> bool:
        """Whether this client supports runtime config overrides for A/B testing."""
        return False
