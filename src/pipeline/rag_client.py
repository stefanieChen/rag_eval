"""RAG client for the rag_2 RAG pipeline.

Provides a concrete RAGClientBase implementation that imports and
drives the rag_2 project's RAGPipeline directly via the filesystem.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

from src.datasets.schema import RAGResponse
from src.llm_judge.judge_base import load_eval_config
from src.logging import get_logger
from src.pipeline.client_base import RAGClientBase


class Rag2Client(RAGClientBase):
    """Concrete client for the rag_2 RAG pipeline.

    Supports direct Python import (same machine) by adding the
    rag_2 project to sys.path and instantiating RAGPipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_eval_config()

        rag_cfg = config.get("rag", {})
        self._project_path = Path(rag_cfg.get("project_path", "../rag_2")).resolve()
        self._config_path = rag_cfg.get("config_path", None)
        self._pipeline = None
        self._rag_config = None
        self._module_cache: Dict[str, ModuleType] = {}
        self._logger = get_logger("pipeline.rag_client")

    def _ensure_pipeline(self, config_override: Optional[Dict[str, Any]] = None) -> None:
        """Lazily initialize the RAG pipeline.

        Args:
            config_override: Optional config values to override in the RAG system.
        """
        if self._pipeline is not None and config_override is None:
            return

        self._logger.info(
            "Initializing RAG pipeline",
            extra={
                "project_path": str(self._project_path),
                "config_path": self._config_path,
                "config_override": config_override,
            },
        )

        # Add rag_2 project to Python path
        project_str = str(self._project_path)
        if project_str not in sys.path:
            sys.path.insert(0, project_str)

        # Ensure the shared 'src' package can locate rag_2 modules
        rag2_src_path = self._project_path / "src"
        if rag2_src_path.exists():
            try:
                src_pkg = importlib.import_module("src")
            except ModuleNotFoundError:
                src_pkg = None
            if src_pkg is not None and hasattr(src_pkg, "__path__"):
                rag2_str = str(rag2_src_path)
                if rag2_str not in src_pkg.__path__:
                    src_pkg.__path__.append(rag2_str)

                # Also extend __path__ for already-loaded sub-packages
                # (e.g. src.logging) so rag_2's sub-modules are discoverable
                for sub_dir in rag2_src_path.iterdir():
                    if sub_dir.is_dir() and (sub_dir / "__init__.py").exists():
                        sub_pkg_name = f"src.{sub_dir.name}"
                        if sub_pkg_name in sys.modules:
                            sub_pkg = sys.modules[sub_pkg_name]
                            sub_str = str(sub_dir)
                            if hasattr(sub_pkg, "__path__") and sub_str not in sub_pkg.__path__:
                                sub_pkg.__path__.append(sub_str)

        config_module = self._load_rag2_module("rag2_config", rag2_src_path / "config.py")
        rag_load_config = getattr(config_module, "load_config")

        pipeline_module = self._load_rag2_module("rag2_pipeline", rag2_src_path / "pipeline.py")
        RAGPipeline = getattr(pipeline_module, "RAGPipeline")

        if self._config_path:
            config_path = Path(self._config_path)
            if not config_path.is_absolute():
                config_path = (self._project_path / config_path).resolve()
            self._rag_config = rag_load_config(str(config_path))
        else:
            self._rag_config = rag_load_config()

        if config_override:
            self._rag_config = _deep_merge(self._rag_config, config_override)

        self._pipeline = RAGPipeline(self._rag_config)
        self._logger.info(
            "RAG pipeline initialized",
            extra={
                "config_override": bool(config_override),
                "collections": self._rag_config.get("retrieval", {}).get("collections", "unknown"),
            },
        )

    def query(self, question: str) -> RAGResponse:
        """Query the RAG system and return a structured response.

        Args:
            question: The user question.

        Returns:
            RAGResponse with answer, retrieved contexts, and metadata.
        """
        import time

        self._ensure_pipeline()
        start = time.perf_counter()

        try:
            self._logger.debug("Dispatching query", extra={"question": question[:80]})
            result = self._pipeline.query(question)
            latency_ms = (time.perf_counter() - start) * 1000

            self._logger.info(
                "Query completed",
                extra={
                    "latency_ms": round(latency_ms, 2),
                    "sources": len(result.get("sources", [])),
                },
            )

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            contexts = []
            context_ids = []
            for src in sources:
                if isinstance(src, dict):
                    contexts.append(src.get("content", str(src)))
                    context_ids.append(src.get("id", src.get("source", "")))
                else:
                    contexts.append(str(src))

            return RAGResponse(
                answer=answer,
                retrieved_contexts=contexts,
                retrieved_ids=context_ids,
                latency_ms=latency_ms,
                metadata={"raw_result": result},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self._logger.exception(
                "RAG query failed",
                extra={"question": question[:80], "latency_ms": round(latency_ms, 2)},
            )
            return RAGResponse(
                answer=f"[Error] {e}",
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    def _load_rag2_module(self, cache_key: str, file_path: Path) -> ModuleType:
        """Load a rag_2 module from file, caching the module object."""
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]

        if not file_path.exists():
            raise FileNotFoundError(f"rag_2 module not found: {file_path}")

        spec = importlib.util.spec_from_file_location(cache_key, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[cache_key] = module
        spec.loader.exec_module(module)

        self._module_cache[cache_key] = module
        self._logger.debug("Loaded rag_2 module", extra={"cache_key": cache_key, "path": str(file_path)})
        return module

    def query_with_config(
        self,
        question: str,
        config_override: Dict[str, Any],
    ) -> RAGResponse:
        """Query the RAG system with a temporary config override (for A/B testing).

        Creates a new pipeline instance with the override applied.

        Args:
            question: The user question.
            config_override: Config values to override (e.g., {"retrieval.hybrid_mode": true}).

        Returns:
            RAGResponse from the overridden configuration.
        """
        # Parse dot-notation keys into nested dict
        nested_override = _unflatten_dict(config_override)

        self._logger.info(
            "Running query with override",
            extra={"override_keys": list(config_override.keys())},
        )

        # Force re-init with override
        old_pipeline = self._pipeline
        self._pipeline = None
        try:
            self._ensure_pipeline(config_override=nested_override)
            return self.query(question)
        finally:
            self._pipeline = old_pipeline

    def get_config(self) -> Dict[str, Any]:
        """Return the current RAG system configuration.

        Returns:
            RAG config dict.
        """
        self._ensure_pipeline()
        return self._rag_config or {}

    # Keep old name for backward compatibility
    get_rag_config = get_config

    @property
    def supports_config_override(self) -> bool:
        """Rag2Client supports runtime config overrides for A/B testing."""
        return True


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts. Override values take precedence.

    Args:
        base: Base dict.
        override: Override dict.

    Returns:
        Merged dict.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _unflatten_dict(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dot-notation keys to nested dict.

    Example: {"retrieval.hybrid_mode": true} → {"retrieval": {"hybrid_mode": true}}

    Args:
        flat: Dict with potentially dot-notation keys.

    Returns:
        Nested dict.
    """
    result: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


# Backward-compatible alias
RAGClient = Rag2Client
