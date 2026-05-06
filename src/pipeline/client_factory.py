"""Factory for creating RAG client instances based on configuration.

Centralizes client selection so that Evaluator, ExperimentRunner, and CLI
do not need to know which concrete client class to import.
"""

from typing import Any, Dict, Optional

from src.llm_judge.judge_base import load_eval_config
from src.logging import get_logger
from src.pipeline.client_base import RAGClientBase

logger = get_logger("pipeline.client_factory")

# Registry of built-in client types → (module_path, class_name)
_BUILTIN_CLIENTS = {
    "rag2": ("src.pipeline.rag_client", "Rag2Client"),
    "http": ("src.pipeline.http_client", "HttpRAGClient"),
}


def create_rag_client(
    config: Optional[Dict[str, Any]] = None,
    client_type: Optional[str] = None,
) -> RAGClientBase:
    """Create and return a RAGClientBase instance.

    Resolution order for *client_type*:
    1. Explicit ``client_type`` argument.
    2. ``config["rag"]["client_type"]``.
    3. Falls back to ``"rag2"`` for backward compatibility.

    Args:
        config: Full eval config dict. Loaded from file if None.
        client_type: Override the client type from config.

    Returns:
        A concrete RAGClientBase instance.

    Raises:
        ValueError: If the requested client_type is unknown.
    """
    if config is None:
        config = load_eval_config()

    if client_type is None:
        client_type = config.get("rag", {}).get("client_type", "rag2")

    logger.info("Creating RAG client", extra={"client_type": client_type})

    if client_type in _BUILTIN_CLIENTS:
        module_path, class_name = _BUILTIN_CLIENTS[client_type]
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(config=config)

    # Allow fully-qualified class path for custom clients
    # e.g. "my_package.my_module.MyRAGClient"
    if "." in client_type:
        return _load_custom_client(client_type, config)

    raise ValueError(
        f"Unknown RAG client type: '{client_type}'. "
        f"Built-in types: {sorted(_BUILTIN_CLIENTS.keys())}. "
        f"Or provide a fully-qualified class path."
    )


def _load_custom_client(
    class_path: str,
    config: Dict[str, Any],
) -> RAGClientBase:
    """Load a custom RAG client from a fully-qualified class path.

    Args:
        class_path: Dotted path like ``my_package.module.ClassName``.
        config: Full eval config dict.

    Returns:
        Instance of the custom client.

    Raises:
        ImportError: If the module cannot be imported.
        TypeError: If the class does not subclass RAGClientBase.
    """
    import importlib

    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid class path: '{class_path}'. Expected 'module.ClassName'.")

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not (isinstance(cls, type) and issubclass(cls, RAGClientBase)):
        raise TypeError(
            f"{class_path} does not subclass RAGClientBase. "
            f"Custom clients must inherit from src.pipeline.client_base.RAGClientBase."
        )

    logger.info("Loaded custom RAG client", extra={"class_path": class_path})
    return cls(config=config)
