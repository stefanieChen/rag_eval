"""Central logging configuration for the RAG evaluation suite."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

_INITIALIZED = False
_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"


def _load_logging_config() -> Dict[str, Any]:
    try:
        from src.llm_judge.judge_base import load_eval_config  # Local import to avoid cycles

        config = load_eval_config() or {}
        return config.get("logging", {})
    except Exception:
        return {}


def setup_logging(config: Optional[Dict[str, Any]] = None, *, reset: bool = False) -> None:
    """Initialise logging for the evaluation suite.

    Parameters
    ----------
    config:
        Optional logging configuration dictionary. When omitted, the configuration
        is sourced from ``config/eval_config.yaml`` if available.
    reset:
        When True, reinitialise logging even if it has already been configured.
    """

    global _INITIALIZED
    if _INITIALIZED and not reset:
        return

    cfg = dict(_load_logging_config())
    if config:
        cfg.update(config)

    log_dir = Path(cfg.get("log_dir", "logs"))
    level_name = str(cfg.get("level", "INFO")).upper()
    log_format = cfg.get("format", _DEFAULT_FORMAT)
    max_bytes = int(cfg.get("rotation", {}).get("max_bytes", 5 * 1024 * 1024))
    backup_count = int(cfg.get("rotation", {}).get("backup_count", 3))

    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(log_format)

    root = logging.getLogger("rag_eval")
    root.setLevel(level)

    if root.handlers:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_path = log_dir / "evaluation.log"
    file_handler = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Optional per-module file handlers
    per_logger_cfg = cfg.get("modules", {})
    for module_name, module_settings in per_logger_cfg.items():
        module_level = getattr(logging, str(module_settings.get("level", level_name)).upper(), level)
        module_file = module_settings.get("file")
        if not module_file:
            continue
        module_path = log_dir / module_file
        module_handler = RotatingFileHandler(
            module_path,
            maxBytes=int(module_settings.get("max_bytes", max_bytes)),
            backupCount=int(module_settings.get("backup_count", backup_count)),
            encoding="utf-8",
        )
        module_handler.setLevel(module_level)
        module_handler.setFormatter(formatter)
        module_logger = logging.getLogger(f"rag_eval.{module_name}")
        module_logger.setLevel(module_level)
        module_logger.addHandler(module_handler)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger within the ``rag_eval`` namespace."""

    setup_logging()
    return logging.getLogger(f"rag_eval.{name}")
