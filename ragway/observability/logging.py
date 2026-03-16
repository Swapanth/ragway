"""Structured JSON logging utilities for rag-toolkit modules."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

_RESERVED_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class _JsonFormatter(logging.Formatter):
    """Formats log records as one-line JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON string representation of the provided log record."""
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_ATTRS and not key.startswith("_")
        }
        if extra_fields:
            payload["extra"] = extra_fields

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _resolve_level() -> int:
    """Resolve log level from RAG_LOG_LEVEL with INFO fallback."""
    level_name = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
    return int(getattr(logging, level_name, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a configured JSON logger for the provided logger name."""
    level = _resolve_level()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    existing_handler = None
    for handler in logger.handlers:
        if getattr(handler, "_rag_json_handler", False):
            existing_handler = handler
            break

    if existing_handler is None:
        handler = logging.StreamHandler()
        setattr(handler, "_rag_json_handler", True)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        existing_handler = handler

    existing_handler.setLevel(level)
    return logger
