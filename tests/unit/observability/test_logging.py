from __future__ import annotations

import json
import logging

from ragway.observability.logging import get_logger


def test_get_logger_respects_env_level(monkeypatch) -> None:
    """get_logger should honor RAG_LOG_LEVEL for logger level configuration."""
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")

    logger = get_logger("rag.tests.observability.logging.level")

    assert logger.level == logging.DEBUG


def test_get_logger_formats_json_and_reuses_handler(monkeypatch) -> None:
    """get_logger should emit JSON payloads and avoid duplicate handlers."""
    monkeypatch.setenv("RAG_LOG_LEVEL", "INFO")
    name = "rag.tests.observability.logging.json"

    logger_first = get_logger(name)
    logger_second = get_logger(name)

    assert len(logger_second.handlers) == 1
    formatter = logger_second.handlers[0].formatter
    assert formatter is not None

    record = logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="structured message",
        args=(),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == name
    assert payload["message"] == "structured message"
    assert "timestamp" in payload
    assert logger_first is logger_second

