from __future__ import annotations

import time

import pytest

from ragway.observability.tracing import RAGTracer


def test_trace_records_latency_and_exports_span() -> None:
    """RAGTracer should record a successful span with latency metadata."""
    tracer = RAGTracer()

    with tracer.trace("retrieve") as span:
        assert span["name"] == "retrieve"
        time.sleep(0.002)

    traces = tracer.export_traces()

    assert len(traces) == 1
    assert traces[0]["name"] == "retrieve"
    assert traces[0]["status"] == "ok"
    assert float(traces[0]["latency_ms"]) >= 0.0
    assert traces[0]["trace_id"]
    assert traces[0]["span_id"]


def test_trace_records_error_status_when_exception_raised() -> None:
    """RAGTracer should mark the span as error when traced code raises."""
    tracer = RAGTracer()

    with pytest.raises(RuntimeError):
        with tracer.trace("generate"):
            raise RuntimeError("boom")

    traces = tracer.export_traces()

    assert len(traces) == 1
    assert traces[0]["name"] == "generate"
    assert traces[0]["status"] == "error"
    assert "boom" in str(traces[0]["error"])

