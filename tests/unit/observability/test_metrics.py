from __future__ import annotations

from ragway.observability.metrics import RAGMetrics


def test_metrics_increment_and_summary_for_counters() -> None:
    """RAGMetrics should expose counters as numeric summary entries."""
    metrics = RAGMetrics()

    metrics.increment("queries")
    metrics.increment("queries")

    summary = metrics.summary()

    assert summary["queries"] == 2.0


def test_metrics_record_and_summary_for_histograms() -> None:
    """RAGMetrics should summarize histogram metrics with aggregate values."""
    metrics = RAGMetrics()

    metrics.record("latency_ms", 10.0)
    metrics.record("latency_ms", 30.0)

    summary = metrics.summary()

    assert summary["latency_ms.count"] == 2.0
    assert summary["latency_ms.sum"] == 40.0
    assert summary["latency_ms.avg"] == 20.0
    assert summary["latency_ms.min"] == 10.0
    assert summary["latency_ms.max"] == 30.0

