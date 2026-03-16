"""In-memory counters and histograms for RAG observability metrics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RAGMetrics:
    """Tracks counters and numeric observations for pipeline monitoring."""

    _counters: dict[str, int] = field(default_factory=dict)
    _histograms: dict[str, list[float]] = field(default_factory=dict)

    def increment(self, metric: str) -> None:
        """Increment a counter metric by one."""
        self._counters[metric] = self._counters.get(metric, 0) + 1

    def record(self, metric: str, value: float) -> None:
        """Record a numeric value for a histogram metric."""
        bucket = self._histograms.setdefault(metric, [])
        bucket.append(float(value))

    def summary(self) -> dict[str, float]:
        """Return flattened aggregate metrics for counters and histograms."""
        results: dict[str, float] = {}

        for metric, count in self._counters.items():
            results[metric] = float(count)

        for metric, values in self._histograms.items():
            if not values:
                continue
            total = sum(values)
            results[f"{metric}.count"] = float(len(values))
            results[f"{metric}.sum"] = total
            results[f"{metric}.avg"] = total / float(len(values))
            results[f"{metric}.min"] = min(values)
            results[f"{metric}.max"] = max(values)

        return results
