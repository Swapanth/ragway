"""Latency evaluator for measuring and scoring pipeline stage durations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from ragway.schema.node import Node


T = TypeVar("T")


@dataclass(slots=True)
class LatencyEval:
    """Tracks wall-clock stage latency and converts latency to a normalized score."""

    stage_latencies: dict[str, float] = field(default_factory=dict)

    def measure(self, stage_name: str, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Measure one stage call and store elapsed seconds for the stage."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.stage_latencies[stage_name] = elapsed
        return result

    def evaluate(self, question: str, answer: str, context: list[Node]) -> float:
        """Return normalized latency score in [0.0, 1.0], higher is faster."""
        del question, answer, context
        if not self.stage_latencies:
            return 0.0

        average_seconds = sum(self.stage_latencies.values()) / len(self.stage_latencies)
        # Converts unbounded latency to bounded score; 0s -> 1.0, larger latencies approach 0.0.
        return self._clamp(1.0 / (1.0 + average_seconds))

    def _clamp(self, value: float) -> float:
        """Clamp a numeric score into the inclusive range [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

