"""OpenTelemetry-compatible trace collection for RAG pipeline spans."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True)
class RAGTracer:
    """Collects lightweight span traces for observability and debugging."""

    _spans: list[dict[str, object]] = field(default_factory=list)
    _active_stack: list[dict[str, object]] = field(default_factory=list)

    @contextmanager
    def trace(self, name: str) -> Iterator[dict[str, object]]:
        """Create a span context, record latency, and store the resulting trace."""
        parent_span = self._active_stack[-1] if self._active_stack else None
        trace_id = str(parent_span["trace_id"]) if parent_span else uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        start_unix_ns = time.time_ns()
        start_perf = time.perf_counter()

        span: dict[str, object] = {
            "name": name,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span["span_id"] if parent_span else None,
            "start_time_unix_ns": float(start_unix_ns),
        }

        self._active_stack.append(span)
        try:
            yield span
        except Exception as exc:
            span["status"] = "error"
            span["error"] = str(exc)
            raise
        else:
            span["status"] = "ok"
        finally:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            span["latency_ms"] = elapsed_ms
            span["end_time_unix_ns"] = float(time.time_ns())
            self._active_stack.pop()
            self._spans.append(dict(span))

    def export_traces(self) -> list[dict[str, object]]:
        """Return a copy of all recorded traces in insertion order."""
        return [dict(span) for span in self._spans]
