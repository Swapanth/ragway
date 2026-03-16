"""Pipeline debugging utilities for tracing RAG execution details."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ragway.schema.node import Node


@dataclass(slots=True)
class DebugTrace:
    """Structured trace collected from one debug pipeline execution."""

    query: str
    retrieved_nodes: list[Node] = field(default_factory=list)
    reranked_nodes: list[Node] = field(default_factory=list)
    prompt_used: str = ""
    answer: str = ""
    latency_per_step: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RAGDebugger:
    """Wraps a pipeline runner and captures intermediate debug artifacts."""

    run_pipeline: Callable[[str], object]
    retrieve_hook: Callable[[str], list[Node] | Awaitable[list[Node]]] | None = None
    rerank_hook: Callable[[str, list[Node]], list[Node] | Awaitable[list[Node]]] | None = None
    prompt_hook: Callable[[str, list[Node]], str] | None = None

    def debug_run(self, query: str) -> DebugTrace:
        """Run pipeline with optional hooks and return a populated debug trace."""
        trace = DebugTrace(query=query)

        if self.retrieve_hook is not None:
            started = time.perf_counter()
            retrieved = self._resolve(self.retrieve_hook(query))
            trace.retrieved_nodes = list(retrieved)
            trace.latency_per_step["retrieve"] = time.perf_counter() - started

        if self.rerank_hook is not None:
            started = time.perf_counter()
            reranked = self._resolve(self.rerank_hook(query, trace.retrieved_nodes))
            trace.reranked_nodes = list(reranked)
            trace.latency_per_step["rerank"] = time.perf_counter() - started
        else:
            trace.reranked_nodes = list(trace.retrieved_nodes)

        if self.prompt_hook is not None:
            started = time.perf_counter()
            trace.prompt_used = self.prompt_hook(query, trace.reranked_nodes)
            trace.latency_per_step["prompt"] = time.perf_counter() - started

        started = time.perf_counter()
        output = self._resolve(self.run_pipeline(query))
        trace.latency_per_step["generate"] = time.perf_counter() - started

        if isinstance(output, dict):
            trace.answer = str(output.get("answer", ""))
        else:
            trace.answer = str(output)

        return trace

    def _resolve(self, value: object) -> object:
        """Resolve awaitable values in sync contexts."""
        if asyncio.iscoroutine(value):
            return asyncio.run(value)
        return value

