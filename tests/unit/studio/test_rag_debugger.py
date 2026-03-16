from __future__ import annotations

from ragway.schema.node import Node
from studio.rag_debugger import DebugTrace, RAGDebugger


def _run_pipeline(query: str) -> str:
    del query
    return "final answer"


def _retrieve_hook(query: str) -> list[Node]:
    return [Node(node_id="n1", doc_id="d1", content=f"ctx for {query}")]


def _rerank_hook(query: str, nodes: list[Node]) -> list[Node]:
    del query
    return list(reversed(nodes))


def _prompt_hook(query: str, nodes: list[Node]) -> str:
    return f"Q:{query} N:{len(nodes)}"


def test_rag_debugger_returns_debug_trace() -> None:
    """RAGDebugger should return populated DebugTrace with step latencies."""
    debugger = RAGDebugger(
        run_pipeline=_run_pipeline,
        retrieve_hook=_retrieve_hook,
        rerank_hook=_rerank_hook,
        prompt_hook=_prompt_hook,
    )

    trace = debugger.debug_run("what is rag")

    assert isinstance(trace, DebugTrace)
    assert trace.query == "what is rag"
    assert trace.answer == "final answer"
    assert trace.prompt_used.startswith("Q:what is rag")
    assert "retrieve" in trace.latency_per_step
    assert "generate" in trace.latency_per_step

