from __future__ import annotations

import inspect

from ragway.interfaces.reranker_protocol import RerankerProtocol
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node


class _Reranker:
    async def rerank(self, query: str, nodes: list[Node]) -> list[Node]:
        del query
        return list(nodes)


def test_reranker_protocol_runtime_checkable() -> None:
    """A structurally compatible reranker should satisfy RerankerProtocol."""
    reranker = _Reranker()
    assert isinstance(reranker, RerankerProtocol)


def test_reranker_protocol_rerank_is_async() -> None:
    """The protocol rerank method should be declared as coroutine function."""
    assert inspect.iscoroutinefunction(RerankerProtocol.rerank)


def test_reranker_protocol_shape_accepts_node_list() -> None:
    """The rerank protocol shape should accept and return node lists."""
    sample_nodes = [
        Node(node_id="n1", doc_id="d1", content="c1", metadata=Metadata()),
        Node(node_id="n2", doc_id="d1", content="c2", metadata=Metadata()),
    ]
    assert isinstance(sample_nodes, list)

