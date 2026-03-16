from __future__ import annotations

import asyncio

from ragway.retrieval.hybrid_retriever import HybridRetriever
from ragway.schema.node import Node


class _StubRetriever:
    def __init__(self, output: list[Node]) -> None:
        self._output = output

    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        del query
        return self._output[:top_k]


async def test_hybrid_retriever_uses_rrf_merge() -> None:
    """Hybrid retriever should fuse BM25 and vector rankings using RRF."""
    n1 = Node(node_id="n1", doc_id="d1", content="one")
    n2 = Node(node_id="n2", doc_id="d2", content="two")
    n3 = Node(node_id="n3", doc_id="d3", content="three")

    bm25 = _StubRetriever([n1, n2])
    vector = _StubRetriever([n2, n3])

    retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector, rrf_k=60)
    result = await retriever.retrieve("q", top_k=3)

    assert [node.node_id for node in result] == ["n2", "n1", "n3"]

