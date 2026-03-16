from __future__ import annotations

import asyncio

from ragway.retrieval.bm25_retriever import BM25Retriever
from ragway.schema.node import Node


async def test_bm25_retriever_ranks_relevant_document_first() -> None:
    """BM25 should rank the document with stronger term match first."""
    nodes = [
        Node(node_id="n1", doc_id="d1", content="apple orange apple"),
        Node(node_id="n2", doc_id="d2", content="banana pear"),
    ]
    retriever = BM25Retriever(nodes=nodes)

    result = await retriever.retrieve("apple", top_k=2)

    assert len(result) >= 1
    assert result[0].node_id == "n1"

