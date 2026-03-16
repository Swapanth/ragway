from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from ragway.retrieval.multi_query_retriever import MultiQueryRetriever
from ragway.schema.node import Node


async def test_multi_query_retriever_merges_query_variants() -> None:
    """Multi-query retriever should expand and fuse variant retrieval results."""
    retriever = AsyncMock()
    llm = AsyncMock()

    n1 = Node(node_id="n1", doc_id="d1", content="one")
    n2 = Node(node_id="n2", doc_id="d2", content="two")
    llm.generate.return_value = "variant one\nvariant two"

    retriever.retrieve.side_effect = [
        [n1],
        [n1, n2],
        [n2],
    ]

    multi = MultiQueryRetriever(retriever=retriever, llm=llm, query_count=3)
    result = await multi.retrieve("original", top_k=2)

    assert [node.node_id for node in result] == ["n1", "n2"]
    assert retriever.retrieve.await_count == 3

