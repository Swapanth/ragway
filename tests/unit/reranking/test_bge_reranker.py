from __future__ import annotations

import asyncio

from ragway.reranking.bge_reranker import BGEReranker
from ragway.schema.node import Node


async def test_bge_reranker_orders_by_overlap() -> None:
    """BGEReranker should rank nodes with higher query overlap first."""
    reranker = BGEReranker()
    nodes = [
        Node(node_id="n1", doc_id="d1", content="banana pear"),
        Node(node_id="n2", doc_id="d1", content="apple orange apple"),
    ]

    result = await reranker.rerank("apple", nodes)

    assert [node.node_id for node in result] == ["n2", "n1"]

