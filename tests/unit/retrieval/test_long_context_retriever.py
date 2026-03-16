from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from ragway.retrieval.long_context_retriever import LongContextRetriever
from ragway.schema.node import Node


async def test_long_context_retriever_sorts_by_doc_and_position() -> None:
    """Long-context retriever should sort retrieved nodes by positional order."""
    wrapped = AsyncMock()
    wrapped.retrieve.return_value = [
        Node(node_id="n3", doc_id="d1", content="c3", position=3),
        Node(node_id="n1", doc_id="d1", content="c1", position=1),
        Node(node_id="n2", doc_id="d2", content="c2", position=2),
    ]

    retriever = LongContextRetriever(retriever=wrapped)
    result = await retriever.retrieve("q", top_k=3)

    assert [node.node_id for node in result] == ["n1", "n3", "n2"]

