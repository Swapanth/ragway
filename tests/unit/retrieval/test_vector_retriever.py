from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from ragway.retrieval.vector_retriever import VectorRetriever
from ragway.schema.node import Node


async def test_vector_retriever_embeds_then_searches() -> None:
    """Vector retriever should call embed and vector search in sequence."""
    embedding = AsyncMock()
    embedding.embed.return_value = [[0.1, 0.9]]

    store = AsyncMock()
    store.search.return_value = [Node(node_id="n1", doc_id="d1", content="alpha", embedding=[0.1, 0.9])]

    retriever = VectorRetriever(embedding_model=embedding, vector_store=store)
    result = await retriever.retrieve("hello", top_k=1)

    assert len(result) == 1
    assert result[0].node_id == "n1"
    embedding.embed.assert_awaited_once_with(["hello"])
    store.search.assert_awaited_once_with([0.1, 0.9], 1)

