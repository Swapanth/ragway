from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.reranking.cohere_reranker import CohereReranker
from ragway.schema.node import Node


async def test_cohere_reranker_requires_client() -> None:
    """Cohere reranker should require COHERE_API_KEY for default client mode."""
    from os import environ

    environ.pop("COHERE_API_KEY", None)
    reranker = CohereReranker(client=None)
    nodes = [Node(node_id="n1", doc_id="d1", content="alpha")]

    with pytest.raises(RagError, match="COHERE_API_KEY"):
        await reranker.rerank("q", nodes)


async def test_cohere_reranker_uses_mocked_api_client() -> None:
    """Cohere reranker should reorder nodes from mocked API rank indices."""
    client = AsyncMock()
    client.rerank.return_value = [1, 0]

    reranker = CohereReranker(client=client)
    nodes = [
        Node(node_id="n1", doc_id="d1", content="low relevance"),
        Node(node_id="n2", doc_id="d1", content="high relevance"),
    ]

    result = await reranker.rerank("relevance", nodes)

    assert [node.node_id for node in result] == ["n2", "n1"]
    client.rerank.assert_awaited_once()
    call_kwargs = client.rerank.await_args.kwargs
    assert call_kwargs["query"] == "relevance"
    assert call_kwargs["documents"] == ["low relevance", "high relevance"]

