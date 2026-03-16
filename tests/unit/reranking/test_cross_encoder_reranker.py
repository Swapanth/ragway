from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.reranking.cross_encoder_reranker import CrossEncoderReranker
from ragway.schema.node import Node


async def test_cross_encoder_reranker_requires_client() -> None:
    """Cross-encoder reranker should fail when no client is configured."""
    reranker = CrossEncoderReranker(client=None)
    nodes = [Node(node_id="n1", doc_id="d1", content="alpha")]

    with pytest.raises(RagError):
        await reranker.rerank("q", nodes)


async def test_cross_encoder_reranker_uses_mocked_client_scores() -> None:
    """Cross-encoder reranker should reorder by mocked score outputs."""
    client = AsyncMock()
    client.score_pairs.return_value = [0.1, 0.9, 0.2]

    reranker = CrossEncoderReranker(client=client)
    nodes = [
        Node(node_id="n1", doc_id="d1", content="doc 1"),
        Node(node_id="n2", doc_id="d1", content="doc 2"),
        Node(node_id="n3", doc_id="d1", content="doc 3"),
    ]

    result = await reranker.rerank("query", nodes)

    assert [node.node_id for node in result] == ["n2", "n3", "n1"]
    client.score_pairs.assert_awaited_once_with(
        [("query", "doc 1"), ("query", "doc 2"), ("query", "doc 3")]
    )

