from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError, ValidationError
from ragway.schema.node import Node
from ragway.vectorstores.pinecone_store import PineconeStore


async def test_pinecone_store_requires_client() -> None:
    """PineconeStore should require API key in default-client mode."""
    from os import environ

    environ.pop("PINECONE_API_KEY", None)
    store = PineconeStore(index_name="idx", client=None)
    with pytest.raises(RagError, match="PINECONE_API_KEY"):
        await store.search([1.0, 0.0], top_k=1)


async def test_pinecone_store_uses_mocked_client_calls() -> None:
    """PineconeStore should delegate all operations to async client."""
    client = AsyncMock()
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=[1.0, 0.0])
    client.query.return_value = [node]

    store = PineconeStore(index_name="idx", namespace="ns", client=client)
    await store.add([node])
    result = await store.search([1.0, 0.0], top_k=1)
    await store.delete(["n1"])

    assert len(result) == 1
    assert result[0].node_id == "n1"
    client.upsert.assert_awaited_once_with("idx", "ns", [node])
    client.query.assert_awaited_once()
    client.delete.assert_awaited_once_with("idx", "ns", ["n1"])


async def test_pinecone_store_rejects_missing_embedding() -> None:
    """PineconeStore add should reject nodes without embeddings."""
    client = AsyncMock()
    store = PineconeStore(index_name="idx", client=client)
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=None)

    with pytest.raises(ValidationError):
        await store.add([node])

