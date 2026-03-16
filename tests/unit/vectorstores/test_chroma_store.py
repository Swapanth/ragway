from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError, ValidationError
from ragway.schema.node import Node
from ragway.vectorstores.chroma_store import ChromaStore


async def test_chroma_store_requires_client() -> None:
    """ChromaStore should lazily construct a default client when none is provided."""
    store = ChromaStore(client=None)
    assert store._client_instance() is not None


async def test_chroma_store_calls_client_methods() -> None:
    """ChromaStore should delegate add/search/delete to async client."""
    client = AsyncMock()
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=[1.0, 0.0])
    client.search.return_value = [node]

    store = ChromaStore(collection="demo", client=client)
    await store.add([node])
    result = await store.search([1.0, 0.0], top_k=1)
    await store.delete(["n1"])

    assert len(result) == 1
    assert result[0].node_id == "n1"
    client.add.assert_awaited_once()
    client.search.assert_awaited_once()
    client.delete.assert_awaited_once()


async def test_chroma_store_rejects_missing_embedding() -> None:
    """ChromaStore add should reject nodes without embeddings."""
    client = AsyncMock()
    store = ChromaStore(client=client)
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=None)

    with pytest.raises(ValidationError):
        await store.add([node])

