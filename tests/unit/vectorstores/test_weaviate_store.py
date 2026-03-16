from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError, ValidationError
from ragway.schema.node import Node
from ragway.vectorstores.weaviate_store import WeaviateStore


async def test_weaviate_store_requires_client() -> None:
    """WeaviateStore should require URL/API key when no client is injected."""
    from os import environ

    environ.pop("WEAVIATE_URL", None)
    environ.pop("WEAVIATE_API_KEY", None)
    store = WeaviateStore(class_name="Node", client=None)
    with pytest.raises(RagError, match="WEAVIATE_URL"):
        await store.search([1.0, 0.0], top_k=1)


async def test_weaviate_store_uses_mocked_client_calls() -> None:
    """WeaviateStore should delegate add/search/delete to async mocked client."""
    client = AsyncMock()
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=[1.0, 0.0])
    client.query.return_value = [node]

    store = WeaviateStore(class_name="Node", client=client)
    await store.add([node])
    result = await store.search([1.0, 0.0], top_k=1)
    await store.delete(["n1"])

    assert len(result) == 1
    assert result[0].node_id == "n1"
    client.insert.assert_awaited_once_with("Node", [node])
    client.query.assert_awaited_once()
    client.delete.assert_awaited_once_with("Node", ["n1"])


async def test_weaviate_store_rejects_missing_embedding() -> None:
    """WeaviateStore add should reject nodes without embeddings."""
    client = AsyncMock()
    store = WeaviateStore(class_name="Node", client=client)
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=None)

    with pytest.raises(ValidationError):
        await store.add([node])

