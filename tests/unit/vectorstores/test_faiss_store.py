from __future__ import annotations

import asyncio

import pytest

from ragway.exceptions import ValidationError
from ragway.schema.node import Node
from ragway.vectorstores.faiss_store import FAISSStore


async def test_faiss_store_add_and_search() -> None:
    """FAISSStore should retrieve the closest node for a query vector."""
    store = FAISSStore()
    nodes = [
        Node(node_id="n1", doc_id="d1", content="alpha", embedding=[1.0, 0.0]),
        Node(node_id="n2", doc_id="d1", content="beta", embedding=[0.0, 1.0]),
    ]

    await store.add(nodes)
    result = await store.search([0.9, 0.1], top_k=1)

    assert len(result) == 1
    assert result[0].node_id == "n1"


async def test_faiss_store_delete_removes_node() -> None:
    """Deleted nodes should not appear in subsequent results."""
    store = FAISSStore()
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=[1.0, 0.0])

    await store.add([node])
    await store.delete(["n1"])
    result = await store.search([1.0, 0.0], top_k=1)

    assert result == []


async def test_faiss_store_rejects_missing_embedding() -> None:
    """Nodes without embeddings should be rejected when adding."""
    store = FAISSStore()
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=None)

    with pytest.raises(ValidationError):
        await store.add([node])

