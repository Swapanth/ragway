from __future__ import annotations

import pytest

from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.vectorstores.base_vectorstore import BaseVectorStore


class _NoImplStore(BaseVectorStore):
    pass


class _MemoryStore(BaseVectorStore):
    def __init__(self) -> None:
        self._nodes: list[Node] = []

    async def add(self, nodes: list[Node]) -> None:
        self._nodes.extend(nodes)

    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        del query_vector
        return self._nodes[:top_k]

    async def delete(self, node_ids: list[str]) -> None:
        node_set = set(node_ids)
        self._nodes = [node for node in self._nodes if node.node_id not in node_set]


async def test_base_vectorstore_is_abstract() -> None:
    """Base vector store should not be instantiable without implementations."""
    with pytest.raises(TypeError):
        _NoImplStore()


async def test_memory_store_satisfies_contract() -> None:
    """Concrete store implementation should support add/search/delete calls."""
    store = _MemoryStore()
    document = Document(doc_id="doc-1", content="hello world")
    node = Node(node_id="n1", doc_id=document.doc_id, content=document.content, embedding=[1.0, 0.0])

    import asyncio

    await store.add([node])
    result = await store.search([1.0, 0.0], top_k=1)
    assert len(result) == 1
    assert result[0].node_id == "n1"

    await store.delete(["n1"])
    result_after = await store.search([1.0, 0.0], top_k=1)
    assert result_after == []

