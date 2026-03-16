from __future__ import annotations

import pytest

from ragway.retrieval.base_retriever import BaseRetriever
from ragway.schema.node import Node


class _NoImplRetriever(BaseRetriever):
    pass


class _SimpleRetriever(BaseRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        del query, top_k
        return [Node(node_id="n1", doc_id="d1", content="alpha")]


async def test_base_retriever_is_abstract() -> None:
    """Base retriever must require concrete retrieve implementation."""
    with pytest.raises(TypeError):
        _NoImplRetriever()


async def test_simple_retriever_implements_contract() -> None:
    """Concrete retriever should return node lists."""
    import asyncio

    retriever = _SimpleRetriever()
    result = await retriever.retrieve("q", top_k=1)
    assert len(result) == 1
    assert result[0].node_id == "n1"

