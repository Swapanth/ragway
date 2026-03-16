from __future__ import annotations

import asyncio

import pytest

from ragway.reranking.base_reranker import BaseReranker
from ragway.schema.node import Node


class _NoImplReranker(BaseReranker):
    pass


class _SimpleReranker(BaseReranker):
    async def rerank(self, query: str, nodes: list[Node]) -> list[Node]:
        del query
        return list(nodes)


async def test_base_reranker_is_abstract() -> None:
    """BaseReranker should not be instantiable without rerank implementation."""
    with pytest.raises(TypeError):
        _NoImplReranker()


async def test_simple_reranker_implements_contract() -> None:
    """Concrete reranker should return node list."""
    reranker = _SimpleReranker()
    nodes = [Node(node_id="n1", doc_id="d1", content="alpha")]
    result = await reranker.rerank("q", nodes)

    assert len(result) == 1
    assert result[0].node_id == "n1"

