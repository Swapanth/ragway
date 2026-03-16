from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from ragway.retrieval.parent_document_retriever import ParentDocumentRetriever
from ragway.schema.node import Node


async def test_parent_document_retriever_collapses_children_to_parent() -> None:
    """Parent retriever should deduplicate child matches into parent results."""
    child_retriever = AsyncMock()
    parent = Node(node_id="p1", doc_id="d1", content="parent")

    child_retriever.retrieve.return_value = [
        Node(node_id="c1", doc_id="d1", content="child-a", parent_id="p1"),
        Node(node_id="c2", doc_id="d1", content="child-b", parent_id="p1"),
    ]

    retriever = ParentDocumentRetriever(child_retriever=child_retriever, parent_nodes={"p1": parent})
    result = await retriever.retrieve("q", top_k=5)

    assert len(result) == 1
    assert result[0].node_id == "p1"

