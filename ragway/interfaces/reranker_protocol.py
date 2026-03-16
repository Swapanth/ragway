"""Protocol defining the reranker interface contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragway.schema.node import Node


@runtime_checkable
class RerankerProtocol(Protocol):
    """Contract for asynchronous reranking providers."""

    async def rerank(self, query: str, nodes: list[Node]) -> list[Node]:
        """Rerank candidate nodes for a given query and return a new ordering."""

