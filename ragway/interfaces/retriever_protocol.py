"""Protocol defining the retriever interface contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragway.schema.node import Node


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Contract for asynchronous retrieval providers."""

    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        """Retrieve the top matching nodes for a user query."""

