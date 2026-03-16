"""Abstract base contract for retrieval adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.schema.node import Node


class BaseRetriever(RetrieverProtocol, ABC):
    """Base retriever contract used by all retrieval implementations."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        """Retrieve the top matching nodes for a query."""

