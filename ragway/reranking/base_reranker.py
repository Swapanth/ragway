"""Abstract base contract for reranking adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragway.interfaces.reranker_protocol import RerankerProtocol
from ragway.schema.node import Node


class BaseReranker(RerankerProtocol, ABC):
    """Base class implemented by all rerankers."""

    @abstractmethod
    async def rerank(self, query: str, nodes: list[Node]) -> list[Node]:
        """Return nodes sorted by descending relevance to query."""

