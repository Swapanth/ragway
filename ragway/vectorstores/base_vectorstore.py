"""Abstract base contract for vector storage adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragway.schema.node import Node


class BaseVectorStore(ABC):
    """Base class for vector stores used by retrieval components."""

    @abstractmethod
    async def add(self, nodes: list[Node]) -> None:
        """Add nodes and their embeddings to the vector index."""

    @abstractmethod
    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        """Search for the top-k most similar nodes for a query vector."""

    @abstractmethod
    async def delete(self, node_ids: list[str]) -> None:
        """Delete nodes from the vector index by node identifier."""

