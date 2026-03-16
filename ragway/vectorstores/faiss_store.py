"""Local FAISS-style vector store with cosine-similarity search."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ragway.exceptions import ValidationError
from ragway.schema.node import Node
from ragway.validators import validate_positive_int
from ragway.vectorstores.base_vectorstore import BaseVectorStore


@dataclass(slots=True)
class FAISSStore(BaseVectorStore):
    """In-memory vector store implementing FAISS-like behavior for local usage."""

    _nodes_by_id: dict[str, Node] = field(default_factory=dict)

    async def add(self, nodes: list[Node]) -> None:
        """Insert or replace nodes that contain embedding vectors."""
        for node in nodes:
            if node.embedding is None:
                raise ValidationError("Node embedding is required for FAISSStore.add")
            normalized = self._normalize(node.embedding)
            self._nodes_by_id[node.node_id] = node.model_copy(update={"embedding": normalized})

    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        """Return top-k nodes ranked by cosine similarity to query vector."""
        top_k = validate_positive_int(top_k, "top_k")
        if not self._nodes_by_id:
            return []

        normalized_query = self._normalize(query_vector)
        scored_nodes: list[tuple[float, Node]] = []
        for node in self._nodes_by_id.values():
            assert node.embedding is not None
            score = self._cosine_similarity(normalized_query, node.embedding)
            scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored_nodes[:top_k]]

    async def delete(self, node_ids: list[str]) -> None:
        """Delete node ids from local index if present."""
        for node_id in node_ids:
            self._nodes_by_id.pop(node_id, None)

    def _normalize(self, vector: list[float]) -> list[float]:
        """Return an L2-normalized vector copy."""
        if not vector:
            raise ValidationError("Vector must not be empty")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return [0.0 for _ in vector]
        return [value / norm for value in vector]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        """Compute cosine similarity between equal-length normalized vectors."""
        if len(left) != len(right):
            raise ValidationError("Vector dimensions must match for cosine similarity")
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

