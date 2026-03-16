"""Cross-encoder reranker adapter using sentence-transformers style scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragway.exceptions import RagError
from ragway.reranking.base_reranker import BaseReranker
from ragway.schema.node import Node


class CrossEncoderClientProtocol(Protocol):
    """Protocol for sentence-transformers CrossEncoder-like clients."""

    async def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return relevance scores for (query, document) pairs."""


@dataclass(slots=True)
class CrossEncoderReranker(BaseReranker):
    """Rerank nodes using sentence-transformers cross-encoder scores."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    client: CrossEncoderClientProtocol | None = None

    async def rerank(self, query: str, nodes: list[Node]) -> list[Node]:
        """Score query-document pairs and sort nodes by descending score."""
        if not nodes:
            return []
        if self.client is None:
            raise RagError("CrossEncoderReranker requires a sentence-transformers client")

        pairs = [(query, node.content) for node in nodes]
        scores = await self.client.score_pairs(pairs)

        scored = list(zip(scores, nodes))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored]

