"""Abstract base class for embedding adapters."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ragway.interfaces.embedding_protocol import EmbeddingProtocol


class BaseEmbedding(EmbeddingProtocol, ABC):
    """Base embedding adapter with common normalization utilities."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vector representations."""

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text value and return one vector."""
        vectors = await self.embed([text])
        return vectors[0] if vectors else []

    def normalize(self, vector: list[float]) -> list[float]:
        """Return a unit-normalized copy of a vector."""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return [0.0 for _ in vector]
        return [value / norm for value in vector]

