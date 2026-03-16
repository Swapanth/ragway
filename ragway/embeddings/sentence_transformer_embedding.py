"""SentenceTransformers-compatible embedding adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragway.embeddings.base_embedding import BaseEmbedding
from ragway.validators import validate_positive_int


class SentenceTransformerClientProtocol(Protocol):
    """Protocol for async sentence-transformer style clients."""

    async def encode(self, texts: list[str], model: str) -> list[list[float]]:
        """Encode a batch of texts into embedding vectors."""


@dataclass(slots=True)
class SentenceTransformerEmbedding(BaseEmbedding):
    """Embedding adapter for SentenceTransformers-like models."""

    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 8
    max_batch_size: int = 32
    client: SentenceTransformerClientProtocol | None = None

    def __post_init__(self) -> None:
        """Validate dimensions and batching configuration."""
        self.dimensions = validate_positive_int(self.dimensions, "dimensions")
        self.max_batch_size = validate_positive_int(self.max_batch_size, "max_batch_size")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via injected client or deterministic local fallback."""
        if not texts:
            return []

        if self.client is None:
            return [self._deterministic_vector(text) for text in texts]

        output: list[list[float]] = []
        for index in range(0, len(texts), self.max_batch_size):
            batch = texts[index : index + self.max_batch_size]
            vectors = await self.client.encode(batch, self.model)
            output.extend(self.normalize(vector) for vector in vectors)
        return output

    def _deterministic_vector(self, text: str) -> list[float]:
        """Generate a normalized deterministic vector without external dependencies."""
        vector = [0.0 for _ in range(self.dimensions)]
        payload = f"{self.model}:{text}"
        for index, character in enumerate(payload):
            bucket = index % self.dimensions
            vector[bucket] += (ord(character) % 29) / 29.0
        return self.normalize(vector)

