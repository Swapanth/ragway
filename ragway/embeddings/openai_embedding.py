"""OpenAI-compatible embedding adapter with async batched requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragway.embeddings.base_embedding import BaseEmbedding
from ragway.exceptions import RagError
from ragway.validators import validate_positive_int


class OpenAIEmbeddingClientProtocol(Protocol):
    """Protocol for async OpenAI embedding client implementations."""

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Return embeddings for a batch of texts and model name."""


@dataclass(slots=True)
class OpenAIEmbedding(BaseEmbedding):
    """Embedding adapter that delegates to an OpenAI-compatible client."""

    model: str = "text-embedding-3-small"
    max_batch_size: int = 32
    client: OpenAIEmbeddingClientProtocol | None = None

    def __post_init__(self) -> None:
        """Validate configuration values for batch execution."""
        self.max_batch_size = validate_positive_int(self.max_batch_size, "max_batch_size")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed input texts in batches and normalize output vectors."""
        if not texts:
            return []
        if self.client is None:
            raise RagError("OpenAIEmbedding requires an async client")

        output: list[list[float]] = []
        for index in range(0, len(texts), self.max_batch_size):
            batch = texts[index : index + self.max_batch_size]
            vectors = await self.client.embed(batch, self.model)
            output.extend(self.normalize(vector) for vector in vectors)
        return output

