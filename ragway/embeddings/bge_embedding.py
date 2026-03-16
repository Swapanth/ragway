"""Deterministic local embedding adapter with BGE-style configuration knobs."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.embeddings.base_embedding import BaseEmbedding
from ragway.validators import validate_positive_int


@dataclass(slots=True)
class BGEEmbedding(BaseEmbedding):
    """Generate deterministic normalized vectors suitable for local testing."""

    dimensions: int = 8
    query_prefix: str = "Represent this sentence for retrieval: "

    def __post_init__(self) -> None:
        """Validate embedding dimensionality."""
        self.dimensions = validate_positive_int(self.dimensions, "dimensions")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed text into deterministic vectors and normalize each result."""
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0 for _ in range(self.dimensions)]
            payload = f"{self.query_prefix}{text}"
            for index, character in enumerate(payload):
                bucket = index % self.dimensions
                vector[bucket] += (ord(character) % 31) / 31.0
            vectors.append(self.normalize(vector))
        return vectors

