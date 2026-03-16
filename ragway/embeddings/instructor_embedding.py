"""Instruction-aware deterministic embedding adapter."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.embeddings.base_embedding import BaseEmbedding
from ragway.validators import validate_positive_int


@dataclass(slots=True)
class InstructorEmbedding(BaseEmbedding):
    """Embed texts by combining a task instruction with each input string."""

    instruction: str = "Represent the document for retrieval"
    dimensions: int = 8

    def __post_init__(self) -> None:
        """Validate embedding dimensionality."""
        self.dimensions = validate_positive_int(self.dimensions, "dimensions")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic vectors conditioned on the instruction text."""
        vectors: list[list[float]] = []
        for text in texts:
            payload = f"{self.instruction}::{text}"
            vector = [0.0 for _ in range(self.dimensions)]
            for index, character in enumerate(payload):
                bucket = index % self.dimensions
                vector[bucket] += ((ord(character) % 17) + 1) / 17.0
            vectors.append(self.normalize(vector))
        return vectors

