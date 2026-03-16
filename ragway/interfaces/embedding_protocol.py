"""Protocol defining the embedding interface contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Contract for asynchronous text embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of input texts into numeric vectors."""
