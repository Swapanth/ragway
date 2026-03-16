from __future__ import annotations

import asyncio
import math

import pytest

from ragway.embeddings.base_embedding import BaseEmbedding


class _NoImplEmbedding(BaseEmbedding):
    pass


class _SimpleEmbedding(BaseEmbedding):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


async def test_base_embedding_is_abstract() -> None:
    """BaseEmbedding cannot be instantiated without embed implementation."""
    with pytest.raises(TypeError):
        _NoImplEmbedding()


async def test_embed_one_uses_batch_embed() -> None:
    """embed_one should delegate to embed and return first vector."""
    adapter = _SimpleEmbedding()
    vector = await adapter.embed_one("hello")
    assert vector == [5.0, 1.0]


async def test_normalize_returns_unit_vector() -> None:
    """normalize should produce a unit-length vector for non-zero input."""
    adapter = _SimpleEmbedding()
    normalized = adapter.normalize([3.0, 4.0])
    length = math.sqrt(sum(value * value for value in normalized))
    assert pytest.approx(length, rel=1e-6) == 1.0

