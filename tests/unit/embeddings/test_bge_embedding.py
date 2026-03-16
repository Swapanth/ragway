from __future__ import annotations

import math

import pytest

from ragway.embeddings.bge_embedding import BGEEmbedding


async def test_bge_embedding_returns_expected_shape() -> None:
    """BGE adapter should return one vector per text with configured dimensions."""
    adapter = BGEEmbedding(dimensions=6)
    vectors = await adapter.embed(["alpha", "beta"])

    assert len(vectors) == 2
    assert all(len(vector) == 6 for vector in vectors)


async def test_bge_embedding_vectors_are_normalized() -> None:
    """BGE adapter vectors should be unit-normalized."""
    adapter = BGEEmbedding(dimensions=10)
    vector = (await adapter.embed(["hello world"]))[0]
    length = math.sqrt(sum(value * value for value in vector))
    assert pytest.approx(length, rel=1e-6) == 1.0

