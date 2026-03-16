from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding


async def test_sentence_transformer_embedding_fallback_shape() -> None:
    """Fallback mode should return deterministic vectors with configured dimensions."""
    adapter = SentenceTransformerEmbedding(dimensions=5, client=None)
    vectors = await adapter.embed(["x", "y"])
    assert len(vectors) == 2
    assert all(len(vector) == 5 for vector in vectors)


async def test_sentence_transformer_embedding_client_batching() -> None:
    """Client mode should batch requests and normalize vectors."""
    client = AsyncMock()
    client.encode.side_effect = [
        [[3.0, 4.0], [8.0, 15.0]],
        [[5.0, 12.0]],
    ]
    adapter = SentenceTransformerEmbedding(client=client, max_batch_size=2)

    vectors = await adapter.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert pytest.approx(vectors[0][0], rel=1e-6) == 0.6
    assert pytest.approx(vectors[0][1], rel=1e-6) == 0.8
    assert client.encode.await_count == 2

