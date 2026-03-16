from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.embeddings.openai_embedding import OpenAIEmbedding
from ragway.exceptions import RagError


async def test_openai_embedding_raises_without_client() -> None:
    """OpenAI adapter should require an async client instance."""
    adapter = OpenAIEmbedding()
    with pytest.raises(RagError):
        await adapter.embed(["hello"])


async def test_openai_embedding_batches_and_normalizes() -> None:
    """OpenAI adapter should batch requests and normalize returned vectors."""
    client = AsyncMock()
    client.embed.side_effect = [
        [[3.0, 4.0], [5.0, 12.0]],
        [[8.0, 15.0]],
    ]
    adapter = OpenAIEmbedding(client=client, max_batch_size=2)

    vectors = await adapter.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert pytest.approx(vectors[0][0], rel=1e-6) == 0.6
    assert pytest.approx(vectors[0][1], rel=1e-6) == 0.8
    assert client.embed.await_count == 2

