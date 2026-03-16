from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.embeddings.cohere_embedding import CohereEmbedding
from ragway.exceptions import RagError


async def test_cohere_embedding_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cohere adapter should require COHERE_API_KEY."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    adapter = CohereEmbedding(client=AsyncMock())

    with pytest.raises(RagError, match="COHERE_API_KEY"):
        await adapter.embed(["hello"])


async def test_cohere_embedding_raises_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cohere adapter should map missing cohere package to RagError."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    adapter = CohereEmbedding(client=None)
    monkeypatch.setattr("ragway.embeddings.cohere_embedding.importlib.import_module", lambda _: (_ for _ in ()).throw(ImportError("cohere")))

    with pytest.raises(RagError, match="cohere package is required"):
        await adapter.embed(["hello"])


async def test_cohere_embedding_batches_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cohere adapter should batch requests and normalize returned vectors."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    client = AsyncMock()
    client.embed.side_effect = [
        [[3.0, 4.0], [5.0, 12.0]],
        [[8.0, 15.0]],
    ]
    adapter = CohereEmbedding(client=client, max_batch_size=2)

    vectors = await adapter.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert pytest.approx(vectors[0][0], rel=1e-6) == 0.6
    assert pytest.approx(vectors[0][1], rel=1e-6) == 0.8
    assert client.embed.await_count == 2
