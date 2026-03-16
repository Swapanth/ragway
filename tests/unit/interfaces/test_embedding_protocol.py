from __future__ import annotations

import inspect

from ragway.interfaces.embedding_protocol import EmbeddingProtocol


class _EmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def test_embedding_protocol_runtime_checkable() -> None:
    """A structurally compatible provider should satisfy EmbeddingProtocol."""
    provider = _EmbeddingProvider()
    assert isinstance(provider, EmbeddingProtocol)


def test_embedding_protocol_embed_is_async() -> None:
    """The protocol embed method should be declared as coroutine function."""
    assert inspect.iscoroutinefunction(EmbeddingProtocol.embed)

