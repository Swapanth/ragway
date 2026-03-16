"""Cohere embedding adapter with async batched requests."""

from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass
from typing import Any, Protocol

from ragway.embeddings.base_embedding import BaseEmbedding
from ragway.exceptions import RagError
from ragway.validators import validate_positive_int


class CohereEmbeddingClientProtocol(Protocol):
    """Protocol for async Cohere embedding client implementations."""

    async def embed(self, texts: list[str], model: str, api_key: str) -> list[list[float]]:
        """Return embeddings for a batch of texts and model name."""


class _DefaultCohereEmbeddingClient:
    """Async client wrapper for Cohere embedding APIs."""

    async def embed(self, texts: list[str], model: str, api_key: str) -> list[list[float]]:
        """Call Cohere embedding APIs and normalize response shapes."""
        try:
            cohere = importlib.import_module("cohere")
        except ImportError as exc:
            raise RagError(f"cohere package is required: {exc}") from exc

        async_client_cls = getattr(cohere, "AsyncClientV2", None)
        response: Any
        if async_client_cls is not None:
            client = async_client_cls(api_key=api_key)
            response = await client.embed(
                texts=texts,
                model=model,
                input_type="search_document",
            )
        else:
            async_client = getattr(cohere, "AsyncClient", None)
            if async_client is not None:
                client = async_client(api_key)
                response = await client.embed(
                    texts=texts,
                    model=model,
                    input_type="search_document",
                )
            else:
                sync_client_cls = getattr(cohere, "ClientV2", None) or getattr(cohere, "Client", None)
                if sync_client_cls is None:
                    raise RagError("Cohere client class was not found")
                client = sync_client_cls(api_key)
                response = await asyncio.to_thread(
                    client.embed,
                    texts=texts,
                    model=model,
                    input_type="search_document",
                )

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if isinstance(embeddings, dict):
            embeddings = embeddings.get("float") or embeddings.get("float_")
        if embeddings is not None and not isinstance(embeddings, list):
            float_vectors = getattr(embeddings, "float_", None) or getattr(embeddings, "float", None)
            if float_vectors is not None:
                embeddings = float_vectors
        if embeddings is None:
            raise RagError("Cohere embedding response did not include embeddings")

        vectors: list[list[float]] = []
        for vector in embeddings:
            vectors.append([float(value) for value in vector])
        return vectors


@dataclass(slots=True)
class CohereEmbedding(BaseEmbedding):
    """Embedding adapter that delegates to a Cohere-compatible async client."""

    model: str = "embed-english-v3.0"
    max_batch_size: int = 32
    client: CohereEmbeddingClientProtocol | None = None

    def __post_init__(self) -> None:
        """Validate configuration values for batch execution."""
        self.max_batch_size = validate_positive_int(self.max_batch_size, "max_batch_size")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed input texts in batches and normalize output vectors."""
        if not texts:
            return []

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RagError("COHERE_API_KEY environment variable is required")
        client = self.client or _DefaultCohereEmbeddingClient()

        output: list[list[float]] = []
        for index in range(0, len(texts), self.max_batch_size):
            batch = texts[index : index + self.max_batch_size]
            vectors = await client.embed(batch, self.model, api_key)
            output.extend(self.normalize(vector) for vector in vectors)
        return output
