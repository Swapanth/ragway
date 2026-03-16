"""Cohere-based reranker adapter."""

from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass
from typing import Any, Protocol

from ragway.exceptions import RagError
from ragway.reranking.base_reranker import BaseReranker
from ragway.schema.node import Node


class CohereClientProtocol(Protocol):
    """Protocol for async Cohere rerank client."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str = "rerank-v3.5",
        api_key: str | None = None,
    ) -> list[int]:
        """Return ranked document indices in descending relevance order."""


class _DefaultCohereRerankClient:
    """Async client wrapper for Cohere rerank APIs."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str = "rerank-v3.5",
        api_key: str | None = None,
    ) -> list[int]:
        """Call Cohere rerank endpoint and map result indices."""
        if not api_key:
            raise RagError("COHERE_API_KEY environment variable is required")

        try:
            cohere = importlib.import_module("cohere")
        except ImportError as exc:
            raise RagError(f"cohere package is required: {exc}") from exc

        response: Any
        async_client_cls = getattr(cohere, "AsyncClientV2", None)
        if async_client_cls is not None:
            client = async_client_cls(api_key=api_key)
            response = await client.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=len(documents),
            )
        else:
            async_client = getattr(cohere, "AsyncClient", None)
            if async_client is not None:
                client = async_client(api_key)
                response = await client.rerank(
                    model=model,
                    query=query,
                    documents=documents,
                    top_n=len(documents),
                )
            else:
                sync_client_cls = getattr(cohere, "ClientV2", None) or getattr(cohere, "Client", None)
                if sync_client_cls is None:
                    raise RagError("Cohere client class was not found")
                client = sync_client_cls(api_key)
                response = await asyncio.to_thread(
                    client.rerank,
                    model=model,
                    query=query,
                    documents=documents,
                    top_n=len(documents),
                )

        results = getattr(response, "results", None)
        if results is None and isinstance(response, dict):
            results = response.get("results")

        indices: list[int] = []
        for item in results or []:
            index = getattr(item, "index", None)
            if index is None and isinstance(item, dict):
                index = item.get("index")
            if isinstance(index, int):
                indices.append(index)
        return indices


@dataclass(slots=True)
class CohereReranker(BaseReranker):
    """Rerank nodes via Cohere API client."""

    model: str = "rerank-v3.5"
    client: CohereClientProtocol | None = None

    async def rerank(self, query: str, nodes: list[Node]) -> list[Node]:
        """Delegate reranking to Cohere and reorder input nodes."""
        if not nodes:
            return []

        documents = [node.content for node in nodes]
        if self.client is None:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise RagError("COHERE_API_KEY environment variable is required")
            client: CohereClientProtocol = _DefaultCohereRerankClient()
            indices = await client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                api_key=api_key,
            )
        else:
            try:
                indices = await self.client.rerank(
                    query=query,
                    documents=documents,
                    model=self.model,
                    api_key=os.getenv("COHERE_API_KEY"),
                )
            except TypeError:
                indices = await self.client.rerank(query=query, documents=documents)

        ordered: list[Node] = []
        for index in indices:
            if 0 <= index < len(nodes):
                ordered.append(nodes[index])

        selected_ids = {node.node_id for node in ordered}
        remaining = [node for node in nodes if node.node_id not in selected_ids]
        return ordered + remaining

