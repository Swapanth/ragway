"""Qdrant vector store adapter with cloud and local modes."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragway.exceptions import RagError, ValidationError
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node
from ragway.validators import validate_positive_int
from ragway.vectorstores.base_vectorstore import BaseVectorStore


class QdrantClientProtocol(Protocol):
    """Protocol for async Qdrant operations used by the adapter."""

    async def upsert(self, collection_name: str, nodes: list[Node]) -> None:
        """Upsert nodes into a collection."""

    async def search(self, collection_name: str, query_vector: list[float], top_k: int) -> list[Node]:
        """Search a collection and return matching nodes."""

    async def delete(self, collection_name: str, node_ids: list[str]) -> None:
        """Delete node ids from a collection."""


class _DefaultQdrantClient:
    """Async wrapper over qdrant-client AsyncQdrantClient implementation."""

    def __init__(
        self,
        *,
        url: str | None,
        api_key: str | None,
        path: str | None,
        in_memory: bool,
        dimension: int,
    ) -> None:
        try:
            qdrant_module = importlib.import_module("qdrant_client")
            qdrant_client_cls = getattr(qdrant_module, "AsyncQdrantClient")
        except (ImportError, AttributeError) as exc:
            raise RagError(f"qdrant-client package is required: {exc}") from exc

        self._dimension = dimension
        if url:
            self._client = qdrant_client_cls(url=url, api_key=api_key)
        elif in_memory:
            self._client = qdrant_client_cls(location=":memory:")
        else:
            storage_path = path or str(Path("./qdrant_data"))
            self._client = qdrant_client_cls(path=storage_path)

    async def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create collection if it does not exist."""
        models = importlib.import_module("qdrant_client.models")
        collections = await self._client.get_collections()
        collection_items = getattr(collections, "collections", [])
        existing_names = [str(getattr(item, "name", "")) for item in collection_items]
        if collection_name in existing_names:
            return

        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    async def upsert(self, collection_name: str, nodes: list[Node]) -> None:
        """Upsert nodes as points into a Qdrant collection."""
        models = importlib.import_module("qdrant_client.models")

        vector_size = len(nodes[0].embedding or []) if nodes else self._dimension
        await self._ensure_collection(collection_name, vector_size)

        points: list[object] = []
        for node in nodes:
            assert node.embedding is not None
            payload = {
                "doc_id": node.doc_id,
                "content": node.content,
                "metadata": node.metadata.model_dump(),
                "parent_id": node.parent_id,
                "position": node.position,
            }
            points.append(models.PointStruct(id=node.node_id, vector=node.embedding, payload=payload))

        await self._client.upsert(collection_name=collection_name, points=points, wait=True)

    async def search(self, collection_name: str, query_vector: list[float], top_k: int) -> list[Node]:
        """Search nearest neighbors in Qdrant and convert payloads to nodes."""
        await self._ensure_collection(collection_name, len(query_vector))

        query_result = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        hits = getattr(query_result, "points", None)
        if hits is None and isinstance(query_result, dict):
            hits = query_result.get("points", [])
        if hits is None:
            hits = []

        nodes: list[Node] = []
        for hit in hits:
            payload = hit.payload or {}
            metadata_raw = payload.get("metadata", {})
            metadata = Metadata.model_validate(metadata_raw if isinstance(metadata_raw, dict) else {})

            node = Node(
                node_id=str(hit.id),
                doc_id=str(payload.get("doc_id", "")),
                content=str(payload.get("content", "")),
                embedding=list(hit.vector) if isinstance(hit.vector, list) else None,
                metadata=metadata,
                parent_id=str(payload["parent_id"]) if payload.get("parent_id") is not None else None,
                position=int(payload["position"]) if payload.get("position") is not None else None,
            )
            nodes.append(node)
        return nodes

    async def delete(self, collection_name: str, node_ids: list[str]) -> None:
        """Delete points from Qdrant collection."""
        models = importlib.import_module("qdrant_client.models")

        selector = models.PointIdsList(points=node_ids)
        await self._client.delete(
            collection_name=collection_name,
            points_selector=selector,
            wait=True,
        )


@dataclass(slots=True)
class QdrantStore(BaseVectorStore):
    """Adapter that routes vector operations to Qdrant in cloud or local mode."""

    collection_name: str = "default"
    url: str | None = None
    api_key: str | None = None
    path: str | None = None
    in_memory: bool = False
    dimension: int = 1536
    client: QdrantClientProtocol | None = None

    def _client_instance(self) -> QdrantClientProtocol:
        """Return configured client or lazily construct default Qdrant client."""
        if self.client is not None:
            return self.client

        url = self.url or os.getenv("QDRANT_URL")
        api_key = self.api_key or os.getenv("QDRANT_API_KEY")
        default_client = _DefaultQdrantClient(
            url=url,
            api_key=api_key,
            path=self.path,
            in_memory=self.in_memory,
            dimension=self.dimension,
        )
        self.client = default_client
        return default_client

    async def add(self, nodes: list[Node]) -> None:
        """Upsert nodes into Qdrant collection."""
        if not nodes:
            return
        for node in nodes:
            if node.embedding is None:
                raise ValidationError("Node embedding is required for QdrantStore.add")
        await self._client_instance().upsert(self.collection_name, nodes)

    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        """Search Qdrant collection by query vector."""
        top_k = validate_positive_int(top_k, "top_k")
        return await self._client_instance().search(self.collection_name, query_vector, top_k)

    async def delete(self, node_ids: list[str]) -> None:
        """Delete node ids from Qdrant collection."""
        await self._client_instance().delete(self.collection_name, node_ids)
