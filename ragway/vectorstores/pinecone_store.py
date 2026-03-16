"""Pinecone vector store adapter."""

from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass
from typing import Any, Protocol

from ragway.exceptions import RagError, ValidationError
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node
from ragway.validators import validate_positive_int
from ragway.vectorstores.base_vectorstore import BaseVectorStore


class PineconeClientProtocol(Protocol):
    """Protocol for async Pinecone client operations used by adapter."""

    async def upsert(self, index_name: str, namespace: str, nodes: list[Node]) -> None:
        """Upsert nodes to Pinecone index and namespace."""

    async def query(
        self,
        index_name: str,
        namespace: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[Node]:
        """Query Pinecone index and namespace for similar nodes."""

    async def delete(self, index_name: str, namespace: str, node_ids: list[str]) -> None:
        """Delete nodes from Pinecone index and namespace."""


class _DefaultPineconeClient:
    """Async wrapper for synchronous Pinecone SDK operations."""

    def __init__(self, *, api_key: str, dimension: int) -> None:
        try:
            pinecone_module = importlib.import_module("pinecone")
        except ImportError as exc:
            raise RagError(f"pinecone package is required: {exc}") from exc

        pinecone_cls = getattr(pinecone_module, "Pinecone", None)
        if pinecone_cls is None:
            raise RagError("pinecone.Pinecone class is required")

        self._serverless_spec = getattr(pinecone_module, "ServerlessSpec", None)
        self._pc = pinecone_cls(api_key=api_key)
        self._dimension = dimension

    def _index_exists(self, index_name: str) -> bool:
        listing = self._pc.list_indexes()
        names: list[str] = []

        names_fn = getattr(listing, "names", None)
        if callable(names_fn):
            names = [str(name) for name in names_fn()]
        elif isinstance(listing, list):
            for item in listing:
                item_name = getattr(item, "name", None)
                if isinstance(item_name, str):
                    names.append(item_name)
                elif isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(str(item["name"]))

        return index_name in names

    def _ensure_index(self, index_name: str, dimension: int) -> Any:
        if not self._index_exists(index_name):
            kwargs: dict[str, object] = {"name": index_name, "dimension": dimension}
            if self._serverless_spec is not None:
                kwargs["spec"] = self._serverless_spec(cloud="aws", region="us-east-1")
            self._pc.create_index(**kwargs)
        return self._pc.Index(index_name)

    async def upsert(self, index_name: str, namespace: str, nodes: list[Node]) -> None:
        """Upsert vectors into Pinecone index/namespace."""

        dimension = self._dimension
        if nodes and nodes[0].embedding is not None:
            dimension = len(nodes[0].embedding)

        def _upsert() -> None:
            index = self._ensure_index(index_name, dimension)
            vectors: list[dict[str, object]] = []
            for node in nodes:
                assert node.embedding is not None
                vectors.append(
                    {
                        "id": node.node_id,
                        "values": node.embedding,
                        "metadata": {
                            "content": node.content,
                            "doc_id": node.doc_id,
                            "source": node.metadata.source,
                        },
                    }
                )
            index.upsert(vectors=vectors, namespace=namespace)

        await asyncio.get_running_loop().run_in_executor(None, _upsert)

    async def query(
        self,
        index_name: str,
        namespace: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[Node]:
        """Query nearest vectors from Pinecone index/namespace."""

        def _query() -> Any:
            index = self._ensure_index(index_name, len(query_vector))
            return index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
            )

        result = await asyncio.get_running_loop().run_in_executor(None, _query)
        matches = getattr(result, "matches", None)
        if matches is None and isinstance(result, dict):
            matches = result.get("matches", [])

        nodes: list[Node] = []
        for match in matches or []:
            node_id = str(getattr(match, "id", "")) if not isinstance(match, dict) else str(match.get("id", ""))
            metadata = (
                getattr(match, "metadata", None) if not isinstance(match, dict) else match.get("metadata")
            ) or {}
            if not isinstance(metadata, dict):
                metadata = {}

            nodes.append(
                Node(
                    node_id=node_id,
                    doc_id=str(metadata.get("doc_id", node_id)),
                    content=str(metadata.get("content", "")),
                    metadata=Metadata.model_validate(metadata),
                )
            )
        return nodes

    async def delete(self, index_name: str, namespace: str, node_ids: list[str]) -> None:
        """Delete vectors from Pinecone index/namespace."""

        def _delete() -> None:
            index = self._ensure_index(index_name, self._dimension)
            index.delete(ids=node_ids, namespace=namespace)

        await asyncio.get_running_loop().run_in_executor(None, _delete)


@dataclass(slots=True)
class PineconeStore(BaseVectorStore):
    """Adapter that routes vector operations to an async Pinecone client."""

    index_name: str
    namespace: str = "default"
    dimension: int = 1536
    client: PineconeClientProtocol | None = None

    def _client_instance(self) -> PineconeClientProtocol:
        """Return injected client or lazily construct the Pinecone SDK wrapper."""
        if self.client is not None:
            return self.client

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RagError("PINECONE_API_KEY environment variable is required")

        self.client = _DefaultPineconeClient(api_key=api_key, dimension=self.dimension)
        return self.client

    async def add(self, nodes: list[Node]) -> None:
        """Upsert nodes into Pinecone."""
        for node in nodes:
            if node.embedding is None:
                raise ValidationError("Node embedding is required for PineconeStore.add")
        await self._client_instance().upsert(self.index_name, self.namespace, nodes)

    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        """Search Pinecone index for nearest neighbors."""
        top_k = validate_positive_int(top_k, "top_k")
        return await self._client_instance().query(self.index_name, self.namespace, query_vector, top_k)

    async def delete(self, node_ids: list[str]) -> None:
        """Delete node ids from Pinecone index."""
        await self._client_instance().delete(self.index_name, self.namespace, node_ids)

