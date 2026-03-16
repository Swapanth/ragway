"""Chroma vector store adapter with async client abstraction."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ragway.exceptions import RagError, ValidationError
from ragway.schema.node import Node
from ragway.schema.metadata import Metadata
from ragway.validators import validate_positive_int
from ragway.vectorstores.base_vectorstore import BaseVectorStore


class ChromaClientProtocol(Protocol):
    """Protocol for async Chroma client operations used by adapter."""

    async def add(self, collection: str, nodes: list[Node]) -> None:
        """Persist nodes into a named collection."""

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[Node]:
        """Search a named collection and return ranked nodes."""

    async def delete(self, collection: str, node_ids: list[str]) -> None:
        """Delete node ids from a named collection."""


class _DefaultChromaClient:
    """Async wrapper for chromadb's synchronous PersistentClient."""

    def __init__(self, path: str) -> None:
        try:
            chromadb = importlib.import_module("chromadb")
        except ImportError as exc:
            raise RagError(f"chromadb package is required: {exc}") from exc

        client_cls = getattr(chromadb, "PersistentClient", None)
        if client_cls is None:
            raise RagError("chromadb.PersistentClient is required")
        self._client = client_cls(path=path)

    def _collection(self, collection: str) -> Any:
        return self._client.get_or_create_collection(collection)

    async def add(self, collection: str, nodes: list[Node]) -> None:
        """Insert documents, vectors, and metadata into Chroma."""

        def _add() -> None:
            self._collection(collection).add(
                ids=[node.node_id for node in nodes],
                embeddings=[node.embedding for node in nodes],
                documents=[node.content for node in nodes],
                metadatas=[{"doc_id": node.doc_id, "source": node.metadata.source} for node in nodes],
            )

        await asyncio.get_running_loop().run_in_executor(None, _add)

    async def search(self, collection: str, query_vector: list[float], top_k: int) -> list[Node]:
        """Run vector similarity query and map results into Node objects."""

        def _search() -> dict[str, Any]:
            col = self._collection(collection)
            try:
                return cast(
                    dict[str, Any],
                    col.query(
                        query_embeddings=[query_vector],
                        n_results=top_k,
                        include=["documents", "metadatas"],
                    ),
                )
            except TypeError:
                return cast(dict[str, Any], col.query(query_embeddings=[query_vector], n_results=top_k))

        result = await asyncio.get_running_loop().run_in_executor(None, _search)

        ids = result.get("ids", [[]])
        documents = result.get("documents", [[]])
        metadatas = result.get("metadatas", [[]])

        rows_ids = ids[0] if ids else []
        rows_docs = documents[0] if documents else []
        rows_meta = metadatas[0] if metadatas else []

        mapped: list[Node] = []
        for idx, (node_id, content) in enumerate(zip(rows_ids, rows_docs)):
            metadata_raw = rows_meta[idx] if idx < len(rows_meta) and isinstance(rows_meta[idx], dict) else {}
            doc_id = str(metadata_raw.get("doc_id", node_id))
            mapped.append(
                Node(
                    node_id=str(node_id),
                    doc_id=doc_id,
                    content=str(content),
                    metadata=Metadata.model_validate(metadata_raw),
                )
            )
        return mapped

    async def delete(self, collection: str, node_ids: list[str]) -> None:
        """Delete vectors from Chroma collection by ids."""

        def _delete() -> None:
            self._collection(collection).delete(ids=node_ids)

        await asyncio.get_running_loop().run_in_executor(None, _delete)


@dataclass(slots=True)
class ChromaStore(BaseVectorStore):
    """Adapter that routes vector operations to an async Chroma client."""

    collection: str = "default"
    path: str = "./chroma_data"
    client: ChromaClientProtocol | None = None

    def _client_instance(self) -> ChromaClientProtocol:
        """Return injected client or lazily build the default Chroma client."""
        if self.client is None:
            self.client = _DefaultChromaClient(path=self.path)
        return self.client

    async def add(self, nodes: list[Node]) -> None:
        """Add nodes into Chroma collection."""
        for node in nodes:
            if node.embedding is None:
                raise ValidationError("Node embedding is required for ChromaStore.add")
        await self._client_instance().add(self.collection, nodes)

    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        """Search Chroma collection by query vector."""
        top_k = validate_positive_int(top_k, "top_k")
        return await self._client_instance().search(self.collection, query_vector, top_k)

    async def delete(self, node_ids: list[str]) -> None:
        """Delete nodes from Chroma collection."""
        await self._client_instance().delete(self.collection, node_ids)

