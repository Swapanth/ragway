"""Weaviate vector store adapter."""

from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, cast

from ragway.exceptions import RagError, ValidationError
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node
from ragway.validators import validate_positive_int
from ragway.vectorstores.base_vectorstore import BaseVectorStore


class WeaviateClientProtocol(Protocol):
    """Protocol for async Weaviate client operations used by adapter."""

    async def insert(self, class_name: str, nodes: list[Node]) -> None:
        """Insert nodes into Weaviate class."""

    async def query(
        self,
        class_name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[Node]:
        """Query Weaviate class for nearest neighbors."""

    async def delete(self, class_name: str, node_ids: list[str]) -> None:
        """Delete nodes from Weaviate class."""


class _DefaultWeaviateClient:
    """Async wrapper for synchronous Weaviate cloud client operations."""

    def __init__(self, *, url: str, api_key: str) -> None:
        try:
            weaviate = importlib.import_module("weaviate")
        except ImportError as exc:
            raise RagError(f"weaviate package is required: {exc}") from exc

        self._client = self._connect_client(weaviate, url=url, api_key=api_key)

    def _connect_client(self, weaviate: Any, *, url: str, api_key: str) -> Any:
        auth = None
        auth_cls = getattr(getattr(weaviate, "auth", object()), "AuthApiKey", None)
        if auth_cls is not None:
            auth = auth_cls(api_key)
        else:
            auth_builder = getattr(getattr(getattr(weaviate, "classes", object()), "init", object()), "Auth", None)
            api_key_fn = getattr(auth_builder, "api_key", None)
            if callable(api_key_fn):
                auth = api_key_fn(api_key)

        connect = getattr(weaviate, "connect_to_weaviate_cloud", None)
        if connect is None:
            raise RagError("weaviate.connect_to_weaviate_cloud is required")

        try:
            return connect(cluster_url=url, auth_credentials=auth)
        except TypeError:
            return connect(url=url, auth_credentials=auth)

    def _ensure_collection(self, class_name: str) -> Any:
        collections = getattr(self._client, "collections", None)
        if collections is None:
            raise RagError("Weaviate client collections API is required")

        exists_fn = getattr(collections, "exists", None)
        create_fn = getattr(collections, "create", None)

        exists = False
        if callable(exists_fn):
            exists = bool(exists_fn(class_name))

        if not exists and callable(create_fn):
            try:
                create_fn(name=class_name)
            except TypeError:
                create_fn(class_name)

        get_fn = getattr(collections, "get", None)
        if not callable(get_fn):
            raise RagError("Weaviate collections.get is required")
        return get_fn(class_name)

    async def insert(self, class_name: str, nodes: list[Node]) -> None:
        """Insert nodes into a Weaviate collection."""

        def _insert() -> None:
            collection = self._ensure_collection(class_name)
            data_api = getattr(collection, "data", None)
            if data_api is None:
                raise RagError("Weaviate collection data API is required")

            insert_many = getattr(data_api, "insert_many", None)
            if callable(insert_many):
                try:
                    data_module = importlib.import_module("weaviate.classes.data")
                    data_object_cls = getattr(data_module, "DataObject")
                    objects = []
                    for node in nodes:
                        assert node.embedding is not None
                        objects.append(
                            data_object_cls(
                                uuid=node.node_id,
                                properties={
                                    "content": node.content,
                                    "doc_id": node.doc_id,
                                    "source": node.metadata.source,
                                },
                                vector=node.embedding,
                            )
                        )
                    insert_many(objects)
                    return
                except Exception:
                    pass

            insert_one = getattr(data_api, "insert", None)
            if callable(insert_one):
                for node in nodes:
                    assert node.embedding is not None
                    insert_one(
                        uuid=node.node_id,
                        properties={
                            "content": node.content,
                            "doc_id": node.doc_id,
                            "source": node.metadata.source,
                        },
                        vector=node.embedding,
                    )
                return

            raise RagError("Weaviate insert API was not found")

        await asyncio.get_running_loop().run_in_executor(None, _insert)

    async def query(self, class_name: str, query_vector: list[float], top_k: int) -> list[Node]:
        """Run near-vector search against Weaviate collection."""

        def _query() -> list[Node]:
            collection = self._ensure_collection(class_name)
            query_api = getattr(collection, "query", None)
            if query_api is None:
                raise RagError("Weaviate query API is required")

            near_vector = getattr(query_api, "near_vector", None)
            if not callable(near_vector):
                raise RagError("Weaviate near_vector query API is required")

            response = near_vector(near_vector=query_vector, limit=top_k)
            objects = getattr(response, "objects", None)
            if objects is None and isinstance(response, dict):
                objects = response.get("objects", [])

            mapped: list[Node] = []
            for obj in objects or []:
                properties = getattr(obj, "properties", None)
                uuid = getattr(obj, "uuid", None)
                if isinstance(obj, dict):
                    properties = obj.get("properties", {})
                    uuid = obj.get("uuid")
                if not isinstance(properties, dict):
                    properties = {}
                node_id = str(uuid or properties.get("node_id") or "")
                mapped.append(
                    Node(
                        node_id=node_id,
                        doc_id=str(properties.get("doc_id", node_id)),
                        content=str(properties.get("content", "")),
                        metadata=Metadata.model_validate(properties),
                    )
                )
            return mapped

        return await asyncio.get_running_loop().run_in_executor(None, _query)

    async def hybrid_query(self, class_name: str, query: str, query_vector: list[float], top_k: int) -> list[Node]:
        """Run hybrid bm25+near-vector search against Weaviate."""

        def _query() -> list[Node]:
            collection = self._ensure_collection(class_name)
            query_api = getattr(collection, "query", None)
            if query_api is None:
                raise RagError("Weaviate query API is required")

            hybrid = getattr(query_api, "hybrid", None)
            if callable(hybrid):
                response = hybrid(query=query, vector=query_vector, limit=top_k)
            else:
                bm25 = getattr(query_api, "bm25", None)
                near_vector = getattr(query_api, "near_vector", None)
                if callable(bm25):
                    response = bm25(query=query, limit=top_k)
                elif callable(near_vector):
                    response = near_vector(near_vector=query_vector, limit=top_k)
                else:
                    raise RagError("Weaviate hybrid search API is required")

            objects = getattr(response, "objects", None)
            if objects is None and isinstance(response, dict):
                objects = response.get("objects", [])

            mapped: list[Node] = []
            for obj in objects or []:
                properties = getattr(obj, "properties", None)
                uuid = getattr(obj, "uuid", None)
                if isinstance(obj, dict):
                    properties = obj.get("properties", {})
                    uuid = obj.get("uuid")
                if not isinstance(properties, dict):
                    properties = {}
                node_id = str(uuid or properties.get("node_id") or "")
                mapped.append(
                    Node(
                        node_id=node_id,
                        doc_id=str(properties.get("doc_id", node_id)),
                        content=str(properties.get("content", "")),
                        metadata=Metadata.model_validate(properties),
                    )
                )
            return mapped

        return await asyncio.get_running_loop().run_in_executor(None, _query)

    async def delete(self, class_name: str, node_ids: list[str]) -> None:
        """Delete nodes by ids from Weaviate collection."""

        def _delete() -> None:
            collection = self._ensure_collection(class_name)
            data_api = getattr(collection, "data", None)
            if data_api is None:
                raise RagError("Weaviate collection data API is required")

            delete_many = getattr(data_api, "delete_many", None)
            if callable(delete_many):
                try:
                    delete_many(where={"path": ["id"], "operator": "ContainsAny", "valueTextArray": node_ids})
                except Exception:
                    for node_id in node_ids:
                        try:
                            data_api.delete_by_id(node_id)
                        except Exception:
                            continue
                return

            delete_by_id = getattr(data_api, "delete_by_id", None)
            if callable(delete_by_id):
                for node_id in node_ids:
                    delete_by_id(node_id)
                return

            raise RagError("Weaviate delete API was not found")

        await asyncio.get_running_loop().run_in_executor(None, _delete)


@dataclass(slots=True)
class WeaviateStore(BaseVectorStore):
    """Adapter that routes vector operations to an async Weaviate client."""

    class_name: str
    url: str | None = None
    api_key: str | None = None
    client: WeaviateClientProtocol | None = None

    def _client_instance(self) -> WeaviateClientProtocol:
        """Return injected client or lazily construct default Weaviate client."""
        if self.client is not None:
            return self.client

        url = self.url or os.getenv("WEAVIATE_URL")
        api_key = self.api_key or os.getenv("WEAVIATE_API_KEY")
        if not url:
            raise RagError("WEAVIATE_URL environment variable is required")
        if not api_key:
            raise RagError("WEAVIATE_API_KEY environment variable is required")

        self.client = _DefaultWeaviateClient(url=url, api_key=api_key)
        return self.client

    async def add(self, nodes: list[Node]) -> None:
        """Insert nodes into Weaviate class."""
        for node in nodes:
            if node.embedding is None:
                raise ValidationError("Node embedding is required for WeaviateStore.add")
        await self._client_instance().insert(self.class_name, nodes)

    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        """Search Weaviate class for top-k nearest nodes."""
        top_k = validate_positive_int(top_k, "top_k")
        return await self._client_instance().query(self.class_name, query_vector, top_k)

    async def hybrid_search(self, query: str, query_vector: list[float], top_k: int) -> list[Node]:
        """Search Weaviate with a hybrid near-vector plus BM25 query."""
        top_k = validate_positive_int(top_k, "top_k")
        client = self._client_instance()
        hybrid_query = getattr(client, "hybrid_query", None)
        if not callable(hybrid_query):
            raise RagError("Configured Weaviate client does not support hybrid search")
        typed_hybrid_query = cast(
            Callable[[str, str, list[float], int], Awaitable[list[Node]]],
            hybrid_query,
        )
        return await typed_hybrid_query(self.class_name, query, query_vector, top_k)

    async def delete(self, node_ids: list[str]) -> None:
        """Delete node ids from Weaviate class."""
        await self._client_instance().delete(self.class_name, node_ids)

