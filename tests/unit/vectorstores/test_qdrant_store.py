from __future__ import annotations

import importlib
import types
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.exceptions import ValidationError
from ragway.schema.node import Node
from ragway.vectorstores.qdrant_store import QdrantStore, _DefaultQdrantClient


async def test_qdrant_store_calls_client_methods() -> None:
    """QdrantStore should delegate add/search/delete to async client."""
    client = AsyncMock()
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=[1.0, 0.0])
    client.search.return_value = [node]

    store = QdrantStore(collection_name="demo", client=client)
    await store.add([node])
    result = await store.search([1.0, 0.0], top_k=1)
    await store.delete(["n1"])

    assert len(result) == 1
    assert result[0].node_id == "n1"
    client.upsert.assert_awaited_once_with("demo", [node])
    client.search.assert_awaited_once_with("demo", [1.0, 0.0], 1)
    client.delete.assert_awaited_once_with("demo", ["n1"])


async def test_qdrant_store_rejects_missing_embedding() -> None:
    """QdrantStore add should reject nodes without embeddings."""
    client = AsyncMock()
    store = QdrantStore(client=client)
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=None)

    with pytest.raises(ValidationError):
        await store.add([node])


async def test_qdrant_store_allows_empty_add() -> None:
    """QdrantStore add should no-op for empty input."""
    client = AsyncMock()
    store = QdrantStore(client=client)

    await store.add([])

    client.upsert.assert_not_called()


async def test_qdrant_store_search_validates_top_k() -> None:
    """QdrantStore search should validate positive top_k."""
    store = QdrantStore(client=AsyncMock())
    with pytest.raises(ValidationError):
        await store.search([1.0], top_k=0)


async def test_qdrant_store_default_client_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """QdrantStore should construct default client using env-backed settings."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "q-key")

    class _FakeDefaultClient:
        def __init__(self, *, url, api_key, path, in_memory, dimension) -> None:
            self.args = (url, api_key, path, in_memory, dimension)

        async def upsert(self, collection_name: str, nodes: list[Node]) -> None:
            del collection_name, nodes

        async def search(self, collection_name: str, query_vector: list[float], top_k: int) -> list[Node]:
            del collection_name, query_vector, top_k
            return []

        async def delete(self, collection_name: str, node_ids: list[str]) -> None:
            del collection_name, node_ids

    monkeypatch.setattr("ragway.vectorstores.qdrant_store._DefaultQdrantClient", _FakeDefaultClient)
    store = QdrantStore(collection_name="demo")
    client = store._client_instance()
    assert isinstance(client, _FakeDefaultClient)


async def test_default_qdrant_client_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default qdrant client should map missing package to RagError."""

    def _import_module(name: str):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", _import_module)
    with pytest.raises(RagError, match="qdrant-client package is required"):
        _DefaultQdrantClient(url=None, api_key=None, path=None, in_memory=True, dimension=8)


async def test_default_qdrant_client_core_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default qdrant client should run ensure/upsert/search/delete with translated payloads."""

    class _FakeDistance:
        COSINE = "cosine"

    class _VectorParams:
        def __init__(self, size: int, distance: str) -> None:
            self.size = size
            self.distance = distance

    class _PointStruct:
        def __init__(self, id: str, vector: list[float], payload: dict[str, object]) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    class _PointIdsList:
        def __init__(self, points: list[str]) -> None:
            self.points = points

    class _CollectionsPayload:
        def __init__(self, names: list[str]) -> None:
            self.collections = [types.SimpleNamespace(name=name) for name in names]

    class _FakeAsyncQdrantClient:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.created_collections: list[str] = []
            self.upsert_points: list[object] = []
            self.deleted: list[str] = []

        async def get_collections(self):
            return _CollectionsPayload(self.created_collections)

        async def create_collection(self, collection_name: str, vectors_config: object) -> None:
            del vectors_config
            self.created_collections.append(collection_name)

        async def upsert(self, *, collection_name: str, points: list[object], wait: bool) -> None:
            del collection_name, wait
            self.upsert_points = points

        async def query_points(self, **kwargs):
            del kwargs
            return types.SimpleNamespace(
                points=[
                    types.SimpleNamespace(
                        id="n1",
                        payload={
                            "doc_id": "d1",
                            "content": "alpha",
                            "metadata": {"source": "unit"},
                            "parent_id": "p1",
                            "position": 1,
                        },
                        vector=[1.0, 0.0],
                    )
                ]
            )

        async def delete(self, *, collection_name: str, points_selector: object, wait: bool) -> None:
            del collection_name, wait
            self.deleted = list(points_selector.points)

    models_module = types.SimpleNamespace(
        VectorParams=_VectorParams,
        Distance=_FakeDistance,
        PointStruct=_PointStruct,
        PointIdsList=_PointIdsList,
    )
    qdrant_module = types.SimpleNamespace(AsyncQdrantClient=_FakeAsyncQdrantClient)

    def _import_module(name: str):
        if name == "qdrant_client":
            return qdrant_module
        if name == "qdrant_client.models":
            return models_module
        raise ImportError(name)

    monkeypatch.setattr("ragway.vectorstores.qdrant_store.importlib.import_module", _import_module)

    client = _DefaultQdrantClient(url="http://localhost", api_key="k", path=None, in_memory=False, dimension=8)
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=[1.0, 0.0])

    await client.upsert("demo", [node])
    results = await client.search("demo", [1.0, 0.0], 1)
    await client.delete("demo", ["n1"])

    assert len(results) == 1
    assert results[0].node_id == "n1"
    assert results[0].parent_id == "p1"


async def test_default_qdrant_client_constructor_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default qdrant client should support url, in-memory, and path-based constructor modes."""

    class _FakeAsyncQdrantClient:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    qdrant_module = types.SimpleNamespace(AsyncQdrantClient=_FakeAsyncQdrantClient)
    monkeypatch.setattr("ragway.vectorstores.qdrant_store.importlib.import_module", lambda name: qdrant_module)

    by_url = _DefaultQdrantClient(url="http://localhost", api_key="k", path=None, in_memory=False, dimension=8)
    by_memory = _DefaultQdrantClient(url=None, api_key=None, path=None, in_memory=True, dimension=8)
    by_path = _DefaultQdrantClient(url=None, api_key=None, path="./qdrant", in_memory=False, dimension=8)

    assert by_url._client.kwargs["url"] == "http://localhost"
    assert by_memory._client.kwargs["location"] == ":memory:"
    assert by_path._client.kwargs["path"] == "./qdrant"
