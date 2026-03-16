from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.exceptions import ValidationError
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node
from ragway.vectorstores.pgvector_store import (
    PGVectorStore,
    _DefaultPGVectorClient,
    _parse_vector_text,
    _validate_identifier,
    _vector_literal,
)


async def test_pgvector_store_calls_client_methods() -> None:
    """PGVectorStore should delegate add/search/delete to async client."""
    client = AsyncMock()
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=[1.0, 0.0])
    client.search.return_value = [node]

    store = PGVectorStore(table_name="rag_nodes", client=client)
    await store.add([node])
    result = await store.search([1.0, 0.0], top_k=1)
    await store.delete(["n1"])

    assert len(result) == 1
    assert result[0].node_id == "n1"
    client.initialize.assert_any_await("rag_nodes", 2)
    client.upsert.assert_awaited_once_with("rag_nodes", [node])
    client.search.assert_awaited_once_with("rag_nodes", [1.0, 0.0], 1)
    client.delete.assert_awaited_once_with("rag_nodes", ["n1"])


async def test_pgvector_store_rejects_missing_embedding() -> None:
    """PGVectorStore add should reject nodes without embeddings."""
    client = AsyncMock()
    store = PGVectorStore(table_name="rag_nodes", client=client)
    node = Node(node_id="n1", doc_id="d1", content="alpha", embedding=None)

    with pytest.raises(ValidationError):
        await store.add([node])


async def test_pgvector_store_rejects_empty_query_vector() -> None:
    """PGVectorStore search should reject empty query vectors."""
    client = AsyncMock()
    store = PGVectorStore(table_name="rag_nodes", client=client)

    with pytest.raises(ValidationError):
        await store.search([], top_k=1)


async def test_pgvector_helpers_validate_and_parse() -> None:
    """Helper functions should validate identifiers and round-trip vectors."""
    assert _validate_identifier("rag_nodes", "table_name") == "rag_nodes"
    with pytest.raises(ValidationError):
        _validate_identifier("bad-name", "table_name")

    literal = _vector_literal([1.0, 2.5, 3.0])
    assert literal == "[1.0,2.5,3.0]"
    assert _parse_vector_text("[1.0,2.5,3.0]") == [1.0, 2.5, 3.0]
    assert _parse_vector_text("[]") == []


async def test_pgvector_store_requires_connection_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """PGVectorStore should require a connection string when no client is injected."""
    monkeypatch.delenv("PGVECTOR_CONNECTION_STRING", raising=False)
    store = PGVectorStore(table_name="rag_nodes")
    with pytest.raises(RagError, match="PGVECTOR_CONNECTION_STRING"):
        store._client_instance()


class _Acquire:
    def __init__(self, connection: object) -> None:
        self.connection = connection

    async def __aenter__(self) -> object:
        return self.connection

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.rows: list[dict[str, object]] = []

    async def execute(self, query: str, *args: object) -> None:
        self.executed.append((query, args))

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.executed.append((query, args))
        return self.rows


class _FakePool:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> _Acquire:
        return _Acquire(self.connection)


async def test_default_pgvector_client_initialize_upsert_search_delete() -> None:
    """Default pgvector client should execute initialize/upsert/search/delete SQL paths."""
    conn = _FakeConnection()
    pool = _FakePool(conn)
    client = _DefaultPGVectorClient("postgresql://demo")
    client._pool = pool

    await client.initialize("rag_nodes", 3)

    node = Node(
        node_id="n1",
        doc_id="d1",
        content="alpha",
        embedding=[1.0, 0.0, 0.5],
        metadata=Metadata(source="s"),
    )
    await client.upsert("rag_nodes", [node])

    conn.rows = [
        {
            "node_id": "n1",
            "doc_id": "d1",
            "content": "alpha",
            "embedding_text": "[1.0,0.0,0.5]",
            "metadata": json.loads(node.metadata.model_dump_json()),
            "parent_id": None,
            "position": None,
        }
    ]
    results = await client.search("rag_nodes", [1.0, 0.0, 0.5], 1)
    await client.delete("rag_nodes", ["n1"])

    assert len(results) == 1
    assert results[0].node_id == "n1"
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in query for query, _ in conn.executed)


async def test_default_pgvector_client_initialize_rejects_non_positive_vector_size() -> None:
    """initialize should reject non-positive vector sizes."""
    client = _DefaultPGVectorClient("postgresql://demo")
    with pytest.raises(ValidationError, match="vector_size must be positive"):
        await client.initialize("rag_nodes", 0)
