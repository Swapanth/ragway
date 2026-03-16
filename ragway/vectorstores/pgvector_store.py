"""PostgreSQL pgvector store adapter using asyncpg."""

from __future__ import annotations

import json
import importlib
import os
import re
from dataclasses import dataclass
from typing import Protocol

from ragway.exceptions import RagError, ValidationError
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node
from ragway.validators import validate_positive_int
from ragway.vectorstores.base_vectorstore import BaseVectorStore


def _validate_identifier(identifier: str, field_name: str) -> str:
    """Validate SQL identifier-like names used for table names."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier):
        raise ValidationError(f"{field_name} must be a valid SQL identifier")
    return identifier


def _vector_literal(values: list[float]) -> str:
    """Convert a vector into pgvector text literal format."""
    return "[" + ",".join(str(value) for value in values) + "]"


def _parse_vector_text(value: str) -> list[float]:
    """Parse pgvector text output like [1,2,3] into a float list."""
    stripped = value.strip().strip("[]")
    if not stripped:
        return []
    return [float(part.strip()) for part in stripped.split(",")]


class PGVectorClientProtocol(Protocol):
    """Protocol for async pgvector client operations."""

    async def initialize(self, table_name: str, vector_size: int) -> None:
        """Ensure pgvector extension and target table exist."""

    async def upsert(self, table_name: str, nodes: list[Node]) -> None:
        """Insert or update nodes in the target table."""

    async def search(self, table_name: str, query_vector: list[float], top_k: int) -> list[Node]:
        """Search nearest neighbors in the target table."""

    async def delete(self, table_name: str, node_ids: list[str]) -> None:
        """Delete nodes by ids from the target table."""


class _DefaultPGVectorClient:
    """Default asyncpg-backed implementation for pgvector operations."""

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._pool: object | None = None
        self._initialized_tables: set[str] = set()

    async def _pool_instance(self) -> object:
        """Create asyncpg pool on first use."""
        if self._pool is None:
            try:
                asyncpg = importlib.import_module("asyncpg")
            except ImportError as exc:
                raise RagError(f"asyncpg package is required: {exc}") from exc
            self._pool = await asyncpg.create_pool(dsn=self._connection_string)
        return self._pool

    async def initialize(self, table_name: str, vector_size: int) -> None:
        """Create pgvector extension and target table if missing."""
        if table_name in self._initialized_tables:
            return
        if vector_size <= 0:
            raise ValidationError("vector_size must be positive")

        pool = await self._pool_instance()
        async with pool.acquire() as connection:  # type: ignore[attr-defined]
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    node_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({vector_size}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    parent_id TEXT NULL,
                    position INTEGER NULL
                )
                """
            )
        self._initialized_tables.add(table_name)

    async def upsert(self, table_name: str, nodes: list[Node]) -> None:
        """Insert or update nodes in pgvector table."""
        pool = await self._pool_instance()
        async with pool.acquire() as connection:  # type: ignore[attr-defined]
            for node in nodes:
                assert node.embedding is not None
                await connection.execute(
                    f"""
                    INSERT INTO {table_name}
                        (node_id, doc_id, content, embedding, metadata, parent_id, position)
                    VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6, $7)
                    ON CONFLICT (node_id)
                    DO UPDATE SET
                        doc_id = EXCLUDED.doc_id,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        parent_id = EXCLUDED.parent_id,
                        position = EXCLUDED.position
                    """,
                    node.node_id,
                    node.doc_id,
                    node.content,
                    _vector_literal(node.embedding),
                    json.dumps(node.metadata.model_dump(mode="json")),
                    node.parent_id,
                    node.position,
                )

    async def search(self, table_name: str, query_vector: list[float], top_k: int) -> list[Node]:
        """Search nodes by cosine distance using pgvector operator."""
        pool = await self._pool_instance()
        async with pool.acquire() as connection:  # type: ignore[attr-defined]
            rows = await connection.fetch(
                f"""
                SELECT
                    node_id,
                    doc_id,
                    content,
                    embedding::text AS embedding_text,
                    metadata,
                    parent_id,
                    position
                FROM {table_name}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                _vector_literal(query_vector),
                top_k,
            )

        results: list[Node] = []
        for row in rows:
            metadata_raw = row["metadata"]
            metadata_dict = metadata_raw if isinstance(metadata_raw, dict) else {}
            metadata = Metadata.model_validate(metadata_dict)
            embedding = _parse_vector_text(str(row["embedding_text"]))
            results.append(
                Node(
                    node_id=str(row["node_id"]),
                    doc_id=str(row["doc_id"]),
                    content=str(row["content"]),
                    embedding=embedding,
                    metadata=metadata,
                    parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
                    position=int(row["position"]) if row["position"] is not None else None,
                )
            )
        return results

    async def delete(self, table_name: str, node_ids: list[str]) -> None:
        """Delete nodes by id from pgvector table."""
        if not node_ids:
            return
        pool = await self._pool_instance()
        async with pool.acquire() as connection:  # type: ignore[attr-defined]
            await connection.execute(f"DELETE FROM {table_name} WHERE node_id = ANY($1::text[])", node_ids)


@dataclass(slots=True)
class PGVectorStore(BaseVectorStore):
    """Adapter that routes vector operations to a pgvector-backed PostgreSQL table."""

    table_name: str = "rag_nodes"
    connection_string: str | None = None
    client: PGVectorClientProtocol | None = None

    def __post_init__(self) -> None:
        """Validate static configuration values."""
        self.table_name = _validate_identifier(self.table_name, "table_name")

    def _client_instance(self) -> PGVectorClientProtocol:
        """Return configured client or lazily construct default asyncpg client."""
        if self.client is not None:
            return self.client

        conn = self.connection_string or os.getenv("PGVECTOR_CONNECTION_STRING")
        if not conn:
            raise RagError("PGVECTOR_CONNECTION_STRING environment variable is required")
        default_client = _DefaultPGVectorClient(connection_string=conn)
        self.client = default_client
        return default_client

    async def add(self, nodes: list[Node]) -> None:
        """Insert or update nodes in pgvector table."""
        if not nodes:
            return
        for node in nodes:
            if node.embedding is None:
                raise ValidationError("Node embedding is required for PGVectorStore.add")

        vector_size = len(nodes[0].embedding or [])
        await self._client_instance().initialize(self.table_name, vector_size)
        await self._client_instance().upsert(self.table_name, nodes)

    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        """Search pgvector table for nearest neighbors."""
        if not query_vector:
            raise ValidationError("query_vector must not be empty")
        top_k = validate_positive_int(top_k, "top_k")

        await self._client_instance().initialize(self.table_name, len(query_vector))
        return await self._client_instance().search(self.table_name, query_vector, top_k)

    async def delete(self, node_ids: list[str]) -> None:
        """Delete node ids from pgvector table."""
        await self._client_instance().delete(self.table_name, node_ids)
