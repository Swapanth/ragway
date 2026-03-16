"""Disk-backed retrieval cache with LRU eviction."""

from __future__ import annotations

import hashlib
import os
import shelve
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from ragway.schema.node import Node

_ORDER_KEY = "__order__"


@dataclass(slots=True)
class RetrievalCache:
    """Caches retrieved nodes by query hash and top-k using disk-backed storage."""

    max_size: int = 1000
    _db_path: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize cache storage path from environment configuration."""
        cache_root = Path(os.getenv("RAG_CACHE_DIR", ".cache/rag"))
        cache_root.mkdir(parents=True, exist_ok=True)
        self._db_path = str(cache_root / "retrieval_cache")

    def get(self, query: str, top_k: int) -> list[Node] | None:
        """Return cached retrieval results for query and top_k if present."""
        key = self._key_for_query(query, top_k)
        with shelve.open(self._db_path) as db:
            if key not in db:
                return None
            rows = cast(list[dict[str, object]], db[key])
            self._touch_key(db, key)
            return [Node.model_validate(row) for row in rows]

    def set(self, query: str, top_k: int, nodes: list[Node]) -> None:
        """Store retrieval results and enforce LRU max size."""
        key = self._key_for_query(query, top_k)
        with shelve.open(self._db_path, writeback=True) as db:
            db[key] = [node.model_dump() for node in nodes]
            self._touch_key(db, key)
            self._evict_if_needed(db)

    def _key_for_query(self, query: str, top_k: int) -> str:
        """Build deterministic cache key for retrieval request."""
        payload = f"{query}|{top_k}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _touch_key(self, db: shelve.Shelf[object], key: str) -> None:
        """Mark key as most recently used in LRU order."""
        order = cast(list[str], db.get(_ORDER_KEY, []))
        if key in order:
            order.remove(key)
        order.append(key)
        db[_ORDER_KEY] = order

    def _evict_if_needed(self, db: shelve.Shelf[object]) -> None:
        """Evict least recently used items until cache size is within limit."""
        order = cast(list[str], db.get(_ORDER_KEY, []))
        while len(order) > self.max_size:
            oldest = order.pop(0)
            if oldest in db:
                del db[oldest]
        db[_ORDER_KEY] = order

