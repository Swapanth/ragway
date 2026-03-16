"""Disk-backed embedding cache with LRU eviction."""

from __future__ import annotations

import hashlib
import os
import shelve
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

_ORDER_KEY = "__order__"


@dataclass(slots=True)
class EmbeddingCache:
    """Caches embedding vectors by text hash using disk-backed key-value storage."""

    max_size: int = 1000
    _db_path: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize cache storage path from environment configuration."""
        cache_root = Path(os.getenv("RAG_CACHE_DIR", ".cache/rag"))
        cache_root.mkdir(parents=True, exist_ok=True)
        self._db_path = str(cache_root / "embedding_cache")

    def get(self, text: str) -> list[float] | None:
        """Return cached embedding vector for text if present."""
        key = self._key_for_text(text)
        with shelve.open(self._db_path) as db:
            if key not in db:
                return None
            vector = cast(list[float], db[key])
            self._touch_key(db, key)
            return list(vector)

    def set(self, text: str, vector: list[float]) -> None:
        """Store embedding vector for text and enforce LRU max size."""
        key = self._key_for_text(text)
        with shelve.open(self._db_path, writeback=True) as db:
            db[key] = list(vector)
            self._touch_key(db, key)
            self._evict_if_needed(db)

    def _key_for_text(self, text: str) -> str:
        """Build deterministic cache key for text payload."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
