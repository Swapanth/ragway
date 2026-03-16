"""Disk-backed LLM response cache with LRU eviction."""

from __future__ import annotations

import hashlib
import os
import shelve
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

_ORDER_KEY = "__order__"


@dataclass(slots=True)
class LLMCache:
    """Caches generated LLM responses by prompt hash using disk-backed storage."""

    max_size: int = 1000
    _db_path: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize cache storage path from environment configuration."""
        cache_root = Path(os.getenv("RAG_CACHE_DIR", ".cache/rag"))
        cache_root.mkdir(parents=True, exist_ok=True)
        self._db_path = str(cache_root / "llm_cache")

    def get(self, prompt: str) -> str | None:
        """Return cached response for prompt if present."""
        key = self._key_for_prompt(prompt)
        with shelve.open(self._db_path) as db:
            if key not in db:
                return None
            response = cast(str, db[key])
            self._touch_key(db, key)
            return response

    def set(self, prompt: str, response: str) -> None:
        """Store response for prompt and enforce LRU max size."""
        key = self._key_for_prompt(prompt)
        with shelve.open(self._db_path, writeback=True) as db:
            db[key] = response
            self._touch_key(db, key)
            self._evict_if_needed(db)

    def _key_for_prompt(self, prompt: str) -> str:
        """Build deterministic cache key for prompt payload."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

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
