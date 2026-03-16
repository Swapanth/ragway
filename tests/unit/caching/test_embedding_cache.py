from __future__ import annotations

from pathlib import Path

from ragway.caching.embedding_cache import EmbeddingCache


def test_embedding_cache_get_set(tmp_path: Path, monkeypatch) -> None:
    """EmbeddingCache should persist vectors on disk and retrieve them by text."""
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))
    cache = EmbeddingCache()

    cache.set("hello", [0.1, 0.2])
    result = cache.get("hello")

    assert result == [0.1, 0.2]


def test_embedding_cache_lru_eviction(tmp_path: Path, monkeypatch) -> None:
    """EmbeddingCache should evict least recently used item when capacity is exceeded."""
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))
    cache = EmbeddingCache(max_size=2)

    cache.set("a", [1.0])
    cache.set("b", [2.0])
    _ = cache.get("a")
    cache.set("c", [3.0])

    assert cache.get("a") == [1.0]
    assert cache.get("b") is None
    assert cache.get("c") == [3.0]

