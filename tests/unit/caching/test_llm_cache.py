from __future__ import annotations

from pathlib import Path

from ragway.caching.llm_cache import LLMCache


def test_llm_cache_get_set(tmp_path: Path, monkeypatch) -> None:
    """LLMCache should persist responses on disk and retrieve by prompt."""
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))
    cache = LLMCache()

    cache.set("prompt", "response")
    result = cache.get("prompt")

    assert result == "response"


def test_llm_cache_lru_eviction(tmp_path: Path, monkeypatch) -> None:
    """LLMCache should evict least recently used prompt when full."""
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))
    cache = LLMCache(max_size=2)

    cache.set("p1", "r1")
    cache.set("p2", "r2")
    _ = cache.get("p1")
    cache.set("p3", "r3")

    assert cache.get("p1") == "r1"
    assert cache.get("p2") is None
    assert cache.get("p3") == "r3"

