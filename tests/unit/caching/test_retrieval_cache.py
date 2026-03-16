from __future__ import annotations

from pathlib import Path

from ragway.caching.retrieval_cache import RetrievalCache
from ragway.schema.node import Node


def test_retrieval_cache_get_set(tmp_path: Path, monkeypatch) -> None:
    """RetrievalCache should persist node lists by query and top_k."""
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))
    cache = RetrievalCache()
    nodes = [Node(node_id="n1", doc_id="d1", content="alpha")]

    cache.set("what", 5, nodes)
    result = cache.get("what", 5)

    assert result is not None
    assert len(result) == 1
    assert result[0].node_id == "n1"


def test_retrieval_cache_lru_eviction(tmp_path: Path, monkeypatch) -> None:
    """RetrievalCache should evict least recently used entry beyond max_size."""
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))
    cache = RetrievalCache(max_size=2)

    cache.set("q1", 5, [Node(node_id="n1", doc_id="d1", content="a")])
    cache.set("q2", 5, [Node(node_id="n2", doc_id="d1", content="b")])
    _ = cache.get("q1", 5)
    cache.set("q3", 5, [Node(node_id="n3", doc_id="d1", content="c")])

    assert cache.get("q1", 5) is not None
    assert cache.get("q2", 5) is None
    assert cache.get("q3", 5) is not None

