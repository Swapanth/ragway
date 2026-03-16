from __future__ import annotations

import asyncio
import types
from typing import AsyncIterator

import pytest

from ragway.exceptions import RagError
from ragway.rag import RAG


class _FakeEngine:
    def __init__(self) -> None:
        self.ingest_calls: list[str] = []

    async def ingest(self, source: str) -> int:
        self.ingest_calls.append(source)
        return 1

    async def query(self, question: str, stream: bool = False) -> str | AsyncIterator[str]:
        if stream:
            async def _iterator() -> AsyncIterator[str]:
                yield "answer:"
                yield question

            return _iterator()
        return f"answer:{question}"

    async def query_with_sources(self, question: str) -> dict[str, object]:
        return {
            "answer": f"answer:{question}",
            "sources": [{"node_id": "n1", "doc_id": "d1", "content": "ctx"}],
        }

    async def evaluate(self, dataset_path: str) -> dict[str, float]:
        del dataset_path
        return {"overall_score": 0.8}


async def test_rag_lazy_init_and_api_key_shortcut(monkeypatch) -> None:
    rag = RAG(llm="anthropic", api_key="sk-ant-test")

    assert rag._engine is None
    assert rag._api_keys["anthropic"] == "sk-ant-test"


async def test_rag_switch_returns_new_instance() -> None:
    rag = RAG(llm="anthropic", top_k=5)
    switched = rag.switch(llm="groq", top_k=10)

    assert rag is not switched
    assert rag._settings["llm"] == "anthropic"
    assert switched._settings["llm"] == "groq"
    assert rag._settings["top_k"] == 5
    assert switched._settings["top_k"] == 10


async def test_query_with_sources_includes_shape(monkeypatch) -> None:
    rag = RAG()
    rag._engine = _FakeEngine()

    payload = await rag.query_with_sources("What is RAG?")

    assert payload["answer"] == "answer:What is RAG?"
    assert isinstance(payload["sources"], list)
    assert isinstance(payload["scores"], list)
    assert isinstance(payload["latency_ms"], float)


async def test_ingest_with_glob(monkeypatch, tmp_path) -> None:
    rag = RAG()
    engine = _FakeEngine()
    rag._engine = engine

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "a.md").write_text("a", encoding="utf-8")
    (docs_dir / "b.txt").write_text("b", encoding="utf-8")

    count = await rag.ingest(str(docs_dir), glob="*.md")

    assert count == 1
    assert len(engine.ingest_calls) == 1
    assert engine.ingest_calls[0].endswith("a.md")


async def test_query_stream_mode_returns_iterator() -> None:
    rag = RAG()
    rag._engine = _FakeEngine()

    async def _collect() -> list[str]:
        stream = await rag.query("What is RAG?", stream=True)
        chunks: list[str] = []
        async for token in stream:
            chunks.append(token)
        return chunks

    assert await _collect() == ["answer:", "What is RAG?"]


async def test_from_dict_sectioned_and_flat_configs() -> None:
    sectioned = {
        "pipeline": "hybrid",
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.1},
        "vectorstore": {"provider": "faiss"},
        "retrieval": {"strategy": "vector", "top_k": 7},
        "reranking": {"enabled": True, "provider": "cohere"},
        "chunking": {"strategy": "fixed", "chunk_size": 256, "overlap": 16},
    }
    rag_a = RAG.from_dict(sectioned)
    assert rag_a._settings["pipeline"] == "hybrid"
    assert rag_a._settings["llm"] == "openai"
    assert rag_a._settings["top_k"] == 7

    flat = {
        "pipeline": "naive",
        "chunk_size": "300",
        "overlap": "20",
        "top_k": "4",
        "temperature": "0.5",
    }
    rag_b = RAG.from_dict(flat)
    assert rag_b._settings["chunk_size"] == 300
    assert rag_b._settings["overlap"] == 20
    assert rag_b._settings["top_k"] == 4


async def test_from_config_file_happy_path(tmp_path) -> None:
    config_file = tmp_path / "rag_config.yaml"
    config_file.write_text("pipeline: naive\nllm: anthropic\n", encoding="utf-8")
    rag = RAG.from_config(str(config_file))
    assert rag._settings["pipeline"] == "naive"


async def test_set_key_unknown_provider_raises() -> None:
    rag = RAG()
    with pytest.raises(ValueError, match="Unknown provider"):
        rag.set_key("unknown-provider", "x")


async def test_set_keys_and_switch_api_keys() -> None:
    rag = RAG()
    rag.set_keys(openai="sk-openai", anthropic="sk-ant")
    switched = rag.switch(api_keys={"groq": "sk-groq"}, api_key="sk-openai-new", llm="openai")

    assert rag._api_keys["openai"] == "sk-openai"
    assert switched._api_keys["groq"] == "sk-groq"
    assert switched._api_keys["openai"] == "sk-openai-new"


async def test_build_config_handles_disabled_reranker() -> None:
    rag = RAG(reranker=None, top_k=9, chunk_size=222, overlap=11)
    cfg = rag._build_config()
    assert cfg["pipeline"] == "naive"
    assert isinstance(cfg["reranking"], dict)
    assert cfg["reranking"]["enabled"] is False
    assert cfg["retrieval"]["top_k"] == 9
    assert cfg["chunking"]["chunk_size"] == 222


async def test_dependency_error_mapping_and_fallback() -> None:
    rag = RAG(llm="openai", vectorstore="")
    mapped = rag._dependency_error(ImportError("openai required"))
    assert "ragway[openai]" in str(mapped)

    generic = rag._dependency_error(RuntimeError("mystery failure"))
    assert "Missing optional dependency" in str(generic)


async def test_evaluate_filters_metrics_with_in_memory_rows() -> None:
    rag = RAG()
    rag._engine = _FakeEngine()

    dataset = [{"question": "What is RAG?", "gold_answer": "Retrieval augmented generation"}]
    scores = await rag.evaluate(dataset, metrics=["overall_score"])
    assert list(scores.keys()) == ["overall_score"]


async def test_from_config_missing_file_raises(tmp_path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        RAG.from_config(str(missing))


async def test_from_config_invalid_root_raises(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(RagError):
        RAG.from_config(str(path))


async def test_evaluate_dataset_path_branch_uses_engine() -> None:
    rag = RAG()
    rag._engine = _FakeEngine()
    scores = await rag.evaluate("/tmp/eval.json")
    assert scores["overall_score"] == 0.8


async def test_ensure_engine_import_errors_are_mapped(monkeypatch) -> None:
    rag = RAG(llm="openai", vectorstore="")
    rag._engine = None

    def _fake_import(name, *args, **kwargs):
        if name == "ragway.raglab":
            raise ImportError("openai required")
        return original_import(name, *args, **kwargs)

    import builtins

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match=r"ragway\[openai\]"):
        rag._ensure_engine()


async def test_ensure_engine_ragerror_dependency_path(monkeypatch) -> None:
    rag = RAG(llm="openai", vectorstore="")
    rag._engine = None

    class _FakeRAGLab:
        @classmethod
        def from_dict(cls, config):
            del config
            raise RagError("provider required install")

    monkeypatch.setitem(__import__("sys").modules, "ragway.raglab", types.SimpleNamespace(RAGLab=_FakeRAGLab))

    with pytest.raises(ImportError, match=r"ragway\[openai\]"):
        rag._ensure_engine()


async def test_as_int_and_as_float_handle_magic_methods() -> None:
    """Numeric coercion helpers should handle __int__/__float__ and fallback values."""

    class _IntLike:
        def __int__(self) -> int:
            return 9

    class _FloatLike:
        def __float__(self) -> float:
            return 0.75

    assert RAG._as_int(_IntLike(), 1) == 9
    assert RAG._as_float(_FloatLike(), 0.1) == pytest.approx(0.75)


async def test_dependency_error_generic_when_no_provider_match() -> None:
    """Dependency mapping should return generic error when provider is not inferable."""
    rag = RAG(llm="llama", vectorstore="faiss", reranker=None)
    err = rag._dependency_error(RuntimeError("something unexpected"))
    assert "Missing optional dependency" in str(err)
