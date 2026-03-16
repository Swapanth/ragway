from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path

import pytest

from ragway.core.component_registry import ComponentRegistry
from ragway.exceptions import RagError
from ragway.raglab import RAGLab
from ragway.schema.document import Document
from ragway.schema.node import Node


class _FakeLLM:
    async def generate(self, prompt: str) -> str:
        del prompt
        return "mock-answer"

    def stream(self, prompt: str):
        async def _iterator():
            yield "stream:"
            yield prompt[:8]

        return _iterator()


class _FakeEmbedding:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] for index, _ in enumerate(texts)]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.nodes: list[Node] = []

    async def add(self, nodes: list[Node]) -> None:
        self.nodes.extend(nodes)

    async def search(self, query_vector: list[float], top_k: int) -> list[Node]:
        del query_vector
        return self.nodes[:top_k]

    async def delete(self, node_ids: list[str]) -> None:
        self.nodes = [node for node in self.nodes if node.node_id not in node_ids]


class _FakeRetriever:
    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = nodes

    async def retrieve(self, query: str, top_k: int) -> list[Node]:
        del query
        return self.nodes[:top_k]


class _FakeChunker:
    def chunk(self, document: Document) -> list[Node]:
        return [
            Node(
                node_id=f"{document.doc_id}-0",
                doc_id=document.doc_id,
                content=document.content,
                metadata=document.metadata,
                position=0,
            )
        ]


def _mock_registry(monkeypatch) -> _FakeVectorStore:
    store = _FakeVectorStore()

    monkeypatch.setattr(ComponentRegistry, "get_llm", classmethod(lambda cls, cfg: _FakeLLM()))
    monkeypatch.setattr(ComponentRegistry, "get_embedding", classmethod(lambda cls, cfg: _FakeEmbedding()))
    monkeypatch.setattr(ComponentRegistry, "get_vectorstore", classmethod(lambda cls, cfg: store))
    monkeypatch.setattr(ComponentRegistry, "get_chunker", classmethod(lambda cls, cfg: _FakeChunker()))
    monkeypatch.setattr(ComponentRegistry, "get_reranker", classmethod(lambda cls, cfg: None))
    monkeypatch.setattr(
        ComponentRegistry,
        "get_retriever",
        classmethod(
            lambda cls, cfg, vectorstore, embedding_model=None, llm=None, nodes=None: _FakeRetriever(nodes or [])
        ),
    )

    return store


async def test_raglab_from_config_ingest_and_query(monkeypatch, tmp_path: Path) -> None:
    """RAGLab should load from config, ingest documents, and answer queries."""
    _mock_registry(monkeypatch)

    config_path = tmp_path / "rag_config.yaml"
    config_path.write_text(
        """
pipeline: naive
llm:
  provider: anthropic
embedding:
  provider: bge
vectorstore:
  provider: faiss
retrieval:
  strategy: vector
  top_k: 5
reranking:
  enabled: false
chunking:
  strategy: recursive
  chunk_size: 512
""".strip(),
        encoding="utf-8",
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "sample.txt").write_text("Retrieval augmented generation", encoding="utf-8")

    lab = RAGLab.from_config(str(config_path))
    chunk_count = await lab.ingest(str(docs_dir))
    answer = await lab.query("What is RAG?")
    with_sources = await lab.query_with_sources("What is RAG?")

    assert chunk_count == 1
    assert answer == "mock-answer"
    assert with_sources["answer"] == "mock-answer"
    assert isinstance(with_sources["sources"], list)
    assert len(with_sources["sources"]) == 1


async def test_raglab_from_dict_and_switch(monkeypatch) -> None:
    """RAGLab should build from dict and support component switching."""
    _mock_registry(monkeypatch)

    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )
    switched = lab.switch(llm="mistral", vectorstore="pinecone")

    assert switched.config["llm"]["provider"] == "mistral"
    assert switched.config["vectorstore"]["provider"] == "pinecone"


async def test_raglab_evaluate(monkeypatch, tmp_path: Path) -> None:
    """RAGLab should evaluate a dataset path and return metric summary."""
    _mock_registry(monkeypatch)

    monkeypatch.setattr(
        "ragway.raglab.RagasEval.run",
        lambda self, dataset, pipeline_name: {"overall_score": 0.9, "faithfulness": 0.8},
    )

    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(
        json.dumps([{"question": "q1", "gold_answer": "g1", "context": ["ctx"]}]),
        encoding="utf-8",
    )

    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    scores = await lab.evaluate(str(dataset_path))

    assert scores["overall_score"] == 0.9
    assert scores["faithfulness"] == 0.8


async def test_raglab_section_requires_mapping(monkeypatch) -> None:
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )
    lab.config["retrieval"] = "not-a-mapping"
    with pytest.raises(RagError):
        lab._section("retrieval")


async def test_raglab_switch_rejects_bad_section_types(monkeypatch) -> None:
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )
    lab.config["llm"] = "bad"
    with pytest.raises(RagError):
        lab.switch(llm="openai")


async def test_raglab_switch_updates_dict_branches(monkeypatch) -> None:
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    switched = lab.switch(
        pipeline="hybrid",
        llm={"provider": "openai", "temperature": 0.3},
        vectorstore={"provider": "chroma", "index_name": "idx"},
        embedding={"provider": "openai", "model": "text-embedding-3-small"},
        retriever={"strategy": "bm25", "top_k": 9},
        reranker={"enabled": True, "provider": "bge"},
        chunker={"strategy": "fixed", "chunk_size": 128},
    )

    assert switched.config["pipeline"] == "hybrid"
    assert switched.config["llm"]["provider"] == "openai"
    assert switched.config["retrieval"]["strategy"] == "bm25"
    assert switched.config["reranking"]["provider"] == "bge"
    assert switched.config["chunking"]["strategy"] == "fixed"


async def test_raglab_stream_query(monkeypatch) -> None:
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    async def _collect() -> list[str]:
        stream = await lab.query("What is RAG?", stream=True)
        chunks: list[str] = []
        async for item in stream:
            chunks.append(item)
        return chunks

    chunks = await _collect()
    assert chunks


async def test_raglab_load_documents_path_variants(monkeypatch, tmp_path: Path) -> None:
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("markdown text", encoding="utf-8")
    (docs / "b.rst").write_text("rst text", encoding="utf-8")
    (docs / "c.txt").write_text("txt text", encoding="utf-8")

    parsed_pdf = Document(doc_id="p1", content="pdf text")
    monkeypatch.setattr("ragway.raglab.PDFParser.parse", lambda self, b, source=None, doc_id=None: parsed_pdf)
    (docs / "d.pdf").write_bytes(b"%PDF")

    loaded = await lab._load_documents(str(docs))
    assert len(loaded) >= 4


async def test_raglab_load_documents_http_and_missing(monkeypatch) -> None:
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    class _FakeWebLoader:
        async def load(self, source: str) -> list[Document]:
            return [Document(doc_id="w1", content=source)]

    monkeypatch.setitem(__import__("sys").modules, "ragway.ingestion.web_loader", types.SimpleNamespace(WebLoader=_FakeWebLoader))

    docs = await lab._load_documents("https://example.com")
    assert docs[0].doc_id == "w1"

    with pytest.raises(RagError):
        await lab._load_documents("/path/does/not/exist")


async def test_raglab_from_config_errors(tmp_path: Path) -> None:
    """from_config should validate missing files and YAML root type."""
    with pytest.raises(RagError, match="Config file not found"):
        RAGLab.from_config(str(tmp_path / "missing.yaml"))

    bad = tmp_path / "bad.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(RagError, match="must contain a mapping"):
        RAGLab.from_config(str(bad))


async def test_raglab_evaluate_errors(monkeypatch, tmp_path: Path) -> None:
    """evaluate should validate dataset file presence and list payload format."""
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    with pytest.raises(RagError, match="Dataset file not found"):
        await lab.evaluate(str(tmp_path / "missing.json"))

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"question": "q1"}), encoding="utf-8")
    with pytest.raises(RagError, match="must be a list"):
        await lab.evaluate(str(bad))


async def test_raglab_ingest_handles_empty_documents(monkeypatch) -> None:
    """ingest should return 0 when no documents are loaded."""
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    async def _empty_load(source: str) -> list[Document]:
        del source
        return []

    monkeypatch.setattr(lab, "_load_documents", _empty_load)
    assert await lab.ingest("ignored") == 0


async def test_raglab_ingest_validates_chunker_contract(monkeypatch) -> None:
    """ingest should validate chunker callable and payload shape."""
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    async def _one_doc(source: str) -> list[Document]:
        del source
        return [Document(doc_id="d1", content="text")]

    monkeypatch.setattr(lab, "_load_documents", _one_doc)

    lab.chunker = object()
    with pytest.raises(RagError, match="does not expose chunk"):
        await lab.ingest("ignored")

    class _BadChunker:
        def chunk(self, document: Document) -> str:
            del document
            return "bad"

    lab.chunker = _BadChunker()
    with pytest.raises(RagError, match="invalid chunk payload"):
        await lab.ingest("ignored")


async def test_raglab_switch_handles_string_branches(monkeypatch) -> None:
    """switch should handle string branch updates for all configurable sections."""
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    switched = lab.switch(
        vectorstore="qdrant",
        embedding="openai",
        retriever="hybrid",
        reranker="cohere",
        chunker="fixed",
    )

    assert switched.config["vectorstore"]["provider"] == "qdrant"
    assert switched.config["embedding"]["provider"] == "openai"
    assert switched.config["retrieval"]["strategy"] == "hybrid"
    assert switched.config["reranking"]["enabled"] is True
    assert switched.config["chunking"]["strategy"] == "fixed"


@pytest.mark.parametrize("section,key,value", [
    ("vectorstore", "vectorstore", "faiss"),
    ("embedding", "embedding", "bge"),
    ("retrieval", "retriever", "vector"),
    ("reranking", "reranker", "cohere"),
    ("chunking", "chunker", "fixed"),
])
async def test_raglab_switch_rejects_non_mapping_sections(monkeypatch, section: str, key: str, value: str) -> None:
    """switch should reject non-mapping config values for each mutable section."""
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )
    lab.config[section] = "bad"
    with pytest.raises(RagError, match="must be a mapping"):
        lab.switch(**{key: value})


async def test_raglab_as_int_magic_and_fallback() -> None:
    """_as_int should use __int__ values and fallback on conversion errors."""

    class _IntLike:
        def __int__(self) -> int:
            return 7

    class _BadInt:
        def __int__(self) -> str:
            return "bad"

    assert RAGLab._as_int(_IntLike(), 1) == 7
    assert RAGLab._as_int(_BadInt(), 5) == 5


async def test_raglab_load_documents_single_file_and_skip_empty_txt(monkeypatch, tmp_path: Path) -> None:
    """_load_documents should support single-file input and ignore empty text files."""
    _mock_registry(monkeypatch)
    lab = RAGLab.from_dict(
        {
            "pipeline": "naive",
            "llm": {"provider": "groq"},
            "vectorstore": {"provider": "faiss"},
            "embedding": {"provider": "bge"},
            "retrieval": {"strategy": "vector", "top_k": 5},
            "reranking": {"enabled": False},
            "chunking": {"strategy": "recursive", "chunk_size": 512},
        }
    )

    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   ", encoding="utf-8")
    assert await lab._load_documents(str(empty_file)) == []

    filled_file = tmp_path / "note.txt"
    filled_file.write_text("hello", encoding="utf-8")
    docs = await lab._load_documents(str(filled_file))
    assert len(docs) == 1
    assert docs[0].content == "hello"

