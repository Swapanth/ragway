from __future__ import annotations

import asyncio
import os
import sys
import types

import pytest

from ragway.core.component_registry import ComponentRegistry
from ragway.embeddings.bge_embedding import BGEEmbedding
from ragway.embeddings.cohere_embedding import CohereEmbedding
from ragway.embeddings.instructor_embedding import InstructorEmbedding
from ragway.embeddings.openai_embedding import OpenAIEmbedding
from ragway.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from ragway.exceptions import RagError
from ragway.generation.anthropic_llm import AnthropicLLM
from ragway.generation.azure_openai_llm import AzureOpenAILLM
from ragway.generation.bedrock_llm import BedrockLLM
from ragway.generation.groq_llm import GroqLLM
from ragway.generation.llama_llm import LlamaLLM
from ragway.generation.local_llm import LocalLLM
from ragway.generation.mistral_llm import MistralLLM
from ragway.generation.openai_llm import OpenAILLM
from ragway.generation.vertex_ai_llm import VertexAILLM
from ragway.ingestion.docx_loader import DocxLoader
from ragway.ingestion.excel_loader import ExcelLoader
from ragway.ingestion.notion_loader import NotionLoader
from ragway.ingestion.pdf_loader import PDFLoader
from ragway.ingestion.web_loader import WebLoader
from ragway.ingestion.youtube_loader import YouTubeLoader
from ragway.reranking.bge_reranker import BGEReranker
from ragway.reranking.cohere_reranker import CohereReranker
from ragway.reranking.cross_encoder_reranker import CrossEncoderReranker
from ragway.schema.node import Node
from ragway.retrieval.bm25_retriever import BM25Retriever
from ragway.retrieval.hybrid_retriever import HybridRetriever
from ragway.retrieval.multi_query_retriever import MultiQueryRetriever
from ragway.retrieval.parent_document_retriever import ParentDocumentRetriever
from ragway.retrieval.vector_retriever import VectorRetriever
from ragway.vectorstores.chroma_store import ChromaStore
from ragway.vectorstores.faiss_store import FAISSStore
from ragway.vectorstores.pinecone_store import PineconeStore
from ragway.vectorstores.pgvector_store import PGVectorStore
from ragway.vectorstores.qdrant_store import QdrantStore
from ragway.vectorstores.weaviate_store import WeaviateStore


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("anthropic", AnthropicLLM),
        ("openai", OpenAILLM),
        ("mistral", MistralLLM),
        ("groq", GroqLLM),
        ("vertex_ai", VertexAILLM),
        ("azure_openai", AzureOpenAILLM),
        ("bedrock", BedrockLLM),
        ("llama", LlamaLLM),
        ("local", LocalLLM),
    ],
)
async def test_llm_provider_resolution(provider: str, expected: type) -> None:
    """Every LLM provider string should resolve to the expected class."""
    instance = ComponentRegistry.get_llm({"provider": provider, "model": "test-model"})
    assert isinstance(instance, expected)


@pytest.mark.parametrize(
    ("provider", "expected", "config"),
    [
        ("faiss", FAISSStore, {}),
        ("chroma", ChromaStore, {"index_name": "idx"}),
        ("pinecone", PineconeStore, {"index_name": "idx"}),
        ("weaviate", WeaviateStore, {"index_name": "idx"}),
        ("qdrant", QdrantStore, {"index_name": "idx"}),
        ("pgvector", PGVectorStore, {"index_name": "idx"}),
    ],
)
async def test_vectorstore_provider_resolution(provider: str, expected: type, config: dict[str, object]) -> None:
    """Every vectorstore provider string should resolve to the expected class."""
    payload = {"provider": provider, **config}
    instance = ComponentRegistry.get_vectorstore(payload)
    assert isinstance(instance, expected)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("cohere", CohereReranker),
        ("bge", BGEReranker),
        ("cross_encoder", CrossEncoderReranker),
    ],
)
async def test_reranker_provider_resolution(provider: str, expected: type) -> None:
    """Every reranker provider string should resolve to the expected class."""
    instance = ComponentRegistry.get_reranker({"provider": provider})
    assert isinstance(instance, expected)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", OpenAIEmbedding),
        ("cohere", CohereEmbedding),
        ("bge", BGEEmbedding),
        ("sentence_transformer", SentenceTransformerEmbedding),
        ("instructor", InstructorEmbedding),
    ],
)
async def test_embedding_provider_resolution(provider: str, expected: type) -> None:
    """Every embedding provider string should resolve to the expected class."""
    instance = ComponentRegistry.get_embedding({"provider": provider})
    assert isinstance(instance, expected)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("web", WebLoader),
        ("pdf", PDFLoader),
        ("docx", DocxLoader),
        ("excel", ExcelLoader),
        ("youtube", YouTubeLoader),
        ("notion", NotionLoader),
    ],
)
async def test_loader_provider_resolution(provider: str, expected: type) -> None:
    """Every loader provider string should resolve to the expected class."""
    instance = ComponentRegistry.get_loader({"provider": provider})
    assert isinstance(instance, expected)


async def test_unknown_provider_raises_helpful_error() -> None:
    """Unknown providers should raise RagError with available options."""
    with pytest.raises(RagError, match="Unknown llm provider"):
        ComponentRegistry.get_llm({"provider": "does-not-exist"})


async def test_numeric_coercion_helpers_handle_invalid_values() -> None:
    assert ComponentRegistry._as_int("7", 1) == 7
    assert ComponentRegistry._as_int("bad", 3) == 3
    assert ComponentRegistry._as_float("0.7", 0.0) == pytest.approx(0.7)
    assert ComponentRegistry._as_float("bad", 0.3) == pytest.approx(0.3)


async def test_get_llm_uses_numeric_fallbacks() -> None:
    llm = ComponentRegistry.get_llm({"provider": "openai", "temperature": "bad", "max_tokens": "bad"})
    assert isinstance(llm, OpenAILLM)
    assert llm.config.temperature == 0.0
    assert llm.config.max_tokens == 256


async def test_get_chunker_all_branch_strategies() -> None:
    recursive = ComponentRegistry.get_chunker({"strategy": "recursive", "chunk_size": 111})
    semantic = ComponentRegistry.get_chunker({"strategy": "semantic", "chunk_size": 222})
    sliding = ComponentRegistry.get_chunker({"strategy": "sliding_window", "chunk_size": 100, "overlap": 20})
    hierarchical = ComponentRegistry.get_chunker({"strategy": "hierarchical", "chunk_size": 300, "overlap": 30})

    assert recursive.__class__.__name__ == "RecursiveChunker"
    assert semantic.__class__.__name__ == "SemanticChunker"
    assert sliding.__class__.__name__ == "SlidingWindowChunker"
    assert hierarchical.__class__.__name__ == "HierarchicalChunker"


async def test_get_retriever_all_supported_strategies() -> None:
    vectorstore = ComponentRegistry.get_vectorstore({"provider": "faiss"})
    embedding = ComponentRegistry.get_embedding({"provider": "openai"})
    llm = ComponentRegistry.get_llm({"provider": "openai"})
    nodes = [
        Node(node_id="n1", doc_id="d1", content="alpha"),
        Node(node_id="n2", doc_id="d1", content="beta", parent_id="n1"),
    ]

    vector_retriever = ComponentRegistry.get_retriever(
        {"strategy": "vector", "top_k": "5"},
        vectorstore,
        embedding_model=embedding,
        llm=llm,
        nodes=nodes,
    )
    bm25_retriever = ComponentRegistry.get_retriever(
        {"strategy": "bm25", "top_k": "5"},
        vectorstore,
        embedding_model=embedding,
        llm=llm,
        nodes=nodes,
    )
    hybrid_retriever = ComponentRegistry.get_retriever(
        {"strategy": "hybrid", "rrf_k": "60"},
        vectorstore,
        embedding_model=embedding,
        llm=llm,
        nodes=nodes,
    )
    multi_query_retriever = ComponentRegistry.get_retriever(
        {"strategy": "multi_query", "query_count": "2", "rrf_k": "55"},
        vectorstore,
        embedding_model=embedding,
        llm=llm,
        nodes=nodes,
    )
    parent_document_retriever = ComponentRegistry.get_retriever(
        {"strategy": "parent_document"},
        vectorstore,
        embedding_model=embedding,
        llm=llm,
        nodes=nodes,
    )

    assert isinstance(vector_retriever, VectorRetriever)
    assert isinstance(bm25_retriever, BM25Retriever)
    assert isinstance(hybrid_retriever, HybridRetriever)
    assert isinstance(multi_query_retriever, MultiQueryRetriever)
    assert isinstance(parent_document_retriever, ParentDocumentRetriever)


async def test_get_retriever_validates_required_dependencies() -> None:
    """Retriever strategies should raise when required dependencies are missing."""
    vectorstore = ComponentRegistry.get_vectorstore({"provider": "faiss"})
    embedding = ComponentRegistry.get_embedding({"provider": "openai"})
    llm = ComponentRegistry.get_llm({"provider": "openai"})

    with pytest.raises(RagError, match="Vector retriever requires"):
        ComponentRegistry.get_retriever({"strategy": "vector"}, vectorstore, embedding_model=None, llm=llm)

    with pytest.raises(RagError, match="Hybrid retriever requires"):
        ComponentRegistry.get_retriever({"strategy": "hybrid"}, vectorstore, embedding_model=None, llm=llm)

    with pytest.raises(RagError, match="MultiQuery retriever requires an llm"):
        ComponentRegistry.get_retriever(
            {"strategy": "multi_query"}, vectorstore, embedding_model=embedding, llm=None
        )

    with pytest.raises(RagError, match="ParentDocument retriever requires"):
        ComponentRegistry.get_retriever(
            {"strategy": "parent_document"}, vectorstore, embedding_model=None, llm=llm
        )


async def test_get_retriever_unknown_strategy_raises() -> None:
    """Unknown retrieval strategies should raise a helpful RagError."""
    vectorstore = ComponentRegistry.get_vectorstore({"provider": "faiss"})
    with pytest.raises(RagError, match="Unknown retrieval provider"):
        ComponentRegistry.get_retriever({"strategy": "unknown"}, vectorstore)


async def test_get_reranker_empty_provider_returns_none() -> None:
    """get_reranker should return None when provider is missing."""
    assert ComponentRegistry.get_reranker({}) is None


async def test_openai_compatible_embedding_client_fallback_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI-compatible embedding client should produce deterministic fallback vectors without API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding = ComponentRegistry.get_embedding({"provider": "openai", "max_batch_size": 2})

    vectors = await embedding.embed(["abc", "xyz"])
    assert len(vectors) == 2
    assert all(isinstance(item, list) for item in vectors)
    assert all(len(item) > 0 for item in vectors)


async def test_openai_compatible_embedding_client_sdk_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI-compatible embedding client should use AsyncOpenAI when key is configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Embeddings:
        async def create(self, *, model: str, input: list[str]):
            del model, input
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.1, 0.2])])

    class _AsyncOpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.embeddings = _Embeddings()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_AsyncOpenAI))
    embedding = ComponentRegistry.get_embedding({"provider": "openai", "model": "text-embedding-3-small"})

    vectors = await embedding.embed(["hello"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 2
    assert vectors[0][0] == pytest.approx(0.44721359, rel=1e-5)
    assert vectors[0][1] == pytest.approx(0.89442719, rel=1e-5)

