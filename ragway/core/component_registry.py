"""Registry for mapping YAML config provider keys to concrete component instances."""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from typing import Any, cast

from ragway.chunking.base_chunker import BaseChunker
from ragway.chunking.fixed_chunker import FixedChunker
from ragway.chunking.hierarchical_chunker import HierarchicalChunker
from ragway.chunking.recursive_chunker import RecursiveChunker
from ragway.chunking.semantic_chunker import SemanticChunker
from ragway.chunking.sliding_window_chunker import SlidingWindowChunker
from ragway.embeddings.bge_embedding import BGEEmbedding
from ragway.embeddings.cohere_embedding import CohereEmbedding
from ragway.embeddings.instructor_embedding import InstructorEmbedding
from ragway.embeddings.openai_embedding import OpenAIEmbedding
from ragway.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from ragway.exceptions import RagError
from ragway.generation.anthropic_llm import AnthropicLLM
from ragway.generation.azure_openai_llm import AzureOpenAILLM
from ragway.generation.base_llm import LLMConfig
from ragway.generation.bedrock_llm import BedrockLLM
from ragway.generation.groq_llm import GroqLLM
from ragway.generation.llama_llm import LlamaLLM
from ragway.generation.local_llm import LocalLLM
from ragway.generation.mistral_llm import MistralLLM
from ragway.generation.openai_llm import OpenAILLM
from ragway.generation.vertex_ai_llm import VertexAILLM
from ragway.ingestion.base_loader import BaseLoader
from ragway.ingestion.docx_loader import DocxLoader
from ragway.ingestion.excel_loader import ExcelLoader
from ragway.ingestion.notion_loader import NotionLoader
from ragway.ingestion.pdf_loader import PDFLoader
from ragway.ingestion.web_loader import WebLoader
from ragway.ingestion.youtube_loader import YouTubeLoader
from ragway.interfaces.embedding_protocol import EmbeddingProtocol
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.interfaces.reranker_protocol import RerankerProtocol
from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.reranking.bge_reranker import BGEReranker
from ragway.reranking.cohere_reranker import CohereReranker
from ragway.reranking.cross_encoder_reranker import CrossEncoderReranker
from ragway.retrieval.bm25_retriever import BM25Retriever
from ragway.retrieval.hybrid_retriever import HybridRetriever
from ragway.retrieval.multi_query_retriever import MultiQueryRetriever
from ragway.retrieval.parent_document_retriever import ParentDocumentRetriever
from ragway.retrieval.vector_retriever import VectorRetriever
from ragway.schema.node import Node
from ragway.vectorstores.base_vectorstore import BaseVectorStore
from ragway.vectorstores.chroma_store import ChromaStore
from ragway.vectorstores.faiss_store import FAISSStore
from ragway.vectorstores.pinecone_store import PineconeStore
from ragway.vectorstores.pgvector_store import PGVectorStore
from ragway.vectorstores.qdrant_store import QdrantStore
from ragway.vectorstores.weaviate_store import WeaviateStore


class _OpenAICompatibleEmbeddingClient:
    """Async embedding client with OpenAI API support and deterministic fallback."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed via OpenAI API when configured, otherwise use deterministic local vectors."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=api_key)
                response = await client.embeddings.create(model=model, input=texts)
                return [list(item.embedding) for item in response.data]
            except Exception:
                pass

        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0 for _ in range(self.dimensions)]
            for index, character in enumerate(text):
                bucket = index % self.dimensions
                vector[bucket] += (ord(character) % 29) / 29.0
            vectors.append(vector)
        return vectors


class ComponentRegistry:
    """Maps config provider keys to concrete component classes and instances."""

    LLM_PROVIDERS = {
        "anthropic": AnthropicLLM,
        "openai": OpenAILLM,
        "mistral": MistralLLM,
        "groq": GroqLLM,
        "vertex_ai": VertexAILLM,
        "azure_openai": AzureOpenAILLM,
        "bedrock": BedrockLLM,
        "llama": LlamaLLM,
        "local": LocalLLM,
    }
    VECTORSTORES = {
        "faiss": FAISSStore,
        "chroma": ChromaStore,
        "pinecone": PineconeStore,
        "weaviate": WeaviateStore,
        "qdrant": QdrantStore,
        "pgvector": PGVectorStore,
    }
    RETRIEVERS = {
        "vector": VectorRetriever,
        "bm25": BM25Retriever,
        "hybrid": HybridRetriever,
        "multi_query": MultiQueryRetriever,
        "parent_document": ParentDocumentRetriever,
    }
    RERANKERS = {
        "cohere": CohereReranker,
        "bge": BGEReranker,
        "cross_encoder": CrossEncoderReranker,
    }
    CHUNKERS = {
        "fixed": FixedChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "sliding_window": SlidingWindowChunker,
        "hierarchical": HierarchicalChunker,
    }
    EMBEDDINGS = {
        "openai": OpenAIEmbedding,
        "cohere": CohereEmbedding,
        "bge": BGEEmbedding,
        "sentence_transformer": SentenceTransformerEmbedding,
        "instructor": InstructorEmbedding,
    }
    LOADERS = {
        "web": WebLoader,
        "pdf": PDFLoader,
        "docx": DocxLoader,
        "excel": ExcelLoader,
        "youtube": YouTubeLoader,
        "notion": NotionLoader,
    }
    PIPELINES = {
        "naive": "pipelines.naive_rag_pipeline",
        "hybrid": "pipelines.hybrid_rag_pipeline",
        "self": "pipelines.self_rag_pipeline",
        "long_context": "pipelines.long_context_rag_pipeline",
        "agentic": "pipelines.agentic_rag_pipeline",
    }

    LLM_DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
        "mistral": "mistral-large-latest",
        "groq": "llama-3.3-70b-versatile",
        "vertex_ai": "gemini-1.5-pro",
        "azure_openai": "gpt-4o",
        "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "llama": "llama-3.1-8b-instruct",
        "local": "local-default-model",
    }

    @classmethod
    def _filter_constructor_kwargs(cls, component_cls: type[object], kwargs: dict[str, object]) -> dict[str, object]:
        """Filter kwargs to those accepted by a component constructor."""
        signature = inspect.signature(component_cls)
        accepted = set(signature.parameters.keys())
        return {key: value for key, value in kwargs.items() if key in accepted}

    @classmethod
    def _as_int(cls, value: object, default: int) -> int:
        """Safely coerce config value to int with fallback default."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, (str, bytes, bytearray)):
            try:
                return int(value)
            except ValueError:
                return default
        try:
            int_method = getattr(value, "__int__")
            parsed = int_method()
            return int(parsed) if isinstance(parsed, int) else default
        except Exception:
            return default

    @classmethod
    def _as_float(cls, value: object, default: float) -> float:
        """Safely coerce config value to float with fallback default."""
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (str, bytes, bytearray)):
            try:
                return float(value)
            except ValueError:
                return default
        try:
            float_method = getattr(value, "__float__")
            parsed = float_method()
            return float(parsed) if isinstance(parsed, float) else default
        except Exception:
            return default

    @classmethod
    def _provider_class(cls, providers: Mapping[str, type[Any]], provider: str, kind: str) -> type[Any]:
        """Resolve provider class or raise a helpful RagError."""
        if provider not in providers:
            options = ", ".join(sorted(providers.keys()))
            raise RagError(f"Unknown {kind} provider '{provider}'. Choose from: {options}")
        return providers[provider]

    @classmethod
    def get_llm(cls, config: dict[str, object]) -> LLMProtocol:
        """Instantiate an LLM adapter from provider and model config."""
        provider = str(config.get("provider", "anthropic"))
        llm_cls = cls._provider_class(cls.LLM_PROVIDERS, provider, "llm")
        default_model = cls.LLM_DEFAULT_MODELS.get(provider, "gpt-4o")

        llm_config = LLMConfig(
            model=str(config.get("model", "")) or default_model,
            temperature=cls._as_float(config.get("temperature", 0.0), 0.0),
            max_tokens=cls._as_int(config.get("max_tokens", 256), 256),
        )
        return cast(LLMProtocol, llm_cls(config=llm_config))

    @classmethod
    def get_vectorstore(cls, config: dict[str, object]) -> BaseVectorStore:
        """Instantiate a vector store from provider config."""
        provider = str(config.get("provider", "faiss"))
        store_cls = cls._provider_class(cls.VECTORSTORES, provider, "vectorstore")

        kwargs = {key: value for key, value in config.items() if key != "provider"}
        # Friendly alias support for stores with required names.
        if provider == "chroma" and "collection" not in kwargs:
            kwargs["collection"] = str(kwargs.get("index_name", "default"))
        if provider == "weaviate" and "class_name" not in kwargs:
            kwargs["class_name"] = str(kwargs.get("index_name", "RagNode"))
        if provider == "qdrant" and "collection_name" not in kwargs:
            kwargs["collection_name"] = str(kwargs.get("index_name", "default"))
        if provider == "pgvector" and "table_name" not in kwargs:
            kwargs["table_name"] = str(kwargs.get("index_name", "rag_nodes"))

        filtered = cls._filter_constructor_kwargs(store_cls, kwargs)
        return cast(BaseVectorStore, store_cls(**filtered))

    @classmethod
    def get_retriever(
        cls,
        config: dict[str, object],
        vectorstore: BaseVectorStore,
        *,
        embedding_model: EmbeddingProtocol | None = None,
        llm: LLMProtocol | None = None,
        nodes: list[Node] | None = None,
    ) -> RetrieverProtocol:
        """Instantiate a retriever strategy from config and dependencies."""
        strategy = str(config.get("strategy", "vector"))
        retriever_cls = cls._provider_class(cls.RETRIEVERS, strategy, "retrieval")
        top_k = cls._as_int(config.get("top_k", 5), 5)

        corpus_nodes = list(nodes) if nodes is not None else []

        if strategy == "vector":
            if embedding_model is None:
                raise RagError("Vector retriever requires an embedding_model instance")
            _ = top_k
            return cast(RetrieverProtocol, retriever_cls(embedding_model=embedding_model, vector_store=vectorstore))

        if strategy == "bm25":
            _ = top_k
            return cast(RetrieverProtocol, retriever_cls(nodes=corpus_nodes))

        if strategy == "hybrid":
            if embedding_model is None:
                raise RagError("Hybrid retriever requires an embedding_model instance")
            vector_retriever = VectorRetriever(embedding_model=embedding_model, vector_store=vectorstore)
            bm25_retriever = BM25Retriever(nodes=corpus_nodes)
            rrf_k = cls._as_int(config.get("rrf_k", 60), 60)
            _ = top_k
            return cast(
                RetrieverProtocol,
                retriever_cls(bm25_retriever=bm25_retriever, vector_retriever=vector_retriever, rrf_k=rrf_k),
            )

        if strategy == "multi_query":
            if llm is None:
                raise RagError("MultiQuery retriever requires an llm instance")
            if embedding_model is None:
                raise RagError("MultiQuery retriever requires an embedding_model instance")
            base = VectorRetriever(embedding_model=embedding_model, vector_store=vectorstore)
            query_count = cls._as_int(config.get("query_count", 3), 3)
            rrf_k = cls._as_int(config.get("rrf_k", 60), 60)
            _ = top_k
            return cast(
                RetrieverProtocol,
                retriever_cls(retriever=base, llm=llm, query_count=query_count, rrf_k=rrf_k),
            )

        if strategy == "parent_document":
            if embedding_model is None:
                raise RagError("ParentDocument retriever requires an embedding_model instance")
            base = VectorRetriever(embedding_model=embedding_model, vector_store=vectorstore)
            parent_nodes = {node.node_id: node for node in corpus_nodes if node.parent_id is None}
            _ = top_k
            return cast(RetrieverProtocol, retriever_cls(child_retriever=base, parent_nodes=parent_nodes))

        raise RagError(f"Unsupported retrieval strategy '{strategy}'")

    @classmethod
    def get_reranker(cls, config: dict[str, object]) -> RerankerProtocol | None:
        """Instantiate an optional reranker from provider config."""
        provider = str(config.get("provider", ""))
        if not provider:
            return None

        reranker_cls = cls._provider_class(cls.RERANKERS, provider, "reranker")
        kwargs = {key: value for key, value in config.items() if key not in {"provider", "enabled", "top_k"}}
        if provider == "cross_encoder" and "model_name" not in kwargs and "model" in config:
            kwargs["model_name"] = config["model"]

        filtered = cls._filter_constructor_kwargs(reranker_cls, kwargs)
        return cast(RerankerProtocol, reranker_cls(**filtered))

    @classmethod
    def get_chunker(cls, config: dict[str, object]) -> BaseChunker:
        """Instantiate a chunker strategy from config."""
        strategy = str(config.get("strategy", "fixed"))
        chunker_cls = cls._provider_class(cls.CHUNKERS, strategy, "chunker")

        kwargs: dict[str, object] = {key: value for key, value in config.items() if key != "strategy"}
        if strategy == "recursive" and "max_tokens" not in kwargs and "chunk_size" in kwargs:
            kwargs["max_tokens"] = kwargs["chunk_size"]
        if strategy == "semantic" and "max_tokens" not in kwargs and "chunk_size" in kwargs:
            kwargs["max_tokens"] = kwargs["chunk_size"]
        if strategy == "sliding_window":
            if "window_size" not in kwargs and "chunk_size" in kwargs:
                kwargs["window_size"] = kwargs["chunk_size"]
            if "stride" not in kwargs:
                chunk_size = cls._as_int(kwargs.get("chunk_size", 200), 200)
                overlap = cls._as_int(kwargs.get("overlap", 0), 0)
                kwargs["stride"] = max(1, chunk_size - overlap)
        if strategy == "hierarchical":
            if "parent_chunk_size" not in kwargs and "chunk_size" in kwargs:
                kwargs["parent_chunk_size"] = kwargs["chunk_size"]
            if "child_overlap" not in kwargs and "overlap" in kwargs:
                kwargs["child_overlap"] = kwargs["overlap"]

        filtered = cls._filter_constructor_kwargs(chunker_cls, kwargs)
        return cast(BaseChunker, chunker_cls(**filtered))

    @classmethod
    def get_embedding(cls, config: dict[str, object]) -> EmbeddingProtocol:
        """Instantiate an embedding adapter from provider config."""
        provider = str(config.get("provider", "openai"))
        embedding_cls = cls._provider_class(cls.EMBEDDINGS, provider, "embedding")

        kwargs = {key: value for key, value in config.items() if key != "provider"}
        if provider == "openai" and "max_batch_size" not in kwargs and "batch_size" in kwargs:
            kwargs["max_batch_size"] = kwargs["batch_size"]
        if provider == "openai" and "client" not in kwargs:
            kwargs["client"] = _OpenAICompatibleEmbeddingClient()

        filtered = cls._filter_constructor_kwargs(embedding_cls, kwargs)
        return cast(EmbeddingProtocol, embedding_cls(**filtered))

    @classmethod
    def get_loader(cls, config: dict[str, object]) -> BaseLoader:
        """Instantiate an ingestion loader from provider config."""
        provider = str(config.get("provider", "web"))
        loader_cls = cls._provider_class(cls.LOADERS, provider, "loader")
        kwargs = {key: value for key, value in config.items() if key != "provider"}
        filtered = cls._filter_constructor_kwargs(loader_cls, kwargs)
        return cast(BaseLoader, loader_cls(**filtered))

