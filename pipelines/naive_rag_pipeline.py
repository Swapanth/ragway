"""Naive RAG pipeline wiring with fixed chunking, vector retrieval, and generation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ragway.chunking.fixed_chunker import FixedChunker
from ragway.core.dependency_container import DependencyContainer
from ragway.core.rag_engine import RAGEngine, RagConfig
from ragway.core.rag_pipeline import RAGPipeline
from ragway.embeddings.openai_embedding import OpenAIEmbedding
from ragway.generation.anthropic_llm import AnthropicLLM
from ragway.generation.base_llm import LLMConfig
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.prompting.prompt_builder import PromptBuilder
from ragway.retrieval.vector_retriever import VectorRetriever
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.vectorstores.faiss_store import FAISSStore


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalOpenAIEmbeddingClient:
    """Local async embedding client used for runnable pipeline demos."""

    dimensions: int = 8

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Return deterministic vectors for each text to simulate embeddings."""
        del model
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0 for _ in range(self.dimensions)]
            for index, character in enumerate(text):
                bucket = index % self.dimensions
                vector[bucket] += (ord(character) % 29) / 29.0
            vectors.append(vector)
        return vectors


@dataclass(slots=True)
class LocalAnthropicClient:
    """Local async Anthropic-compatible client used for runnable demos."""

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
    ) -> str:
        """Return a deterministic answer string from prompt content."""
        del model, temperature, max_tokens, api_key
        prompt_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        if not prompt_lines:
            return "No prompt content provided."
        return f"Answer based on context: {prompt_lines[-1]}"


async def _seed_nodes(embedding_model: OpenAIEmbedding, vector_store: FAISSStore) -> list[Node]:
    """Create and index baseline chunks used by the naive pipeline."""
    document = Document(
        doc_id="doc-001",
        content=(
            "rag-toolkit is a modular Retrieval-Augmented Generation framework. "
            "It supports chunking, embeddings, vector retrieval, reranking, and generation. "
            "The naive pipeline demonstrates a straightforward retrieve-then-generate flow."
        ),
    )

    chunker = FixedChunker(chunk_size=512, overlap=50)
    nodes = chunker.chunk(document)

    vectors = await embedding_model.embed([node.content for node in nodes])
    nodes_with_embeddings = [
        node.model_copy(update={"embedding": vector})
        for node, vector in zip(nodes, vectors)
    ]
    await vector_store.add(nodes_with_embeddings)
    return nodes


async def _build_container(llm: LLMProtocol | None = None) -> DependencyContainer:
    """Construct and register all naive pipeline dependencies in DI container."""
    container = DependencyContainer()

    embedding_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        max_batch_size=32,
        client=LocalOpenAIEmbeddingClient(),
    )
    vector_store = FAISSStore()
    await _seed_nodes(embedding_model=embedding_model, vector_store=vector_store)

    retriever = VectorRetriever(embedding_model=embedding_model, vector_store=vector_store)
    prompt_builder = PromptBuilder(template_name="default")
    llm_instance = llm or AnthropicLLM(
        config=LLMConfig(model="claude-sonnet-4-6", temperature=0.0, max_tokens=256),
        client=LocalAnthropicClient(),
    )

    container.register_instance("embedding_model", embedding_model)
    container.register_instance("retriever", retriever)
    container.register_instance("prompt_builder", prompt_builder)
    container.register_instance("llm", llm_instance)
    return container


def build_pipeline(
    llm: LLMProtocol | None = None,
    vectorstore: object | None = None,
    embedding: object | None = None,
    retriever: object | None = None,
    reranker: object | None = None,
    chunker: object | None = None,
) -> RAGPipeline:
    """Build and return the naive RAG pipeline definition."""
    llm = llm or AnthropicLLM()
    _ = (llm, vectorstore, embedding, retriever, reranker, chunker)
    return RAGPipeline(
        name="naive",
        config=RagConfig(top_k=5, enable_rerank=False, include_citations=False),
    )


async def run(
    query: str,
    llm: LLMProtocol | None = None,
    vectorstore: object | None = None,
    embedding: object | None = None,
    retriever: object | None = None,
    reranker: object | None = None,
    chunker: object | None = None,
) -> str:
    """Run the naive RAG pipeline asynchronously for one query."""
    _ = (vectorstore, embedding, retriever, reranker, chunker)
    container = await _build_container(llm=llm)
    pipeline = build_pipeline(llm=llm)
    engine: RAGEngine = pipeline.build_engine(container)
    return await engine._run_async(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run("What is rag-toolkit?"))
    logger.info(result)

