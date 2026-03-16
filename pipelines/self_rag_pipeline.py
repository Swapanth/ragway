"""Self-RAG pipeline with context grading and self-correction loop."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ragway.chunking.fixed_chunker import FixedChunker
from ragway.components.hallucination_detector import HallucinationDetector
from ragway.components.query_expansion import QueryExpansion
from ragway.core.dependency_container import DependencyContainer
from ragway.core.rag_engine import RagConfig
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
        """Return deterministic outputs for grading and answer generation."""
        del model, temperature, max_tokens, api_key

        lowered = prompt.lower()
        if "is this context relevant to answer" in lowered:
            # Simple lexical heuristic for Yes/No grading.
            try:
                query_part = prompt.split("answer:", maxsplit=1)[1].split("?", maxsplit=1)[0].lower()
            except IndexError:
                query_part = ""
            context_part = prompt.lower()
            query_terms = {term for term in query_part.split() if term.isalpha()}
            if query_terms and any(term in context_part for term in query_terms):
                return "Yes"
            return "No"

        prompt_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        if not prompt_lines:
            return "No prompt content provided."
        return f"Answer based on context: {prompt_lines[-1]}"


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from .env into process environment."""
    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


async def _seed_nodes(embedding_model: OpenAIEmbedding, vector_store: FAISSStore) -> list[Node]:
    """Create and index baseline chunks used by the self-RAG pipeline."""
    document = Document(
        doc_id="doc-001",
        content=(
            "rag-toolkit is a modular Retrieval-Augmented Generation framework. "
            "Self-RAG adds self-correction through context relevance checks and hallucination checks. "
            "Query expansion can retry retrieval when relevant context is missing."
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
    return nodes_with_embeddings


async def _build_container(llm: LLMProtocol | None = None) -> DependencyContainer:
    """Construct and register all self-RAG dependencies in DI container."""
    _load_env_file()

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
    query_expansion = QueryExpansion(llm=llm_instance, variant_count=3)
    hallucination_detector = HallucinationDetector()

    container.register_instance("retriever", retriever)
    container.register_instance("prompt_builder", prompt_builder)
    container.register_instance("llm", llm_instance)
    container.register_instance("query_expansion", query_expansion)
    container.register_instance("hallucination_detector", hallucination_detector)
    return container


def build_pipeline(
    llm: LLMProtocol | None = None,
    vectorstore: object | None = None,
    embedding: object | None = None,
    retriever: object | None = None,
    reranker: object | None = None,
    chunker: object | None = None,
) -> RAGPipeline:
    """Build and return the self-RAG pipeline definition."""
    llm = llm or AnthropicLLM()
    _ = (llm, vectorstore, embedding, retriever, reranker, chunker)
    return RAGPipeline(
        name="self_rag",
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
) -> dict:
    """Run self-RAG flow and return answer with metadata."""
    _ = (vectorstore, embedding, retriever, reranker, chunker)
    container = await _build_container(llm=llm)
    pipeline = build_pipeline(llm=llm)

    retriever = container.resolve("retriever")
    prompt_builder = container.resolve("prompt_builder")
    llm = container.resolve("llm")
    query_expansion = container.resolve("query_expansion")
    hallucination_detector = container.resolve("hallucination_detector")

    config = pipeline.config
    candidate_queries = await query_expansion.expand(query)

    relevant_nodes: list[Node] = []
    used_query = query
    retries_used = 0

    for attempt, candidate_query in enumerate(candidate_queries[:3]):
        used_query = candidate_query
        retrieved_nodes = await retriever.retrieve(candidate_query, top_k=config.top_k)

        graded_nodes: list[Node] = []
        for node in retrieved_nodes:
            grade_prompt = f"Is this context relevant to answer: {candidate_query}? Yes/No\n\nContext:\n{node.content}"
            grade = (await llm.generate(grade_prompt)).strip().lower()
            if grade.startswith("yes"):
                graded_nodes.append(node)

        if graded_nodes:
            relevant_nodes = graded_nodes
            retries_used = attempt
            break

        retries_used = attempt + 1

    if not relevant_nodes:
        relevant_nodes = await retriever.retrieve(query, top_k=config.top_k)

    base_prompt = prompt_builder.build(query=used_query, nodes=relevant_nodes)
    answer = await llm.generate(base_prompt)

    grounding_score = hallucination_detector.score(answer, relevant_nodes)
    hallucination_score = 1.0 - grounding_score

    if hallucination_score > 0.4:
        strict_prompt = (
            base_prompt
            + "\n\nAnswer only using the provided context."
        )
        answer = await llm.generate(strict_prompt)
        grounding_score = hallucination_detector.score(answer, relevant_nodes)
        hallucination_score = 1.0 - grounding_score

    return {
        "answer": answer,
        "metadata": {
            "query_used": used_query,
            "retries_used": retries_used,
            "relevant_nodes": len(relevant_nodes),
            "hallucination_score": round(hallucination_score, 4),
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = asyncio.run(run("How does self-rag reduce hallucination?"))
    logger.info("Answer: %s", output["answer"])
    logger.info("Metadata: %s", output["metadata"])

