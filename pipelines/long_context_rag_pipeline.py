"""Long-context RAG pipeline with ordered retrieval and context compression."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ragway.chunking.fixed_chunker import FixedChunker
from ragway.components.context_compression import ContextCompression
from ragway.core.dependency_container import DependencyContainer
from ragway.core.rag_engine import RAGEngine, RagConfig
from ragway.core.rag_pipeline import RAGPipeline
from ragway.embeddings.openai_embedding import OpenAIEmbedding
from ragway.generation.anthropic_llm import AnthropicLLM
from ragway.generation.base_llm import LLMConfig
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.prompting.prompt_registry import PromptRegistry
from ragway.prompting.page_context_formatter import PageContextFormatter
from ragway.retrieval.long_context_retriever import LongContextRetriever
from ragway.retrieval.vector_retriever import VectorRetriever
from ragway.schema.metadata import Metadata
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


@dataclass(slots=True)
class LongContextPromptBuilder:
    """Build prompts from long ordered context with optional compression."""

    registry: PromptRegistry
    page_context_formatter: PageContextFormatter
    context_compression: ContextCompression
    template_name: str = "default"

    def build(self, query: str, nodes: list[Node]) -> str:
        """Build prompt from sorted nodes, compressing if token budget is exceeded."""
        ordered_nodes = sorted(
            nodes,
            key=lambda node: (node.doc_id, node.position if node.position is not None else 10**9),
        )

        total_tokens = sum(len(node.content.split()) for node in ordered_nodes)
        context_nodes = ordered_nodes

        if total_tokens > self.context_compression.token_limit:
            compressed_tokens = self.context_compression.compress(ordered_nodes).split()
            context_nodes = []
            cursor = 0
            for node in ordered_nodes:
                if cursor >= len(compressed_tokens):
                    break
                original_count = len(node.content.split())
                take_count = min(original_count, len(compressed_tokens) - cursor)
                trimmed_content = " ".join(compressed_tokens[cursor : cursor + take_count])
                cursor += take_count
                if trimmed_content:
                    context_nodes.append(node.model_copy(update={"content": trimmed_content}))

        context = self.page_context_formatter.format(context_nodes)
        template = self.registry.get(self.template_name)
        return template.render(query=query, context=context)


async def _seed_nodes(embedding_model: OpenAIEmbedding, vector_store: FAISSStore) -> list[Node]:
    """Create and index baseline chunks used by the long-context pipeline."""
    document = Document(
        doc_id="doc-001",
        content=(
            "rag-toolkit supports long-context retrieval by preserving document order. "
            "Chunks should be read in natural page sequence for coherent answers. "
            "Long-context pipelines may compress context to fit model token limits."
        ),
    )

    chunker = FixedChunker(chunk_size=512, overlap=50)
    raw_nodes = chunker.chunk(document)
    page_nodes: list[Node] = []
    for index, node in enumerate(raw_nodes, start=1):
        attributes = dict(node.metadata.attributes)
        attributes["page"] = index
        page_nodes.append(
            node.model_copy(update={"metadata": Metadata(source=node.metadata.source, attributes=attributes)})
        )

    vectors = await embedding_model.embed([node.content for node in page_nodes])
    nodes_with_embeddings = [
        node.model_copy(update={"embedding": vector})
        for node, vector in zip(page_nodes, vectors)
    ]
    await vector_store.add(nodes_with_embeddings)
    return nodes_with_embeddings


async def _build_container(llm: LLMProtocol | None = None) -> DependencyContainer:
    """Construct and register all long-context pipeline dependencies in DI container."""
    container = DependencyContainer()

    embedding_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        max_batch_size=32,
        client=LocalOpenAIEmbeddingClient(),
    )
    vector_store = FAISSStore()
    await _seed_nodes(embedding_model=embedding_model, vector_store=vector_store)

    vector_retriever = VectorRetriever(embedding_model=embedding_model, vector_store=vector_store)
    retriever = LongContextRetriever(retriever=vector_retriever)

    prompt_builder = LongContextPromptBuilder(
        registry=PromptRegistry(),
        page_context_formatter=PageContextFormatter(),
        context_compression=ContextCompression(token_limit=12000),
    )

    llm_instance = llm or AnthropicLLM(
        config=LLMConfig(model="claude-opus-4-6", temperature=0.0, max_tokens=2048),
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
    """Build and return the long-context RAG pipeline definition."""
    llm = llm or AnthropicLLM()
    _ = (llm, vectorstore, embedding, retriever, reranker, chunker)
    return RAGPipeline(
        name="long_context",
        config=RagConfig(top_k=20, enable_rerank=False, include_citations=False),
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
    """Run the long-context RAG pipeline asynchronously for one query."""
    _ = (vectorstore, embedding, retriever, reranker, chunker)
    container = await _build_container(llm=llm)
    pipeline = build_pipeline(llm=llm)
    engine: RAGEngine = pipeline.build_engine(container)
    return await engine._run_async(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run("How does long-context retrieval work in rag-toolkit?"))
    logger.info(result)

