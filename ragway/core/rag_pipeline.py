"""Declarative pipeline definition for constructing a configured RAG engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ragway.core.dependency_container import DependencyContainer
from ragway.core.rag_engine import CitationBuilderProtocol, PromptBuilderProtocol, RAGEngine, RagConfig
from ragway.interfaces.embedding_protocol import EmbeddingProtocol
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.interfaces.reranker_protocol import RerankerProtocol
from ragway.interfaces.retriever_protocol import RetrieverProtocol


@dataclass(slots=True)
class RAGPipeline:
    """Declarative pipeline that maps component names to engine dependencies."""

    name: str
    config: RagConfig
    embedding_name: str = "embedding_model"
    retriever_name: str = "retriever"
    prompt_builder_name: str = "prompt_builder"
    llm_name: str = "llm"
    reranker_name: str = "reranker"
    citation_builder_name: str = "citation_builder"

    def build_engine(self, container: DependencyContainer) -> RAGEngine:
        """Resolve dependencies from container and create a RAGEngine instance."""
        embedding_model = cast(EmbeddingProtocol, container.resolve(self.embedding_name))
        retriever = cast(RetrieverProtocol, container.resolve(self.retriever_name))
        prompt_builder = cast(PromptBuilderProtocol, container.resolve(self.prompt_builder_name))
        llm = cast(LLMProtocol, container.resolve(self.llm_name))

        reranker = (
            cast(RerankerProtocol, container.resolve(self.reranker_name)) if container.has(self.reranker_name) else None
        )
        citation_builder = (
            cast(CitationBuilderProtocol, container.resolve(self.citation_builder_name))
            if container.has(self.citation_builder_name)
            else None
        )

        return RAGEngine(
            config=self.config,
            embedding_model=embedding_model,
            retriever=retriever,
            prompt_builder=prompt_builder,
            llm=llm,
            reranker=reranker,
            citation_builder=citation_builder,
        )

