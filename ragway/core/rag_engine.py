"""Core RAG engine orchestrating retrieval, reranking, prompting, and generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from ragway.interfaces.embedding_protocol import EmbeddingProtocol
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.interfaces.reranker_protocol import RerankerProtocol
from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.schema.node import Node
from ragway.validators import validate_positive_int, validate_probability


class PromptBuilderProtocol(Protocol):
    """Protocol for prompt builder dependencies used by the engine."""

    def build(self, query: str, nodes: list[Node]) -> str:
        """Build a full prompt from query and retrieved nodes."""


class CitationBuilderProtocol(Protocol):
    """Protocol for citation builder dependencies used by the engine."""

    def build(self, answer: str, nodes: list[Node]) -> dict[str, str]:
        """Build answer-sentence to source-label citations."""


@dataclass(slots=True)
class RagConfig:
    """Configuration for core RAG engine orchestration."""

    top_k: int = 5
    enable_rerank: bool = True
    include_citations: bool = True
    citation_limit: int = 3
    temperature: float = 0.0
    max_tokens: int = 256

    def __post_init__(self) -> None:
        """Validate RAG execution settings."""
        self.top_k = validate_positive_int(self.top_k, "top_k")
        self.citation_limit = validate_positive_int(self.citation_limit, "citation_limit")
        self.max_tokens = validate_positive_int(self.max_tokens, "max_tokens")
        self.temperature = validate_probability(self.temperature, "temperature")


@dataclass(slots=True)
class RAGEngine:
    """Orchestrates the end-to-end RAG flow for a single query."""

    config: RagConfig
    embedding_model: EmbeddingProtocol
    retriever: RetrieverProtocol
    prompt_builder: PromptBuilderProtocol
    llm: LLMProtocol
    reranker: RerankerProtocol | None = None
    citation_builder: CitationBuilderProtocol | None = None

    def run(self, query: str) -> str:
        """Execute full RAG flow and return answer text with optional citations."""
        return asyncio.run(self._run_async(query))

    async def _run_async(self, query: str) -> str:
        """Internal async execution of all RAG pipeline steps."""
        await self.embedding_model.embed([query])

        nodes = await self.retriever.retrieve(query=query, top_k=self.config.top_k)

        if self.config.enable_rerank and self.reranker is not None:
            nodes = await self.reranker.rerank(query=query, nodes=nodes)

        prompt = self.prompt_builder.build(query=query, nodes=nodes)
        answer = await self.llm.generate(prompt)

        if not self.config.include_citations or self.citation_builder is None:
            return answer

        citations = self.citation_builder.build(answer=answer, nodes=nodes)
        if not citations:
            return answer

        citation_lines: list[str] = []
        for sentence, source in list(citations.items())[: self.config.citation_limit]:
            citation_lines.append(f"- {sentence} [{source}]")

        return f"{answer}\n\nCitations:\n" + "\n".join(citation_lines)

