from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from ragway.core.rag_engine import RAGEngine, RagConfig
from ragway.schema.node import Node


def test_rag_engine_run_full_flow_with_citations() -> None:
    """RAGEngine.run should orchestrate full flow and append citations."""
    config = RagConfig(top_k=2, enable_rerank=True, include_citations=True, citation_limit=2)

    embedding = AsyncMock()
    retriever = AsyncMock()
    reranker = AsyncMock()
    llm = AsyncMock()
    prompt_builder = MagicMock()
    citation_builder = MagicMock()

    nodes = [
        Node(node_id="n1", doc_id="d1", content="alpha"),
        Node(node_id="n2", doc_id="d1", content="beta"),
    ]
    retriever.retrieve.return_value = nodes
    reranker.rerank.return_value = list(reversed(nodes))
    prompt_builder.build.return_value = "PROMPT"
    llm.generate.return_value = "ANSWER"
    citation_builder.build.return_value = {"ANSWER": "doc-a"}

    engine = RAGEngine(
        config=config,
        embedding_model=embedding,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=reranker,
        citation_builder=citation_builder,
    )

    result = engine.run("query")

    assert "ANSWER" in result
    assert "Citations:" in result
    embedding.embed.assert_awaited_once_with(["query"])
    retriever.retrieve.assert_awaited_once_with(query="query", top_k=2)
    reranker.rerank.assert_awaited_once()
    llm.generate.assert_awaited_once_with("PROMPT")


def test_rag_engine_run_without_optional_steps() -> None:
    """RAGEngine should skip reranking and citations when disabled."""
    config = RagConfig(top_k=1, enable_rerank=False, include_citations=False)

    embedding = AsyncMock()
    retriever = AsyncMock()
    llm = AsyncMock()
    prompt_builder = MagicMock()

    retriever.retrieve.return_value = [Node(node_id="n1", doc_id="d1", content="alpha")]
    prompt_builder.build.return_value = "PROMPT"
    llm.generate.return_value = "ANSWER"

    engine = RAGEngine(
        config=config,
        embedding_model=embedding,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    result = engine.run("query")

    assert result == "ANSWER"

