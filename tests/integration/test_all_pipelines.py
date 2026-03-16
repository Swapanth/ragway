"""Integration coverage for all primary RAG pipelines and public RAG APIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragway.rag import RAG


QUERY = "What are the main components of a RAG system?"
RAG_PAPER_PATH = Path("data/docs/rag-paper-2005.11401.pdf")


@pytest.mark.integration
@pytest.mark.parametrize(
    "pipeline_name",
    ["naive", "hybrid", "self", "long_context", "agentic"],
)
async def test_pipeline_returns_real_answer(pipeline_name: str) -> None:
    """Each supported pipeline should return a substantial non-template answer."""
    if not RAG_PAPER_PATH.exists():
        pytest.skip(f"RAG paper not found at {RAG_PAPER_PATH}")

    rag = RAG(
        pipeline=pipeline_name,
        llm="anthropic",
        vectorstore="faiss",
        reranker="cohere" if pipeline_name != "naive" else None,
    )

    await rag.ingest(str(RAG_PAPER_PATH))
    answer = await rag.query(QUERY)

    assert isinstance(answer, str)
    assert len(answer) > 50
    assert "context" not in answer.lower()[:30]
    assert "provide" not in answer.lower()[:30]

    print(f"\n{pipeline_name}:\n{answer[:300]}\n")


@pytest.mark.integration
async def test_rag_from_config() -> None:
    """Public API from_config should ingest and return answer with sources."""
    rag = RAG.from_config("configs/default.yaml")

    await rag.ingest("data/docs/")
    result = await rag.query_with_sources(QUERY)

    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 0


@pytest.mark.integration
async def test_rag_switch() -> None:
    """Public API switch should return a modified copy and keep original unchanged."""
    rag = RAG(llm="anthropic", vectorstore="faiss")
    fast = rag.switch(llm="groq")

    assert fast._build_config()["llm"]["provider"] == "groq"
    assert rag._build_config()["llm"]["provider"] == "anthropic"
