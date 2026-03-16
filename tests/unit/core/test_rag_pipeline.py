from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from ragway.core.dependency_container import DependencyContainer
from ragway.core.rag_engine import RagConfig
from ragway.core.rag_pipeline import RAGPipeline


def test_rag_pipeline_build_engine_resolves_dependencies() -> None:
    """RAGPipeline should construct engine from named container dependencies."""
    container = DependencyContainer()
    container.register_instance("embedding_model", AsyncMock())
    container.register_instance("retriever", AsyncMock())
    container.register_instance("prompt_builder", MagicMock())
    container.register_instance("llm", AsyncMock())

    pipeline = RAGPipeline(name="default", config=RagConfig())
    engine = pipeline.build_engine(container)

    assert engine.config.top_k == 5
    assert engine.reranker is None
    assert engine.citation_builder is None


def test_rag_pipeline_build_engine_includes_optional_components() -> None:
    """RAGPipeline should include optional reranker and citation builder when present."""
    container = DependencyContainer()
    container.register_instance("embedding_model", AsyncMock())
    container.register_instance("retriever", AsyncMock())
    container.register_instance("prompt_builder", MagicMock())
    container.register_instance("llm", AsyncMock())
    container.register_instance("reranker", AsyncMock())
    container.register_instance("citation_builder", MagicMock())

    pipeline = RAGPipeline(name="default", config=RagConfig())
    engine = pipeline.build_engine(container)

    assert engine.reranker is not None
    assert engine.citation_builder is not None

