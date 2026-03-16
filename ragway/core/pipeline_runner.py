"""Pipeline runner for executing declarative RAG pipelines end-to-end."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.core.dependency_container import DependencyContainer
from ragway.core.rag_pipeline import RAGPipeline


@dataclass(slots=True)
class PipelineRunner:
    """Executes a configured RAGPipeline using a dependency container."""

    container: DependencyContainer

    def run(self, pipeline: RAGPipeline, query: str) -> str:
        """Build and execute the provided pipeline for a single query."""
        engine = pipeline.build_engine(self.container)
        return engine.run(query)

