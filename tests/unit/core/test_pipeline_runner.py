from __future__ import annotations

from unittest.mock import MagicMock

from ragway.core.dependency_container import DependencyContainer
from ragway.core.pipeline_runner import PipelineRunner
from ragway.core.rag_pipeline import RAGPipeline


def test_pipeline_runner_executes_pipeline_end_to_end() -> None:
    """PipelineRunner should build engine and return query answer."""
    container = DependencyContainer()
    runner = PipelineRunner(container=container)

    pipeline = MagicMock(spec=RAGPipeline)
    engine = MagicMock()
    engine.run.return_value = "answer"
    pipeline.build_engine.return_value = engine

    result = runner.run(pipeline=pipeline, query="hello")

    assert result == "answer"
    pipeline.build_engine.assert_called_once_with(container)
    engine.run.assert_called_once_with("hello")

