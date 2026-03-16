from __future__ import annotations

from dataclasses import dataclass

from studio.pipeline_visualizer import PipelineVisualizer


@dataclass
class _Config:
    enable_rerank: bool


@dataclass
class _Pipeline:
    name: str
    config: _Config


def test_pipeline_visualizer_outputs_mermaid() -> None:
    """PipelineVisualizer should return Mermaid flowchart text."""
    visualizer = PipelineVisualizer()
    pipeline = _Pipeline(name="hybrid", config=_Config(enable_rerank=True))

    output = visualizer.visualize(pipeline)

    assert output.startswith("flowchart TD")
    assert "Retriever (hybrid)" in output
    assert "Reranker" in output
