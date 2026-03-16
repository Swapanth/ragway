"""Pipeline visualization helper that produces Mermaid flowcharts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PipelineVisualizer:
    """Builds Mermaid diagrams describing pipeline component flow."""

    def visualize(self, pipeline: object) -> str:
        """Return a Mermaid flowchart for the provided pipeline object."""
        name = getattr(pipeline, "name", "pipeline")
        config = getattr(pipeline, "config", None)
        enable_rerank = bool(getattr(config, "enable_rerank", False))

        lines = [
            "flowchart TD",
            f"    Q[Query] --> R[Retriever ({name})]",
        ]

        if enable_rerank:
            lines.append("    R --> RR[Reranker]")
            lines.append("    RR --> PB[Prompt Builder]")
        else:
            lines.append("    R --> PB[Prompt Builder]")

        lines.extend(
            [
                "    PB --> LLM[LLM Generator]",
                "    LLM --> A[Answer]",
            ]
        )
        return "\n".join(lines)
