"""Citation-aware context formatting utilities."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.schema.node import Node


@dataclass(slots=True)
class CitationFormatter:
    """Formats node context lines and appends source citations."""

    unknown_source: str = "unknown"

    def format(self, nodes: list[Node]) -> str:
        """Return context blocks with trailing [source] citation tags."""
        if not nodes:
            return ""

        lines: list[str] = []
        for index, node in enumerate(nodes, start=1):
            source = node.metadata.source or self.unknown_source
            lines.append(f"[{index}] {node.content.strip()} [{source}]")
        return "\n\n".join(lines).strip()

