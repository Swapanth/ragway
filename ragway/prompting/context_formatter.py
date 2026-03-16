"""Context formatting utilities for assembling prompt context blocks."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.schema.node import Node


@dataclass(slots=True)
class ContextFormatter:
    """Formats retrieved nodes into a plain context string."""

    include_headers: bool = True

    def format(self, nodes: list[Node]) -> str:
        """Format a node list into a deterministic multi-block context string."""
        if not nodes:
            return ""

        blocks: list[str] = []
        for index, node in enumerate(nodes, start=1):
            if self.include_headers:
                blocks.append(f"[{index}] {node.content.strip()}")
            else:
                blocks.append(node.content.strip())
        return "\n\n".join(blocks).strip()

