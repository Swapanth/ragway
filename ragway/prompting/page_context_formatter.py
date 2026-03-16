"""Page-aware context formatting utilities."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.schema.node import Node


@dataclass(slots=True)
class PageContextFormatter:
    """Formats node context and includes page-number annotations."""

    unknown_page: str = "?"

    def format(self, nodes: list[Node]) -> str:
        """Format context entries with page labels extracted from metadata."""
        if not nodes:
            return ""

        blocks: list[str] = []
        for index, node in enumerate(nodes, start=1):
            page_value = node.metadata.attributes.get("page")
            page_label = str(page_value) if page_value is not None else self.unknown_page
            blocks.append(f"[{index}] (page {page_label}) {node.content.strip()}")
        return "\n\n".join(blocks).strip()

