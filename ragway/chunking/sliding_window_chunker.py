"""Sliding-window chunker for overlapping token windows."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.chunking.base_chunker import BaseChunker
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.validators import validate_positive_int


@dataclass(slots=True)
class SlidingWindowChunker(BaseChunker):
    """Split a document into fixed-size windows advanced by a stride."""

    window_size: int
    stride: int

    def __post_init__(self) -> None:
        """Validate window and stride configuration."""
        self.window_size = validate_positive_int(self.window_size, "window_size")
        self.stride = validate_positive_int(self.stride, "stride")

    def chunk(self, document: Document) -> list[Node]:
        """Generate overlapping windows from document tokens."""
        tokens = document.content.split()
        if not tokens:
            return []

        nodes: list[Node] = []
        position = 0
        for start in range(0, len(tokens), self.stride):
            window = tokens[start : start + self.window_size]
            if not window:
                break
            nodes.append(
                Node(
                    node_id=f"{document.doc_id}:sliding:{position}",
                    doc_id=document.doc_id,
                    content=" ".join(window),
                    metadata=document.metadata,
                    position=position,
                )
            )
            position += 1
            if start + self.window_size >= len(tokens):
                break

        return nodes

