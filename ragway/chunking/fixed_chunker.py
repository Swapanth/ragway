"""Fixed-size token chunker with configurable overlap."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.chunking.base_chunker import BaseChunker
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.validators import validate_positive_int


@dataclass(slots=True)
class FixedChunker(BaseChunker):
    """Split content into fixed token windows with optional overlap."""

    chunk_size: int
    overlap: int = 0

    def __post_init__(self) -> None:
        """Validate chunker sizing constraints."""
        self.chunk_size = validate_positive_int(self.chunk_size, "chunk_size")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    def chunk(self, document: Document) -> list[Node]:
        """Split a document by whitespace-token count."""
        tokens = document.content.split()
        if not tokens:
            return []

        nodes: list[Node] = []
        step = self.chunk_size - self.overlap
        index = 0
        position = 0

        while index < len(tokens):
            token_slice = tokens[index : index + self.chunk_size]
            chunk_text = " ".join(token_slice)
            nodes.append(
                Node(
                    node_id=f"{document.doc_id}:fixed:{position}",
                    doc_id=document.doc_id,
                    content=chunk_text,
                    metadata=document.metadata,
                    position=position,
                )
            )
            index += step
            position += 1

        return nodes

