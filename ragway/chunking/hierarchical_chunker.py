"""Hierarchical chunker that emits both parent and child nodes."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.chunking.base_chunker import BaseChunker
from ragway.chunking.fixed_chunker import FixedChunker
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.validators import validate_positive_int


@dataclass(slots=True)
class HierarchicalChunker(BaseChunker):
    """Build parent chunks first, then child chunks nested under each parent."""

    parent_chunk_size: int = 200
    child_chunk_size: int = 80
    child_overlap: int = 20

    def __post_init__(self) -> None:
        """Validate parent and child chunk configuration."""
        self.parent_chunk_size = validate_positive_int(
            self.parent_chunk_size,
            "parent_chunk_size",
        )
        self.child_chunk_size = validate_positive_int(self.child_chunk_size, "child_chunk_size")
        if self.child_overlap < 0:
            raise ValueError("child_overlap must be >= 0")
        if self.child_overlap >= self.child_chunk_size:
            raise ValueError("child_overlap must be smaller than child_chunk_size")

    def chunk(self, document: Document) -> list[Node]:
        """Emit parent nodes followed by child nodes referencing their parent IDs."""
        parent_chunker = FixedChunker(chunk_size=self.parent_chunk_size, overlap=0)
        parent_nodes = parent_chunker.chunk(document)
        if not parent_nodes:
            return []

        child_chunker = FixedChunker(
            chunk_size=self.child_chunk_size,
            overlap=self.child_overlap,
        )

        all_nodes: list[Node] = []
        child_position = 0

        for parent_index, parent in enumerate(parent_nodes):
            parent_node_id = f"{document.doc_id}:parent:{parent_index}"
            parent_node = Node(
                node_id=parent_node_id,
                doc_id=document.doc_id,
                content=parent.content,
                metadata=document.metadata,
                position=parent_index,
            )
            all_nodes.append(parent_node)

            parent_document = Document(
                doc_id=document.doc_id,
                content=parent.content,
                metadata=document.metadata,
            )
            child_nodes = child_chunker.chunk(parent_document)
            for child in child_nodes:
                all_nodes.append(
                    Node(
                        node_id=f"{parent_node_id}:child:{child_position}",
                        doc_id=document.doc_id,
                        content=child.content,
                        metadata=document.metadata,
                        parent_id=parent_node_id,
                        position=child_position,
                    )
                )
                child_position += 1

        return all_nodes

