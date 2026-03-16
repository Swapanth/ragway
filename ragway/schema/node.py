"""Node schema representing a retrievable chunk derived from a document."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ragway.schema.metadata import Metadata
from ragway.validators import validate_non_empty_text


class Node(BaseModel):
    """A chunk-level unit used by retrievers and rerankers."""

    node_id: str = Field(..., description="Unique node identifier.")
    doc_id: str = Field(..., description="Source document identifier.")
    content: str = Field(..., description="Chunk text content.")
    embedding: list[float] | None = Field(
        default=None,
        description="Optional vector embedding for retrieval.",
    )
    metadata: Metadata = Field(default_factory=Metadata, description="Node metadata.")
    parent_id: str | None = Field(default=None, description="Optional parent node id.")
    position: int | None = Field(
        default=None,
        description="Optional ordinal position within the parent document.",
    )

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        """Validate that the node identifier is non-empty text."""
        return validate_non_empty_text(value, "node_id")

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        """Validate that the source document identifier is non-empty text."""
        return validate_non_empty_text(value, "doc_id")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Validate that node content is non-empty text."""
        return validate_non_empty_text(value, "content")

    def has_embedding(self) -> bool:
        """Return whether this node currently has an embedding vector."""
        return self.embedding is not None

