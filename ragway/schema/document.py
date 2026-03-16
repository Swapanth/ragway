"""Document schema representing immutable raw ingested content."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ragway.schema.metadata import Metadata
from ragway.validators import validate_non_empty_text


class Document(BaseModel):
    """A raw source document before chunking or embedding."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Unique document identifier.")
    content: str = Field(..., description="Raw textual content.")
    metadata: Metadata = Field(default_factory=Metadata, description="Document metadata.")

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        """Validate that the document identifier is non-empty text."""
        return validate_non_empty_text(value, "doc_id")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Validate that document content is non-empty text."""
        return validate_non_empty_text(value, "content")

    def content_length(self) -> int:
        """Return the length of the document content in characters."""
        return len(self.content)

