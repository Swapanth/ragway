"""Typed metadata model used by documents and nodes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


JsonScalar = str | int | float | bool | None


class Metadata(BaseModel):
    """Structured metadata attached to ingested documents and derived nodes."""

    source: str | None = Field(default=None, description="Origin of the content.")
    created_at: datetime | None = Field(
        default=None,
        description="UTC timestamp for when metadata was first created.",
    )
    tags: list[str] = Field(default_factory=list, description="User-defined label tags.")
    attributes: dict[str, JsonScalar] = Field(
        default_factory=dict,
        description="Arbitrary scalar attributes used for filtering and tracing.",
    )

    def with_attribute(self, key: str, value: JsonScalar) -> Metadata:
        """Return a copy of metadata with one attribute inserted or replaced."""
        updated_attributes: dict[str, JsonScalar] = dict(self.attributes)
        updated_attributes[key] = value
        return self.model_copy(update={"attributes": updated_attributes})
