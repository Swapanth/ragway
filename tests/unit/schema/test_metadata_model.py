from __future__ import annotations

from datetime import datetime, timezone

from ragway.schema.metadata import Metadata


def test_metadata_defaults() -> None:
    """Metadata should initialize list/dict fields with safe defaults."""
    metadata = Metadata()
    assert metadata.source is None
    assert metadata.tags == []
    assert metadata.attributes == {}


def test_metadata_with_attribute_returns_copy() -> None:
    """with_attribute should return an updated copy without mutating original."""
    original = Metadata(source="doc")
    updated = original.with_attribute("lang", "en")

    assert original.attributes == {}
    assert updated.attributes == {"lang": "en"}
    assert updated.source == "doc"


def test_metadata_accepts_datetime() -> None:
    """Metadata should preserve provided datetime values."""
    created = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    metadata = Metadata(created_at=created, tags=["a", "b"])

    assert metadata.created_at == created
    assert metadata.tags == ["a", "b"]
