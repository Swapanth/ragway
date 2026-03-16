from __future__ import annotations

import pytest

from ragway.exceptions import RagError, SchemaError, ValidationError


def test_exception_hierarchy() -> None:
    """All project exceptions should inherit from RagError."""
    assert issubclass(ValidationError, RagError)
    assert issubclass(SchemaError, RagError)


def test_can_raise_project_exceptions() -> None:
    """Project exceptions should be raiseable and catchable by type."""
    with pytest.raises(ValidationError):
        raise ValidationError("invalid input")

