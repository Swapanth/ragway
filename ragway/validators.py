"""Reusable validation helpers for rag-toolkit foundation models."""

from __future__ import annotations

from ragway.exceptions import ValidationError


def validate_non_empty_text(value: str, field_name: str) -> str:
    """Validate that a text field is not empty or whitespace-only."""
    if not value.strip():
        raise ValidationError(f"{field_name} must not be empty")
    return value


def validate_positive_int(value: int, field_name: str) -> int:
    """Validate that an integer field is strictly positive."""
    if value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return value


def validate_probability(value: float, field_name: str) -> float:
    """Validate that a numeric field is in the inclusive range [0.0, 1.0]."""
    if value < 0.0 or value > 1.0:
        raise ValidationError(f"{field_name} must be between 0.0 and 1.0")
    return value

