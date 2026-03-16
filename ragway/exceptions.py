"""Project-specific exception hierarchy for rag-toolkit."""

from __future__ import annotations


class RagError(Exception):
    """Base exception for all rag-toolkit errors."""


class ValidationError(RagError):
    """Raised when data validation fails."""


class SchemaError(RagError):
    """Raised when schema construction or transformation fails."""
