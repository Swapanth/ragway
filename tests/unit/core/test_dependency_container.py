from __future__ import annotations

import pytest

from ragway.core.dependency_container import DependencyContainer
from ragway.exceptions import ValidationError


def test_dependency_container_register_and_resolve_instance() -> None:
    """Container should return registered instances by name."""
    container = DependencyContainer()
    instance = object()
    container.register_instance("x", instance)

    assert container.resolve("x") is instance


def test_dependency_container_resolves_provider_once() -> None:
    """Provider resolution should memoize created instance."""
    container = DependencyContainer()
    calls = {"count": 0}

    def factory() -> object:
        calls["count"] += 1
        return object()

    container.register_provider("y", factory)
    first = container.resolve("y")
    second = container.resolve("y")

    assert first is second
    assert calls["count"] == 1


def test_dependency_container_raises_for_missing_name() -> None:
    """Resolving an unknown dependency should raise ValidationError."""
    container = DependencyContainer()
    with pytest.raises(ValidationError):
        container.resolve("missing")

