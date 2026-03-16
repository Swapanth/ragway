"""Dependency injection container for registering and resolving components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from ragway.exceptions import ValidationError


T = TypeVar("T")


@dataclass(slots=True)
class Provider(Generic[T]):
    """Represents a lazily evaluated provider function for a dependency."""

    factory: Callable[[], T]

    def get(self) -> T:
        """Instantiate and return the dependency from its factory."""
        return self.factory()


@dataclass(slots=True)
class DependencyContainer:
    """Simple DI container for component registration and lookup by name."""

    _instances: dict[str, object] = field(default_factory=dict)
    _providers: dict[str, Provider[object]] = field(default_factory=dict)

    def register_instance(self, name: str, instance: object) -> None:
        """Register a concrete instance under a name."""
        self._instances[name] = instance

    def register_provider(self, name: str, factory: Callable[[], object]) -> None:
        """Register a lazy provider function under a name."""
        self._providers[name] = Provider(factory=factory)

    def resolve(self, name: str) -> object:
        """Resolve a dependency by name from instances or providers."""
        if name in self._instances:
            return self._instances[name]

        if name in self._providers:
            instance = self._providers[name].get()
            self._instances[name] = instance
            return instance

        raise ValidationError(f"Dependency not registered: {name}")

    def has(self, name: str) -> bool:
        """Return whether a dependency name is registered."""
        return name in self._instances or name in self._providers

