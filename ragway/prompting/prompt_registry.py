"""Registry for prompt templates addressed by name."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragway.exceptions import ValidationError
from ragway.prompting.templates import PromptTemplate, built_in_templates


@dataclass(slots=True)
class PromptRegistry:
    """Stores prompt templates and resolves templates by unique name."""

    _templates: dict[str, PromptTemplate] = field(default_factory=built_in_templates)

    def register(self, template: PromptTemplate) -> None:
        """Register or replace a template by its name."""
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """Return a template by name or raise if missing."""
        if name not in self._templates:
            raise ValidationError(f"Unknown prompt template: {name}")
        return self._templates[name]

    def list_names(self) -> list[str]:
        """Return all registered template names in sorted order."""
        return sorted(self._templates.keys())

