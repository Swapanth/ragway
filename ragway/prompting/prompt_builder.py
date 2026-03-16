"""Prompt assembly helpers for combining templates and formatted context."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.prompting.context_formatter import ContextFormatter
from ragway.prompting.prompt_registry import PromptRegistry
from ragway.schema.node import Node


@dataclass(slots=True)
class PromptBuilder:
    """Builds final prompts from query, nodes, and selected template."""

    template_name: str = "default"
    registry: PromptRegistry | None = None
    context_formatter: ContextFormatter | None = None

    def __post_init__(self) -> None:
        """Initialize default dependencies when not explicitly supplied."""
        if self.registry is None:
            self.registry = PromptRegistry()
        if self.context_formatter is None:
            self.context_formatter = ContextFormatter()

    def build(self, query: str, nodes: list[Node]) -> str:
        """Build a complete prompt string from query and retrieved nodes."""
        assert self.registry is not None
        assert self.context_formatter is not None

        template = self.registry.get(self.template_name)
        context = self.context_formatter.format(nodes)
        return template.render(query=query, context=context)

