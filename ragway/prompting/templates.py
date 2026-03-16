"""Prompt templates and rendering helpers for RAG generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Represents a named prompt template with system and user sections."""

    name: str
    system_template: str
    user_template: str

    def render(self, query: str, context: str) -> str:
        """Render a final prompt string using query and context values."""
        system_text = self.system_template.strip()
        user_text = self.user_template.format(query=query, context=context).strip()
        return f"System:\n{system_text}\n\nUser:\n{user_text}".strip()


DEFAULT_TEMPLATE = PromptTemplate(
    name="default",
    system_template=(
        "You are a careful retrieval-augmented assistant. "
        "Answer only from the provided context and cite sources when possible."
    ),
    user_template=(
        "Question:\n{query}\n\n"
        "Context:\n{context}\n\n"
        "Instructions:\nProvide a concise answer grounded in context."
    ),
)


QA_TEMPLATE = PromptTemplate(
    name="qa",
    system_template="You answer user questions using provided context only.",
    user_template="Q: {query}\n\nContext:\n{context}\n\nA:",
)


def built_in_templates() -> dict[str, PromptTemplate]:
    """Return the built-in template mapping keyed by template name."""
    return {
        DEFAULT_TEMPLATE.name: DEFAULT_TEMPLATE,
        QA_TEMPLATE.name: QA_TEMPLATE,
    }
