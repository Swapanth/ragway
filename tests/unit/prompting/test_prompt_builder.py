from __future__ import annotations

from ragway.prompting.prompt_builder import PromptBuilder
from ragway.prompting.prompt_registry import PromptRegistry
from ragway.prompting.templates import PromptTemplate
from ragway.schema.node import Node


def test_prompt_builder_builds_prompt_from_query_and_nodes() -> None:
    """PromptBuilder should combine query and formatted context into template."""
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(
            name="custom",
            system_template="System",
            user_template="Q:{query}\nC:{context}",
        )
    )
    builder = PromptBuilder(template_name="custom", registry=registry)
    nodes = [Node(node_id="n1", doc_id="d1", content="Context line")]

    prompt = builder.build("What is this?", nodes)

    assert "Q:What is this?" in prompt
    assert "Context line" in prompt


def test_prompt_builder_with_empty_nodes_still_renders() -> None:
    """PromptBuilder should still render with empty context."""
    builder = PromptBuilder()
    prompt = builder.build("Question", [])

    assert "Question" in prompt
    assert "Context:" in prompt

