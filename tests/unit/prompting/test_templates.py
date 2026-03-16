from __future__ import annotations

from ragway.prompting.templates import DEFAULT_TEMPLATE, QA_TEMPLATE, PromptTemplate, built_in_templates


def test_prompt_template_renders_query_and_context() -> None:
    """PromptTemplate should inject query and context into user template."""
    template = PromptTemplate(
        name="t1",
        system_template="System text",
        user_template="Q={query} C={context}",
    )

    output = template.render(query="What?", context="Some context")

    assert "System text" in output
    assert "Q=What? C=Some context" in output


def test_built_in_templates_contains_defaults() -> None:
    """Built-in templates should expose default and QA templates."""
    templates = built_in_templates()
    assert DEFAULT_TEMPLATE.name in templates
    assert QA_TEMPLATE.name in templates

