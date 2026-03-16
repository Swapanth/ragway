from __future__ import annotations

import pytest

from ragway.exceptions import ValidationError
from ragway.prompting.prompt_registry import PromptRegistry
from ragway.prompting.templates import PromptTemplate


def test_prompt_registry_register_and_get() -> None:
    """PromptRegistry should return templates previously registered."""
    registry = PromptRegistry()
    template = PromptTemplate(name="custom", system_template="S", user_template="{query} {context}")

    registry.register(template)
    result = registry.get("custom")

    assert result.name == "custom"


def test_prompt_registry_missing_template_raises() -> None:
    """PromptRegistry should raise ValidationError for unknown names."""
    registry = PromptRegistry()
    with pytest.raises(ValidationError):
        registry.get("missing")

