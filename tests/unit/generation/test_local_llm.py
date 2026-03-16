from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.base_llm import LLMConfig
from ragway.generation.local_llm import LocalLLM


async def test_local_llm_requires_client() -> None:
    """LocalLLM should raise when no local client is configured."""
    llm = LocalLLM(client=None)

    with pytest.raises(RagError):
        await llm.generate("hello")


async def test_local_llm_uses_mocked_client() -> None:
    """LocalLLM should call generic async local client with config values."""
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="my-local-model", temperature=0.5, max_tokens=90)
    llm = LocalLLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        model="my-local-model",
        prompt="hello",
        temperature=0.5,
        max_tokens=90,
    )

