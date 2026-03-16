from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.base_llm import LLMConfig
from ragway.generation.llama_llm import LlamaLLM


async def test_llama_llm_requires_client() -> None:
    """LlamaLLM should raise when no local client is configured."""
    llm = LlamaLLM(client=None)

    with pytest.raises(RagError):
        await llm.generate("hello")


async def test_llama_llm_uses_mocked_client() -> None:
    """LlamaLLM should call local async client with config values."""
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="llama-3.1-8b-instruct", temperature=0.4, max_tokens=77)
    llm = LlamaLLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        model="llama-3.1-8b-instruct",
        prompt="hello",
        temperature=0.4,
        max_tokens=77,
    )

