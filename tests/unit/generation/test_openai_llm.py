from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.base_llm import LLMConfig
from ragway.generation.openai_llm import OpenAILLM


async def test_openai_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAILLM should raise when OPENAI_API_KEY is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = OpenAILLM(client=AsyncMock())

    with pytest.raises(RagError):
        await llm.generate("hello")


async def test_openai_llm_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAILLM should call async client with config-derived values."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="gpt-4o", temperature=0.3, max_tokens=111)
    llm = OpenAILLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        model="gpt-4o",
        prompt="hello",
        temperature=0.3,
        max_tokens=111,
        api_key="test-key",
    )


async def test_openai_llm_sdk_path_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAILLM should parse chat completion text from AsyncOpenAI."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Completions:
        async def create(self, **kwargs):
            del kwargs
            message = types.SimpleNamespace(content="hello")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _AsyncOpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_AsyncOpenAI))
    llm = OpenAILLM()

    assert await llm.generate("prompt") == "hello"


async def test_openai_llm_sdk_empty_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAILLM should raise when SDK response has no text content."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Completions:
        async def create(self, **kwargs):
            del kwargs
            message = types.SimpleNamespace(content="   ")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _AsyncOpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_AsyncOpenAI))
    llm = OpenAILLM()

    with pytest.raises(RagError, match="did not include text content"):
        await llm.generate("prompt")


async def test_openai_llm_stream_yields_generate_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAILLM stream should yield one chunk delegated from generate()."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(client=AsyncMock())
    llm.client.generate.return_value = "stream-value"

    async def _collect() -> list[str]:
        return [part async for part in llm.stream("prompt")]

    assert await _collect() == ["stream-value"]

