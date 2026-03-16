from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.base_llm import LLMConfig
from ragway.generation.groq_llm import GroqLLM


async def test_groq_llm_default_model_name() -> None:
    """GroqLLM should default to llama-3.3-70b-versatile model."""
    llm = GroqLLM(client=AsyncMock())
    assert llm.config.model == "llama-3.3-70b-versatile"


async def test_groq_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """GroqLLM should raise when GROQ_API_KEY is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm = GroqLLM(client=AsyncMock())

    with pytest.raises(RagError):
        await llm.generate("hello")


async def test_groq_llm_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """GroqLLM should call async client with config-derived values."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=200)
    llm = GroqLLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        model="llama-3.1-8b-instant",
        prompt="hello",
        temperature=0.1,
        max_tokens=200,
        api_key="test-key",
    )


async def test_groq_llm_sdk_path_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """GroqLLM should parse completion text from OpenAI-compatible SDK."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class _Completions:
        async def create(self, **kwargs):
            del kwargs
            msg = types.SimpleNamespace(content="hello")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class _AsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_AsyncOpenAI))
    llm = GroqLLM()

    assert await llm.generate("prompt") == "hello"


async def test_groq_llm_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """GroqLLM stream should yield one generate chunk."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm = GroqLLM(client=AsyncMock())
    llm.client.generate.return_value = "streamed"

    async def _collect() -> list[str]:
        return [part async for part in llm.stream("prompt")]

    assert await _collect() == ["streamed"]

