from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.base_llm import LLMConfig
from ragway.generation.mistral_llm import MistralLLM


async def test_mistral_llm_default_model_name() -> None:
    """MistralLLM should default to mistral-large-latest model."""
    llm = MistralLLM(client=AsyncMock())
    assert llm.config.model == "mistral-large-latest"


async def test_mistral_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """MistralLLM should raise when MISTRAL_API_KEY is missing."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    llm = MistralLLM(client=AsyncMock())

    with pytest.raises(RagError):
        await llm.generate("hello")


async def test_mistral_llm_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """MistralLLM should call async client with config-derived values."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="mistral-small-latest", temperature=0.2, max_tokens=123)
    llm = MistralLLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        model="mistral-small-latest",
        prompt="hello",
        temperature=0.2,
        max_tokens=123,
        api_key="test-key",
    )


async def test_mistral_llm_sdk_path_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """MistralLLM should parse completion text from OpenAI-compatible SDK."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

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
    llm = MistralLLM()

    assert await llm.generate("prompt") == "hello"


async def test_mistral_llm_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """MistralLLM stream should yield one generate chunk."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    llm = MistralLLM(client=AsyncMock())
    llm.client.generate.return_value = "streamed"

    async def _collect() -> list[str]:
        return [part async for part in llm.stream("prompt")]

    assert await _collect() == ["streamed"]

