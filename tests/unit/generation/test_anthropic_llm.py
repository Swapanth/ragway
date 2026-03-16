from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.anthropic_llm import AnthropicLLM
from ragway.generation.base_llm import LLMConfig


async def test_anthropic_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicLLM should raise when ANTHROPIC_API_KEY is missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = AnthropicLLM(client=AsyncMock())

    with pytest.raises(RagError):
        await llm.generate("hello")


async def test_anthropic_llm_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicLLM should call async client with config-derived values."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="claude-sonnet-4-6", temperature=0.2, max_tokens=123)
    llm = AnthropicLLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        model="claude-sonnet-4-6",
        prompt="hello",
        temperature=0.2,
        max_tokens=123,
        api_key="test-key",
    )


async def test_anthropic_llm_sdk_path_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicLLM should parse text blocks from AsyncAnthropic responses."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _Block:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Messages:
        async def create(self, **kwargs):
            del kwargs
            return types.SimpleNamespace(content=[_Block("hello"), _Block("world")])

    class _AsyncAnthropic:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(AsyncAnthropic=_AsyncAnthropic))
    llm = AnthropicLLM()

    result = await llm.generate("prompt")
    assert result == "hello\nworld"


async def test_anthropic_llm_returns_fallback_on_sdk_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicLLM should return fallback text when SDK call fails."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _Messages:
        async def create(self, **kwargs):
            del kwargs
            raise RuntimeError("network")

    class _AsyncAnthropic:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(AsyncAnthropic=_AsyncAnthropic))
    llm = AnthropicLLM()
    result = await llm.generate("context" * 100)

    assert result.startswith("Fallback answer based on available context:")


async def test_anthropic_llm_stream_yields_generate_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicLLM stream should yield one chunk delegated from generate()."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = AnthropicLLM(client=AsyncMock())
    llm.client.generate.return_value = "stream-value"

    async def _collect() -> list[str]:
        return [part async for part in llm.stream("prompt")]

    assert await _collect() == ["stream-value"]

