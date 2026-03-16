from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.azure_openai_llm import AzureOpenAILLM
from ragway.generation.base_llm import LLMConfig


async def test_azure_openai_llm_requires_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """AzureOpenAILLM should require key, endpoint, and deployment settings."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    llm = AzureOpenAILLM(client=AsyncMock())

    with pytest.raises(RagError):
        await llm.generate("hello")


async def test_azure_openai_llm_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """AzureOpenAILLM should call async client with config-derived values."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-deploy")
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="gpt-4o", temperature=0.2, max_tokens=111)
    llm = AzureOpenAILLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        deployment="gpt-4o-deploy",
        prompt="hello",
        temperature=0.2,
        max_tokens=111,
        endpoint="https://example.azure.com",
        api_key="test-key",
    )


async def test_azure_openai_llm_requires_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    """AzureOpenAILLM should raise when deployment name is missing."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.azure.com")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    llm = AzureOpenAILLM(client=AsyncMock())

    with pytest.raises(RagError, match="AZURE_OPENAI_DEPLOYMENT"):
        await llm.generate("hello")


async def test_azure_openai_llm_sdk_path_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """AzureOpenAILLM should parse completion text from AsyncAzureOpenAI."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-deploy")

    class _Completions:
        async def create(self, **kwargs):
            del kwargs
            message = types.SimpleNamespace(content="hello")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _AsyncAzureOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncAzureOpenAI=_AsyncAzureOpenAI))
    llm = AzureOpenAILLM()

    assert await llm.generate("prompt") == "hello"
