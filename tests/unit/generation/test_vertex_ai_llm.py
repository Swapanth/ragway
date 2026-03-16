from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.base_llm import LLMConfig
from ragway.generation.vertex_ai_llm import VertexAILLM


async def test_vertex_ai_llm_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """VertexAILLM should require GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    llm = VertexAILLM(client=AsyncMock())

    with pytest.raises(RagError):
        await llm.generate("hello")


async def test_vertex_ai_llm_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """VertexAILLM should call async client with config-derived values."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="gemini-1.5-pro", temperature=0.1, max_tokens=150)
    llm = VertexAILLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        model="gemini-1.5-pro",
        prompt="hello",
        temperature=0.1,
        max_tokens=150,
        credentials_path=None,
        api_key="test-key",
    )


async def test_vertex_ai_llm_requires_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """VertexAILLM should require GOOGLE_CLOUD_PROJECT on SDK path."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    fake_vertex = types.SimpleNamespace(init=lambda **kwargs: None)
    fake_gen = types.SimpleNamespace(GenerationConfig=object, GenerativeModel=object)

    def _import_module(name: str):
        if name == "vertexai":
            return fake_vertex
        if name == "vertexai.generative_models":
            return fake_gen
        raise ImportError(name)

    monkeypatch.setattr("ragway.generation.vertex_ai_llm.importlib.import_module", _import_module)
    llm = VertexAILLM()
    with pytest.raises(RagError, match="GOOGLE_CLOUD_PROJECT"):
        await llm.generate("hello")


async def test_vertex_ai_llm_sdk_path_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """VertexAILLM should parse text from generate_content_async response."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")

    class _GenerationConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Model:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def generate_content_async(self, prompt: str, generation_config: object):
            del prompt, generation_config
            return types.SimpleNamespace(text="vertex-answer")

    fake_vertex = types.SimpleNamespace(init=lambda **kwargs: kwargs)
    fake_gen = types.SimpleNamespace(GenerationConfig=_GenerationConfig, GenerativeModel=_Model)

    def _import_module(name: str):
        if name == "vertexai":
            return fake_vertex
        if name == "vertexai.generative_models":
            return fake_gen
        raise ImportError(name)

    monkeypatch.setattr("ragway.generation.vertex_ai_llm.importlib.import_module", _import_module)
    llm = VertexAILLM()
    assert await llm.generate("hello") == "vertex-answer"
