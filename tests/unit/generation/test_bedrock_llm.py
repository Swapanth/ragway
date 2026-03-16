from __future__ import annotations

import asyncio
import json
from io import BytesIO
from unittest.mock import AsyncMock

import pytest

from ragway.exceptions import RagError
from ragway.generation.base_llm import LLMConfig
from ragway.generation.bedrock_llm import BedrockLLM


async def test_bedrock_llm_uses_mocked_client() -> None:
    """BedrockLLM should call async client with config-derived values."""
    client = AsyncMock()
    client.generate.return_value = "response"

    config = LLMConfig(model="anthropic.claude-3-5-sonnet-20241022-v2:0", temperature=0.3, max_tokens=121)
    llm = BedrockLLM(config=config, client=client)
    result = await llm.generate("hello")

    assert result == "response"
    client.generate.assert_awaited_once_with(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        prompt="hello",
        temperature=0.3,
        max_tokens=121,
    )


async def test_bedrock_llm_import_error_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """BedrockLLM should map missing boto3 dependency to RagError."""
    monkeypatch.setattr("ragway.generation.bedrock_llm.importlib.import_module", lambda _: (_ for _ in ()).throw(ImportError("missing")))
    llm = BedrockLLM()

    with pytest.raises(RagError, match="boto3 package is required"):
        await llm.generate("hello")


async def test_bedrock_llm_sdk_path_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """BedrockLLM should parse model response body into text output."""

    class _Runtime:
        def invoke_model(self, **kwargs):
            del kwargs
            payload = {"content": [{"text": "hello"}, {"text": "world"}]}
            return {"body": BytesIO(json.dumps(payload).encode("utf-8"))}

    class _Boto3:
        @staticmethod
        def client(name: str, region_name: str):
            del name, region_name
            return _Runtime()

    monkeypatch.setattr("ragway.generation.bedrock_llm.importlib.import_module", lambda _: _Boto3)
    llm = BedrockLLM()

    assert await llm.generate("prompt") == "hello\nworld"


async def test_bedrock_llm_missing_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """BedrockLLM should raise when response body is missing."""

    class _Runtime:
        def invoke_model(self, **kwargs):
            del kwargs
            return {}

    class _Boto3:
        @staticmethod
        def client(name: str, region_name: str):
            del name, region_name
            return _Runtime()

    monkeypatch.setattr("ragway.generation.bedrock_llm.importlib.import_module", lambda _: _Boto3)
    llm = BedrockLLM()

    with pytest.raises(RagError, match="response body"):
        await llm.generate("prompt")
