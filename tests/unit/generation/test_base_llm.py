from __future__ import annotations

import asyncio

import pytest

from ragway.generation.base_llm import BaseLLM, LLMConfig


class _NoImplLLM(BaseLLM):
    pass


class _EchoLLM(BaseLLM):
    def __init__(self) -> None:
        self.config = LLMConfig(model="echo-model", temperature=0.1, max_tokens=64)

    async def generate(self, prompt: str, stream: bool = False) -> str:
        del stream
        return prompt


async def test_base_llm_is_abstract() -> None:
    """BaseLLM should require an implementation of generate."""
    with pytest.raises(TypeError):
        _NoImplLLM()


async def test_llm_config_validates_values() -> None:
    """LLMConfig should validate max tokens and temperature bounds."""
    with pytest.raises(Exception):
        LLMConfig(model="x", temperature=2.0, max_tokens=10)


async def test_echo_llm_returns_prompt() -> None:
    """Concrete LLM should asynchronously return generated text."""
    llm = _EchoLLM()
    assert await llm.generate("hello") == "hello"

