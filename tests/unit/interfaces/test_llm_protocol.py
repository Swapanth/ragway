from __future__ import annotations

import asyncio
import inspect

from ragway.interfaces.llm_protocol import LLMProtocol


class _LLM:
    async def generate(self, prompt: str, stream: bool = False) -> str:
        del stream
        return prompt

    async def stream(self, prompt: str):
        yield prompt


async def test_llm_protocol_runtime_checkable() -> None:
    """A structurally compatible LLM should satisfy LLMProtocol."""
    llm = _LLM()
    assert isinstance(llm, LLMProtocol)


async def test_llm_protocol_generate_is_async() -> None:
    """The protocol generate method should be declared as coroutine function."""
    assert inspect.iscoroutinefunction(LLMProtocol.generate)


async def test_llm_protocol_stream_returns_chunks() -> None:
    """A structurally compatible stream method should yield prompt chunks."""
    llm = _LLM()
    chunks = await _collect(llm.stream("hello"))
    assert chunks == ["hello"]


async def _collect(iterator) -> list[str]:
    items: list[str] = []
    async for chunk in iterator:
        items.append(chunk)
    return items

