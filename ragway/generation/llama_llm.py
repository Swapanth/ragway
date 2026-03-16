"""Local LLaMA adapter compatible with llama-cpp-python style clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from ragway.exceptions import RagError
from ragway.generation.base_llm import BaseLLM, LLMConfig


class LlamaClientProtocol(Protocol):
    """Protocol for async llama-cpp-python style generation clients."""

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a completion string from a local LLaMA model."""


@dataclass(slots=True)
class LlamaLLM(BaseLLM):
    """LLaMA adapter for local inference via injected client."""

    config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model="llama-3.1-8b-instruct",
            temperature=0.0,
            max_tokens=256,
        )
    )
    client: LlamaClientProtocol | None = None

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text through local llama client using configured parameters."""
        if stream:
            chunks = [chunk async for chunk in self.stream(prompt)]
            return "".join(chunks)

        if self.client is None:
            raise RagError("LlamaLLM requires a llama-cpp-python compatible client")

        return await self.client.generate(
            model=self.config.model,
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated chunks for local llama responses."""
        yield await self.generate(prompt, stream=False)

