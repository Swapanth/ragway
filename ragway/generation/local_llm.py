"""Generic local LLM adapter wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from ragway.exceptions import RagError
from ragway.generation.base_llm import BaseLLM, LLMConfig


class LocalClientProtocol(Protocol):
    """Protocol for generic async local model clients."""

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate text from a local model runtime."""


@dataclass(slots=True)
class LocalLLM(BaseLLM):
    """Adapter for arbitrary local model runtimes."""

    config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model="local-default-model",
            temperature=0.0,
            max_tokens=256,
        )
    )
    client: LocalClientProtocol | None = None

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text through generic local client using configured parameters."""
        if stream:
            chunks = [chunk async for chunk in self.stream(prompt)]
            return "".join(chunks)

        if self.client is None:
            raise RagError("LocalLLM requires an async local client")

        return await self.client.generate(
            model=self.config.model,
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated chunks for local model responses."""
        yield await self.generate(prompt, stream=False)

