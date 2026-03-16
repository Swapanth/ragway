"""OpenAI LLM adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from dotenv import load_dotenv

from ragway.exceptions import RagError
from ragway.generation.base_llm import BaseLLM, LLMConfig

load_dotenv()


class OpenAIClientProtocol(Protocol):
    """Protocol for async OpenAI chat-completion clients."""

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
    ) -> str:
        """Generate a completion string with OpenAI API."""


@dataclass(slots=True)
class OpenAILLM(BaseLLM):
    """OpenAI adapter that reads API key from environment."""

    config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=256,
        )
    )
    client: OpenAIClientProtocol | None = None

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text through OpenAI client using configured parameters."""
        if stream:
            chunks = [chunk async for chunk in self.stream(prompt)]
            return "".join(chunks)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RagError("OPENAI_API_KEY environment variable is required")

        if self.client is not None:
            return await self.client.generate(
                model=self.config.model,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RagError(f"openai package is required: {exc}") from exc

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        message = response.choices[0].message.content if response.choices else None
        if not isinstance(message, str) or not message.strip():
            raise RagError("OpenAI response did not include text content")
        return message.strip()

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated chunks for OpenAI responses."""
        yield await self.generate(prompt, stream=False)

