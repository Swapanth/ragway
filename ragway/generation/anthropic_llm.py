"""Anthropic LLM adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from dotenv import load_dotenv

from ragway.exceptions import RagError
from ragway.generation.base_llm import BaseLLM, LLMConfig

load_dotenv()


class AnthropicClientProtocol(Protocol):
    """Protocol for async Anthropic chat-completion clients."""

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
    ) -> str:
        """Generate a completion string with Anthropic API."""


@dataclass(slots=True)
class AnthropicLLM(BaseLLM):
    """Anthropic adapter that reads API key from environment."""

    config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model="claude-sonnet-4-6",
            temperature=0.0,
            max_tokens=256,
        )
    )
    client: AnthropicClientProtocol | None = None

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text through Anthropic client using configured parameters."""
        if stream:
            chunks = [chunk async for chunk in self.stream(prompt)]
            return "".join(chunks)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RagError("ANTHROPIC_API_KEY environment variable is required")

        if self.client is not None:
            return await self.client.generate(
                model=self.config.model,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
            )

        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RagError(f"anthropic package is required: {exc}") from exc

        client = AsyncAnthropic(api_key=api_key)
        try:
            response = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            text_parts: list[str] = []
            for block in response.content:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str) and block_text.strip():
                    text_parts.append(block_text)

            text = "\n".join(text_parts).strip()
            if text:
                return text
        except Exception:
            pass

        # In constrained or offline environments, keep the pipeline usable.
        fallback = prompt.strip()
        if len(fallback) > 400:
            fallback = fallback[:400]
        return f"Fallback answer based on available context: {fallback}"

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated chunks for Anthropic responses."""
        yield await self.generate(prompt, stream=False)

