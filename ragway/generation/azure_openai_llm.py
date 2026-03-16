"""Azure OpenAI LLM adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from dotenv import load_dotenv

from ragway.exceptions import RagError
from ragway.generation.base_llm import BaseLLM, LLMConfig

load_dotenv()


class AzureOpenAIClientProtocol(Protocol):
    """Protocol for async Azure OpenAI client implementations."""

    async def generate(
        self,
        deployment: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        endpoint: str,
        api_key: str,
    ) -> str:
        """Generate a completion string through Azure OpenAI."""


@dataclass(slots=True)
class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI adapter using deployment-based chat completions."""

    config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=256,
        )
    )
    client: AzureOpenAIClientProtocol | None = None

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text through Azure OpenAI client using configured parameters."""
        if stream:
            chunks = [chunk async for chunk in self.stream(prompt)]
            return "".join(chunks)

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not api_key:
            raise RagError("AZURE_OPENAI_API_KEY environment variable is required")
        if not endpoint:
            raise RagError("AZURE_OPENAI_ENDPOINT environment variable is required")
        if not deployment:
            raise RagError("AZURE_OPENAI_DEPLOYMENT environment variable is required")

        if self.client is not None:
            return await self.client.generate(
                deployment=deployment,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                endpoint=endpoint,
                api_key=api_key,
            )

        try:
            from openai import AsyncAzureOpenAI
        except ImportError as exc:
            raise RagError(f"openai package is required: {exc}") from exc

        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        response = await client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        message = response.choices[0].message.content if response.choices else None
        if not isinstance(message, str) or not message.strip():
            raise RagError("Azure OpenAI response did not include text content")
        return message.strip()

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated chunks for Azure OpenAI responses."""
        yield await self.generate(prompt, stream=False)
