"""AWS Bedrock LLM adapter."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from dotenv import load_dotenv

from ragway.exceptions import RagError
from ragway.generation.base_llm import BaseLLM, LLMConfig

load_dotenv()


class BedrockClientProtocol(Protocol):
    """Protocol for async Bedrock client implementations."""

    async def generate(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate a completion string via AWS Bedrock."""


@dataclass(slots=True)
class BedrockLLM(BaseLLM):
    """Bedrock adapter that reads AWS credentials from environment variables."""

    config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            temperature=0.0,
            max_tokens=256,
        )
    )
    client: BedrockClientProtocol | None = None

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text through Bedrock runtime using configured parameters."""
        if stream:
            chunks = [chunk async for chunk in self.stream(prompt)]
            return "".join(chunks)

        if self.client is not None:
            return await self.client.generate(
                model=self.config.model,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

        try:
            boto3 = importlib.import_module("boto3")
        except ImportError as exc:
            raise RagError(f"boto3 package is required: {exc}") from exc

        runtime = boto3.client("bedrock-runtime", region_name=region)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }

        response = await asyncio.to_thread(
            runtime.invoke_model,
            modelId=self.config.model,
            body=json.dumps(body),
        )

        stream_raw = response.get("body")
        if stream_raw is None or not hasattr(stream_raw, "read"):
            raise RagError("Bedrock response did not include a response body")

        payload_bytes = await asyncio.to_thread(stream_raw.read)
        payload = json.loads(payload_bytes.decode("utf-8"))
        content_blocks = payload.get("content", [])
        if not isinstance(content_blocks, list):
            raise RagError("Bedrock response content format is invalid")

        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)

        text_out = "\n".join(text_parts).strip()
        if not text_out:
            raise RagError("Bedrock response did not include text content")
        return text_out

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated chunks for Bedrock responses."""
        yield await self.generate(prompt, stream=False)
