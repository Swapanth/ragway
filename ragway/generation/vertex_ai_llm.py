"""Vertex AI Gemini LLM adapter."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from dotenv import load_dotenv

from ragway.exceptions import RagError
from ragway.generation.base_llm import BaseLLM, LLMConfig

load_dotenv()


class VertexAIClientProtocol(Protocol):
    """Protocol for async Vertex AI client implementations."""

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        credentials_path: str | None,
        api_key: str | None,
    ) -> str:
        """Generate a completion string with Vertex-compatible backend."""


@dataclass(slots=True)
class VertexAILLM(BaseLLM):
    """Vertex AI adapter supporting service-account or API-key authentication."""

    config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model="gemini-1.5-pro",
            temperature=0.0,
            max_tokens=256,
        )
    )
    client: VertexAIClientProtocol | None = None

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text through Vertex AI with configured parameters."""
        if stream:
            chunks = [chunk async for chunk in self.stream(prompt)]
            return "".join(chunks)

        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not credentials_path and not api_key:
            raise RagError("GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY is required")

        if self.client is not None:
            return await self.client.generate(
                model=self.config.model,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                credentials_path=credentials_path,
                api_key=api_key,
            )

        try:
            vertexai = importlib.import_module("vertexai")
            generative_models = importlib.import_module("vertexai.generative_models")
            generation_config_cls = getattr(generative_models, "GenerationConfig")
            generative_model_cls = getattr(generative_models, "GenerativeModel")
        except (ImportError, AttributeError) as exc:
            raise RagError(f"google-cloud-aiplatform package is required: {exc}") from exc

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project_id:
            raise RagError("GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI")

        vertexai.init(project=project_id, location=location)
        model = generative_model_cls(self.config.model)
        response = await model.generate_content_async(
            prompt,
            generation_config=generation_config_cls(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            ),
        )

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RagError("Vertex AI response did not include text content")
        return text.strip()

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated chunks for Vertex AI responses."""
        yield await self.generate(prompt, stream=False)
