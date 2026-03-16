"""Abstract base for language model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.validators import validate_positive_int, validate_probability


@dataclass(slots=True)
class LLMConfig:
    """Configuration shared by all language model adapters."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 256

    def __post_init__(self) -> None:
        """Validate model generation parameters."""
        self.temperature = validate_probability(self.temperature, "temperature")
        self.max_tokens = validate_positive_int(self.max_tokens, "max_tokens")


class BaseLLM(LLMProtocol, ABC):
    """Base class for all concrete LLM adapters."""

    config: LLMConfig

    @abstractmethod
    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate a completion for the input prompt."""

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield one fallback chunk when provider streaming is unavailable."""
        yield await self.generate(prompt, stream=False)

