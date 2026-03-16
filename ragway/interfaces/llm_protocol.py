"""Protocol defining the language model generation contract."""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    """Contract for asynchronous language model providers."""

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate a response string for the supplied prompt."""

    def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield response chunks for the supplied prompt as they arrive."""
