"""Context compression component for fitting retrieved context to token limits."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.schema.node import Node
from ragway.validators import validate_positive_int


@dataclass(slots=True)
class ContextCompression:
    """Compress retrieved context blocks to a target token budget."""

    token_limit: int = 256

    def __post_init__(self) -> None:
        """Validate configured token limit."""
        self.token_limit = validate_positive_int(self.token_limit, "token_limit")

    def compress(self, nodes: list[Node]) -> str:
        """Return concatenated node text truncated to token_limit words."""
        if not nodes:
            return ""

        output_tokens: list[str] = []
        for node in nodes:
            for token in node.content.split():
                if len(output_tokens) >= self.token_limit:
                    return " ".join(output_tokens)
                output_tokens.append(token)

        return " ".join(output_tokens)

