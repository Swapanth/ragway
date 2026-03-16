"""Query expansion component for generating search variants via LLM."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.validators import validate_positive_int


@dataclass(slots=True)
class QueryExpansion:
    """Generate query variants with an LLM for improved retrieval coverage."""

    llm: LLMProtocol
    variant_count: int = 3

    def __post_init__(self) -> None:
        """Validate expansion configuration."""
        self.variant_count = validate_positive_int(self.variant_count, "variant_count")

    async def expand(self, query: str) -> list[str]:
        """Return up to N unique query variants including the original query."""
        prompt = (
            "Generate concise alternate search queries that preserve user intent. "
            "Return one variant per line with no numbering.\n"
            f"Original query: {query}"
        )
        response = await self.llm.generate(prompt)

        variants: list[str] = [query]
        for line in response.splitlines():
            candidate = line.strip(" -\t")
            if candidate and candidate not in variants:
                variants.append(candidate)
            if len(variants) >= self.variant_count:
                break

        return variants

