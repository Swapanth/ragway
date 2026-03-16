from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from ragway.components.query_expansion import QueryExpansion


async def test_query_expansion_generates_unique_variants() -> None:
    """QueryExpansion should parse and deduplicate LLM-generated variants."""
    llm = AsyncMock()
    llm.generate.return_value = "variant one\nvariant two\nvariant one"

    component = QueryExpansion(llm=llm, variant_count=3)
    result = await component.expand("original")

    assert result == ["original", "variant one", "variant two"]
    llm.generate.assert_awaited_once()

