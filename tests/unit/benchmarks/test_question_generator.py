from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from benchmarks.question_generator import QuestionGenerator
from ragway.exceptions import ValidationError


async def test_generate_returns_normalized_rows() -> None:
    """generate should produce canonical rows with context_docs."""
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value='[{"question": "What is RAG?", "gold_answer": "Retrieval-augmented generation"}]'
    )

    generator = QuestionGenerator(llm=llm, pairs_per_document=1)
    rows = await generator.generate(["RAG combines retrieval and generation."])

    assert rows == [
        {
            "question": "What is RAG?",
            "gold_answer": "Retrieval-augmented generation",
            "context_docs": ["RAG combines retrieval and generation."],
        }
    ]
    llm.generate.assert_awaited_once()


async def test_generate_parses_fenced_json() -> None:
    """generate should parse JSON wrapped in markdown code fences."""
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value='```json\n[{"question":"Q","gold_answer":"A"}]\n```'
    )

    generator = QuestionGenerator(llm=llm, pairs_per_document=1)
    rows = await generator.generate(["doc"])

    assert rows[0]["question"] == "Q"
    assert rows[0]["gold_answer"] == "A"


async def test_generate_raises_when_output_invalid() -> None:
    """generate should raise ValidationError for malformed LLM output."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="not-json")

    generator = QuestionGenerator(llm=llm, pairs_per_document=1)

    with pytest.raises(ValidationError):
        await generator.generate(["doc"])


async def test_pairs_per_document_must_be_positive() -> None:
    """QuestionGenerator should validate pairs_per_document."""
    with pytest.raises(ValidationError):
        QuestionGenerator(llm=AsyncMock(), pairs_per_document=0)

