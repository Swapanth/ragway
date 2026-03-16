from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.synthetic_dataset_generator import SyntheticDatasetGenerator


async def test_generate_saves_json_and_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """SyntheticDatasetGenerator should persist generated rows and return them."""
    llm = AsyncMock()
    generator = SyntheticDatasetGenerator(llm=llm, pairs_per_document=1)

    expected_rows = [{"question": "Q", "gold_answer": "A", "context_docs": ["doc"]}]
    generator.question_generator = SimpleNamespace(generate=AsyncMock(return_value=expected_rows))

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> object:
        calls.append((getattr(func, "__name__", str(func)), args, kwargs))
        return None

    monkeypatch.setattr("benchmarks.synthetic_dataset_generator.asyncio.to_thread", fake_to_thread)

    output_path = Path("benchmarks_output/synthetic.json")
    rows = await generator.generate(["doc"], output_path)

    assert rows == expected_rows
    assert calls[0][0] == "mkdir"
    assert calls[1][0] == "write_text"

    written_payload = calls[1][1][0]
    assert json.loads(str(written_payload)) == expected_rows
