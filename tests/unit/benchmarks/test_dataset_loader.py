from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks import dataset_loader
from ragway.exceptions import ValidationError


async def test_load_eval_dataset_uses_disk_loader_when_path_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_eval_dataset should route to disk loader for existing paths."""
    expected_rows = [{"question": "Q", "gold_answer": "A", "context_docs": ["doc"]}]
    disk_loader = AsyncMock(return_value=expected_rows)

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(dataset_loader, "_load_from_disk", disk_loader)

    rows = await dataset_loader.load_eval_dataset("dataset.json")

    assert rows == expected_rows
    disk_loader.assert_awaited_once()


async def test_load_eval_dataset_uses_hf_loader_when_path_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_eval_dataset should route to HuggingFace loader for non-path sources."""
    expected_rows = [{"question": "Q", "gold_answer": "A", "context_docs": ["doc"]}]
    hf_loader = AsyncMock(return_value=expected_rows)

    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(dataset_loader, "_load_from_huggingface", hf_loader)

    rows = await dataset_loader.load_eval_dataset("org/dataset", split="test")

    assert rows == expected_rows
    hf_loader.assert_awaited_once_with("org/dataset", split="test")


async def test_load_from_json_normalizes_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON loader should parse rows and keep required benchmark keys."""

    async def fake_to_thread(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return '[{"question": "What?", "gold_answer": "Because", "context_docs": ["doc-1"]}]'

    monkeypatch.setattr(dataset_loader.asyncio, "to_thread", fake_to_thread)

    rows = await dataset_loader._load_from_json(Path("mock.json"))

    assert rows == [{"question": "What?", "gold_answer": "Because", "context_docs": ["doc-1"]}]


async def test_load_from_csv_uses_context_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """CSV loader should map context column into context_docs."""

    async def fake_to_thread(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return "question,gold_answer,context\nQ1,A1,doc one"

    monkeypatch.setattr(dataset_loader.asyncio, "to_thread", fake_to_thread)

    rows = await dataset_loader._load_from_csv(Path("mock.csv"))

    assert rows == [{"question": "Q1", "gold_answer": "A1", "context_docs": ["doc one"]}]


async def test_load_from_huggingface_normalizes_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """HuggingFace loader should normalize dataset rows into canonical keys."""

    def fake_load_dataset(name: str, *, split: str) -> list[dict[str, object]]:
        assert name == "org/my-dataset"
        assert split == "train"
        return [{"question": "Q", "gold_answer": "A", "context_docs": ["doc"]}]

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> object:
        assert callable(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(dataset_loader.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    rows = await dataset_loader._load_from_huggingface("org/my-dataset", split="train")

    assert rows == [{"question": "Q", "gold_answer": "A", "context_docs": ["doc"]}]


async def test_normalize_rows_raises_for_missing_question() -> None:
    """normalize_rows should raise if required text fields are empty."""
    with pytest.raises(ValidationError):
        dataset_loader._normalize_rows([
            {"question": "", "gold_answer": "A", "context_docs": ["doc"]},
        ])

