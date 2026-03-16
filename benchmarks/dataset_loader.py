"""Load benchmark evaluation datasets from local files or HuggingFace datasets."""

from __future__ import annotations

import asyncio
import csv
import io
import importlib
import json
from pathlib import Path
from typing import Mapping

from ragway.exceptions import RagError, ValidationError


_REQUIRED_KEYS = ("question", "gold_answer", "context_docs")


async def load_eval_dataset(source: str | Path, *, split: str = "train") -> list[dict[str, object]]:
    """Load an evaluation dataset from JSON/CSV files or a HuggingFace dataset."""
    source_path = Path(source)
    if source_path.exists():
        return await _load_from_disk(source_path)
    return await _load_from_huggingface(str(source), split=split)


async def _load_from_disk(path: Path) -> list[dict[str, object]]:
    """Load a dataset from supported file formats on disk."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return await _load_from_json(path)
    if suffix == ".csv":
        return await _load_from_csv(path)
    raise ValidationError(f"Unsupported dataset file extension: {suffix}")


async def _load_from_json(path: Path) -> list[dict[str, object]]:
    """Load and normalize rows from a JSON dataset file."""
    try:
        raw_content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except OSError as exc:
        raise RagError(f"Failed to read dataset JSON file {path}: {exc}") from exc

    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in dataset file {path}: {exc}") from exc

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        candidate = payload.get("data")
        if not isinstance(candidate, list):
            raise ValidationError(f"JSON dataset at {path} must contain a list or a 'data' list")
        rows = candidate
    else:
        raise ValidationError(f"JSON dataset at {path} must be a list or object with a 'data' list")

    return _normalize_rows(rows)


async def _load_from_csv(path: Path) -> list[dict[str, object]]:
    """Load and normalize rows from a CSV dataset file."""
    try:
        raw_content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except OSError as exc:
        raise RagError(f"Failed to read dataset CSV file {path}: {exc}") from exc

    reader = csv.DictReader(io.StringIO(raw_content))
    rows: list[Mapping[str, object]] = [row for row in reader]
    return _normalize_rows(rows)


async def _load_from_huggingface(dataset_name: str, *, split: str) -> list[dict[str, object]]:
    """Load and normalize rows from a HuggingFace dataset split."""
    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = getattr(datasets_module, "load_dataset", None)
        if load_dataset is None:
            raise RagError("datasets.load_dataset is unavailable")
    except ImportError as exc:
        raise RagError(f"datasets is required to load HuggingFace datasets: {exc}") from exc

    try:
        dataset = await asyncio.to_thread(load_dataset, dataset_name, split=split)
    except Exception as exc:
        raise RagError(f"Failed to load HuggingFace dataset '{dataset_name}' split '{split}': {exc}") from exc

    rows: list[Mapping[str, object]] = []
    for item in dataset:
        if not isinstance(item, Mapping):
            raise ValidationError("HuggingFace dataset rows must be mapping-like")
        rows.append(item)

    return _normalize_rows(rows)


def _normalize_rows(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    """Normalize arbitrary row mappings into canonical evaluation row dictionaries."""
    normalized: list[dict[str, object]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValidationError(f"Dataset row at index {index} must be a mapping")

        question = _coerce_text(row.get("question"))
        gold_answer = _coerce_text(row.get("gold_answer"))
        context_docs = _coerce_context_docs(row)

        normalized_row = {
            "question": question,
            "gold_answer": gold_answer,
            "context_docs": context_docs,
        }

        missing_keys = [key for key in _REQUIRED_KEYS if key not in normalized_row]
        if missing_keys:
            raise ValidationError(f"Dataset row at index {index} is missing keys: {missing_keys}")

        normalized.append(normalized_row)

    return normalized


def _coerce_text(value: object) -> str:
    """Coerce a value into a non-empty string."""
    if value is None:
        raise ValidationError("Dataset rows must include non-empty question and gold_answer fields")

    text = str(value).strip()
    if not text:
        raise ValidationError("Dataset rows must include non-empty question and gold_answer fields")
    return text


def _coerce_context_docs(row: Mapping[str, object]) -> list[str]:
    """Normalize context documents field from common dataset schemas."""
    raw_context = row.get("context_docs", row.get("context", []))

    if isinstance(raw_context, str):
        context_docs = [raw_context.strip()] if raw_context.strip() else []
    elif isinstance(raw_context, list):
        context_docs = [str(item).strip() for item in raw_context if str(item).strip()]
    else:
        raise ValidationError("Dataset row context_docs/context must be a string or list")

    return context_docs

