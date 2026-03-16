from __future__ import annotations

import pytest

from benchmarks.rag_eval_dataset import RagEvalDataset
from ragway.exceptions import ValidationError


def test_len_returns_dataset_size() -> None:
    """RagEvalDataset __len__ should return the number of rows."""
    dataset = RagEvalDataset(rows=[{"question": "Q", "gold_answer": "A", "context_docs": []}])
    assert len(dataset) == 1


def test_getitem_returns_row_by_index() -> None:
    """RagEvalDataset __getitem__ should return rows by index."""
    row = {"question": "Q", "gold_answer": "A", "context_docs": []}
    dataset = RagEvalDataset(rows=[row])
    assert dataset[0] == row


def test_split_returns_train_and_test_subsets() -> None:
    """split should partition rows according to the provided ratio."""
    rows = [
        {"question": "Q1", "gold_answer": "A1", "context_docs": []},
        {"question": "Q2", "gold_answer": "A2", "context_docs": []},
        {"question": "Q3", "gold_answer": "A3", "context_docs": []},
        {"question": "Q4", "gold_answer": "A4", "context_docs": []},
    ]
    dataset = RagEvalDataset(rows=rows)

    train, test = dataset.split(ratio=0.5)

    assert len(train) == 2
    assert len(test) == 2
    assert train[0]["question"] == "Q1"
    assert test[0]["question"] == "Q3"


def test_split_raises_for_invalid_ratio() -> None:
    """split should validate ratio is strictly between 0 and 1."""
    dataset = RagEvalDataset(rows=[])
    with pytest.raises(ValidationError):
        dataset.split(ratio=1.0)

