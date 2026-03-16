"""Container wrapper for benchmark RAG evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.exceptions import ValidationError


@dataclass(slots=True)
class RagEvalDataset:
    """In-memory wrapper around normalized benchmark dataset rows."""

    rows: list[dict[str, object]]

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return a dataset row by index."""
        return self.rows[index]

    def split(self, ratio: float = 0.8) -> tuple[RagEvalDataset, RagEvalDataset]:
        """Split dataset into train/test subsets using a deterministic cutoff ratio."""
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValidationError("split ratio must be between 0.0 and 1.0, exclusive")

        cutoff = int(len(self.rows) * ratio)
        train = RagEvalDataset(rows=self.rows[:cutoff])
        test = RagEvalDataset(rows=self.rows[cutoff:])
        return train, test

