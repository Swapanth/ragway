from __future__ import annotations

import pytest

from ragway.chunking.fixed_chunker import FixedChunker
from ragway.schema.document import Document


def test_fixed_chunker_splits_by_token_count() -> None:
    """Fixed chunker should split text into windows by token count."""
    document = Document(doc_id="doc-1", content="one two three four five six seven")
    chunker = FixedChunker(chunk_size=3, overlap=0)

    chunks = chunker.chunk(document)

    assert [node.content for node in chunks] == [
        "one two three",
        "four five six",
        "seven",
    ]


def test_fixed_chunker_overlap_advances_by_stride() -> None:
    """Overlap should produce overlapping windows with stride chunk_size-overlap."""
    document = Document(doc_id="doc-2", content="a b c d e f")
    chunker = FixedChunker(chunk_size=4, overlap=2)

    chunks = chunker.chunk(document)

    assert [node.content for node in chunks] == ["a b c d", "c d e f", "e f"]


def test_fixed_chunker_rejects_invalid_overlap() -> None:
    """Overlap equal to or larger than chunk size should be invalid."""
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=2, overlap=2)

