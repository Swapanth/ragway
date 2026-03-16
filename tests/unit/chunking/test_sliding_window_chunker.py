from __future__ import annotations

from ragway.chunking.sliding_window_chunker import SlidingWindowChunker
from ragway.schema.document import Document


def test_sliding_window_chunker_uses_window_and_stride() -> None:
    """Sliding windows should advance by stride and keep window size where possible."""
    document = Document(doc_id="doc-1", content="a b c d e f g")
    chunker = SlidingWindowChunker(window_size=4, stride=2)

    chunks = chunker.chunk(document)

    assert [node.content for node in chunks] == ["a b c d", "c d e f", "e f g"]


def test_sliding_window_chunker_single_token_input() -> None:
    """Single-token content should produce exactly one chunk."""
    document = Document(doc_id="doc-2", content="only")
    chunker = SlidingWindowChunker(window_size=3, stride=1)

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == "only"

