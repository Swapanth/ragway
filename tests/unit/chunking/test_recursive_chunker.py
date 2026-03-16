from __future__ import annotations

from ragway.chunking.recursive_chunker import RecursiveChunker
from ragway.schema.document import Document


def test_recursive_chunker_prefers_paragraph_boundaries() -> None:
    """Paragraph boundaries should be preserved when each paragraph fits."""
    text = "one two three\n\nfour five six"
    document = Document(doc_id="doc-1", content=text)
    chunker = RecursiveChunker(max_tokens=3)

    chunks = chunker.chunk(document)

    assert [node.content for node in chunks] == ["one two three", "four five six"]


def test_recursive_chunker_falls_back_to_sentence_and_words() -> None:
    """Large text without paragraph splits should break by sentence then words."""
    text = "one two three four. five six seven eight."
    document = Document(doc_id="doc-2", content=text)
    chunker = RecursiveChunker(max_tokens=3)

    chunks = chunker.chunk(document)

    assert all(len(node.content.split()) <= 3 for node in chunks)
    assert len(chunks) >= 3

