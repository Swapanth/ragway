from __future__ import annotations

from ragway.chunking.semantic_chunker import SemanticChunker
from ragway.schema.document import Document


def test_semantic_chunker_splits_low_similarity_sentences() -> None:
    """Different-topic adjacent sentences should split at high threshold."""
    text = "apple orange fruit. quantum field particle."
    document = Document(doc_id="doc-1", content=text)
    chunker = SemanticChunker(similarity_threshold=0.9, max_tokens=50)

    chunks = chunker.chunk(document)

    assert len(chunks) == 2


def test_semantic_chunker_merges_related_sentences() -> None:
    """Related adjacent sentences should merge when similarity is sufficient."""
    text = "apple fruit orange. orange fruit apple."
    document = Document(doc_id="doc-2", content=text)
    chunker = SemanticChunker(similarity_threshold=0.1, max_tokens=50)

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert "orange fruit apple" in chunks[0].content

