from __future__ import annotations

import pytest

from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class _NoImplParser(BaseDocumentParser):
    pass


class _EchoParser(BaseDocumentParser):
    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content
        return self._build_document(text, source=source, doc_id=doc_id)


def test_document_parser_is_abstract() -> None:
    """Base parser should not be instantiated without parse()."""
    with pytest.raises(TypeError):
        _NoImplParser()


def test_build_document_sets_source_and_doc_id() -> None:
    """Concrete parser should build validated document with metadata source."""
    parser = _EchoParser()
    document = parser.parse("Hello parser", source="notes.md", doc_id="doc-fixed")

    assert document.doc_id == "doc-fixed"
    assert document.content == "Hello parser"
    assert document.metadata.source == "notes.md"


def test_build_document_generates_stable_identifier() -> None:
    """Generated IDs should be deterministic for same content and source."""
    parser = _EchoParser()

    left = parser.parse("alpha", source="a.md")
    right = parser.parse("alpha", source="a.md")

    assert left.doc_id == right.doc_id
    assert left.doc_id.startswith("doc-")

