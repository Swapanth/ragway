from __future__ import annotations

import asyncio
import sys
import types

import pytest

from ragway.exceptions import RagError
from ragway.ingestion.docx_loader import DocxLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class _TextParser(BaseDocumentParser):
    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content
        return self._build_document(text, source=source, doc_id=doc_id)


class _FakeParagraph:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeCell:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRow:
    def __init__(self, cells: list[str]) -> None:
        self.cells = [_FakeCell(cell) for cell in cells]


class _FakeTable:
    def __init__(self, rows: list[list[str]]) -> None:
        self.rows = [_FakeRow(row) for row in rows]


class _FakeDoc:
    def __init__(self) -> None:
        self.paragraphs = [_FakeParagraph("Heading"), _FakeParagraph("Body")]
        self.tables = [_FakeTable([["a", "b"], ["c", "d"]])]


async def test_docx_loader_loads_docx(monkeypatch: pytest.MonkeyPatch) -> None:
    """DocxLoader should parse text extracted from DOCX structures."""
    fake_docx = types.ModuleType("docx")

    def _document(path: object) -> _FakeDoc:
        assert str(path).endswith("notes.docx")
        return _FakeDoc()

    fake_docx.Document = _document
    monkeypatch.setitem(sys.modules, "docx", fake_docx)

    loader = DocxLoader(parser=_TextParser())
    docs = await loader.load("notes.docx")

    assert len(docs) == 1
    assert docs[0].doc_id == "docx-notes"
    assert "Heading" in docs[0].content
    assert "a | b" in docs[0].content


async def test_docx_loader_rejects_wrong_suffix() -> None:
    """DocxLoader should reject non-DOCX paths."""
    loader = DocxLoader(parser=_TextParser())
    with pytest.raises(RagError):
        await loader.load("notes.txt")
