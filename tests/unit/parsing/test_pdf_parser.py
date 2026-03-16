from __future__ import annotations

import sys
import types

import pytest

from ragway.exceptions import RagError
from ragway.parsing.pdf_parser import PDFParser


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, _: object) -> None:
        self.pages = [_FakePage("First"), _FakePage("Second")]


def test_pdf_parser_uses_pypdf_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDFParser should parse page text via pypdf when available."""
    parser = PDFParser()
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = _FakeReader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    document = parser.parse(b"%PDF-test", source="doc.pdf", doc_id="pdf-1")

    assert document.doc_id == "pdf-1"
    assert document.content == "First\nSecond"
    assert document.metadata.source == "doc.pdf"


def test_pdf_parser_falls_back_to_pdfminer(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDFParser should use pdfminer when pypdf backend is unavailable."""
    parser = PDFParser()
    monkeypatch.delitem(sys.modules, "pypdf", raising=False)

    fake_pdfminer_high_level = types.ModuleType("pdfminer.high_level")
    fake_pdfminer_high_level.extract_text = lambda _stream: "Fallback text"
    monkeypatch.setitem(sys.modules, "pdfminer.high_level", fake_pdfminer_high_level)

    document = parser.parse(b"%PDF-test", source="fallback.pdf")

    assert document.content == "Fallback text"


def test_pdf_parser_raises_when_backends_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDFParser should raise RagError if no backend can parse data."""
    parser = PDFParser()

    fake_pypdf = types.ModuleType("pypdf")
    fake_pdfminer_high_level = types.ModuleType("pdfminer.high_level")
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    monkeypatch.setitem(sys.modules, "pdfminer.high_level", fake_pdfminer_high_level)

    with pytest.raises(RagError):
        parser.parse(b"%PDF-test")

