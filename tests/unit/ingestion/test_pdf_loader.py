from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ragway.exceptions import RagError
from ragway.ingestion.pdf_loader import PDFLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class _SpyParser(BaseDocumentParser):
    def __init__(self) -> None:
        self.calls: list[tuple[str | bytes, str | None, str | None]] = []

    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        self.calls.append((raw_content, source, doc_id))
        text = "pdf text"
        return self._build_document(text, source=source, doc_id=doc_id)


async def test_pdf_loader_reads_file_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDFLoader should read bytes from disk and parse into one document."""
    parser = _SpyParser()
    loader = PDFLoader(parser=parser)

    def _read_bytes(self: Path) -> bytes:
        return b"%PDF-1.7"

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)

    documents = await loader.load("reports/sample.pdf")

    assert len(documents) == 1
    assert documents[0].doc_id == "pdf-sample"
    assert parser.calls[0][0] == b"%PDF-1.7"


async def test_pdf_loader_rejects_non_pdf_source() -> None:
    """PDFLoader should reject unsupported file extensions."""
    loader = PDFLoader(parser=_SpyParser())

    with pytest.raises(RagError):
        await loader.load("notes.txt")


async def test_pdf_loader_wraps_file_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDFLoader should convert OSError into RagError with context."""
    loader = PDFLoader(parser=_SpyParser())

    def _raise_error(self: Path) -> bytes:
        raise OSError("not found")

    monkeypatch.setattr(Path, "read_bytes", _raise_error)

    with pytest.raises(RagError):
        await loader.load("missing.pdf")

