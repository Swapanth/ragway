from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ragway.exceptions import RagError
from ragway.ingestion.markdown_loader import MarkdownLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class _SpyParser(BaseDocumentParser):
    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content
        return self._build_document(text, source=source, doc_id=doc_id)


async def test_markdown_loader_loads_single_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """MarkdownLoader should read and parse a single markdown file."""
    loader = MarkdownLoader(parser=_SpyParser())

    def _read_text(self: Path, encoding: str) -> str:
        assert encoding == "utf-8"
        return "# Title"

    monkeypatch.setattr(Path, "read_text", _read_text)

    documents = await loader.load("docs/readme.md")

    assert len(documents) == 1
    assert documents[0].doc_id == "md-readme"
    assert documents[0].metadata.source == str(Path("docs/readme.md"))


async def test_markdown_loader_loads_markdown_files_from_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """MarkdownLoader should discover markdown files recursively."""
    loader = MarkdownLoader(parser=_SpyParser())

    def _rglob(self: Path, pattern: str) -> list[Path]:
        assert pattern == "*.md"
        return [Path("a.md"), Path("b.md")]

    def _read_text(self: Path, encoding: str) -> str:
        assert encoding == "utf-8"
        return f"content-{self.name}"

    monkeypatch.setattr(Path, "rglob", _rglob)
    monkeypatch.setattr(Path, "read_text", _read_text)

    documents = await loader.load("docs")

    assert len(documents) == 2
    assert [doc.doc_id for doc in documents] == ["md-a", "md-b"]


async def test_markdown_loader_raises_when_no_files_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """MarkdownLoader should raise RagError when no markdown files exist."""
    loader = MarkdownLoader(parser=_SpyParser())

    def _rglob(self: Path, pattern: str) -> list[Path]:
        assert pattern == "*.md"
        return []

    monkeypatch.setattr(Path, "rglob", _rglob)

    with pytest.raises(RagError):
        await loader.load("empty-dir")

