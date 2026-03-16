"""Asynchronous loader for DOCX files stored on disk."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

from ragway.exceptions import RagError
from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.parsing.markdown_parser import MarkdownParser
from ragway.schema.document import Document


class DocxLoader(BaseLoader):
    """Load DOCX files and parse extracted text into documents."""

    def __init__(self, parser: BaseDocumentParser | None = None) -> None:
        """Initialize loader with optional custom parser."""
        super().__init__(parser or MarkdownParser())

    async def load(self, source: object) -> list[Document]:
        """Load one DOCX file from disk and return a parsed document."""
        if not isinstance(source, (str, Path)):
            raise RagError("DocxLoader source must be a file path string or Path")

        path = Path(source)
        if path.suffix.lower() != ".docx":
            raise RagError(f"DocxLoader only supports .docx files: {path}")

        text = await self._extract_text(path)
        document = self.parser.parse(text, source=str(path), doc_id=f"docx-{path.stem}")
        return [document]

    async def _extract_text(self, path: Path) -> str:
        """Extract visible paragraph and table text from DOCX."""
        try:
            docx_module = importlib.import_module("docx")
            docx_document = getattr(docx_module, "Document")
        except (ImportError, AttributeError) as exc:
            raise RagError(f"python-docx is required for DOCX loading: {exc}") from exc

        def _read_sync() -> str:
            doc = docx_document(path)
            lines: list[str] = []
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    lines.append(text)
            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        lines.append(" | ".join(cells))
            return "\n".join(lines).strip()

        try:
            extracted = await asyncio.to_thread(_read_sync)
        except Exception as exc:
            raise RagError(f"Failed to read DOCX file {path}: {exc}") from exc

        if not extracted:
            raise RagError(f"DOCX file has no textual content: {path}")
        return extracted
