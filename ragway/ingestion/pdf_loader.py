"""Asynchronous loader for PDF files stored on disk."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ragway.exceptions import RagError
from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.parsing.pdf_parser import PDFParser
from ragway.schema.document import Document


class PDFLoader(BaseLoader):
    """Load PDF files from disk and parse them into documents."""

    def __init__(self, parser: BaseDocumentParser | None = None) -> None:
        """Initialize loader with an optional custom parser."""
        super().__init__(parser or PDFParser())

    async def load(self, source: object) -> list[Document]:
        """Load one PDF file from disk and return a single parsed document."""
        if not isinstance(source, (str, Path)):
            raise RagError("PDFLoader source must be a file path string or Path")

        path = Path(source)
        if path.suffix.lower() != ".pdf":
            raise RagError(f"PDFLoader only supports .pdf files: {path}")

        try:
            pdf_bytes = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise RagError(f"Failed to read PDF file {path}: {exc}") from exc

        document = self.parser.parse(pdf_bytes, source=str(path), doc_id=f"pdf-{path.stem}")
        return [document]

