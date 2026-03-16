"""Asynchronous loader for markdown files from disk."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ragway.exceptions import RagError
from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.parsing.markdown_parser import MarkdownParser
from ragway.schema.document import Document


class MarkdownLoader(BaseLoader):
    """Load markdown files from a file path or directory."""

    def __init__(self, parser: BaseDocumentParser | None = None) -> None:
        """Initialize loader with optional markdown parser override."""
        super().__init__(parser or MarkdownParser())

    async def load(self, source: object) -> list[Document]:
        """Load markdown files and return parsed documents."""
        if not isinstance(source, (str, Path)):
            raise RagError("MarkdownLoader source must be a file path string or Path")

        path = Path(source)
        markdown_files = await self._collect_markdown_files(path)
        if not markdown_files:
            raise RagError(f"No markdown files found for source: {path}")

        documents: list[Document] = []
        for markdown_file in markdown_files:
            try:
                content = await asyncio.to_thread(markdown_file.read_text, encoding="utf-8")
            except OSError as exc:
                raise RagError(f"Failed to read markdown file {markdown_file}: {exc}") from exc

            document = self.parser.parse(
                content,
                source=str(markdown_file),
                doc_id=f"md-{markdown_file.stem}",
            )
            documents.append(document)

        return documents

    async def _collect_markdown_files(self, source_path: Path) -> list[Path]:
        """Collect markdown files from file or directory input."""
        if source_path.suffix.lower() == ".md":
            return [source_path]

        try:
            files = await asyncio.to_thread(lambda: list(source_path.rglob("*.md")))
        except OSError as exc:
            raise RagError(f"Failed to scan markdown source {source_path}: {exc}") from exc
        return sorted(files)

