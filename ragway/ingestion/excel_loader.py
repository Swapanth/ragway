"""Asynchronous loader for Excel files stored on disk."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

from ragway.exceptions import RagError
from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.parsing.markdown_parser import MarkdownParser
from ragway.schema.document import Document


class ExcelLoader(BaseLoader):
    """Load XLSX files and parse tabular content into documents."""

    def __init__(self, parser: BaseDocumentParser | None = None) -> None:
        """Initialize loader with optional custom parser."""
        super().__init__(parser or MarkdownParser())

    async def load(self, source: object) -> list[Document]:
        """Load one Excel workbook from disk and return a parsed document."""
        if not isinstance(source, (str, Path)):
            raise RagError("ExcelLoader source must be a file path string or Path")

        path = Path(source)
        if path.suffix.lower() not in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
            raise RagError(f"ExcelLoader only supports Excel files: {path}")

        text = await self._extract_text(path)
        document = self.parser.parse(text, source=str(path), doc_id=f"excel-{path.stem}")
        return [document]

    async def _extract_text(self, path: Path) -> str:
        """Extract worksheet rows to text blocks."""
        try:
            openpyxl_module = importlib.import_module("openpyxl")
            load_workbook = getattr(openpyxl_module, "load_workbook")
        except (ImportError, AttributeError) as exc:
            raise RagError(f"openpyxl is required for Excel loading: {exc}") from exc

        def _read_sync() -> str:
            workbook = load_workbook(path, read_only=True, data_only=True)
            sections: list[str] = []
            for worksheet in workbook.worksheets:
                rows: list[str] = []
                rows.append(f"# Sheet: {worksheet.title}")
                for row in worksheet.iter_rows(values_only=True):
                    cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                    if cells:
                        rows.append("\t".join(cells))
                if len(rows) > 1:
                    sections.append("\n".join(rows))
            workbook.close()
            return "\n\n".join(sections).strip()

        try:
            extracted = await asyncio.to_thread(_read_sync)
        except Exception as exc:
            raise RagError(f"Failed to read Excel file {path}: {exc}") from exc

        if not extracted:
            raise RagError(f"Excel file has no textual content: {path}")
        return extracted
