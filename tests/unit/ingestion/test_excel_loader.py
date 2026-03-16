from __future__ import annotations

import asyncio
import sys
import types

import pytest

from ragway.exceptions import RagError
from ragway.ingestion.excel_loader import ExcelLoader
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


class _FakeWorksheet:
    def __init__(self, title: str, rows: list[tuple[object, ...]]) -> None:
        self.title = title
        self._rows = rows

    def iter_rows(self, values_only: bool = True) -> list[tuple[object, ...]]:
        assert values_only is True
        return self._rows


class _FakeWorkbook:
    def __init__(self) -> None:
        self.worksheets = [
            _FakeWorksheet("Sheet1", [("a", "b"), (1, 2)]),
        ]

    def close(self) -> None:
        return None


async def test_excel_loader_loads_xlsx(monkeypatch: pytest.MonkeyPatch) -> None:
    """ExcelLoader should serialize worksheet rows into document content."""
    fake_openpyxl = types.ModuleType("openpyxl")

    def _load_workbook(path: object, read_only: bool, data_only: bool) -> _FakeWorkbook:
        assert str(path).endswith("sheet.xlsx")
        assert read_only is True
        assert data_only is True
        return _FakeWorkbook()

    fake_openpyxl.load_workbook = _load_workbook
    monkeypatch.setitem(sys.modules, "openpyxl", fake_openpyxl)

    loader = ExcelLoader(parser=_TextParser())
    docs = await loader.load("sheet.xlsx")

    assert len(docs) == 1
    assert docs[0].doc_id == "excel-sheet"
    assert "# Sheet: Sheet1" in docs[0].content
    assert "a\tb" in docs[0].content


async def test_excel_loader_rejects_wrong_suffix() -> None:
    """ExcelLoader should reject non-Excel paths."""
    loader = ExcelLoader(parser=_TextParser())
    with pytest.raises(RagError):
        await loader.load("sheet.txt")
