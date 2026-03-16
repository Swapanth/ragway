from __future__ import annotations

import asyncio

import pytest

from ragway.exceptions import RagError
from ragway.ingestion.notion_loader import NotionLoader
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


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def json(self) -> dict[str, object]:
        return self._payload

    async def text(self) -> str:
        return str(self._payload)


class _FakeSession:
    def __init__(self, headers: dict[str, str]) -> None:
        assert "Authorization" in headers

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def get(self, url: str) -> _FakeResponse:
        assert "blocks/0123456789abcdef0123456789abcdef/children" in url
        return _FakeResponse(
            {
                "results": [
                    {
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"plain_text": "Line one"},
                                {"plain_text": " line two"},
                            ]
                        },
                    }
                ],
                "has_more": False,
                "next_cursor": None,
            }
        )


async def test_notion_loader_loads_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """NotionLoader should fetch and parse block text."""
    monkeypatch.setenv("NOTION_API_KEY", "test-key")

    import ragway.ingestion.notion_loader as notion_loader_module

    monkeypatch.setattr(notion_loader_module.aiohttp, "ClientSession", _FakeSession)

    loader = NotionLoader(parser=_TextParser())
    docs = await loader.load("0123456789abcdef0123456789abcdef")

    assert len(docs) == 1
    assert docs[0].doc_id == "notion-0123456789abcdef0123456789abcdef"
    assert docs[0].content == "Line one line two"


async def test_notion_loader_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """NotionLoader should require NOTION_API_KEY."""
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    loader = NotionLoader(parser=_TextParser())

    with pytest.raises(RagError, match="NOTION_API_KEY"):
        await loader.load("0123456789abcdef0123456789abcdef")
