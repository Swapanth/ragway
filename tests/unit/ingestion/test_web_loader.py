from __future__ import annotations

import asyncio
import sys
import types

import pytest

from ragway.exceptions import RagError
from ragway.ingestion.web_loader import WebLoader
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
    def __init__(self, text: str, status_ok: bool = True) -> None:
        self.text = text
        self._status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self._status_ok:
            raise RuntimeError("http error")


async def test_web_loader_fetches_single_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebLoader should fetch one URL and parse response text."""
    loader = WebLoader(parser=_TextParser(), timeout_seconds=5)
    fake_requests = types.ModuleType("requests")

    def _get(url: str, timeout: int) -> _FakeResponse:
        assert url == "https://example.com"
        assert timeout == 5
        return _FakeResponse("<html>content</html>")

    fake_requests.get = _get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    documents = await loader.load("https://example.com")

    assert len(documents) == 1
    assert documents[0].doc_id == "web-1"
    assert documents[0].metadata.source == "https://example.com"


async def test_web_loader_fetches_multiple_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebLoader should fetch all URLs in order."""
    loader = WebLoader(parser=_TextParser())
    fake_requests = types.ModuleType("requests")

    def _get(url: str, timeout: int) -> _FakeResponse:
        return _FakeResponse(f"payload:{url}:{timeout}")

    fake_requests.get = _get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    documents = await loader.load(["https://a.test", "https://b.test"])

    assert len(documents) == 2
    assert [doc.doc_id for doc in documents] == ["web-1", "web-2"]


async def test_web_loader_wraps_request_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebLoader should convert request failures into RagError."""
    loader = WebLoader(parser=_TextParser())
    fake_requests = types.ModuleType("requests")

    def _get(url: str, timeout: int) -> _FakeResponse:
        raise RuntimeError(f"boom:{url}:{timeout}")

    fake_requests.get = _get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    with pytest.raises(RagError):
        await loader.load("https://example.com")

