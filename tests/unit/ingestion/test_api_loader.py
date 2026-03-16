from __future__ import annotations

import asyncio
import sys
import types

import pytest

from ragway.exceptions import RagError
from ragway.ingestion.api_loader import APILoader
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
    def __init__(self, text: str = "", json_payload: object | None = None) -> None:
        self.text = text
        self._json_payload = json_payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object | None:
        return self._json_payload


async def test_api_loader_calls_get_for_string_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """APILoader should call requests.request with default GET for URL source."""
    loader = APILoader(parser=_TextParser(), timeout_seconds=7)
    fake_requests = types.ModuleType("requests")

    def _request(
        method: str,
        url: str,
        timeout: int,
        headers: object,
        params: object,
        json: object,
        data: object,
    ) -> _FakeResponse:
        assert method == "GET"
        assert url == "https://api.example.test/items"
        assert timeout == 7
        assert headers is None
        assert params is None
        assert json is None
        assert data is None
        return _FakeResponse(text="plain payload", json_payload=None)

    fake_requests.request = _request
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    documents = await loader.load("https://api.example.test/items")

    assert len(documents) == 1
    assert documents[0].content == "plain payload"


async def test_api_loader_serializes_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """APILoader should convert JSON responses into serialized text."""
    loader = APILoader(parser=_TextParser())
    fake_requests = types.ModuleType("requests")

    def _request(
        method: str,
        url: str,
        timeout: int,
        headers: object,
        params: object,
        json: object,
        data: object,
    ) -> _FakeResponse:
        assert method == "POST"
        assert url == "https://api.example.test/search"
        assert isinstance(headers, dict)
        assert isinstance(params, dict)
        assert isinstance(json, dict)
        return _FakeResponse(json_payload={"ok": True, "items": [1, 2]})

    fake_requests.request = _request
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    documents = await loader.load(
        {
            "url": "https://api.example.test/search",
            "method": "post",
            "headers": {"X-Token": "t"},
            "params": {"page": 1},
            "json": {"q": "rag"},
        }
    )


    assert documents[0].content == '{"ok": true, "items": [1, 2]}'
    assert documents[0].metadata.source == "https://api.example.test/search"


async def test_api_loader_wraps_request_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """APILoader should convert request failures to RagError."""
    loader = APILoader(parser=_TextParser())
    fake_requests = types.ModuleType("requests")

    def _request(
        method: str,
        url: str,
        timeout: int,
        headers: object,
        params: object,
        json: object,
        data: object,
    ) -> _FakeResponse:
        raise RuntimeError(f"boom:{method}:{url}:{timeout}:{headers}:{params}:{json}:{data}")

    fake_requests.request = _request
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    with pytest.raises(RagError):
        await loader.load("https://api.example.test/items")

