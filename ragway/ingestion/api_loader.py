"""Asynchronous generic REST API loader for document ingestion."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol, TypedDict, cast

from ragway.exceptions import RagError
from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class _RequestConfig(TypedDict):
    """Normalized request configuration for API calls."""

    method: str
    url: str
    headers: object | None
    params: object | None
    json: object | None
    data: object | None


class _ResponseProtocol(Protocol):
    """Subset of HTTP response behavior needed by APILoader."""

    text: str

    def raise_for_status(self) -> None:
        """Raise if the HTTP response represents an error status."""

    def json(self) -> object:
        """Return decoded JSON payload when available."""


class _TextDocumentParser(BaseDocumentParser):
    """Default parser for plain API text payloads."""

    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        """Wrap raw API text as a document."""
        text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content
        return self._build_document(text, source=source, doc_id=doc_id)


class APILoader(BaseLoader):
    """Load documents from REST API endpoints."""

    def __init__(
        self,
        parser: BaseDocumentParser | None = None,
        timeout_seconds: int = 20,
        default_method: str = "GET",
    ) -> None:
        """Initialize loader with parser and HTTP defaults."""
        super().__init__(parser or _TextDocumentParser())
        self.timeout_seconds = timeout_seconds
        self.default_method = default_method.upper()

    async def load(self, source: object) -> list[Document]:
        """Load API payload as one document from a URL or request config."""
        request = self._normalize_request(source)

        try:
            import requests
        except ImportError as exc:
            raise RagError(f"requests is required for API loading: {exc}") from exc

        if not hasattr(requests, "request"):
            raise RagError("requests.request is unavailable for API loading")

        def send_request(
            method: str,
            url: str,
            headers: object | None,
            params: object | None,
            payload_json: object | None,
            data: object | None,
        ) -> object:
            """Execute one API request using the optional request payload fields."""
            return requests.request(
                method,
                url,
                timeout=self.timeout_seconds,
                headers=cast(Any, headers),
                params=cast(Any, params),
                json=payload_json,
                data=cast(Any, data),
            )

        try:
            response = await asyncio.to_thread(
                send_request,
                request["method"],
                request["url"],
                headers=request["headers"],
                params=request["params"],
                payload_json=request["json"],
                data=request["data"],
            )
            typed_response = cast(_ResponseProtocol, response)
            typed_response.raise_for_status()
        except Exception as exc:
            raise RagError(f"Failed to call API {request['url']}: {exc}") from exc

        payload_text = self._response_to_text(typed_response)
        document = self.parser.parse(payload_text, source=request["url"], doc_id="api-1")
        return [document]

    def _normalize_request(self, source: object) -> _RequestConfig:
        """Normalize source input into request details."""
        if isinstance(source, str):
            return {
                "method": self.default_method,
                "url": source,
                "headers": None,
                "params": None,
                "json": None,
                "data": None,
            }

        if isinstance(source, dict):
            url = source.get("url")
            if not isinstance(url, str) or not url.strip():
                raise RagError("APILoader config must include non-empty 'url'")

            method = source.get("method", self.default_method)
            if not isinstance(method, str) or not method.strip():
                raise RagError("APILoader 'method' must be a non-empty string")

            return {
                "method": method.upper(),
                "url": url.strip(),
                "headers": source.get("headers"),
                "params": source.get("params"),
                "json": source.get("json"),
                "data": source.get("data"),
            }

        raise RagError("APILoader source must be a URL string or request config dictionary")

    def _response_to_text(self, response: object) -> str:
        """Convert API response object into textual content for parsing."""
        if hasattr(response, "json"):
            try:
                payload = response.json()
                if payload is not None:
                    return json.dumps(payload, ensure_ascii=True)
            except Exception:
                pass

        text = getattr(response, "text", "")
        if not isinstance(text, str) or not text.strip():
            raise RagError("API response contained no textual payload")
        return text

