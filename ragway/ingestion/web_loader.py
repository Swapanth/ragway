"""Asynchronous web loader that fetches URL content into documents."""

from __future__ import annotations

import asyncio

from ragway.exceptions import RagError
from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.parsing.html_parser import HTMLParser
from ragway.schema.document import Document


class WebLoader(BaseLoader):
    """Fetch web pages and parse each response into a document."""

    def __init__(self, parser: BaseDocumentParser | None = None, timeout_seconds: int = 15) -> None:
        """Initialize loader with parser and request timeout."""
        super().__init__(parser or HTMLParser())
        self.timeout_seconds = timeout_seconds

    async def load(self, source: object) -> list[Document]:
        """Load one or many URLs and return parsed documents."""
        urls = self._normalize_urls(source)

        try:
            import requests
        except ImportError as exc:
            raise RagError(f"requests is required for web loading: {exc}") from exc

        if not hasattr(requests, "get"):
            raise RagError("requests.get is unavailable for web loading")

        documents: list[Document] = []
        for index, url in enumerate(urls):
            try:
                response = await asyncio.to_thread(requests.get, url, timeout=self.timeout_seconds)
                response.raise_for_status()
            except Exception as exc:
                raise RagError(f"Failed to fetch URL {url}: {exc}") from exc

            document = self.parser.parse(response.text, source=url, doc_id=f"web-{index + 1}")
            documents.append(document)

        return documents

    def _normalize_urls(self, source: object) -> list[str]:
        """Normalize URL input into a non-empty list of strings."""
        if isinstance(source, str):
            urls = [source]
        elif isinstance(source, list) and all(isinstance(item, str) for item in source):
            urls = source
        else:
            raise RagError("WebLoader source must be a URL string or list of URL strings")

        cleaned_urls = [url.strip() for url in urls if url.strip()]
        if not cleaned_urls:
            raise RagError("WebLoader source must contain at least one non-empty URL")
        return cleaned_urls

