"""Asynchronous loader for Notion page content via REST API."""

from __future__ import annotations

import os
import re

import aiohttp

from ragway.exceptions import RagError
from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.parsing.markdown_parser import MarkdownParser
from ragway.schema.document import Document


class NotionLoader(BaseLoader):
    """Load Notion page block text and parse into documents."""

    def __init__(self, parser: BaseDocumentParser | None = None, notion_version: str = "2022-06-28") -> None:
        """Initialize loader with optional parser and Notion API version."""
        super().__init__(parser or MarkdownParser())
        self.notion_version = notion_version

    async def load(self, source: object) -> list[Document]:
        """Load one Notion page by page id or page URL."""
        if not isinstance(source, str):
            raise RagError("NotionLoader source must be a page id or URL string")

        api_key = os.getenv("NOTION_API_KEY")
        if not api_key:
            raise RagError("NOTION_API_KEY environment variable is required")

        page_id = self._extract_page_id(source)
        text = await self._fetch_page_text(page_id, api_key)
        document = self.parser.parse(text, source=source, doc_id=f"notion-{page_id}")
        return [document]

    def _extract_page_id(self, source: str) -> str:
        """Extract a Notion page id from either id input or URL."""
        text = source.strip()
        if not text:
            raise RagError("Notion source must be non-empty")

        page_id_pattern = re.compile(r"([0-9a-fA-F]{32})")
        match = page_id_pattern.search(text.replace("-", ""))
        if match:
            return match.group(1).lower()

        raise RagError(f"Could not extract Notion page id from source: {source}")

    async def _fetch_page_text(self, page_id: str, api_key: str) -> str:
        """Fetch and flatten Notion block children text for a page."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": self.notion_version,
            "Content-Type": "application/json",
        }

        url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
        lines: list[str] = []
        next_cursor: str | None = None

        async with aiohttp.ClientSession(headers=headers) as session:
            while True:
                request_url = url if not next_cursor else f"{url}&start_cursor={next_cursor}"
                async with session.get(request_url) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RagError(f"Notion API request failed ({response.status}): {body}")
                    payload = await response.json()

                results = payload.get("results", [])
                if not isinstance(results, list):
                    raise RagError("Unexpected Notion API response format: 'results' must be a list")

                for block in results:
                    block_type = block.get("type")
                    if not isinstance(block_type, str):
                        continue
                    block_content = block.get(block_type, {})
                    rich_text = block_content.get("rich_text", []) if isinstance(block_content, dict) else []
                    if not isinstance(rich_text, list):
                        continue
                    fragment = "".join(
                        str(item.get("plain_text", ""))
                        for item in rich_text
                        if isinstance(item, dict)
                    ).strip()
                    if fragment:
                        lines.append(fragment)

                has_more = bool(payload.get("has_more", False))
                next_cursor_raw = payload.get("next_cursor")
                next_cursor = str(next_cursor_raw) if next_cursor_raw is not None else None
                if not has_more or not next_cursor:
                    break

        text = "\n".join(lines).strip()
        if not text:
            raise RagError(f"No textual content found for Notion page {page_id}")
        return text
