"""HTML parser implementation that extracts readable text."""

from __future__ import annotations

from ragway.exceptions import RagError
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class HTMLParser(BaseDocumentParser):
    """Parse HTML content into plain text documents."""

    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        """Parse HTML and return a document with extracted text content."""
        html_text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content

        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RagError(f"beautifulsoup4 is required for HTML parsing: {exc}") from exc

        soup = BeautifulSoup(html_text, "html.parser")
        parsed_text = soup.get_text("\n", strip=True).strip()
        if not parsed_text:
            raise RagError("Failed to parse HTML content: no textual content found")

        return self._build_document(parsed_text, source=source, doc_id=doc_id)

