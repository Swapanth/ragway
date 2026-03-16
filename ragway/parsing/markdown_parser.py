"""Markdown parser implementation that converts markdown to plain text."""

from __future__ import annotations

from ragway.exceptions import RagError
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class MarkdownParser(BaseDocumentParser):
    """Parse markdown text into a document with normalized plain content."""

    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        """Parse markdown content and return a document."""
        markdown_text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content
        plain_text = self._extract_text(markdown_text)
        return self._build_document(plain_text, source=source, doc_id=doc_id)

    def _extract_text(self, markdown_text: str) -> str:
        """Extract textual content from markdown tokens."""
        try:
            from markdown_it import MarkdownIt
        except ImportError as exc:
            raise RagError(f"markdown-it-py is required for markdown parsing: {exc}") from exc

        parser = MarkdownIt("commonmark")
        tokens = parser.parse(markdown_text)

        parts: list[str] = []

        def collect(token_list: list[object]) -> None:
            for token in token_list:
                token_type = getattr(token, "type", "")
                token_content = getattr(token, "content", "")
                if token_type in {"text", "code_inline", "code_block", "fence", "html_block", "html_inline"}:
                    stripped = token_content.strip()
                    if stripped:
                        parts.append(stripped)

                children = getattr(token, "children", None)
                if children:
                    collect(list(children))

        collect(list(tokens))

        combined = "\n".join(parts).strip()
        if not combined:
            raise RagError("Failed to parse markdown content: no textual content found")
        return combined

