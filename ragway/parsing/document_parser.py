"""Abstract parser contract for converting raw content into documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256

from ragway.schema.document import Document
from ragway.schema.metadata import Metadata
from ragway.validators import validate_non_empty_text


class BaseDocumentParser(ABC):
    """Base parser contract for producing Document instances."""

    @abstractmethod
    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        """Parse raw content into a typed document."""

    def _build_document(
        self,
        content: str,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        """Build a validated document from parsed content and source metadata."""
        normalized_content = validate_non_empty_text(content, "content")
        normalized_source = source.strip() if source is not None else None
        document_id = doc_id if doc_id is not None else self._make_doc_id(normalized_content, normalized_source)

        return Document(
            doc_id=document_id,
            content=normalized_content,
            metadata=Metadata(source=normalized_source),
        )

    @staticmethod
    def _make_doc_id(content: str, source: str | None) -> str:
        """Create a deterministic identifier from source and content."""
        basis = f"{source or ''}\n{content}".encode("utf-8")
        digest = sha256(basis).hexdigest()[:16]
        return f"doc-{digest}"

