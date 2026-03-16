"""PDF parser implementation that extracts text into documents."""

from __future__ import annotations

from io import BytesIO

from ragway.exceptions import RagError
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class PDFParser(BaseDocumentParser):
    """Parse PDF content bytes into a Document instance."""

    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        """Parse PDF data and return a document with extracted text."""
        pdf_bytes = raw_content if isinstance(raw_content, bytes) else raw_content.encode("utf-8")
        parsed_text = self._extract_text(pdf_bytes)
        return self._build_document(parsed_text, source=source, doc_id=doc_id)

    def _extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text with available PDF backends in priority order."""
        errors: list[str] = []

        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(pdf_bytes))
            text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
            if text:
                return text
            errors.append("pypdf extracted no text")
        except ImportError:
            errors.append("pypdf is unavailable")
        except Exception as exc:  # pragma: no cover - defensive integration path
            errors.append(f"pypdf failed: {exc}")

        try:
            from pdfminer.high_level import extract_text

            text = (extract_text(BytesIO(pdf_bytes)) or "").strip()
            if text:
                return text
            errors.append("pdfminer extracted no text")
        except ImportError:
            errors.append("pdfminer is unavailable")
        except Exception as exc:  # pragma: no cover - defensive integration path
            errors.append(f"pdfminer failed: {exc}")

        detail = "; ".join(errors)
        raise RagError(f"Failed to parse PDF content: {detail}")

