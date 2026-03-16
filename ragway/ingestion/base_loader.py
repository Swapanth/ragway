"""Abstract base loader contract for ingestion components."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class BaseLoader(ABC):
    """Base class for all data loaders that emit documents."""

    def __init__(self, parser: BaseDocumentParser) -> None:
        """Initialize loader with a parser used to build documents."""
        self.parser = parser

    @abstractmethod
    async def load(self, source: object) -> list[Document]:
        """Load one or more documents from a source descriptor."""

