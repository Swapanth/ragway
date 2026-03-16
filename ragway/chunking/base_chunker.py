"""Abstract contract for chunking a document into retrievable nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragway.schema.document import Document
from ragway.schema.node import Node


class BaseChunker(ABC):
    """Base class implemented by all concrete chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Node]:
        """Split a document into one or more nodes."""

