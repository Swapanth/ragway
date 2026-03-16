"""Recursive chunker that tries paragraph, sentence, then word-level splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragway.chunking.base_chunker import BaseChunker
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.validators import validate_positive_int


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class RecursiveChunker(BaseChunker):
    """Chunk by increasingly finer granularity until pieces fit size limits."""

    max_tokens: int = 200

    def __post_init__(self) -> None:
        """Validate the configured token limit."""
        self.max_tokens = validate_positive_int(self.max_tokens, "max_tokens")

    def chunk(self, document: Document) -> list[Node]:
        """Split a document recursively by paragraph, sentence, then words."""
        if not document.content.strip():
            return []

        pieces = self._split_piece(document.content.strip())
        nodes: list[Node] = []
        for index, piece in enumerate(pieces):
            nodes.append(
                Node(
                    node_id=f"{document.doc_id}:recursive:{index}",
                    doc_id=document.doc_id,
                    content=piece,
                    metadata=document.metadata,
                    position=index,
                )
            )
        return nodes

    def _split_piece(self, text: str) -> list[str]:
        """Recursively split one text piece to respect max token count."""
        token_count = len(text.split())
        if token_count <= self.max_tokens:
            return [text]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            output: list[str] = []
            for paragraph in paragraphs:
                output.extend(self._split_piece(paragraph))
            return output

        sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        if len(sentences) > 1:
            output = []
            current = ""
            for sentence in sentences:
                candidate = sentence if not current else f"{current} {sentence}"
                if len(candidate.split()) <= self.max_tokens:
                    current = candidate
                else:
                    if current:
                        output.extend(self._split_piece(current))
                    current = sentence
            if current:
                output.extend(self._split_piece(current))
            return output

        words = text.split()
        output = []
        for start in range(0, len(words), self.max_tokens):
            output.append(" ".join(words[start : start + self.max_tokens]))
        return output

