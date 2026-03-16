"""Semantic chunker using adjacent sentence embedding similarity."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ragway.chunking.base_chunker import BaseChunker
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.validators import validate_positive_int, validate_probability


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class SemanticChunker(BaseChunker):
    """Split text at semantic boundaries using cosine similarity thresholds."""

    similarity_threshold: float = 0.6
    max_tokens: int = 200

    def __post_init__(self) -> None:
        """Validate semantic chunker configuration."""
        self.similarity_threshold = validate_probability(
            self.similarity_threshold,
            "similarity_threshold",
        )
        self.max_tokens = validate_positive_int(self.max_tokens, "max_tokens")

    def chunk(self, document: Document) -> list[Node]:
        """Split document sentences where semantic similarity drops."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(document.content) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                Node(
                    node_id=f"{document.doc_id}:semantic:0",
                    doc_id=document.doc_id,
                    content=sentences[0],
                    metadata=document.metadata,
                    position=0,
                )
            ]

        chunks: list[str] = []
        current = sentences[0]
        previous_vector = self._sentence_vector(sentences[0])

        for sentence in sentences[1:]:
            sentence_vector = self._sentence_vector(sentence)
            similarity = self._cosine_similarity(previous_vector, sentence_vector)
            candidate = f"{current} {sentence}"
            if similarity >= self.similarity_threshold and len(candidate.split()) <= self.max_tokens:
                current = candidate
            else:
                chunks.append(current)
                current = sentence
            previous_vector = sentence_vector
        chunks.append(current)

        nodes: list[Node] = []
        for index, content in enumerate(chunks):
            nodes.append(
                Node(
                    node_id=f"{document.doc_id}:semantic:{index}",
                    doc_id=document.doc_id,
                    content=content,
                    metadata=document.metadata,
                    position=index,
                )
            )
        return nodes

    def _sentence_vector(self, sentence: str) -> dict[str, float]:
        """Build a sparse normalized bag-of-words vector for one sentence."""
        tokens = [token for token in re.findall(r"\w+", sentence.lower()) if token]
        counts: dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0

        norm = math.sqrt(sum(value * value for value in counts.values()))
        if norm == 0.0:
            return {}

        return {token: value / norm for token, value in counts.items()}

    def _cosine_similarity(self, left: dict[str, float], right: dict[str, float]) -> float:
        """Compute cosine similarity of two sparse normalized vectors."""
        if not left or not right:
            return 0.0

        smaller = left if len(left) <= len(right) else right
        larger = right if smaller is left else left
        return sum(value * larger.get(token, 0.0) for token, value in smaller.items())

