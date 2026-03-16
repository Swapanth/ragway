"""Citation builder for mapping answer spans to source nodes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragway.schema.node import Node


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class CitationBuilder:
    """Build sentence-level citations by lexical overlap with source nodes."""

    def build(self, answer: str, nodes: list[Node]) -> dict[str, str]:
        """Map answer sentences to best matching source labels."""
        if not answer.strip() or not nodes:
            return {}

        citations: dict[str, str] = {}
        sentence_candidates = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(answer) if s.strip()]
        for sentence in sentence_candidates:
            sentence_terms = self._terms(sentence)
            best_score = -1.0
            best_label = "unknown"

            for node in nodes:
                node_terms = self._terms(node.content)
                score = self._overlap(sentence_terms, node_terms)
                if score > best_score:
                    best_score = score
                    best_label = node.metadata.source or node.node_id

            citations[sentence] = best_label

        return citations

    def _terms(self, text: str) -> set[str]:
        """Tokenize text into lowercase terms."""
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}

    def _overlap(self, left: set[str], right: set[str]) -> float:
        """Compute overlap ratio of left terms found in right terms."""
        if not left:
            return 0.0
        return len(left.intersection(right)) / len(left)

