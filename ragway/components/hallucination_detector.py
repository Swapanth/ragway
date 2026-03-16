"""Hallucination detector scoring answer grounding against retrieved context."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragway.schema.node import Node


_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class HallucinationDetector:
    """Estimate hallucination risk using lexical grounding ratio."""

    def score(self, answer: str, nodes: list[Node]) -> float:
        """Return grounding score in [0, 1], higher means better support."""
        answer_terms = self._terms(answer)
        if not answer_terms:
            return 0.0

        context_terms: set[str] = set()
        for node in nodes:
            context_terms.update(self._terms(node.content))

        if not context_terms:
            return 0.0

        supported = len(answer_terms.intersection(context_terms))
        return supported / len(answer_terms)

    def is_hallucinated(self, answer: str, nodes: list[Node], threshold: float = 0.5) -> bool:
        """Return True when grounding score falls below threshold."""
        return self.score(answer, nodes) < threshold

    def _terms(self, text: str) -> set[str]:
        """Tokenize text into lowercase lexical terms."""
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}

