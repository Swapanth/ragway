"""Context precision evaluator estimating relevance density of retrieved context."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragway.schema.node import Node


_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class ContextPrecision:
    """Scores how much retrieved context is relevant to question and answer terms."""

    def evaluate(self, question: str, answer: str, context: list[Node]) -> float:
        """Return context precision score in the inclusive range [0.0, 1.0]."""
        anchor_terms = self._terms(question).union(self._terms(answer))
        if not anchor_terms:
            return 0.0

        context_terms: set[str] = set()
        for node in context:
            context_terms.update(self._terms(node.content))
        if not context_terms:
            return 0.0

        relevant = len(context_terms.intersection(anchor_terms))
        return self._clamp(relevant / len(context_terms))

    def _terms(self, text: str) -> set[str]:
        """Tokenize text into lowercase lexical terms."""
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}

    def _clamp(self, value: float) -> float:
        """Clamp a numeric score into the inclusive range [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

