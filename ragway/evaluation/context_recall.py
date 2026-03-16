"""Context recall evaluator for measuring coverage of answer content by context."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragway.schema.node import Node


_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class ContextRecall:
    """Scores how much answer content is recoverable from retrieved context."""

    def evaluate(self, question: str, answer: str, context: list[Node]) -> float:
        """Return context recall score in the inclusive range [0.0, 1.0]."""
        del question
        answer_terms = self._terms(answer)
        if not answer_terms:
            return 0.0

        context_terms: set[str] = set()
        for node in context:
            context_terms.update(self._terms(node.content))
        if not context_terms:
            return 0.0

        covered = len(answer_terms.intersection(context_terms))
        return self._clamp(covered / len(answer_terms))

    def _terms(self, text: str) -> set[str]:
        """Tokenize text into lowercase lexical terms."""
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}

    def _clamp(self, value: float) -> float:
        """Clamp a numeric score into the inclusive range [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

