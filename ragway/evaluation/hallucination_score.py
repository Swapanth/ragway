"""Hallucination evaluator measuring unsupported claims in answer text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragway.schema.node import Node


_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class HallucinationScore:
    """Scores hallucination ratio in [0.0, 1.0], where higher means more hallucination."""

    def evaluate(self, question: str, answer: str, context: list[Node]) -> float:
        """Return hallucination ratio for answer terms unsupported by context."""
        del question
        answer_terms = self._terms(answer)
        if not answer_terms:
            return 0.0

        context_terms: set[str] = set()
        for node in context:
            context_terms.update(self._terms(node.content))
        if not context_terms:
            return 1.0

        unsupported = len(answer_terms.difference(context_terms))
        return self._clamp(unsupported / len(answer_terms))

    def _terms(self, text: str) -> set[str]:
        """Tokenize text into lowercase lexical terms."""
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}

    def _clamp(self, value: float) -> float:
        """Clamp a numeric score into the inclusive range [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

