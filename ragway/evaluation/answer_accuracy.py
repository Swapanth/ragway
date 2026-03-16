"""Answer accuracy evaluator comparing answer text to a gold answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragway.schema.node import Node


_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class AnswerAccuracy:
    """Scores factual answer overlap with a provided gold answer in [0.0, 1.0]."""

    gold_answer: str

    def evaluate(self, question: str, answer: str, context: list[Node]) -> float:
        """Return lexical overlap score between answer and gold answer."""
        del question, context
        gold_terms = self._terms(self.gold_answer)
        answer_terms = self._terms(answer)
        if not gold_terms:
            return 0.0
        matched = len(gold_terms.intersection(answer_terms))
        return self._clamp(matched / len(gold_terms))

    def _terms(self, text: str) -> set[str]:
        """Tokenize text into lowercase lexical terms."""
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}

    def _clamp(self, value: float) -> float:
        """Clamp a numeric score into the inclusive range [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

