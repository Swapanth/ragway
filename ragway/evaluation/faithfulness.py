"""Faithfulness evaluator estimating whether answer claims are grounded in context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ragway.schema.node import Node


_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class FaithfulnessEval:
    """Scores answer faithfulness to retrieved context in the range [0.0, 1.0]."""

    judge: Callable[[str], str] | None = None

    def evaluate(self, question: str, answer: str, context: list[Node]) -> float:
        """Return a faithfulness score for answer groundedness against context."""
        del question
        if self.judge is not None:
            prompt = self._build_judge_prompt(answer=answer, context=context)
            verdict = self.judge(prompt).strip().lower()
            if verdict.startswith("yes"):
                return 1.0
            if verdict.startswith("no"):
                return 0.0

        answer_terms = self._terms(answer)
        if not answer_terms:
            return 0.0
        context_terms: set[str] = set()
        for node in context:
            context_terms.update(self._terms(node.content))
        if not context_terms:
            return 0.0

        supported = len(answer_terms.intersection(context_terms))
        return self._clamp(supported / len(answer_terms))

    def _build_judge_prompt(self, answer: str, context: list[Node]) -> str:
        """Build the LLM-as-judge prompt for faithfulness verdict."""
        context_text = "\n\n".join(node.content for node in context)
        return (
            "Is the answer fully supported by the context? Reply Yes or No.\n\n"
            f"Answer:\n{answer}\n\n"
            f"Context:\n{context_text}"
        )

    def _terms(self, text: str) -> set[str]:
        """Tokenize text into lowercase lexical terms."""
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}

    def _clamp(self, value: float) -> float:
        """Clamp a numeric score into the inclusive range [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

