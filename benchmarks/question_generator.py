"""Generate evaluation question-answer pairs from source documents using an LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ragway.exceptions import ValidationError
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.validators import validate_positive_int

_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(slots=True)
class QuestionGenerator:
    """Generate question and gold-answer examples from document text."""

    llm: LLMProtocol
    pairs_per_document: int = 1

    def __post_init__(self) -> None:
        """Validate generation configuration."""
        self.pairs_per_document = validate_positive_int(self.pairs_per_document, "pairs_per_document")

    async def generate(self, documents: list[str]) -> list[dict[str, object]]:
        """Generate normalized evaluation rows from input documents."""
        generated_rows: list[dict[str, object]] = []

        for document in documents:
            prompt = self._build_prompt(document)
            response = await self.llm.generate(prompt)
            pairs = self._parse_pairs(response)

            for pair in pairs:
                generated_rows.append(
                    {
                        "question": pair["question"],
                        "gold_answer": pair["gold_answer"],
                        "context_docs": [document],
                    }
                )

        return generated_rows

    def _build_prompt(self, document: str) -> str:
        """Build a prompt that asks the LLM for synthetic QA pairs in JSON format."""
        return (
            "Generate evaluation examples for a RAG benchmark. "
            f"Create exactly {self.pairs_per_document} question-answer pairs grounded only in the document. "
            "Return strict JSON as a list of objects with keys question and gold_answer.\n\n"
            f"Document:\n{document}"
        )

    def _parse_pairs(self, response: str) -> list[dict[str, str]]:
        """Parse and validate LLM output into normalized question-answer pairs."""
        payload = self._parse_json_payload(response)
        if isinstance(payload, dict):
            candidate_pairs = [payload]
        elif isinstance(payload, list):
            candidate_pairs = payload
        else:
            raise ValidationError("LLM output must be a JSON object or list")

        valid_pairs: list[dict[str, str]] = []
        for item in candidate_pairs:
            if not isinstance(item, dict):
                raise ValidationError("Each generated pair must be a JSON object")

            question = str(item.get("question", "")).strip()
            gold_answer = str(item.get("gold_answer", "")).strip()
            if not question or not gold_answer:
                raise ValidationError("Generated pairs must include non-empty question and gold_answer")

            valid_pairs.append({"question": question, "gold_answer": gold_answer})

        if not valid_pairs:
            raise ValidationError("LLM output produced no valid question-answer pairs")

        return valid_pairs

    def _parse_json_payload(self, response: str) -> object:
        """Parse JSON payload from raw LLM response, including fenced JSON blocks."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        match = _JSON_BLOCK_PATTERN.search(response)
        if match is None:
            raise ValidationError("LLM output is not valid JSON")

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"LLM output contains invalid JSON: {exc}") from exc

