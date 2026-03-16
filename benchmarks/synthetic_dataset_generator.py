"""Generate and persist full synthetic RAG evaluation datasets."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.question_generator import QuestionGenerator
from ragway.interfaces.llm_protocol import LLMProtocol


@dataclass(slots=True)
class SyntheticDatasetGenerator:
    """Build synthetic benchmark datasets using an LLM-backed question generator."""

    llm: LLMProtocol
    pairs_per_document: int = 1
    question_generator: QuestionGenerator = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the internal question generator."""
        self.question_generator = QuestionGenerator(
            llm=self.llm,
            pairs_per_document=self.pairs_per_document,
        )

    async def generate(self, documents: list[str], output_path: str | Path) -> list[dict[str, object]]:
        """Generate a synthetic dataset and save it as JSON at output_path."""
        dataset = await self.question_generator.generate(documents)

        destination = Path(output_path)
        await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
        payload = json.dumps(dataset, indent=2, ensure_ascii=True)
        await asyncio.to_thread(destination.write_text, payload, encoding="utf-8")
        return dataset

