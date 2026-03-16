"""CLI command for benchmarking one or all pipelines on an eval dataset."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import click

from ragway.evaluation.ragas_eval import RagasEval
from ragway.generation.llm_factory import get_llm
from ragway.interfaces.llm_protocol import LLMProtocol


PIPELINE_MODULES: dict[str, str] = {
    "naive": "pipelines.naive_rag_pipeline",
    "hybrid": "pipelines.hybrid_rag_pipeline",
    "self": "pipelines.self_rag_pipeline",
    "long_context": "pipelines.long_context_rag_pipeline",
    "agentic": "pipelines.agentic_rag_pipeline",
}


def _resolve_runner(pipeline: str) -> Callable[[str, LLMProtocol | None], Awaitable[object]]:
    """Resolve the async run function from a pipeline module."""
    module = importlib.import_module(PIPELINE_MODULES[pipeline])
    return cast(Callable[[str, LLMProtocol | None], Awaitable[object]], getattr(module, "run"))


async def _benchmark(dataset: list[dict[str, object]], pipelines: list[str], llm_name: str) -> dict[str, float]:
    """Run selected pipelines and return overall score per pipeline."""
    results: dict[str, float] = {}

    for pipeline in pipelines:
        runner = _resolve_runner(pipeline)
        llm = get_llm(llm_name)
        rows: list[dict[str, object]] = []
        for item in dataset:
            question = str(item.get("question", ""))
            output = await runner(question, llm)
            answer = output.get("answer", "") if isinstance(output, dict) else str(output)
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "context": item.get("context", []),
                    "gold_answer": item.get("gold_answer", ""),
                }
            )

        summary = RagasEval().run(dataset=rows, pipeline_name=pipeline)
        results[pipeline] = summary.get("overall_score", 0.0)

    return results


@click.command("benchmark")
@click.option("--pipeline", type=click.Choice(["all", *PIPELINE_MODULES.keys()]), default="all", show_default=True)
@click.option(
    "--dataset",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("./data/eval.json"),
    show_default=True,
)
@click.option(
    "--llm",
    "llm_name",
    type=click.Choice(["anthropic", "openai", "llama", "local", "mistral", "groq"]),
    default="anthropic",
    show_default=True,
)
def benchmark_command(pipeline: str, dataset: Path, llm_name: str) -> None:
    """Benchmark one or all pipelines on an evaluation dataset."""
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise click.ClickException("Dataset must be a JSON list")
    selected = list(PIPELINE_MODULES.keys()) if pipeline == "all" else [pipeline]
    typed_payload = [cast(dict[str, object], item) for item in payload if isinstance(item, dict)]
    scores = asyncio.run(_benchmark(typed_payload, selected, llm_name=llm_name))

    click.echo("pipeline\toverall_score")
    for name, score in scores.items():
        click.echo(f"{name}\t{score:.4f}")

