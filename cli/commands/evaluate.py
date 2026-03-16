"""CLI command for evaluating one pipeline with dataset metrics."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import click

from ragway.evaluation.ragas_eval import RagasEval


PIPELINE_MODULES: dict[str, str] = {
    "naive": "pipelines.naive_rag_pipeline",
    "hybrid": "pipelines.hybrid_rag_pipeline",
    "self": "pipelines.self_rag_pipeline",
    "long_context": "pipelines.long_context_rag_pipeline",
    "agentic": "pipelines.agentic_rag_pipeline",
}


def _resolve_runner(pipeline: str) -> Callable[[str], Awaitable[object]]:
    """Resolve the async run function from a pipeline module."""
    module = importlib.import_module(PIPELINE_MODULES[pipeline])
    return getattr(module, "run")


async def _run_eval(dataset: list[dict], pipeline: str) -> dict[str, float]:
    """Run selected pipeline across dataset and compute evaluation summary."""
    runner = _resolve_runner(pipeline)
    rows: list[dict] = []

    for item in dataset:
        question = str(item.get("question", ""))
        output = await runner(question)
        answer = output.get("answer", "") if isinstance(output, dict) else str(output)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "context": item.get("context", []),
                "gold_answer": item.get("gold_answer", ""),
            }
        )

    return RagasEval().run(dataset=rows, pipeline_name=pipeline)


@click.command("evaluate")
@click.option("--dataset", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--pipeline", type=click.Choice(list(PIPELINE_MODULES.keys())), default="naive", show_default=True)
def evaluate_command(dataset: Path, pipeline: str) -> None:
    """Run ragas-like evaluation on a dataset for one selected pipeline."""
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    summary = asyncio.run(_run_eval(payload, pipeline))

    click.echo(f"Pipeline: {pipeline}")
    for name, value in summary.items():
        click.echo(f"{name}: {value:.4f}")

