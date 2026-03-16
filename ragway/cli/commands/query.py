"""CLI command for querying a chosen RAG pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from ragway.core.pipeline_builder import PipelineBuilder
from ragway.rag import RAG


PIPELINES = ["naive", "hybrid", "self", "long_context", "agentic"]


@click.command("query")
@click.argument("question", type=str)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("rag.yaml"),
    show_default=True,
)
@click.option("--pipeline", type=click.Choice(PIPELINES), default="naive", show_default=True)
@click.option(
    "--llm",
    "llm_name",
    type=click.Choice(["anthropic", "openai", "llama", "local", "mistral", "groq"]),
    default="anthropic",
    show_default=True,
)
def query_command(question: str, config_path: Path, pipeline: str, llm_name: str) -> None:
    """Run a question through the selected pipeline and print its answer."""
    if config_path.exists():
        builder = PipelineBuilder(str(config_path))
        result = asyncio.run(builder.run(question))
        if isinstance(result, dict):
            answer = str(result.get("answer", ""))
            click.echo(answer)
            click.echo(f"metadata: {result}")
            return

        click.echo(str(result))
        return

    rag = RAG(pipeline=pipeline, llm=llm_name)

    try:
        result = asyncio.run(rag.query(question))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if isinstance(result, dict):
        answer = str(result.get("answer", ""))
        click.echo(answer)
        click.echo(f"metadata: {result}")
        return

    click.echo(str(result))

