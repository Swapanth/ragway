"""CLI command for querying a chosen RAG pipeline."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Callable
from pathlib import Path

import click

from ragway.core.pipeline_builder import PipelineBuilder
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
    return getattr(module, "run")


@click.command("query")
@click.argument("question", type=str)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("rag_config.yaml"),
    show_default=True,
)
@click.option("--pipeline", type=click.Choice(list(PIPELINE_MODULES.keys())), default="naive", show_default=True)
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

    runner = _resolve_runner(pipeline)
    llm = get_llm(llm_name)
    result = asyncio.run(runner(question, llm=llm))

    if isinstance(result, dict):
        answer = str(result.get("answer", ""))
        click.echo(answer)
        click.echo(f"metadata: {result}")
        return

    click.echo(str(result))

