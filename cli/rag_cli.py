"""Main CLI entry point for rag-toolkit commands."""

from __future__ import annotations

import click

from cli.commands.benchmark import benchmark_command
from cli.commands.evaluate import evaluate_command
from cli.commands.ingest import ingest_command
from cli.commands.query import query_command


@click.group(name="rag")
def cli() -> None:
    """RAG toolkit command line interface."""


cli.add_command(ingest_command)
cli.add_command(query_command)
cli.add_command(evaluate_command)
cli.add_command(benchmark_command)


if __name__ == "__main__":
    cli()
