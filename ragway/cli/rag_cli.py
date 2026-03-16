"""Main CLI entry point for rag-toolkit commands."""

from __future__ import annotations

import click

from ragway.cli.commands.benchmark import benchmark_command
from ragway.cli.commands.evaluate import evaluate_command
from ragway.cli.commands.ingest import ingest_command
from ragway.cli.commands.query import query_command


@click.group(name="rag")
def cli() -> None:
    """RAG toolkit command line interface."""


cli.add_command(ingest_command)
cli.add_command(query_command)
cli.add_command(evaluate_command)
cli.add_command(benchmark_command)


def main() -> None:
    """Console script entry point."""
    cli()


if __name__ == "__main__":
    main()

