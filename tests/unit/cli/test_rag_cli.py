from __future__ import annotations

from click.testing import CliRunner

from ragway.cli.rag_cli import cli


def test_rag_cli_root_help() -> None:
    """CLI root should be invokable and show command list."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "query" in result.output
    assert "evaluate" in result.output
    assert "benchmark" in result.output
