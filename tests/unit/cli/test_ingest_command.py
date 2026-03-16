from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from ragway.cli.commands.ingest import ingest_command


def test_ingest_command_reports_counts(tmp_path: Path) -> None:
    """Ingest command should process source files and print summary counts."""
    source = tmp_path / "docs"
    source.mkdir()
    (source / "a.md").write_text("hello world", encoding="utf-8")
    (source / "b.txt").write_text("another file", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(ingest_command, ["--source", str(source), "--pipeline", "naive"])

    assert result.exit_code == 0
    assert "Ingested documents:" in result.output
    assert "Stored chunks:" in result.output
