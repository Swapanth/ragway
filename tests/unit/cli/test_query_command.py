from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ragway.cli.commands import query as query_module
from ragway.cli.commands.query import query_command


class _FakeRAG:
    def __init__(self, pipeline: str, llm: str) -> None:
        self.pipeline = pipeline
        self.llm = llm

    async def query(self, question: str) -> str:
        return f"mock answer:{self.pipeline}:{self.llm}:{question}"


def test_query_command_uses_mocked_pipeline_runner(monkeypatch) -> None:
    """Query command should call RAG.query when config file does not exist."""
    monkeypatch.setattr(query_module, "RAG", _FakeRAG)

    runner = CliRunner()
    result = runner.invoke(query_command, ["what is rag?", "--config", "does-not-exist.yaml", "--pipeline", "naive"])

    assert result.exit_code == 0
    assert "mock answer" in result.output


def test_query_command_uses_pipeline_builder_with_dict_payload(monkeypatch, tmp_path: Path) -> None:
    """When config exists, query command should route through PipelineBuilder and print metadata."""

    class _FakeBuilder:
        def __init__(self, config_path: str) -> None:
            self.config_path = config_path

        async def run(self, question: str) -> object:
            return {"answer": f"ans:{question}", "score": 0.8}

    config_path = tmp_path / "rag.yaml"
    config_path.write_text("pipeline: naive\n", encoding="utf-8")
    monkeypatch.setattr(query_module, "PipelineBuilder", _FakeBuilder)

    runner = CliRunner()
    result = runner.invoke(query_command, ["what is rag?", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "ans:what is rag?" in result.output
    assert "metadata:" in result.output


def test_query_command_uses_pipeline_builder_with_string_payload(monkeypatch, tmp_path: Path) -> None:
    """When config exists and pipeline returns a string, command should print that value."""

    class _FakeBuilder:
        def __init__(self, config_path: str) -> None:
            self.config_path = config_path

        async def run(self, question: str) -> object:
            return f"plain:{question}"

    config_path = tmp_path / "rag.yaml"
    config_path.write_text("pipeline: naive\n", encoding="utf-8")
    monkeypatch.setattr(query_module, "PipelineBuilder", _FakeBuilder)

    runner = CliRunner()
    result = runner.invoke(query_command, ["what is rag?", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "plain:what is rag?" in result.output


@pytest.mark.parametrize("llm_name", ["mistral", "groq"])
def test_query_command_accepts_new_llm_options(monkeypatch, llm_name: str) -> None:
    """Query command should accept newly added LLM switch values."""
    monkeypatch.setattr(query_module, "RAG", _FakeRAG)

    runner = CliRunner()
    result = runner.invoke(
        query_command,
        ["what is rag?", "--config", "does-not-exist.yaml", "--pipeline", "naive", "--llm", llm_name],
    )

    assert result.exit_code == 0
    assert "mock answer" in result.output
