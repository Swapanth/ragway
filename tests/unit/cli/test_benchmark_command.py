from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from ragway.cli.commands import benchmark as benchmark_module
from ragway.cli.commands.benchmark import benchmark_command


async def _fake_run(question: str, llm=None) -> str:
    del llm
    return f"answer for {question}"


def test_benchmark_command_uses_mocked_pipeline_calls(monkeypatch, tmp_path: Path) -> None:
    """Benchmark command should run mocked pipelines and print score table."""
    dataset = tmp_path / "eval.json"
    payload = [{"question": "q1", "gold_answer": "g1", "context": ["ctx"]}]
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    fake_module = types.SimpleNamespace(run=_fake_run)
    monkeypatch.setattr(benchmark_module.importlib, "import_module", lambda _: fake_module)

    runner = CliRunner()
    result = runner.invoke(benchmark_command, ["--pipeline", "all", "--dataset", str(dataset)])

    assert result.exit_code == 0
    assert "pipeline\toverall_score" in result.output


@pytest.mark.parametrize("llm_name", ["mistral", "groq"])
def test_benchmark_command_accepts_new_llm_options(monkeypatch, tmp_path: Path, llm_name: str) -> None:
    """Benchmark command should accept newly added LLM switch values."""
    dataset = tmp_path / "eval.json"
    payload = [{"question": "q1", "gold_answer": "g1", "context": ["ctx"]}]
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    fake_module = types.SimpleNamespace(run=_fake_run)
    monkeypatch.setattr(benchmark_module.importlib, "import_module", lambda _: fake_module)

    runner = CliRunner()
    result = runner.invoke(
        benchmark_command,
        ["--pipeline", "all", "--dataset", str(dataset), "--llm", llm_name],
    )

    assert result.exit_code == 0
    assert "pipeline\toverall_score" in result.output
