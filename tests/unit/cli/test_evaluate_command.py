from __future__ import annotations

import json
import types
from pathlib import Path

from click.testing import CliRunner

from ragway.cli.commands import evaluate as evaluate_module
from ragway.cli.commands.evaluate import evaluate_command


async def _fake_run(question: str) -> str:
    return f"answer for {question}"


def test_evaluate_command_uses_mocked_pipeline_calls(monkeypatch, tmp_path: Path) -> None:
    """Evaluate command should run with mocked pipeline and print metrics."""
    dataset = tmp_path / "eval.json"
    payload = [{"question": "q1", "gold_answer": "g1", "context": ["ctx"]}]
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    fake_module = types.SimpleNamespace(run=_fake_run)
    monkeypatch.setattr(evaluate_module.importlib, "import_module", lambda _: fake_module)

    runner = CliRunner()
    result = runner.invoke(evaluate_command, ["--dataset", str(dataset), "--pipeline", "naive"])

    assert result.exit_code == 0
    assert "overall_score" in result.output
