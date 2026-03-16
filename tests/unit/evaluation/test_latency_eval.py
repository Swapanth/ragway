from __future__ import annotations

import time

from ragway.evaluation.latency_eval import LatencyEval


def test_latency_eval_measures_stage_latency_and_scores() -> None:
    """LatencyEval should record stage latency and produce bounded score."""
    evaluator = LatencyEval()

    def stage() -> int:
        time.sleep(0.01)
        return 1

    result = evaluator.measure("retrieve", stage)
    score = evaluator.evaluate("q", "a", [])

    assert result == 1
    assert "retrieve" in evaluator.stage_latencies
    assert 0.0 <= score <= 1.0

