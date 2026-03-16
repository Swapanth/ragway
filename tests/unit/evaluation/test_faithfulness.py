from __future__ import annotations

from unittest.mock import MagicMock

from ragway.evaluation.faithfulness import FaithfulnessEval
from ragway.schema.node import Node


def test_faithfulness_uses_llm_judge_when_provided() -> None:
    """FaithfulnessEval should use mocked judge verdict when available."""
    judge = MagicMock(return_value="Yes")
    evaluator = FaithfulnessEval(judge=judge)
    context = [Node(node_id="n1", doc_id="d1", content="Paris is in France")]

    score = evaluator.evaluate("Where is Paris?", "Paris is in France", context)

    assert score == 1.0
    judge.assert_called_once()


def test_faithfulness_heuristic_fallback() -> None:
    """FaithfulnessEval should compute heuristic overlap without a judge."""
    evaluator = FaithfulnessEval()
    context = [Node(node_id="n1", doc_id="d1", content="Paris is in France")]

    score = evaluator.evaluate("Where is Paris?", "Paris is in France", context)

    assert 0.0 <= score <= 1.0
    assert score > 0.5

