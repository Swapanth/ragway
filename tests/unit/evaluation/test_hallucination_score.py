from __future__ import annotations

from ragway.evaluation.hallucination_score import HallucinationScore
from ragway.schema.node import Node


def test_hallucination_score_is_low_when_supported() -> None:
    """HallucinationScore should be lower for supported answers."""
    evaluator = HallucinationScore()
    context = [Node(node_id="n1", doc_id="d1", content="Paris is in France")]

    score = evaluator.evaluate("Where is Paris?", "Paris is in France", context)

    assert 0.0 <= score <= 1.0
    assert score < 0.5


def test_hallucination_score_is_high_when_unsupported() -> None:
    """HallucinationScore should be higher for unsupported answers."""
    evaluator = HallucinationScore()
    context = [Node(node_id="n1", doc_id="d1", content="Paris is in France")]

    score = evaluator.evaluate("Where is Paris?", "Mars has oceans", context)

    assert score > 0.5

