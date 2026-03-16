from __future__ import annotations

from ragway.evaluation.answer_accuracy import AnswerAccuracy
from ragway.schema.node import Node


def test_answer_accuracy_scores_overlap_with_gold() -> None:
    """AnswerAccuracy should score higher when answer matches gold answer terms."""
    evaluator = AnswerAccuracy(gold_answer="Paris is in France")
    context = [Node(node_id="n1", doc_id="d1", content="unused context")]

    score = evaluator.evaluate("Where is Paris?", "Paris is in France", context)

    assert 0.0 <= score <= 1.0
    assert score == 1.0

