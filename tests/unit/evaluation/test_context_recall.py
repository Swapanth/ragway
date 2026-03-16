from __future__ import annotations

from ragway.evaluation.context_recall import ContextRecall
from ragway.schema.node import Node


def test_context_recall_measures_answer_coverage() -> None:
    """ContextRecall should score coverage of answer terms by context terms."""
    evaluator = ContextRecall()
    context = [Node(node_id="n1", doc_id="d1", content="Paris is in France")]

    score = evaluator.evaluate("Where is Paris?", "Paris is in France", context)

    assert score == 1.0

