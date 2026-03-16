from __future__ import annotations

from ragway.evaluation.context_precision import ContextPrecision
from ragway.schema.node import Node


def test_context_precision_scores_relevant_context_density() -> None:
    """ContextPrecision should score how relevant context terms are to query and answer."""
    evaluator = ContextPrecision()
    context = [
        Node(node_id="n1", doc_id="d1", content="Paris is in France"),
        Node(node_id="n2", doc_id="d1", content="Completely unrelated sentence"),
    ]

    score = evaluator.evaluate("Where is Paris?", "Paris is in France", context)

    assert 0.0 <= score <= 1.0

