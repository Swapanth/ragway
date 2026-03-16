from __future__ import annotations

from ragway.components.hallucination_detector import HallucinationDetector
from ragway.schema.node import Node


def test_hallucination_detector_scores_grounded_answer_higher() -> None:
    """Grounded answers should receive higher support score than unrelated text."""
    detector = HallucinationDetector()
    nodes = [Node(node_id="n1", doc_id="d1", content="Paris is in France")]

    grounded = detector.score("Paris is in France", nodes)
    ungrounded = detector.score("Mars has oceans", nodes)

    assert grounded > ungrounded


def test_hallucination_detector_flags_low_support() -> None:
    """Detector should flag answers with low context support."""
    detector = HallucinationDetector()
    nodes = [Node(node_id="n1", doc_id="d1", content="Paris is in France")]

    assert detector.is_hallucinated("Mars has oceans", nodes, threshold=0.4)

