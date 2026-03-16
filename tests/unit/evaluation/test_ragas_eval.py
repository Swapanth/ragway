from __future__ import annotations

from unittest.mock import MagicMock

from ragway.evaluation.faithfulness import FaithfulnessEval
from ragway.evaluation.ragas_eval import RagasEval


def test_ragas_eval_returns_metric_summary() -> None:
    """RagasEval should return averaged metric summary dict."""
    faithfulness = FaithfulnessEval(judge=MagicMock(return_value="Yes"))
    evaluator = RagasEval(faithfulness_eval=faithfulness)

    dataset = [
        {
            "question": "Where is Paris?",
            "answer": "Paris is in France",
            "gold_answer": "Paris is in France",
            "context": [{"content": "Paris is in France", "node_id": "n1", "doc_id": "d1"}],
        },
        {
            "question": "Where is Berlin?",
            "answer": "Berlin is in Germany",
            "gold_answer": "Berlin is in Germany",
            "context": ["Berlin is in Germany"],
        },
    ]

    summary = evaluator.run(dataset=dataset, pipeline_name="naive")

    expected_keys = {
        "faithfulness",
        "answer_accuracy",
        "context_recall",
        "context_precision",
        "hallucination_score",
        "latency_score",
        "overall_score",
    }
    assert set(summary.keys()) == expected_keys
    assert all(0.0 <= value <= 1.0 for value in summary.values())

