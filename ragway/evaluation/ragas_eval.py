"""Batch evaluator that runs all RAG metrics and summarizes average scores."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragway.evaluation.answer_accuracy import AnswerAccuracy
from ragway.evaluation.context_precision import ContextPrecision
from ragway.evaluation.context_recall import ContextRecall
from ragway.evaluation.faithfulness import FaithfulnessEval
from ragway.evaluation.hallucination_score import HallucinationScore
from ragway.evaluation.latency_eval import LatencyEval
from ragway.schema.node import Node


@dataclass(slots=True)
class RagasEval:
    """Runs evaluation metrics over a dataset and returns summary scores."""

    faithfulness_eval: FaithfulnessEval = field(default_factory=FaithfulnessEval)
    context_recall_eval: ContextRecall = field(default_factory=ContextRecall)
    context_precision_eval: ContextPrecision = field(default_factory=ContextPrecision)
    hallucination_eval: HallucinationScore = field(default_factory=HallucinationScore)
    latency_eval: LatencyEval = field(default_factory=LatencyEval)

    def run(self, dataset: list[dict[str, object]], pipeline_name: str) -> dict[str, float]:
        """Evaluate a dataset and return mean score per metric."""
        del pipeline_name
        if not dataset:
            return {
                "faithfulness": 0.0,
                "answer_accuracy": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
                "hallucination_score": 0.0,
                "latency_score": 0.0,
                "overall_score": 0.0,
            }

        faithfulness_scores: list[float] = []
        accuracy_scores: list[float] = []
        recall_scores: list[float] = []
        precision_scores: list[float] = []
        hallucination_scores: list[float] = []
        latency_scores: list[float] = []

        for row in dataset:
            question = str(row.get("question", ""))
            answer = str(row.get("answer", ""))
            gold_answer = str(row.get("gold_answer", ""))
            context = self._normalize_context(row.get("context", []))

            faithfulness_scores.append(self.faithfulness_eval.evaluate(question, answer, context))
            accuracy_scores.append(AnswerAccuracy(gold_answer=gold_answer).evaluate(question, answer, context))
            recall_scores.append(self.context_recall_eval.evaluate(question, answer, context))
            precision_scores.append(self.context_precision_eval.evaluate(question, answer, context))
            hallucination_scores.append(self.hallucination_eval.evaluate(question, answer, context))
            latency_scores.append(self.latency_eval.evaluate(question, answer, context))

        summary = {
            "faithfulness": self._mean(faithfulness_scores),
            "answer_accuracy": self._mean(accuracy_scores),
            "context_recall": self._mean(recall_scores),
            "context_precision": self._mean(precision_scores),
            "hallucination_score": self._mean(hallucination_scores),
            "latency_score": self._mean(latency_scores),
        }

        summary["overall_score"] = self._mean(
            [
                summary["faithfulness"],
                summary["answer_accuracy"],
                summary["context_recall"],
                summary["context_precision"],
                1.0 - summary["hallucination_score"],
                summary["latency_score"],
            ]
        )
        return summary

    def _normalize_context(self, raw_context: object) -> list[Node]:
        """Normalize dataset context into a list of Node objects."""
        if not isinstance(raw_context, list):
            return []

        nodes: list[Node] = []
        for index, item in enumerate(raw_context):
            if isinstance(item, Node):
                nodes.append(item)
                continue
            if isinstance(item, str):
                nodes.append(Node(node_id=f"ctx-{index}", doc_id="ctx", content=item))
                continue
            if isinstance(item, dict):
                content = str(item.get("content", ""))
                if content:
                    nodes.append(
                        Node(
                            node_id=str(item.get("node_id", f"ctx-{index}")),
                            doc_id=str(item.get("doc_id", "ctx")),
                            content=content,
                        )
                    )
        return nodes

    def _mean(self, values: list[float]) -> float:
        """Return arithmetic mean for a list of floats."""
        if not values:
            return 0.0
        return sum(values) / len(values)

