"""BM25 keyword retriever over in-memory nodes."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ragway.schema.node import Node
from ragway.validators import validate_positive_int

from ragway.retrieval.base_retriever import BaseRetriever


_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class BM25Retriever(BaseRetriever):
    """Retrieve nodes using BM25 ranking over tokenized content."""

    nodes: list[Node]
    k1: float = 1.5
    b: float = 0.75
    _doc_freq: dict[str, int] = field(default_factory=dict, init=False)
    _avg_doc_len: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Precompute corpus statistics used by BM25 scoring."""
        doc_lengths: list[int] = []
        for node in self.nodes:
            tokens = set(self._tokenize(node.content))
            doc_lengths.append(len(self._tokenize(node.content)))
            for token in tokens:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._avg_doc_len = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0

    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        """Rank nodes by BM25 score for query terms."""
        top_k = validate_positive_int(top_k, "top_k")
        if not self.nodes:
            return []

        query_terms = self._tokenize(query)
        scored: list[tuple[float, Node]] = []
        for node in self.nodes:
            score = self._score_node(node, query_terms)
            scored.append((score, node))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for score, node in scored[:top_k] if score > 0.0]

    def _score_node(self, node: Node, query_terms: list[str]) -> float:
        """Compute BM25 score of one node for provided query terms."""
        terms = self._tokenize(node.content)
        term_counts: dict[str, int] = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        score = 0.0
        doc_len = len(terms)
        corpus_size = max(len(self.nodes), 1)
        avg_len = self._avg_doc_len if self._avg_doc_len > 0 else 1.0

        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_len))
            score += idf * (numerator / denominator)

        return score

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize input text into lower-cased alphanumeric terms."""
        return [token.lower() for token in _TOKEN_PATTERN.findall(text)]

