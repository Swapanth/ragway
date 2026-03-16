"""Local BGE-style reranker without external API dependencies."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragway.reranking.base_reranker import BaseReranker
from ragway.schema.node import Node


_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(slots=True)
class BGEReranker(BaseReranker):
    """Rerank nodes using lexical overlap with the query."""

    async def rerank(self, query: str, nodes: list[Node]) -> list[Node]:
        """Sort nodes by descending token-overlap score."""
        query_terms = self._tokenize(query)
        scored = [(self._score(query_terms, self._tokenize(node.content)), node) for node in nodes]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored]

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into lowercase term set."""
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}

    def _score(self, query_terms: set[str], node_terms: set[str]) -> float:
        """Compute overlap score between query and node terms."""
        if not query_terms:
            return 0.0
        overlap = len(query_terms.intersection(node_terms))
        return overlap / len(query_terms)

