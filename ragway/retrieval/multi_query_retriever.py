"""Multi-query retriever that expands a query then fuses rankings."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.schema.node import Node
from ragway.validators import validate_positive_int

from ragway.retrieval.base_retriever import BaseRetriever


@dataclass(slots=True)
class MultiQueryRetriever(BaseRetriever):
    """Expand one query into variants and merge results with rank-based fusion."""

    retriever: RetrieverProtocol
    llm: LLMProtocol
    query_count: int = 3
    rrf_k: int = 60

    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        """Generate query variants, retrieve for each, and fuse into final ranking."""
        top_k = validate_positive_int(top_k, "top_k")
        query_count = validate_positive_int(self.query_count, "query_count")
        rrf_k = validate_positive_int(self.rrf_k, "rrf_k")

        prompt = (
            "Generate concise alternate search queries for the same intent. "
            "Return one query per line.\n"
            f"Original query: {query}"
        )
        llm_text = await self.llm.generate(prompt)
        variants = [line.strip() for line in llm_text.splitlines() if line.strip()]

        effective_queries = [query]
        for variant in variants:
            if variant not in effective_queries:
                effective_queries.append(variant)
            if len(effective_queries) >= query_count:
                break

        score_by_id: dict[str, float] = {}
        node_by_id: dict[str, Node] = {}

        for candidate in effective_queries:
            nodes = await self.retriever.retrieve(candidate, top_k)
            for rank, node in enumerate(nodes, start=1):
                node_by_id[node.node_id] = node
                score_by_id[node.node_id] = score_by_id.get(node.node_id, 0.0) + (1.0 / (rrf_k + rank))

        ranked_ids = sorted(score_by_id.keys(), key=lambda node_id: score_by_id[node_id], reverse=True)
        return [node_by_id[node_id] for node_id in ranked_ids[:top_k]]

