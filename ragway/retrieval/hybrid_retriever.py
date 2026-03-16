"""Hybrid retriever combining vector and BM25 rankings via RRF."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.schema.node import Node
from ragway.validators import validate_positive_int

from ragway.retrieval.base_retriever import BaseRetriever


@dataclass(slots=True)
class HybridRetriever(BaseRetriever):
    """Fuse multiple retriever results using Reciprocal Rank Fusion."""

    bm25_retriever: RetrieverProtocol
    vector_retriever: RetrieverProtocol
    rrf_k: int = 60

    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        """Retrieve nodes from BM25 and vector retrievers, then fuse with RRF."""
        top_k = validate_positive_int(top_k, "top_k")
        rrf_k = validate_positive_int(self.rrf_k, "rrf_k")

        bm25_nodes = await self.bm25_retriever.retrieve(query, top_k)
        vector_nodes = await self.vector_retriever.retrieve(query, top_k)

        score_by_id: dict[str, float] = {}
        node_by_id: dict[str, Node] = {}

        for rank, node in enumerate(bm25_nodes, start=1):
            node_by_id[node.node_id] = node
            score_by_id[node.node_id] = score_by_id.get(node.node_id, 0.0) + (1.0 / (rrf_k + rank))

        for rank, node in enumerate(vector_nodes, start=1):
            node_by_id[node.node_id] = node
            score_by_id[node.node_id] = score_by_id.get(node.node_id, 0.0) + (1.0 / (rrf_k + rank))

        ranked_ids = sorted(score_by_id.keys(), key=lambda node_id: score_by_id[node_id], reverse=True)
        return [node_by_id[node_id] for node_id in ranked_ids[:top_k]]

