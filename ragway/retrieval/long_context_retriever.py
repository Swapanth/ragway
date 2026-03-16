"""Long-context retriever that orders results for downstream long-context models."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.schema.node import Node
from ragway.validators import validate_positive_int

from ragway.retrieval.base_retriever import BaseRetriever


@dataclass(slots=True)
class LongContextRetriever(BaseRetriever):
    """Sort retrieved nodes by document and position for long-context prompting."""

    retriever: RetrieverProtocol

    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        """Retrieve candidate nodes then order by positional context."""
        top_k = validate_positive_int(top_k, "top_k")
        nodes = await self.retriever.retrieve(query, top_k)
        return sorted(nodes, key=lambda node: (node.doc_id, node.position if node.position is not None else 10**9))

