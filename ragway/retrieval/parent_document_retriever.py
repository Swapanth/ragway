"""Parent-document retriever that maps child matches back to parent nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.schema.node import Node
from ragway.validators import validate_positive_int

from ragway.retrieval.base_retriever import BaseRetriever


@dataclass(slots=True)
class ParentDocumentRetriever(BaseRetriever):
    """Retrieve child nodes and return unique parent-level results."""

    child_retriever: RetrieverProtocol
    parent_nodes: dict[str, Node] = field(default_factory=dict)

    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        """Retrieve child nodes, then collapse to unique parent nodes."""
        top_k = validate_positive_int(top_k, "top_k")
        child_nodes = await self.child_retriever.retrieve(query, top_k)

        results: list[Node] = []
        seen: set[str] = set()

        for child in child_nodes:
            parent_key = child.parent_id if child.parent_id is not None else child.doc_id
            if parent_key in seen:
                continue
            seen.add(parent_key)

            if child.parent_id is not None and child.parent_id in self.parent_nodes:
                results.append(self.parent_nodes[child.parent_id])
            else:
                results.append(child)

        return results[:top_k]

