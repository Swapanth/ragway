"""Vector retriever backed by embedding model and vector store."""

from __future__ import annotations

from dataclasses import dataclass

from ragway.interfaces.embedding_protocol import EmbeddingProtocol
from ragway.schema.node import Node
from ragway.validators import validate_positive_int
from ragway.vectorstores.base_vectorstore import BaseVectorStore

from ragway.retrieval.base_retriever import BaseRetriever


@dataclass(slots=True)
class VectorRetriever(BaseRetriever):
    """Retrieve nodes by nearest-neighbor search over vector embeddings."""

    embedding_model: EmbeddingProtocol
    vector_store: BaseVectorStore

    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        """Embed query text and search vector store for closest nodes."""
        top_k = validate_positive_int(top_k, "top_k")
        query_vector = (await self.embedding_model.embed([query]))[0]
        return await self.vector_store.search(query_vector, top_k)

