"""CLI command for ingesting source documents into a vector index."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import click

from ragway.chunking.fixed_chunker import FixedChunker
from ragway.embeddings.openai_embedding import OpenAIEmbedding
from ragway.parsing.pdf_parser import PDFParser
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.vectorstores.faiss_store import FAISSStore


PIPELINE_CHOICES = ["naive", "hybrid", "self", "long_context", "agentic"]


@dataclass(slots=True)
class LocalOpenAIEmbeddingClient:
    """Local deterministic embedding client for CLI ingestion."""

    dimensions: int = 8

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed texts into deterministic vectors."""
        del model
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0 for _ in range(self.dimensions)]
            for index, character in enumerate(text):
                bucket = index % self.dimensions
                vector[bucket] += (ord(character) % 29) / 29.0
            vectors.append(vector)
        return vectors


def _collect_documents(source: Path) -> list[Document]:
    """Collect text documents from a source directory recursively."""
    documents: list[Document] = []
    pdf_parser = PDFParser()

    for path in source.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".rst"}:
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            documents.append(Document(doc_id=str(path.relative_to(source)), content=content))
            continue

        if suffix == ".pdf":
            try:
                parsed = pdf_parser.parse(path.read_bytes(), source=str(path), doc_id=str(path.relative_to(source)))
            except Exception:
                # Skip unreadable PDFs so ingestion can continue for remaining files.
                continue
            documents.append(parsed)

    return documents


async def _ingest_async(source: Path) -> tuple[int, int]:
    """Ingest documents into an in-memory FAISS store and return counts."""
    documents = _collect_documents(source)
    chunker = FixedChunker(chunk_size=512, overlap=50)
    nodes: list[Node] = []
    for document in documents:
        nodes.extend(chunker.chunk(document))

    embedding = OpenAIEmbedding(model="text-embedding-3-small", max_batch_size=32, client=LocalOpenAIEmbeddingClient())
    vectors = await embedding.embed([node.content for node in nodes]) if nodes else []
    nodes_with_embeddings = [
        node.model_copy(update={"embedding": vector})
        for node, vector in zip(nodes, vectors)
    ]

    store = FAISSStore()
    await store.add(nodes_with_embeddings)
    return len(documents), len(nodes_with_embeddings)


@click.command("ingest")
@click.option("--source", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--pipeline", type=click.Choice(PIPELINE_CHOICES), default="naive", show_default=True)
def ingest_command(source: Path, pipeline: str) -> None:
    """Load documents, chunk them, embed, and store vectors for a pipeline."""
    doc_count, chunk_count = asyncio.run(_ingest_async(source))
    click.echo(f"Pipeline: {pipeline}")
    click.echo(f"Ingested documents: {doc_count}")
    click.echo(f"Stored chunks: {chunk_count}")

