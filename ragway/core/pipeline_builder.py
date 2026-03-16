"""Build configurable RAG pipelines from YAML files."""

from __future__ import annotations

import importlib
from collections.abc import Awaitable
from pathlib import Path
from typing import cast

from ragway.core.component_registry import ComponentRegistry
from ragway.core.config_loader import ConfigLoader
from ragway.core.rag_pipeline import RAGPipeline
from ragway.exceptions import RagError
from ragway.interfaces.embedding_protocol import EmbeddingProtocol
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.vectorstores.base_vectorstore import BaseVectorStore


class PipelineBuilder:
    """Build and execute pipelines from a declarative YAML config."""

    def __init__(self, config_path: str = "rag.yaml") -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise RagError(f"Config file not found: {self.config_path}")

        loaded = ConfigLoader.load(str(self.config_path))
        plugins_raw = loaded.get("plugins")
        if plugins_raw is not None and not isinstance(plugins_raw, dict):
            raise RagError("Config section 'plugins' must be a mapping")

        self.root_config: dict[str, object] = loaded
        self.config: dict[str, object] = plugins_raw if isinstance(plugins_raw, dict) else loaded
        self.components: dict[str, object] = {}
        self.pipeline_module: object | None = None
        self._seeded_nodes: list[Node] = []
        self._indexed: bool = False

    def _section(self, name: str) -> dict[str, object]:
        """Return a config subsection as a dictionary."""
        raw = self.config.get(name, {})
        if isinstance(raw, dict):
            return raw
        raise RagError(f"Config section '{name}' must be a mapping")

    def _seed_nodes(self, chunker: object) -> list[Node]:
        """Build default local nodes so retrievers can initialize without extra code."""
        content = (
            "rag-toolkit is a modular Retrieval-Augmented Generation framework. "
            "It supports chunking, embeddings, vector retrieval, reranking, and generation. "
            "Hybrid and long-context pipelines provide specialized retrieval behavior."
        )
        document = Document(doc_id="config-doc-001", content=content)

        chunk_fn = getattr(chunker, "chunk", None)
        if not callable(chunk_fn):
            raise RagError("Configured chunker does not expose a chunk(document) method")

        chunked = chunk_fn(document)
        if not isinstance(chunked, list):
            raise RagError("Configured chunker returned an invalid value")
        return [node for node in chunked if isinstance(node, Node)]

    async def _index_nodes(self, nodes: list[Node], embedding: object, vectorstore: object) -> list[Node]:
        """Embed and index nodes for retrieval usage."""
        if not nodes:
            return []

        embed_fn = getattr(embedding, "embed", None)
        if not callable(embed_fn):
            raise RagError("Configured embedding model does not expose embed(texts)")

        vectors = await embed_fn([node.content for node in nodes])
        if not isinstance(vectors, list):
            raise RagError("Configured embedding model returned an invalid vectors payload")

        nodes_with_embeddings: list[Node] = []
        for node, vector in zip(nodes, vectors):
            nodes_with_embeddings.append(node.model_copy(update={"embedding": vector}))

        add_fn = getattr(vectorstore, "add", None)
        if not callable(add_fn):
            raise RagError("Configured vectorstore does not expose add(nodes)")
        await add_fn(nodes_with_embeddings)

        return nodes_with_embeddings

    def build(self) -> RAGPipeline:
        """Build configured component instances and return selected pipeline definition."""
        registry = ComponentRegistry()

        llm = registry.get_llm(self._section("llm"))
        embedding = registry.get_embedding(self._section("embedding"))
        vectorstore = registry.get_vectorstore(self._section("vectorstore"))
        chunker = registry.get_chunker(self._section("chunking"))
        nodes = self._seed_nodes(chunker)

        retriever = registry.get_retriever(
            self._section("retrieval"),
            vectorstore,
            embedding_model=embedding,
            llm=llm,
            nodes=nodes,
        )

        reranking_cfg = self._section("reranking")
        reranker = registry.get_reranker(reranking_cfg) if bool(reranking_cfg.get("enabled", False)) else None

        pipeline_key = str(self.root_config.get("pipeline", self.config.get("pipeline", "naive")))
        if pipeline_key not in ComponentRegistry.PIPELINES:
            options = ", ".join(sorted(ComponentRegistry.PIPELINES.keys()))
            raise RagError(f"Unknown pipeline '{pipeline_key}'. Choose from: {options}")

        module_path = ComponentRegistry.PIPELINES[pipeline_key]
        module = importlib.import_module(module_path)
        build_pipeline = getattr(module, "build_pipeline", None)
        if not callable(build_pipeline):
            raise RagError(f"Pipeline module '{module_path}' does not expose build_pipeline")

        self.components = {
            "llm": llm,
            "vectorstore": vectorstore,
            "embedding": embedding,
            "retriever": retriever,
            "reranker": reranker,
            "chunker": chunker,
        }
        self._seeded_nodes = nodes
        self._indexed = False
        self.pipeline_module = module

        pipeline = build_pipeline(
            llm=llm,
            vectorstore=vectorstore,
            embedding=embedding,
            retriever=retriever,
            reranker=reranker,
            chunker=chunker,
        )
        if not isinstance(pipeline, RAGPipeline):
            raise RagError("build_pipeline() did not return a RAGPipeline instance")
        return pipeline

    async def run(self, query: str) -> object:
        """Execute configured pipeline query using the built runtime components."""
        if self.pipeline_module is None or not self.components:
            self.build()

        if not self._indexed:
            embedding = cast(EmbeddingProtocol, self.components["embedding"])
            vectorstore = cast(BaseVectorStore, self.components["vectorstore"])
            llm = cast(LLMProtocol, self.components["llm"])
            indexed_nodes = await self._index_nodes(
                self._seeded_nodes,
                embedding,
                vectorstore,
            )
            self.components["retriever"] = ComponentRegistry().get_retriever(
                self._section("retrieval"),
                vectorstore,
                embedding_model=embedding,
                llm=llm,
                nodes=indexed_nodes,
            )
            self._indexed = True

        assert self.pipeline_module is not None
        run_fn = getattr(self.pipeline_module, "run", None)
        if not callable(run_fn):
            raise RagError("Configured pipeline module does not expose run(query)")

        result = run_fn(
            query,
            llm=self.components.get("llm"),
            vectorstore=self.components.get("vectorstore"),
            embedding=self.components.get("embedding"),
            retriever=self.components.get("retriever"),
            reranker=self.components.get("reranker"),
            chunker=self.components.get("chunker"),
        )
        if not isinstance(result, Awaitable):
            raise RagError("Configured pipeline run() did not return an awaitable")
        return await result

