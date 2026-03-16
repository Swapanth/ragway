"""Top-level public API for building and running configurable RAG pipelines."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import AsyncIterator

from ragway.core.component_registry import ComponentRegistry
from ragway.core.config_loader import ConfigLoader
from ragway.evaluation.ragas_eval import RagasEval
from ragway.exceptions import RagError
from ragway.interfaces.embedding_protocol import EmbeddingProtocol
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.interfaces.reranker_protocol import RerankerProtocol
from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.parsing.markdown_parser import MarkdownParser
from ragway.parsing.pdf_parser import PDFParser
from ragway.prompting.prompt_builder import PromptBuilder
from ragway.schema.document import Document
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node
from ragway.vectorstores.base_vectorstore import BaseVectorStore


class RAGLab:
    """Main entry point for rag-lab using config-driven component composition."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config: dict[str, object] = deepcopy(config)
        self.registry = ComponentRegistry()

        self.llm: LLMProtocol
        self.embedding: EmbeddingProtocol
        self.vectorstore: BaseVectorStore
        self.retriever: RetrieverProtocol
        self.reranker: RerankerProtocol | None
        self.chunker: object
        self.prompt_builder = PromptBuilder(template_name="default")

        self._indexed_nodes: list[Node] = []
        self._last_nodes: list[Node] = []

        self._build_components()

    @classmethod
    def from_config(cls, config_path: str = "rag.yaml") -> RAGLab:
        """Load everything from a YAML config file."""
        path = Path(config_path)
        if not path.exists():
            raise RagError(f"Config file not found: {path}")

        loaded = ConfigLoader.load(str(path))
        return cls(config=loaded)

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> RAGLab:
        """Build from a Python dict, useful for testing."""
        return cls(config=config)

    async def ingest(self, source: str) -> int:
        """Ingest documents from a path or URL. Returns chunk count."""
        documents = await self._load_documents(source)
        if not documents:
            return 0

        chunk_fn = getattr(self.chunker, "chunk", None)
        if not callable(chunk_fn):
            raise RagError("Configured chunker does not expose chunk(document)")

        nodes: list[Node] = []
        for document in documents:
            chunked = chunk_fn(document)
            if not isinstance(chunked, list):
                raise RagError("Chunker returned invalid chunk payload")
            nodes.extend([node for node in chunked if isinstance(node, Node)])

        if not nodes:
            return 0

        vectors = await self.embedding.embed([node.content for node in nodes])
        nodes_with_embeddings = [
            node.model_copy(update={"embedding": vector})
            for node, vector in zip(nodes, vectors)
        ]
        await self.vectorstore.add(nodes_with_embeddings)

        self._indexed_nodes.extend(nodes_with_embeddings)
        self._refresh_retriever()
        return len(nodes_with_embeddings)

    async def query(self, question: str, stream: bool = False) -> str | AsyncIterator[str]:
        """Run a query through the configured pipeline."""
        if not stream:
            answer, nodes = await self._query_internal(question)
            self._last_nodes = nodes
            return answer

        prompt, nodes = await self._build_prompt(question)
        self._last_nodes = nodes
        return self.llm.stream(prompt)

    async def query_with_sources(self, question: str) -> dict[str, object]:
        """Run a query and return answer + source nodes."""
        answer, nodes = await self._query_internal(question)
        self._last_nodes = nodes

        sources = [
            {
                "node_id": node.node_id,
                "doc_id": node.doc_id,
                "content": node.content,
            }
            for node in nodes
        ]
        return {"answer": answer, "sources": sources}

    async def evaluate(self, dataset_path: str) -> dict[str, float]:
        """Evaluate pipeline on a dataset. Returns metric scores."""
        path = Path(dataset_path)
        if not path.exists():
            raise RagError(f"Dataset file not found: {path}")

        dataset = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(dataset, list):
            raise RagError("Evaluation dataset must be a list of rows")

        rows: list[dict[str, object]] = []
        for item in dataset:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", ""))
            answer, nodes = await self._query_internal(question)
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "gold_answer": item.get("gold_answer", ""),
                    "context": [node.content for node in nodes],
                }
            )

        pipeline_name = str(self.config.get("pipeline", "naive"))
        return RagasEval().run(dataset=rows, pipeline_name=pipeline_name)

    def switch(self, **kwargs: object) -> RAGLab:
        """Return a new RAGLab with selected components swapped by provider."""
        updated = deepcopy(self.config)

        if "pipeline" in kwargs:
            updated["pipeline"] = kwargs["pipeline"]

        if "llm" in kwargs:
            updated.setdefault("llm", {})
            llm_cfg = updated["llm"]
            if not isinstance(llm_cfg, dict):
                raise RagError("Config section 'llm' must be a mapping")
            if isinstance(kwargs["llm"], str):
                llm_cfg["provider"] = kwargs["llm"]
            elif isinstance(kwargs["llm"], dict):
                llm_cfg.update(kwargs["llm"])

        if "vectorstore" in kwargs:
            updated.setdefault("vectorstore", {})
            store_cfg = updated["vectorstore"]
            if not isinstance(store_cfg, dict):
                raise RagError("Config section 'vectorstore' must be a mapping")
            if isinstance(kwargs["vectorstore"], str):
                store_cfg["provider"] = kwargs["vectorstore"]
            elif isinstance(kwargs["vectorstore"], dict):
                store_cfg.update(kwargs["vectorstore"])

        if "embedding" in kwargs:
            updated.setdefault("embedding", {})
            embedding_cfg = updated["embedding"]
            if not isinstance(embedding_cfg, dict):
                raise RagError("Config section 'embedding' must be a mapping")
            if isinstance(kwargs["embedding"], str):
                embedding_cfg["provider"] = kwargs["embedding"]
            elif isinstance(kwargs["embedding"], dict):
                embedding_cfg.update(kwargs["embedding"])

        if "retriever" in kwargs:
            updated.setdefault("retrieval", {})
            retrieval_cfg = updated["retrieval"]
            if not isinstance(retrieval_cfg, dict):
                raise RagError("Config section 'retrieval' must be a mapping")
            if isinstance(kwargs["retriever"], str):
                retrieval_cfg["strategy"] = kwargs["retriever"]
            elif isinstance(kwargs["retriever"], dict):
                retrieval_cfg.update(kwargs["retriever"])

        if "reranker" in kwargs:
            updated.setdefault("reranking", {})
            rerank_cfg = updated["reranking"]
            if not isinstance(rerank_cfg, dict):
                raise RagError("Config section 'reranking' must be a mapping")
            if isinstance(kwargs["reranker"], str):
                rerank_cfg["provider"] = kwargs["reranker"]
                rerank_cfg["enabled"] = True
            elif isinstance(kwargs["reranker"], dict):
                rerank_cfg.update(kwargs["reranker"])

        if "chunker" in kwargs:
            updated.setdefault("chunking", {})
            chunk_cfg = updated["chunking"]
            if not isinstance(chunk_cfg, dict):
                raise RagError("Config section 'chunking' must be a mapping")
            if isinstance(kwargs["chunker"], str):
                chunk_cfg["strategy"] = kwargs["chunker"]
            elif isinstance(kwargs["chunker"], dict):
                chunk_cfg.update(kwargs["chunker"])

        return RAGLab.from_dict(updated)

    def _section(self, name: str) -> dict[str, object]:
        """Return one config subsection as a dictionary."""
        raw = self.config.get(name, {})
        if isinstance(raw, dict):
            return raw
        raise RagError(f"Config section '{name}' must be a mapping")

    def _build_components(self) -> None:
        """Instantiate all configurable components from the current config."""
        llm_obj = self.registry.get_llm(self._section("llm"))
        embedding_obj = self.registry.get_embedding(self._section("embedding"))
        vectorstore_obj = self.registry.get_vectorstore(self._section("vectorstore"))
        chunker_obj = self.registry.get_chunker(self._section("chunking"))

        self.llm = llm_obj
        self.embedding = embedding_obj
        self.vectorstore = vectorstore_obj
        self.chunker = chunker_obj

        reranking_cfg = self._section("reranking")
        if bool(reranking_cfg.get("enabled", False)):
            reranker_obj = self.registry.get_reranker(reranking_cfg)
        else:
            reranker_obj = None
        self.reranker = reranker_obj

        self.retriever = self.registry.get_retriever(
            self._section("retrieval"),
            self.vectorstore,
            embedding_model=self.embedding,
            llm=self.llm,
            nodes=self._indexed_nodes,
        )

    def _refresh_retriever(self) -> None:
        """Rebuild retriever after ingestion so corpus-aware strategies update."""
        self.retriever = self.registry.get_retriever(
            self._section("retrieval"),
            self.vectorstore,
            embedding_model=self.embedding,
            llm=self.llm,
            nodes=self._indexed_nodes,
        )

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        """Safely parse config integers with a fallback value."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, (str, bytes, bytearray)):
            try:
                return int(value)
            except ValueError:
                return default
        try:
            int_method = getattr(value, "__int__")
            parsed = int_method()
            return int(parsed) if isinstance(parsed, int) else default
        except Exception:
            return default

    async def _build_prompt(self, question: str) -> tuple[str, list[Node]]:
        """Execute retrieve-rerank flow and return prompt plus source nodes."""
        top_k = self._as_int(self._section("retrieval").get("top_k", 5), 5)
        nodes = await self.retriever.retrieve(question, top_k=top_k)

        if self.reranker is not None:
            nodes = await self.reranker.rerank(question, nodes)

        prompt = self.prompt_builder.build(query=question, nodes=nodes)
        return prompt, nodes

    async def _query_internal(self, question: str) -> tuple[str, list[Node]]:
        """Execute retrieve-rerank-generate flow and return answer with nodes."""
        prompt, nodes = await self._build_prompt(question)
        answer = await self.llm.generate(prompt)
        return answer, nodes
    async def _load_documents(self, source: str) -> list[Document]:
        """Load documents from URL, file, or directory source."""
        if source.startswith("http://") or source.startswith("https://"):
            from ragway.ingestion.web_loader import WebLoader

            return await WebLoader().load(source)

        path = Path(source)
        if not path.exists():
            raise RagError(f"Source path not found: {source}")

        documents: list[Document] = []
        markdown_parser = MarkdownParser()
        pdf_parser = PDFParser()

        files: list[Path]
        if path.is_file():
            files = [path]
        else:
            files = [item for item in path.rglob("*") if item.is_file()]

        for item in files:
            suffix = item.suffix.lower()
            if suffix == ".md":
                text = item.read_text(encoding="utf-8", errors="ignore")
                documents.append(markdown_parser.parse(text, source=str(item), doc_id=item.stem))
                continue
            if suffix == ".pdf":
                documents.append(pdf_parser.parse(item.read_bytes(), source=str(item), doc_id=item.stem))
                continue
            if suffix in {".txt", ".rst"}:
                text = item.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue
                metadata = Metadata(source=str(item))
                documents.append(Document(doc_id=item.stem, content=text, metadata=metadata))

        return documents


RAG = RAGLab

