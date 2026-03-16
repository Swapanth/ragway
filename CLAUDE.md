# CLAUDE.md — ragway

This file is read automatically by Claude Code at the start of every session.
It defines everything about how to build, test, and extend this project.

---

## Project overview

`ragway` is a modular, research-grade Python library for building Retrieval-Augmented
Generation (RAG) pipelines. It supports naive, hybrid, agentic, graph, self-correcting,
and long-context RAG patterns. The codebase is designed for experimentation — every
component is swappable via Protocol interfaces.

---

## Stack

| Concern         | Tool                                      |
|-----------------|-------------------------------------------|
| Language        | Python 3.11                               |
| Package manager | pip + pyproject.toml                      |
| Testing         | pytest + pytest-asyncio                   |
| Linting         | ruff                                      |
| Type checking   | mypy (strict mode)                        |
| Docs site       | Next.js + Nextra (docs-site/)             |
| Containerisation| docker-compose.yml                        |

---

## Essential commands

```bash
# Install in editable mode (run once after cloning)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run a specific module's tests
pytest tests/unit/test_chunking.py

# Lint
ruff check ragway/

# Auto-fix lint issues
ruff check ragway/ --fix

# Type check
mypy ragway/

# Run a specific pipeline end-to-end
python -m pipelines.naive_rag_pipeline

# Start docs site locally
cd docs-site && npm run dev
```

---

## Build order — CRITICAL

Always build in this dependency order. Later modules import from earlier ones.
Never write a consumer before its dependency exists.

```
LAYER 0 — foundations (no internal imports)
  ragway/exceptions.py
  ragway/validators.py
  ragway/schema/document.py
  ragway/schema/node.py
  ragway/schema/metadata.py

LAYER 1 — protocols (depend only on schema)
  ragway/interfaces/embedding_protocol.py
  ragway/interfaces/retriever_protocol.py
  ragway/interfaces/reranker_protocol.py
  ragway/interfaces/llm_protocol.py

LAYER 2 — config (depend on schema + interfaces)
  ragway/config/rag_config.py
  ragway/config/model_config.py
  ragway/config/retriever_config.py
  ragway/config/pipeline_config.py

LAYER 3 — utilities (no domain imports)
  ragway/utils/token_counter.py
  ragway/utils/text_utils.py
  ragway/utils/async_executor.py

LAYER 4 — core engine (depends on interfaces + config)
  ragway/core/config.py
  ragway/core/dependency_container.py
  ragway/core/rag_engine.py
  ragway/core/rag_pipeline.py
  ragway/core/pipeline_runner.py

LAYER 5 — leaf modules (each depends on layers 0–4 only)
  ragway/parsing/        — all parsers
  ragway/ingestion/      — all loaders
  ragway/preprocessing/  — cleaners, enrichers
  ragway/chunking/       — all chunkers
  ragway/embeddings/     — all embedding adapters
  ragway/vectorstores/   — all vectorstore adapters
  ragway/document_store/ — SQLite, Postgres stores
  ragway/indexing/       — index builders
  ragway/retrieval/      — all retrievers
  ragway/reranking/      — all rerankers
  ragway/generation/     — all LLM adapters
  ragway/prompting/      — prompt builders and templates
  ragway/components/     — query expansion, memory, citations
  ragway/tools/          — agent tools
  ragway/caching/        — embedding, retrieval, LLM caches
  ragway/evaluation/     — all eval metrics
  ragway/observability/  — tracing, metrics, logging
  ragway/plugins/        — plugin system

LAYER 6 — pipelines (depend on everything above)
  pipelines/naive_rag_pipeline.py
  pipelines/hybrid_rag_pipeline.py
  pipelines/agentic_rag_pipeline.py
  pipelines/graph_rag_pipeline.py
  pipelines/self_rag_pipeline.py
  pipelines/long_context_rag_pipeline.py

LAYER 7 — CLI, studio, benchmarks (end-user layer)
  cli/
  studio/
  benchmarks/
```

---

## Coding conventions

### General
- Every module file must have a module-level docstring explaining its purpose.
- All public classes and functions must have docstrings.
- All function signatures must have full type hints — no `Any` unless unavoidable.
- Use `from __future__ import annotations` at the top of every file.
- Never use bare `except:` — always catch specific exception types.
- Prefer `pathlib.Path` over `os.path` everywhere.

### Classes
- Base classes in `ragway/` use `ABC` with `@abstractmethod`.
- Interfaces in `ragway/interfaces/` use `typing.Protocol` (structural subtyping).
- Concrete implementations inherit from the corresponding base class in their module folder.
- Use `@dataclass` or `pydantic.BaseModel` for data containers — never plain dicts.

### Async
- All I/O-bound operations (embedding calls, LLM calls, vectorstore queries) must be `async`.
- Use `ragway/utils/async_executor.py` to run async code from sync contexts.
- Never use `asyncio.run()` inside library code — only in CLI entry points and tests.

### Imports
- Internal imports are always absolute: `from ragway.schema.document import Document`
- Never use relative imports (`from ..schema import ...`)
- Group imports: stdlib → third-party → internal, separated by blank lines

### Error handling
- Raise project-specific exceptions from `ragway/exceptions.py`, not built-ins.
- Always add context to exceptions: `raise EmbeddingError(f"Model {model} failed: {e}") from e`

---

## Testing conventions

- Mirror the module structure: `ragway/chunking/fixed_chunker.py` → `tests/unit/chunking/test_fixed_chunker.py`
- Every public method must have at least one test.
- Use `pytest.fixture` for shared test data (documents, nodes, configs).
- Mock all external calls (OpenAI, Cohere, Pinecone, etc.) using `unittest.mock.AsyncMock`.
- Integration tests in `tests/integration/` may make real network calls — mark with `@pytest.mark.integration`.
- Benchmark tests go in `tests/benchmark/` — mark with `@pytest.mark.benchmark`.
- Use `pytest-asyncio` with `@pytest.mark.asyncio` for all async test functions.

### Test file template
```python
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from ragway.schema.document import Document
from ragway.chunking.fixed_chunker import FixedChunker


@pytest.fixture
def sample_document() -> Document:
    return Document(content="Sample text " * 100, metadata={})


class TestFixedChunker:
    def test_chunks_by_size(self, sample_document: Document) -> None:
        chunker = FixedChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(sample_document)
        assert len(chunks) > 1
        assert all(len(c.content) <= 110 for c in chunks)

    def test_empty_document_returns_empty(self) -> None:
        chunker = FixedChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk(Document(content="", metadata={}))
        assert chunks == []
```

---

## Module-by-module guidance

### ragway/schema/
- `Document` — raw ingested content with source metadata. Immutable after creation.
- `Node` — a chunk derived from a Document. Has `node_id`, `doc_id`, `content`, `embedding`, `metadata`.
- `Metadata` — typed metadata container. Use pydantic with strict validation.

### ragway/interfaces/
- `EmbeddingProtocol` — must define `async def embed(self, texts: list[str]) -> list[list[float]]`
- `RetrieverProtocol` — must define `async def retrieve(self, query: str, top_k: int) -> list[Node]`
- `RerankerProtocol` — must define `async def rerank(self, query: str, nodes: list[Node]) -> list[Node]`
- `LLMProtocol` — must define `async def generate(self, prompt: str) -> str`

### ragway/chunking/
- All chunkers inherit from `BaseChunker` and implement `def chunk(self, document: Document) -> list[Node]`
- `FixedChunker` — splits by token count with optional overlap
- `SemanticChunker` — splits at semantic boundaries using embedding cosine similarity
- `RecursiveChunker` — tries paragraph → sentence → word splits recursively
- `SlidingWindowChunker` — overlapping windows, configurable stride
- `HierarchicalChunker` — produces parent + child nodes for parent-document retrieval

### ragway/embeddings/
- All adapters inherit from `BaseEmbedding` and implement `EmbeddingProtocol`
- Cache embedding calls via `ragway/caching/embedding_cache.py` — never re-embed the same text twice
- Always normalise embeddings to unit length before storing

### ragway/vectorstores/
- All stores inherit from `BaseVectorStore`
- Must implement: `async def add(nodes)`, `async def search(query_vector, top_k)`, `async def delete(node_ids)`
- FAISS store is the default for local/testing use
- Pinecone and Weaviate stores use environment variables for credentials — never hardcode keys

### ragway/retrieval/
- `VectorRetriever` — pure ANN search via vectorstore
- `BM25Retriever` — keyword search, no embeddings required
- `HybridRetriever` — combines vector + BM25 with RRF (Reciprocal Rank Fusion) by default
- `MultiQueryRetriever` — generates N query variants via LLM, merges results
- `ParentDocumentRetriever` — retrieves child chunks, returns parent documents
- `LongContextRetriever` — retrieves and sorts by position for long-context LLMs

### ragway/generation/
- All LLM adapters inherit from `BaseLLM` and implement `LLMProtocol`
- `AnthropicLLM` — uses `claude-sonnet-4-6` by default, configurable via `ModelConfig`
- `OpenAILLM` — uses `gpt-4o` by default
- Always pass `max_tokens` and `temperature` from config — never hardcode

### ragway/evaluation/
- All evaluators accept `(question, answer, context)` triples
- `FaithfulnessEval` — checks answer is grounded in context (LLM-as-judge)
- `AnswerAccuracy` — checks factual correctness against a gold answer
- `ContextRecall` — checks retrieved context covers the answer
- `HallucinationScore` — flags claims not supported by retrieved context
- Use `ragas_eval.py` for batch evaluation against full datasets

### pipelines/
- Each pipeline file is a self-contained runnable script.
- Must define a `build_pipeline() -> RAGPipeline` factory function.
- Must define a `run(query: str) -> str` entry point.
- Add `if __name__ == "__main__"` block for direct execution.

---

## Environment variables

```bash
# Required for cloud services (add to .env, never commit)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=...
WEAVIATE_URL=...
WEAVIATE_API_KEY=...

# Optional tuning
RAG_LOG_LEVEL=INFO          # DEBUG | INFO | WARNING | ERROR
RAG_CACHE_DIR=.cache/ragway    # local cache for embeddings
RAG_DEFAULT_MODEL=claude-sonnet-4-6
```

---

## What NOT to do

- Do not put business logic in `__init__.py` files — keep them as clean re-exports only.
- Do not add `print()` statements — use `ragway/observability/logging.py`.
- Do not hardcode model names, API keys, or file paths — use config or env vars.
- Do not write synchronous wrappers around async functions in library code.
- Do not import `pipelines/` modules from inside `ragway/` — pipelines are consumers, not library code.
- Do not add new dependencies to `pyproject.toml` without a comment explaining why.
- Do not skip writing tests when implementing a new module. Tests are part of the task.

---

## Docs conventions (docs-site/)

- Every module in `ragway/` has a corresponding `.mdx` page in `docs/modules/`.
- Every pipeline in `pipelines/` has a corresponding `.mdx` page in `docs/pipelines/`.
- Doc pages follow the structure: Overview → When to use → Configuration → Example → API reference.
- Code examples in docs must be runnable — test them before committing.
- The docs site is Next.js + Nextra. Run `npm run dev` in `docs-site/` to preview.

---

## Session startup checklist

When starting a new Claude Code session, always:

1. Run `pip install -e ".[dev]"` if this is a fresh environment.
2. Run `pytest tests/` to see what currently passes.
3. Run `mypy ragway/` to see current type errors.
4. Check the build order above before writing any new file.
5. After writing a module, run its tests before moving to the next module.