# ragway

The way to build RAG.

## Install

```bash
pip install ragway
```

## Quickstart

```python
from ragway import RAG
import asyncio, pathlib
pathlib.Path("example_docs").mkdir(exist_ok=True); pathlib.Path("example_docs/intro.md").write_text("# RAG\nRetrieval-Augmented Generation combines retrieval with generation.", encoding="utf-8")
rag = RAG(llm="openai", api_key="YOUR_OPENAI_KEY")
print(asyncio.run(rag.ingest("example_docs")), asyncio.run(rag.query("What is RAG?")))
```

## Why ragway

- Compared to LangChain: smaller public surface, fewer framework abstractions, explicit config and component wiring.
- Compared to LlamaIndex: direct control over retrieval/rerank/vectorstore choices without committing to one indexing model.
- For production code: you can start simple with `RAG(...)`, then move to YAML config and provider-specific tuning without rewriting app code.

## What You Can Swap

| Component | Options |
| --- | --- |
| LLM | anthropic, openai, mistral, groq, llama, local |
| Vectorstore | faiss, chroma, pinecone, weaviate |
| Retrieval | vector, bm25, hybrid, multi_query, parent_document |
| Reranker | cohere, bge, cross_encoder (or None) |
| Chunking | fixed, recursive, semantic, sliding_window, hierarchical |
| Pipeline | naive, hybrid, self, long_context, agentic |

## Install Options

```bash
# Base package
pip install ragway

# Provider extras
pip install ragway[anthropic]
pip install ragway[openai]
pip install ragway[mistral]
pip install ragway[groq]
pip install ragway[cohere]
pip install ragway[pinecone]
pip install ragway[weaviate]
pip install ragway[faiss]
pip install ragway[chroma]
pip install ragway[llama]
pip install ragway[bge]

# Bundles
pip install ragway[all-cloud]
pip install ragway[all-local]
pip install ragway[all]
pip install ragway[dev]
```

## Config File Example

```yaml
version: "1.0"
pipeline: hybrid

plugins:
  llm:
    provider: groq
    model: llama-3.1-8b-instant
    api_key: ${GROQ_API_KEY}
    temperature: 0.2
    max_tokens: 512

  embedding:
    provider: openai
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
    batch_size: 32

  vectorstore:
    provider: faiss
    index_name: rag-lab
    top_k: 5

  retrieval:
    strategy: hybrid
    top_k: 5
    rrf_k: 60

  reranker:
    enabled: true
    provider: cohere
    api_key: ${COHERE_API_KEY}
    top_k: 3

  chunking:
    strategy: recursive
    chunk_size: 512
    overlap: 50
```

```python
from ragway import RAG
import asyncio

rag = RAG.from_config("rag.yaml")
print(asyncio.run(rag.query("Summarize the key idea.")))
```

## Links

- Docs: https://ragway.dev
- Issues: https://github.com/swapanth/ragway/issues
- Changelog: https://github.com/swapanth/ragway/blob/main/CHANGELOG.md
