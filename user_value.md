# ragforge overview

## quick stats

| Feature | Description |
|-------|-------------|
| **6** | Ready-to-use pipelines |
| **4** | Working examples |
| **pip install** | One command setup |
| **CLI** | No-code interface |

---

## 1 — Build any RAG app in minutes

```python
from ragforge import RAGEngine

engine = RAGEngine.from_config("config.yaml")
engine.ingest("docs/")

answer = engine.query("What is our refund policy?")

# Done. That's it.
```

---

## 2 — Production-grade pipelines included

| Pipeline | Purpose |
|--------|--------|
| **Naive RAG** | Quick prototypes |
| **Hybrid RAG** | Better accuracy |
| **Agentic RAG** | Multi-step reasoning |
| **Graph RAG** | Relationship aware retrieval |
| **Self RAG** | Self-correcting responses |
| **Long-context RAG** | Works with large documents |

---

## 3 — Fully swappable components (no lock-in)

### Vector Stores
- FAISS  
- Chroma  
- Pinecone  
- Weaviate  

### LLMs
- OpenAI  
- Anthropic  
- LLaMA  
- Local models  

### Embeddings
- OpenAI  
- BGE  
- Instructor  
- SentenceTransformers  

### Retrievers
- Vector  
- BM25  
- Hybrid  
- Multi-query  
- Parent-document  

---

## 4 — Built-in evaluation

Evaluate RAG performance automatically using **RAGAS**.

Metrics included:

- Faithfulness
- Answer accuracy
- Context recall
- Context precision
- Hallucination score
- Latency

No extra setup required.

---

## 5 — CLI for non-code workflows

```bash
ragforge ingest ./docs --pipeline hybrid

ragforge query "summarise the report"

ragforge evaluate --dataset benchmarks/

ragforge benchmark --config experiment.yaml
```

---

## 6 — Copy-paste real-world examples

| Example | Description |
|-------|-------------|
| **Chat with PDF** | Upload any PDF and ask questions |
| **Research assistant** | Multi-document synthesis |
| **Codebase assistant** | Ask questions about your code |
| **Company knowledgebase** | Internal Q&A over docs |

---

## The one-line pitch

A developer gets a **complete RAG toolkit — from loading a PDF to evaluating answer quality — without stitching together 10 different libraries.**