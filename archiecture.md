raglab/
|
|-- README.md
|-- CLAUDE.md
|-- pyproject.toml
|-- rag.yaml
|-- archiecture.md
|-- om.md
|-- user_value.md
|-- coverage.json
|
|-- .github/
|   |-- instructions/
|       |-- copilot-instructions.instructions.md
|
|-- configs/
|   |-- default.yaml
|   |-- fast_local.yaml
|   |-- fully_local.yaml
|   |-- high_quality.yaml
|   |-- long_doc.yaml
|   |-- agentic_groq.yaml
|   |-- pinecone_openai.yaml
|   |-- weaviate_mistral.yaml
|
|-- data/
|   |-- docs/
|       |-- rag-paper-2005.11401.pdf
|       |-- rag-survey-2312.10997.pdf
|
|-- ragway/
|   |-- __init__.py
|   |-- exceptions.py
|   |-- validators.py
|   |-- rag.py
|   |-- raglab.py
|   |
|   |-- schema/
|   |   |-- document.py
|   |   |-- metadata.py
|   |   |-- node.py
|   |
|   |-- interfaces/
|   |   |-- embedding_protocol.py
|   |   |-- retriever_protocol.py
|   |   |-- reranker_protocol.py
|   |   |-- llm_protocol.py
|   |
|   |-- core/
|   |   |-- component_registry.py
|   |   |-- config_loader.py
|   |   |-- dependency_container.py
|   |   |-- pipeline_builder.py
|   |   |-- pipeline_runner.py
|   |   |-- rag_engine.py
|   |   |-- rag_pipeline.py
|   |
|   |-- ingestion/
|   |   |-- base_loader.py
|   |   |-- api_loader.py
|   |   |-- pdf_loader.py
|   |   |-- markdown_loader.py
|   |   |-- web_loader.py
|   |   |-- docx_loader.py
|   |   |-- excel_loader.py
|   |   |-- notion_loader.py
|   |   |-- youtube_loader.py
|   |
|   |-- parsing/
|   |   |-- document_parser.py
|   |   |-- pdf_parser.py
|   |   |-- markdown_parser.py
|   |   |-- html_parser.py
|   |
|   |-- chunking/
|   |   |-- base_chunker.py
|   |   |-- fixed_chunker.py
|   |   |-- recursive_chunker.py
|   |   |-- semantic_chunker.py
|   |   |-- sliding_window_chunker.py
|   |   |-- hierarchical_chunker.py
|   |
|   |-- embeddings/
|   |   |-- base_embedding.py
|   |   |-- openai_embedding.py
|   |   |-- bge_embedding.py
|   |   |-- instructor_embedding.py
|   |   |-- sentence_transformer_embedding.py
|   |   |-- cohere_embedding.py
|   |
|   |-- vectorstores/
|   |   |-- base_vectorstore.py
|   |   |-- faiss_store.py
|   |   |-- chroma_store.py
|   |   |-- pinecone_store.py
|   |   |-- weaviate_store.py
|   |   |-- qdrant_store.py
|   |   |-- pgvector_store.py
|   |
|   |-- retrieval/
|   |   |-- base_retriever.py
|   |   |-- vector_retriever.py
|   |   |-- hybrid_retriever.py
|   |   |-- bm25_retriever.py
|   |   |-- multi_query_retriever.py
|   |   |-- parent_document_retriever.py
|   |   |-- long_context_retriever.py
|   |
|   |-- reranking/
|   |   |-- base_reranker.py
|   |   |-- bge_reranker.py
|   |   |-- cohere_reranker.py
|   |   |-- cross_encoder_reranker.py
|   |
|   |-- generation/
|   |   |-- base_llm.py
|   |   |-- llm_factory.py
|   |   |-- openai_llm.py
|   |   |-- anthropic_llm.py
|   |   |-- azure_openai_llm.py
|   |   |-- bedrock_llm.py
|   |   |-- groq_llm.py
|   |   |-- mistral_llm.py
|   |   |-- vertex_ai_llm.py
|   |   |-- llama_llm.py
|   |   |-- local_llm.py
|   |
|   |-- prompting/
|   |   |-- prompt_builder.py
|   |   |-- prompt_registry.py
|   |   |-- templates.py
|   |   |-- context_formatter.py
|   |   |-- page_context_formatter.py
|   |   |-- citation_formatter.py
|   |
|   |-- components/
|   |   |-- query_expansion.py
|   |   |-- context_compression.py
|   |   |-- memory_module.py
|   |   |-- citation_builder.py
|   |   |-- guardrails.py
|   |   |-- hallucination_detector.py
|   |
|   |-- evaluation/
|   |   |-- ragas_eval.py
|   |   |-- faithfulness.py
|   |   |-- answer_accuracy.py
|   |   |-- context_recall.py
|   |   |-- context_precision.py
|   |   |-- hallucination_score.py
|   |   |-- latency_eval.py
|   |
|   |-- caching/
|   |   |-- embedding_cache.py
|   |   |-- retrieval_cache.py
|   |   |-- llm_cache.py
|   |
|   |-- observability/
|   |   |-- logging.py
|   |   |-- tracing.py
|   |   |-- metrics.py
|   |   |-- cost_tracker.py
|   |
|   |-- cli/
|   |   |-- __init__.py
|   |   |-- rag_cli.py
|   |   |-- commands/
|   |       |-- __init__.py
|   |       |-- ingest.py
|   |       |-- query.py
|   |       |-- evaluate.py
|   |       |-- benchmark.py
|
|-- pipelines/
|   |-- naive_rag_pipeline.py
|   |-- hybrid_rag_pipeline.py
|   |-- agentic_rag_pipeline.py
|   |-- self_rag_pipeline.py
|   |-- long_context_rag_pipeline.py
|
|-- cli/
|   |-- rag_cli.py
|   |-- commands/
|       |-- ingest.py
|       |-- query.py
|       |-- evaluate.py
|       |-- benchmark.py
|
|-- studio/
|   |-- rag_debugger.py
|   |-- pipeline_visualizer.py
|   |-- experiment_dashboard.py
|
|-- benchmarks/
|   |-- __init__.py
|   |-- rag_eval_dataset.py
|   |-- dataset_loader.py
|   |-- synthetic_dataset_generator.py
|   |-- question_generator.py
|
|-- tests/
|   |-- unit/
|   |   |-- (mirrors ragway/ modules with detailed unit tests)
|   |   |-- test_rag_public_api.py
|   |   |-- test_raglab.py
|   |-- integration/
|   |   |-- test_all_pipelines.py
|   |   |-- test_all_providers.py
|
Added and expanded features reflected above:
- New vector store integrations: Qdrant and PgVector.
- Expanded LLM provider support: Azure OpenAI, AWS Bedrock, Groq, Mistral, and Vertex AI.
- Broader ingestion support: DOCX, Excel, Notion, and YouTube loaders.
- Stronger orchestration layer: component registry, config loader, and pipeline builder.
- Provider-specific runtime configs under configs/ for local, cloud, and hybrid setups.
