from __future__ import annotations

import asyncio
import types

import pytest

from ragway.core.pipeline_builder import PipelineBuilder
from ragway.core.rag_pipeline import RAGPipeline
from ragway.exceptions import RagError
from ragway.schema.node import Node


async def test_pipeline_builder_build_with_minimal_config(monkeypatch, tmp_path) -> None:
    """PipelineBuilder should build a pipeline from config with mocked components."""
    config_path = tmp_path / "rag_config.yaml"
    config_path.write_text(
        """
pipeline: naive
llm:
  provider: anthropic
embedding:
  provider: bge
vectorstore:
  provider: faiss
retrieval:
  strategy: vector
reranking:
  enabled: false
chunking:
  strategy: fixed
  chunk_size: 20
  overlap: 5
""".strip(),
        encoding="utf-8",
    )

    class FakePipelineModule:
        @staticmethod
        def build_pipeline(**kwargs):
            _ = kwargs
            from ragway.core.rag_engine import RagConfig

            return RAGPipeline(name="naive", config=RagConfig())

        @staticmethod
        async def run(query: str, **kwargs):
            _ = kwargs
            return f"ok:{query}"

    monkeypatch.setattr(
      "ragway.core.pipeline_builder.importlib.import_module",
        lambda _: FakePipelineModule,
    )

    builder = PipelineBuilder(str(config_path))
    pipeline = builder.build()
    result = await builder.run("hello")

    assert isinstance(pipeline, RAGPipeline)
    assert result == "ok:hello"


async def test_pipeline_builder_provider_switch_changes_component_type(monkeypatch, tmp_path) -> None:
    """Changing provider string should change the class type selected by builder."""
    config_path = tmp_path / "rag_config.yaml"
    config_path.write_text(
        """
pipeline: naive
llm:
  provider: anthropic
embedding:
  provider: bge
vectorstore:
  provider: faiss
retrieval:
  strategy: vector
reranking:
  enabled: false
chunking:
  strategy: fixed
  chunk_size: 20
  overlap: 5
""".strip(),
        encoding="utf-8",
    )

    from ragway.core.rag_engine import RagConfig

    async def _fake_run(query: str, **kwargs):
      del kwargs
      return query

    fake_module = types.SimpleNamespace(
      build_pipeline=lambda **kwargs: RAGPipeline(name="naive", config=RagConfig()),
      run=_fake_run,
    )
    monkeypatch.setattr("ragway.core.pipeline_builder.importlib.import_module", lambda _: fake_module)

    builder = PipelineBuilder(str(config_path))
    builder.build()
    first_type = type(builder.components["llm"])

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("provider: anthropic", "provider: openai"),
        encoding="utf-8",
    )
    builder_two = PipelineBuilder(str(config_path))
    builder_two.build()
    second_type = type(builder_two.components["llm"])

    assert first_type is not second_type


async def test_pipeline_builder_missing_config_raises(tmp_path) -> None:
    """PipelineBuilder should raise when config path does not exist."""
    missing = tmp_path / "missing.yaml"
    with pytest.raises(RagError, match="Config file not found"):
        PipelineBuilder(str(missing))


async def test_pipeline_builder_root_must_be_mapping(tmp_path) -> None:
    """PipelineBuilder should reject YAML with non-mapping root."""
    config_path = tmp_path / "rag_config.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(RagError, match="mapping"):
        PipelineBuilder(str(config_path))


async def test_pipeline_builder_section_requires_mapping(monkeypatch, tmp_path) -> None:
    """_section should reject non-mapping config sections."""
    del monkeypatch
    config_path = tmp_path / "rag_config.yaml"
    config_path.write_text("pipeline: naive\n", encoding="utf-8")
    builder = PipelineBuilder(str(config_path))
    builder.config["llm"] = "not-a-mapping"
    with pytest.raises(RagError, match="must be a mapping"):
        builder._section("llm")


async def test_pipeline_builder_seed_nodes_validates_chunker(tmp_path) -> None:
    """_seed_nodes should reject chunkers without callable chunk()."""
    config_path = tmp_path / "rag_config.yaml"
    config_path.write_text("pipeline: naive\n", encoding="utf-8")
    builder = PipelineBuilder(str(config_path))

    with pytest.raises(RagError, match="does not expose a chunk"):
        builder._seed_nodes(object())

    class _BadChunker:
        def chunk(self, document):
            del document
            return "bad"

    with pytest.raises(RagError, match="returned an invalid value"):
        builder._seed_nodes(_BadChunker())


async def test_pipeline_builder_index_nodes_validates_embedding_and_vectorstore(tmp_path) -> None:
    """_index_nodes should validate embedding output and vectorstore add callable."""
    config_path = tmp_path / "rag_config.yaml"
    config_path.write_text("pipeline: naive\n", encoding="utf-8")
    builder = PipelineBuilder(str(config_path))
    nodes = [Node(node_id="n1", doc_id="d1", content="alpha")]

    class _BadEmbeddingA:
        pass

    with pytest.raises(RagError, match="does not expose embed"):
        await builder._index_nodes(nodes, _BadEmbeddingA(), object())

    class _BadEmbeddingB:
        async def embed(self, texts):
            del texts
            return "bad"

    with pytest.raises(RagError, match="invalid vectors payload"):
        await builder._index_nodes(nodes, _BadEmbeddingB(), object())

    class _GoodEmbedding:
        async def embed(self, texts):
            del texts
            return [[1.0, 0.0]]

    with pytest.raises(RagError, match="does not expose add"):
        await builder._index_nodes(nodes, _GoodEmbedding(), object())

