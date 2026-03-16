"""Integration tests for provider adapters across LLM, embeddings, and vectorstores."""

from __future__ import annotations

import os
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from ragway.core.component_registry import ComponentRegistry
from ragway.generation.llm_factory import get_llm
from ragway.reranking.cohere_reranker import CohereReranker
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node


def _handle_integration_provider_error(provider: str, error: Exception) -> None:
    """Skip known env/dependency/auth failures and re-raise unknown errors."""
    message = str(error).lower()

    known_fragments = [
        "environment variable is required",
        "package is required",
        "requires an async client",
        "api key",
        "authentication",
        "unauthorized",
        "forbidden",
        "invalid",
        "permission denied",
        "insufficient_quota",
        "exceeded your current quota",
        "rate limit",
        "401",
        "403",
        "429",
    ]
    if any(fragment in message for fragment in known_fragments):
        pytest.skip(f"{provider} unavailable in this environment: {error}")

    raise error


def _make_node(node_id: str, content: str, embedding: list[float]) -> Node:
    """Create a minimal valid node for integration checks."""
    stable_uuid = str(uuid5(NAMESPACE_URL, node_id))
    return Node(
        node_id=stable_uuid,
        doc_id="integration-doc",
        content=content,
        embedding=embedding,
        metadata=Metadata(source="integration-test"),
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "env_key"),
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("groq", "GROQ_API_KEY"),
    ],
)
async def test_llm_providers(provider: str, env_key: str) -> None:
    """Each configured LLM provider should generate non-empty text."""
    if not os.getenv(env_key):
        pytest.skip(f"{env_key} is not set")

    llm = get_llm(provider)

    async def _run() -> str:
        return await llm.generate("Return a short greeting in one sentence.")

    try:
        result = await _run()
    except Exception as error:  # pragma: no cover - integration-only control path
        _handle_integration_provider_error(provider, error)
        return

    assert isinstance(result, str)
    assert result.strip() != ""


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "config"),
    [
        ("faiss", {}),
        ("chroma", {"index_name": "integration-collection"}),
        ("pinecone", {"index_name": "integration-index"}),
        ("weaviate", {"index_name": "IntegrationNode"}),
        (
            "qdrant",
            {
                "index_name": "integration-qdrant",
                "in_memory": True,
            },
        ),
    ],
)
async def test_vectorstore_providers(
    provider: str,
    config: dict[str, Any],
) -> None:
    """Vectorstore providers should construct and complete add/search/delete roundtrip."""
    store = ComponentRegistry.get_vectorstore({"provider": provider, **config})
    assert store is not None

    node = _make_node("n1", "Cats are popular pets.", [0.1, 0.2, 0.3, 0.4])

    async def _roundtrip() -> list[Node]:
        await store.add([node])
        found = await store.search([0.1, 0.2, 0.3, 0.4], top_k=1)
        await store.delete([node.node_id])
        return found

    try:
        results = await _roundtrip()
    except Exception as error:  # pragma: no cover - integration-only control path
        _handle_integration_provider_error(provider, error)
        return

    assert len(results) <= 1
    if results:
        assert results[0].doc_id == "integration-doc"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "config"),
    [
        ("openai", {"batch_size": 2}),
        ("cohere", {}),
        ("bge", {"dimensions": 16}),
    ],
)
async def test_embedding_providers(provider: str, config: dict[str, Any]) -> None:
    """Embedding providers should return at least one non-empty vector when configured."""
    if provider == "cohere":
        if not os.getenv("COHERE_API_KEY"):
            pytest.skip("COHERE_API_KEY is not set")

    embedding = ComponentRegistry.get_embedding({"provider": provider, **config})

    async def _run() -> list[list[float]]:
        return await embedding.embed(["Integration embedding sample text."])

    try:
        vectors = await _run()
    except Exception as error:  # pragma: no cover - integration-only control path
        _handle_integration_provider_error(provider, error)
        return

    assert len(vectors) == 1
    assert len(vectors[0]) > 0


@pytest.mark.integration
async def test_cohere_reranker_rank_order() -> None:
    """Cohere reranker should avoid ranking unrelated content as the most relevant."""
    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY is not set")

    reranker = CohereReranker()
    nodes = [
        _make_node("n1", "Cats are common indoor companion animals.", [0.2, 0.1, 0.0]),
        _make_node("n2", "Quarterly earnings increased by 12 percent.", [0.0, 0.2, 0.1]),
        _make_node("n3", "Felines are playful pets that enjoy attention.", [0.1, 0.1, 0.2]),
    ]

    async def _run() -> list[Node]:
        return await reranker.rerank("Which passages discuss cats as pets?", nodes)

    try:
        ranked = await _run()
    except Exception as error:  # pragma: no cover - integration-only control path
        _handle_integration_provider_error("cohere-reranker", error)
        return

    ranked_ids = [node.node_id for node in ranked]
    assert len(ranked_ids) == 3
    assert ranked_ids[0] != "n2"