from __future__ import annotations

import os

import pytest

from ragway.core.config_loader import ConfigLoader
from ragway.exceptions import RagError


def test_resolves_env_vars(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """${VAR} placeholders should resolve against current environment values."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-value")

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        """
version: "1.0"
pipeline: naive
plugins:
  llm:
    provider: anthropic
    api_key: ${ANTHROPIC_API_KEY}
  retrieval:
    strategy: vector
""".strip(),
        encoding="utf-8",
    )

    loaded = ConfigLoader.load(str(config_path))
    plugins = loaded.get("plugins")
    assert isinstance(plugins, dict)
    llm = plugins.get("llm")
    assert isinstance(llm, dict)
    assert "api_key" not in llm
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-env-value"


def test_injects_api_keys(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """api_key values under known plugins should be injected into os.environ."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        """
version: "1.0"
pipeline: naive
plugins:
  llm:
    provider: mistral
    api_key: sk-mistral-inline
  retrieval:
    strategy: vector
""".strip(),
        encoding="utf-8",
    )

    ConfigLoader.load(str(config_path))
    assert os.environ.get("MISTRAL_API_KEY") == "sk-mistral-inline"


def test_removes_keys_from_config(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw api_key entries should be removed from config payload after injection."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        """
version: "1.0"
pipeline: hybrid
plugins:
  reranker:
    enabled: true
    provider: cohere
    api_key: cohere-inline
""".strip(),
        encoding="utf-8",
    )

    loaded = ConfigLoader.load(str(config_path))
    plugins = loaded.get("plugins")
    assert isinstance(plugins, dict)
    reranker = plugins.get("reranker")
    assert isinstance(reranker, dict)
    assert "api_key" not in reranker


def test_missing_env_var_raises_error(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing env vars referenced by ${VAR} should raise a helpful RagError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        """
version: "1.0"
pipeline: naive
plugins:
  llm:
    provider: anthropic
    api_key: ${ANTHROPIC_API_KEY}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RagError, match=r"Config references \$\{ANTHROPIC_API_KEY\} but it is not set"):
        ConfigLoader.load(str(config_path))


def test_direct_key_works(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct api_key values in YAML should inject without env substitution."""
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        """
version: "1.0"
pipeline: naive
plugins:
  vectorstore:
    provider: pinecone
    api_key: pinecone-inline
""".strip(),
        encoding="utf-8",
    )

    ConfigLoader.load(str(config_path))
    assert os.environ.get("PINECONE_API_KEY") == "pinecone-inline"


def test_env_var_takes_priority(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing env vars should not be overwritten by api_key values from YAML."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-priority-key")

    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        """
version: "1.0"
pipeline: naive
plugins:
  embedding:
    provider: openai
    api_key: yaml-key-that-should-not-win
""".strip(),
        encoding="utf-8",
    )

    ConfigLoader.load(str(config_path))
    assert os.environ.get("OPENAI_API_KEY") == "env-priority-key"
