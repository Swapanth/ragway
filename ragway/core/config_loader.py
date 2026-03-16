"""Configuration loader with environment variable resolution and API key injection."""

from __future__ import annotations

import os
import re
from typing import Any

import yaml

from ragway.exceptions import RagError


class ConfigLoader:
    """Load YAML config, resolve env substitutions, and inject provider API keys."""

    _PLUGIN_ENV_MAP: dict[tuple[str, str], str] = {
        ("llm", "anthropic"): "ANTHROPIC_API_KEY",
        ("llm", "openai"): "OPENAI_API_KEY",
        ("llm", "mistral"): "MISTRAL_API_KEY",
        ("llm", "groq"): "GROQ_API_KEY",
        ("embedding", "openai"): "OPENAI_API_KEY",
        ("embedding", "cohere"): "COHERE_API_KEY",
        ("vectorstore", "pinecone"): "PINECONE_API_KEY",
        ("vectorstore", "weaviate"): "WEAVIATE_API_KEY",
        ("vectorstore", "qdrant"): "QDRANT_API_KEY",
        ("reranker", "cohere"): "COHERE_API_KEY",
    }

    @classmethod
    def load(cls, path: str) -> dict[str, object]:
        """Read config file, resolve substitutions, and expose API keys via env vars."""
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()

        resolved = cls._resolve_env_vars(raw)
        loaded = yaml.safe_load(resolved) or {}
        if not isinstance(loaded, dict):
            raise RagError("Config file must contain a mapping at the root")

        config: dict[str, object] = loaded
        cls._inject_api_keys(config)
        return config

    @classmethod
    def _resolve_env_vars(cls, raw: str) -> str:
        """Replace ${VAR_NAME} placeholders with values from os.environ."""

        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            value = os.environ.get(var)
            if not value:
                raise RagError(
                    f"Config references ${{{var}}} but it is not set in environment. "
                    f"Set it with:\n"
                    f"  export {var}=your-key\n"
                    f"Or put the key directly in rag.yaml."
                )
            return value

        return re.sub(r"\$\{([^}]+)\}", _replace, raw)

    @classmethod
    def _inject_api_keys(cls, config: dict[str, object]) -> None:
        """Move api_key values from plugin sections into environment variables."""
        plugins_raw = config.get("plugins")
        if not isinstance(plugins_raw, dict):
            return

        plugins: dict[str, Any] = plugins_raw
        for plugin_name in ["llm", "embedding", "vectorstore", "reranker"]:
            section_raw = plugins.get(plugin_name)
            if not isinstance(section_raw, dict):
                continue

            section: dict[str, Any] = section_raw
            provider = str(section.get("provider", "")).strip().lower()
            api_key = section.get("api_key")

            if not isinstance(api_key, str) or not api_key.strip():
                section.pop("api_key", None)
                continue

            env_name = cls._PLUGIN_ENV_MAP.get((plugin_name, provider))
            if env_name and not os.environ.get(env_name):
                os.environ[env_name] = api_key.strip()

            section.pop("api_key", None)
