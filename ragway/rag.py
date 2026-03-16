"""Main public RAG class intended for everyday ragway usage."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, AsyncIterator

from ragway.core.config_loader import ConfigLoader
from ragway.evaluation.ragas_eval import RagasEval
from ragway.exceptions import RagError

if TYPE_CHECKING:
    from ragway.raglab import RAGLab


_PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "pinecone": "PINECONE_API_KEY",
    "weaviate": "WEAVIATE_API_KEY",
    "qdrant": "QDRANT_API_KEY",
}


_PROVIDER_EXTRA_HINTS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "mistral": "mistral",
    "groq": "groq",
    "cohere": "cohere",
    "pinecone": "pinecone",
    "weaviate": "weaviate",
    "qdrant": "qdrant",
    "faiss": "faiss",
    "chroma": "chroma",
    "llama": "llama",
    "bge": "bge",
}


class RAG:
    """Main user-facing class for config-free RAG setup and execution."""

    def __init__(
        self,
        pipeline: str = "naive",
        llm: str = "anthropic",
        llm_model: str | None = None,
        vectorstore: str = "faiss",
        retrieval: str = "vector",
        reranker: str | None = "cohere",
        chunking: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 50,
        top_k: int = 5,
        temperature: float = 0.2,
        api_key: str | None = None,
        api_keys: dict[str, str] | None = None,
    ) -> None:
        """Create a lazily initialized RAG instance with ergonomic defaults."""
        self._settings: dict[str, object] = {
            "pipeline": pipeline,
            "llm": llm,
            "llm_model": llm_model,
            "vectorstore": vectorstore,
            "retrieval": retrieval,
            "reranker": reranker,
            "chunking": chunking,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "temperature": temperature,
        }

        self._api_keys: dict[str, str] = {}
        if api_keys is not None:
            for provider, key in api_keys.items():
                self.set_key(provider, key)

        if api_key is not None:
            self.set_key(llm, api_key)

        self._engine: RAGLab | None = None

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        """Safely parse integers from dynamic config values."""
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

    @staticmethod
    def _as_float(value: object, default: float) -> float:
        """Safely parse floats from dynamic config values."""
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (str, bytes, bytearray)):
            try:
                return float(value)
            except ValueError:
                return default
        try:
            float_method = getattr(value, "__float__")
            parsed = float_method()
            return float(parsed) if isinstance(parsed, float) else default
        except Exception:
            return default

    @classmethod
    def from_config(cls, path: str = "rag.yaml") -> RAG:
        """Load from YAML config file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        loaded = ConfigLoader.load(str(config_path))
        return cls.from_dict(loaded)

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> RAG:
        """Build from a Python dict."""
        if "plugins" in config and isinstance(config.get("plugins"), dict):
            merged: dict[str, object] = dict(config)
            plugin_cfg = config.get("plugins")
            assert isinstance(plugin_cfg, dict)
            merged.update(plugin_cfg)
            config = merged

        if any(section in config for section in ["llm", "vectorstore", "retrieval", "reranking", "chunking"]):
            llm_cfg = config.get("llm", {})
            vector_cfg = config.get("vectorstore", {})
            retrieval_cfg = config.get("retrieval", {})
            reranking_cfg = config.get("reranking", {})
            chunking_cfg = config.get("chunking", {})

            llm_provider = str(llm_cfg.get("provider", "anthropic")) if isinstance(llm_cfg, dict) else "anthropic"
            llm_model = str(llm_cfg.get("model")) if isinstance(llm_cfg, dict) and llm_cfg.get("model") else None
            vector_provider = (
                str(vector_cfg.get("provider", "faiss")) if isinstance(vector_cfg, dict) else "faiss"
            )
            retrieval_strategy = (
                str(retrieval_cfg.get("strategy", "vector")) if isinstance(retrieval_cfg, dict) else "vector"
            )
            if isinstance(reranking_cfg, dict):
                rerank_enabled = bool(reranking_cfg.get("enabled", False))
                rerank_provider = str(reranking_cfg.get("provider", "cohere")) if rerank_enabled else None
            else:
                rerank_provider = None
            chunking_strategy = (
                str(chunking_cfg.get("strategy", "recursive")) if isinstance(chunking_cfg, dict) else "recursive"
            )
            chunk_size = cls._as_int(chunking_cfg.get("chunk_size", 512), 512) if isinstance(chunking_cfg, dict) else 512
            overlap = cls._as_int(chunking_cfg.get("overlap", 50), 50) if isinstance(chunking_cfg, dict) else 50
            top_k = cls._as_int(retrieval_cfg.get("top_k", 5), 5) if isinstance(retrieval_cfg, dict) else 5
            temperature = cls._as_float(llm_cfg.get("temperature", 0.2), 0.2) if isinstance(llm_cfg, dict) else 0.2

            rag = cls(
                pipeline=str(config.get("pipeline", "naive")),
                llm=llm_provider,
                llm_model=llm_model,
                vectorstore=vector_provider,
                retrieval=retrieval_strategy,
                reranker=rerank_provider,
                chunking=chunking_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                top_k=top_k,
                temperature=temperature,
            )
            return rag

        return cls(
            pipeline=str(config.get("pipeline", "naive")),
            llm=str(config.get("llm", "anthropic")),
            llm_model=str(config.get("llm_model")) if config.get("llm_model") else None,
            vectorstore=str(config.get("vectorstore", "faiss")),
            retrieval=str(config.get("retrieval", "vector")),
            reranker=str(config.get("reranker")) if config.get("reranker") else None,
            chunking=str(config.get("chunking", "recursive")),
            chunk_size=cls._as_int(config.get("chunk_size", 512), 512),
            overlap=cls._as_int(config.get("overlap", 50), 50),
            top_k=cls._as_int(config.get("top_k", 5), 5),
            temperature=cls._as_float(config.get("temperature", 0.2), 0.2),
        )

    def switch(self, **kwargs: object) -> RAG:
        """Return a new RAG with some settings changed."""
        updated = deepcopy(self._settings)
        updated.update(kwargs)

        switched_keys = deepcopy(self._api_keys)
        if "api_keys" in kwargs and isinstance(kwargs["api_keys"], dict):
            for provider, key in kwargs["api_keys"].items():
                switched_keys[str(provider)] = str(key)

        llm_provider = str(updated.get("llm", self._settings["llm"]))
        if "api_key" in kwargs and kwargs["api_key"] is not None:
            switched_keys[llm_provider] = str(kwargs["api_key"])

        return RAG(
            pipeline=str(updated["pipeline"]),
            llm=llm_provider,
            llm_model=str(updated["llm_model"]) if updated.get("llm_model") else None,
            vectorstore=str(updated["vectorstore"]),
            retrieval=str(updated["retrieval"]),
            reranker=str(updated["reranker"]) if updated.get("reranker") else None,
            chunking=str(updated["chunking"]),
            chunk_size=self._as_int(updated["chunk_size"], 512),
            overlap=self._as_int(updated["overlap"], 50),
            top_k=self._as_int(updated["top_k"], 5),
            temperature=self._as_float(updated["temperature"], 0.2),
            api_keys=switched_keys,
        )

    async def ingest(self, source: str, glob: str = "**/*") -> int:
        """Ingest documents. Returns number of chunks stored."""
        engine = self._ensure_engine()
        source_path = Path(source)
        if source_path.exists() and source_path.is_dir() and glob != "**/*":
            ingested = 0
            for item in source_path.glob(glob):
                if item.is_file():
                    ingested += await engine.ingest(str(item))
            return ingested
        return await engine.ingest(source)

    async def query(self, question: str, stream: bool = False) -> str | AsyncIterator[str]:
        """Run a question through the pipeline. Returns answer string."""
        engine = self._ensure_engine()
        return await engine.query(question, stream=stream)

    async def query_with_sources(self, question: str) -> dict[str, object]:
        """Returns: {answer, sources, scores, latency_ms}."""
        engine = self._ensure_engine()
        start = perf_counter()
        payload = await engine.query_with_sources(question)
        latency_ms = (perf_counter() - start) * 1000.0

        sources = payload.get("sources", []) if isinstance(payload, dict) else []
        if not isinstance(sources, list):
            sources = []

        scores: list[float] = []
        return {
            "answer": str(payload.get("answer", "")) if isinstance(payload, dict) else "",
            "sources": sources,
            "scores": scores,
            "latency_ms": latency_ms,
        }

    async def evaluate(self, dataset: str | list[dict[str, object]], metrics: list[str] | None = None) -> dict[str, float]:
        """Evaluate pipeline. Returns metric_name -> score dict."""
        if isinstance(dataset, str):
            engine = self._ensure_engine()
            results = await engine.evaluate(dataset)
        else:
            rows: list[dict[str, object]] = []
            for item in dataset:
                question = str(item.get("question", ""))
                gold_answer = str(item.get("gold_answer", ""))
                response = await self.query_with_sources(question)
                contexts: list[str] = []
                raw_sources = response.get("sources", [])
                source_items = raw_sources if isinstance(raw_sources, list) else []
                for source in source_items:
                    if isinstance(source, dict):
                        contexts.append(str(source.get("content", "")))

                rows.append(
                    {
                        "question": question,
                        "answer": str(response.get("answer", "")),
                        "gold_answer": gold_answer,
                        "context": contexts,
                    }
                )

            evaluator = RagasEval()
            results = evaluator.run(rows, pipeline_name=str(self._settings["pipeline"]))

        if metrics is None:
            return results

        allowed = set(metrics)
        return {name: score for name, score in results.items() if name in allowed}

    def set_key(self, provider: str, key: str) -> None:
        """Set one provider API key in process environment."""
        provider_name = provider.strip().lower()
        env_name = _PROVIDER_ENV_KEYS.get(provider_name)
        if env_name is None:
            raise ValueError(f"Unknown provider for API key: {provider}")

        os.environ[env_name] = key
        self._api_keys[provider_name] = key

    def set_keys(self, **kwargs: object) -> None:
        """Set multiple provider API keys in process environment."""
        for provider, key in kwargs.items():
            self.set_key(str(provider), str(key))

    def _build_config(self) -> dict[str, object]:
        """Convert constructor settings into structured RAGLab config."""
        llm_cfg: dict[str, object] = {
            "provider": str(self._settings["llm"]),
            "temperature": self._as_float(self._settings["temperature"], 0.2),
            "max_tokens": 1024,
        }
        llm_model = self._settings.get("llm_model")
        if isinstance(llm_model, str) and llm_model:
            llm_cfg["model"] = llm_model

        reranker = self._settings.get("reranker")
        reranking_cfg: dict[str, object]
        if isinstance(reranker, str) and reranker:
            # Keep minimal UX working when users only configure an LLM key.
            if reranker == "cohere" and not os.getenv("COHERE_API_KEY"):
                reranking_cfg = {"enabled": False}
            else:
                reranking_cfg = {
                    "enabled": True,
                    "provider": reranker,
                    "top_k": self._as_int(self._settings["top_k"], 5),
                }
        else:
            reranking_cfg = {"enabled": False}

        return {
            "pipeline": str(self._settings["pipeline"]),
            "llm": llm_cfg,
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "batch_size": 32,
            },
            "vectorstore": {
                "provider": str(self._settings["vectorstore"]),
                "index_name": "rag-lab",
                "top_k": self._as_int(self._settings["top_k"], 5),
            },
            "retrieval": {
                "strategy": str(self._settings["retrieval"]),
                "top_k": self._as_int(self._settings["top_k"], 5),
            },
            "reranking": reranking_cfg,
            "chunking": {
                "strategy": str(self._settings["chunking"]),
                "chunk_size": self._as_int(self._settings["chunk_size"], 512),
                "overlap": self._as_int(self._settings["overlap"], 50),
            },
        }

    def _ensure_engine(self) -> RAGLab:
        """Build internal engine on first use to keep import and init lightweight."""
        if self._engine is not None:
            return self._engine

        config = self._build_config()
        try:
            from ragway.raglab import RAGLab

            self._engine = RAGLab.from_dict(config)
            return self._engine
        except ImportError as exc:
            raise self._dependency_error(exc) from exc
        except ModuleNotFoundError as exc:
            raise self._dependency_error(exc) from exc
        except RagError as exc:
            if "required" in str(exc).lower() or "install" in str(exc).lower():
                raise self._dependency_error(exc) from exc
            raise

    def _dependency_error(self, exc: Exception) -> ImportError:
        """Translate provider dependency issues into actionable pip-install hints."""
        text = str(exc).lower()

        candidates = [
            str(self._settings.get("vectorstore", "")).lower(),
            str(self._settings.get("llm", "")).lower(),
            str(self._settings.get("reranker", "")).lower(),
        ]

        for provider in candidates:
            if not provider:
                continue
            extra = _PROVIDER_EXTRA_HINTS.get(provider, provider)
            if provider in text or "required" in text or "not installed" in text:
                name = provider.capitalize()
                return ImportError(f"{name} not installed. Run: pip install ragway[{extra}]")

        return ImportError(f"Missing optional dependency. Install the needed extra package. Details: {exc}")
