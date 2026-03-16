"""Factory helpers for selecting LLM adapters by name."""

from __future__ import annotations

from ragway.generation.anthropic_llm import AnthropicLLM
from ragway.generation.azure_openai_llm import AzureOpenAILLM
from ragway.generation.bedrock_llm import BedrockLLM
from ragway.generation.groq_llm import GroqLLM
from ragway.generation.llama_llm import LlamaLLM
from ragway.generation.local_llm import LocalLLM
from ragway.generation.mistral_llm import MistralLLM
from ragway.generation.openai_llm import OpenAILLM
from ragway.generation.vertex_ai_llm import VertexAILLM
from ragway.interfaces.llm_protocol import LLMProtocol


def get_llm(name: str) -> LLMProtocol:
    """Return a concrete LLM adapter instance for a supported provider name."""
    options: dict[str, type[LLMProtocol]] = {
        "anthropic": AnthropicLLM,
        "openai": OpenAILLM,
        "llama": LlamaLLM,
        "local": LocalLLM,
        "mistral": MistralLLM,
        "groq": GroqLLM,
        "vertex_ai": VertexAILLM,
        "azure_openai": AzureOpenAILLM,
        "bedrock": BedrockLLM,
    }
    if name not in options:
        raise ValueError(f"Unknown LLM: {name}. Choose from {list(options)}")
    return options[name]()

