"""Agentic RAG pipeline with tool-using iterative reasoning loop."""

from __future__ import annotations

import asyncio
import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ragway.chunking.fixed_chunker import FixedChunker
from ragway.components.memory_module import MemoryModule
from ragway.core.dependency_container import DependencyContainer
from ragway.core.rag_engine import RagConfig
from ragway.core.rag_pipeline import RAGPipeline
from ragway.embeddings.openai_embedding import OpenAIEmbedding
from ragway.generation.anthropic_llm import AnthropicLLM
from ragway.generation.base_llm import LLMConfig
from ragway.interfaces.llm_protocol import LLMProtocol
from ragway.schema.document import Document
from ragway.schema.node import Node
from ragway.vectorstores.faiss_store import FAISSStore


logger = logging.getLogger(__name__)


class ToolProtocol(Protocol):
    """Protocol for tools callable by the agent loop."""

    name: str
    description: str

    async def run(self, tool_input: str) -> str:
        """Execute the tool for the given input and return text output."""


@dataclass(slots=True)
class ToolRegistry:
    """Registry holding named tools used by the agent."""

    _tools: dict[str, ToolProtocol] = field(default_factory=dict)

    def register(self, tool: ToolProtocol) -> None:
        """Register one tool by its canonical name."""
        self._tools[tool.name] = tool

    def list_descriptions(self) -> str:
        """Return tool descriptions for prompt injection."""
        lines = [f"- {tool.name}: {tool.description}" for tool in self._tools.values()]
        return "\n".join(lines)

    async def run_tool(self, name: str, tool_input: str) -> str:
        """Execute a registered tool by name."""
        if name not in self._tools:
            return f"Tool not found: {name}"
        return await self._tools[name].run(tool_input)


@dataclass(slots=True)
class LocalOpenAIEmbeddingClient:
    """Local async embedding client used for runnable pipeline demos."""

    dimensions: int = 8

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Return deterministic vectors for each text to simulate embeddings."""
        del model
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0 for _ in range(self.dimensions)]
            for index, character in enumerate(text):
                bucket = index % self.dimensions
                vector[bucket] += (ord(character) % 29) / 29.0
            vectors.append(vector)
        return vectors


@dataclass(slots=True)
class LocalAgentAnthropicClient:
    """Local async Anthropic-compatible client for agent action planning."""

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
    ) -> str:
        """Return tool action or answer based on prompt and memory state."""
        del model, temperature, max_tokens, api_key

        lowered = prompt.lower()
        has_tool_result = "tool_result:" in lowered

        query_text = ""
        for line in prompt.splitlines():
            if line.lower().startswith("query:"):
                query_text = line.split(":", maxsplit=1)[1].strip()
                break
        query_lower = query_text.lower()

        if not has_tool_result:
            if any(token in query_lower for token in ["+", "-", "*", "/", "(", ")"]) and any(
                ch.isdigit() for ch in query_lower
            ):
                return f"TOOL:calculator|{query_text}"
            if any(word in query_lower for word in ["web", "latest", "news", "today"]):
                return f"TOOL:search|{query_text}"
            return f"TOOL:retrieval|{query_text}"

        return "ANSWER:Final answer synthesized from retrieved tool results."


@dataclass(slots=True)
class RetrievalTool:
    """Tool that retrieves relevant nodes from vector store."""

    name: str = "retrieval"
    description: str = "Searches indexed documents and returns relevant context."
    embedding_model: OpenAIEmbedding | None = None
    vector_store: FAISSStore | None = None
    top_k: int = 5

    async def run(self, tool_input: str) -> str:
        """Run vector search for the tool input query."""
        if self.embedding_model is None or self.vector_store is None:
            return "retrieval tool unavailable"

        vector = (await self.embedding_model.embed([tool_input]))[0]
        nodes = await self.vector_store.search(vector, self.top_k)
        if not nodes:
            return "No relevant context found."
        return "\n".join(node.content for node in nodes)


@dataclass(slots=True)
class SearchTool:
    """Mock web search fallback tool."""

    name: str = "search"
    description: str = "Performs a web search fallback for external knowledge."

    async def run(self, tool_input: str) -> str:
        """Return mocked web search output."""
        return f"Mock search result: top web snippets about '{tool_input}'."


@dataclass(slots=True)
class CalculatorTool:
    """Tool that evaluates safe arithmetic expressions."""

    name: str = "calculator"
    description: str = "Evaluates arithmetic expressions."

    async def run(self, tool_input: str) -> str:
        """Evaluate arithmetic expression and return numeric output."""
        expression = tool_input.strip()
        if expression.lower().startswith("calculate"):
            expression = expression.split(" ", maxsplit=1)[1] if " " in expression else ""
        if not expression:
            return "No expression provided."

        try:
            value = self._safe_eval(expression)
            return f"{value}"
        except Exception:
            return "Invalid arithmetic expression."

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate arithmetic AST limited to numeric operations."""
        node = ast.parse(expression, mode="eval")
        return float(self._eval_node(node.body))

    def _eval_node(self, node: ast.AST) -> float:
        """Evaluate allowed AST node types recursively."""
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Unsupported expression")


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from .env into process environment."""
    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


async def _seed_vectorstore(embedding_model: OpenAIEmbedding, vector_store: FAISSStore) -> None:
    """Seed vector store with baseline chunks used by retrieval tool."""
    document = Document(
        doc_id="doc-001",
        content=(
            "Agentic RAG uses tools to gather evidence before answering. "
            "A tool loop can call retrieval, web search fallback, and calculators. "
            "Memory stores tool calls and outputs for iterative decision making."
        ),
    )
    chunker = FixedChunker(chunk_size=512, overlap=50)
    nodes = chunker.chunk(document)

    vectors = await embedding_model.embed([node.content for node in nodes])
    nodes_with_embeddings = [
        node.model_copy(update={"embedding": vector})
        for node, vector in zip(nodes, vectors)
    ]
    await vector_store.add(nodes_with_embeddings)


async def _build_container(llm: LLMProtocol | None = None) -> DependencyContainer:
    """Construct and register dependencies for agentic pipeline."""
    _load_env_file()

    container = DependencyContainer()
    embedding_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        max_batch_size=32,
        client=LocalOpenAIEmbeddingClient(),
    )
    vector_store = FAISSStore()
    await _seed_vectorstore(embedding_model=embedding_model, vector_store=vector_store)

    llm_instance = llm or AnthropicLLM(
        config=LLMConfig(model="claude-sonnet-4-6", temperature=0.0, max_tokens=512),
        client=LocalAgentAnthropicClient(),
    )

    tool_registry = ToolRegistry()
    tool_registry.register(RetrievalTool(embedding_model=embedding_model, vector_store=vector_store, top_k=5))
    tool_registry.register(SearchTool())
    tool_registry.register(CalculatorTool())

    memory = MemoryModule(max_turns=30)

    container.register_instance("llm", llm_instance)
    container.register_instance("tool_registry", tool_registry)
    container.register_instance("memory", memory)
    return container


def build_pipeline(
    llm: LLMProtocol | None = None,
    vectorstore: object | None = None,
    embedding: object | None = None,
    retriever: object | None = None,
    reranker: object | None = None,
    chunker: object | None = None,
) -> RAGPipeline:
    """Build and return the agentic RAG pipeline definition."""
    llm = llm or AnthropicLLM()
    _ = (llm, vectorstore, embedding, retriever, reranker, chunker)
    return RAGPipeline(
        name="agentic",
        config=RagConfig(top_k=5, enable_rerank=False, include_citations=False),
    )


async def run(
    query: str,
    llm: LLMProtocol | None = None,
    vectorstore: object | None = None,
    embedding: object | None = None,
    retriever: object | None = None,
    reranker: object | None = None,
    chunker: object | None = None,
) -> dict:
    """Run agentic loop and return answer with tool/iteration metadata."""
    _ = (vectorstore, embedding, retriever, reranker, chunker)
    container = await _build_container(llm=llm)
    pipeline = build_pipeline(llm=llm)
    _ = pipeline

    llm = container.resolve("llm")
    tool_registry = container.resolve("tool_registry")
    memory = container.resolve("memory")

    max_iterations = 5
    tool_calls_made = 0
    best_answer = "No final answer produced."

    for iteration in range(1, max_iterations + 1):
        memory_text = memory.to_prompt_history()
        prompt = (
            "You are an agent that can use tools. Decide one action.\n"
            "Output either:\n"
            "1) TOOL:<tool_name>|<tool_input>\n"
            "2) ANSWER:<final_answer>\n\n"
            f"Available tools:\n{tool_registry.list_descriptions()}\n\n"
            f"Query: {query}\n\n"
            f"Memory:\n{memory_text}\n"
        )

        decision = (await llm.generate(prompt)).strip()
        if decision.upper().startswith("ANSWER:"):
            best_answer = decision.split(":", maxsplit=1)[1].strip() or best_answer
            return {
                "answer": best_answer,
                "tool_calls_made": tool_calls_made,
                "iterations_used": iteration,
            }

        if decision.upper().startswith("TOOL:"):
            payload = decision.split(":", maxsplit=1)[1]
            if "|" in payload:
                tool_name, tool_input = payload.split("|", maxsplit=1)
            else:
                tool_name, tool_input = payload, query
            tool_name = tool_name.strip().lower()
            tool_input = tool_input.strip()

            tool_result = await tool_registry.run_tool(tool_name, tool_input)
            memory.add_turn(
                user_message=f"TOOL_CALL: {tool_name}({tool_input})",
                assistant_message=f"TOOL_RESULT: {tool_result}",
            )
            tool_calls_made += 1
            continue

        memory.add_turn(user_message="UNPARSEABLE_DECISION", assistant_message=decision)

    if memory.get_recent():
        best_answer = memory.get_recent()[-1][1]

    return {
        "answer": best_answer,
        "tool_calls_made": tool_calls_made,
        "iterations_used": max_iterations,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run("What is agentic rag and what is 12 * (3 + 1)?"))
    logger.info("Answer: %s", result["answer"])
    logger.info("Tool calls: %s", result["tool_calls_made"])
    logger.info("Iterations: %s", result["iterations_used"])

