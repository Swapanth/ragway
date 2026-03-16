from __future__ import annotations

from ragway.components.memory_module import MemoryModule


def test_memory_module_keeps_recent_turns() -> None:
    """MemoryModule should keep only the most recent max_turns entries."""
    memory = MemoryModule(max_turns=2)
    memory.add_turn("u1", "a1")
    memory.add_turn("u2", "a2")
    memory.add_turn("u3", "a3")

    assert memory.get_recent() == [("u2", "a2"), ("u3", "a3")]


def test_memory_module_prompt_history_render() -> None:
    """MemoryModule should render turns into prompt history text."""
    memory = MemoryModule(max_turns=5)
    memory.add_turn("hello", "hi")

    history = memory.to_prompt_history()

    assert "User: hello" in history
    assert "Assistant: hi" in history

