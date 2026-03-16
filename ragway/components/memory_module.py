"""Conversation memory component for storing and retrieving dialogue turns."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragway.validators import validate_positive_int


@dataclass(slots=True)
class MemoryModule:
    """Stores recent conversation history with a bounded turn buffer."""

    max_turns: int = 20
    _turns: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate memory buffer size."""
        self.max_turns = validate_positive_int(self.max_turns, "max_turns")

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Append one conversation turn and trim to max_turns."""
        self._turns.append((user_message, assistant_message))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def get_recent(self) -> list[tuple[str, str]]:
        """Return a copy of recent conversation turns."""
        return list(self._turns)

    def clear(self) -> None:
        """Remove all stored conversation turns."""
        self._turns.clear()

    def to_prompt_history(self) -> str:
        """Render conversation turns into a prompt-friendly history string."""
        lines: list[str] = []
        for user_message, assistant_message in self._turns:
            lines.append(f"User: {user_message}")
            lines.append(f"Assistant: {assistant_message}")
        return "\n".join(lines)

