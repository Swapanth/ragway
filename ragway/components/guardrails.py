"""Guardrails component for basic input and output safety checks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Guardrails:
    """Apply lightweight policy checks to user input and model output."""

    blocked_terms: set[str] = field(default_factory=lambda: {"drop table", "api_key", "password"})
    max_input_chars: int = 4000
    max_output_chars: int = 8000

    def check_input(self, text: str) -> tuple[bool, str]:
        """Validate input text against blocked terms and size constraints."""
        return self._check_text(text=text, max_chars=self.max_input_chars, channel="input")

    def check_output(self, text: str) -> tuple[bool, str]:
        """Validate output text against blocked terms and size constraints."""
        return self._check_text(text=text, max_chars=self.max_output_chars, channel="output")

    def _check_text(self, text: str, max_chars: int, channel: str) -> tuple[bool, str]:
        """Run common checks and return (is_safe, reason)."""
        if len(text) > max_chars:
            return False, f"{channel} exceeds maximum character limit"

        lowered = text.lower()
        for term in self.blocked_terms:
            if term in lowered:
                return False, f"{channel} contains blocked term: {term}"

        return True, "ok"
