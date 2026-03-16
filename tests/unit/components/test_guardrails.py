from __future__ import annotations

from ragway.components.guardrails import Guardrails


def test_guardrails_flags_blocked_terms() -> None:
    """Guardrails should reject text containing blocked terms."""
    guardrails = Guardrails(blocked_terms={"secret"})
    is_safe, reason = guardrails.check_input("This has a secret inside")

    assert not is_safe
    assert "blocked term" in reason


def test_guardrails_accepts_clean_text() -> None:
    """Guardrails should allow safe text that passes checks."""
    guardrails = Guardrails(blocked_terms={"secret"})
    is_safe, reason = guardrails.check_output("Safe response")

    assert is_safe
    assert reason == "ok"

