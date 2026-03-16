"""API token cost estimation utilities for RAG pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragway.exceptions import ValidationError

_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "gpt-4o": (5.0, 15.0),
}


@dataclass(slots=True)
class CostTracker:
    """Accumulates estimated LLM token costs across one or more runs."""

    _line_items: list[float] = field(default_factory=list)

    def track_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record a billable token usage event for a supported model."""
        if model not in _PRICING_PER_MILLION_TOKENS:
            supported = ", ".join(sorted(_PRICING_PER_MILLION_TOKENS))
            raise ValidationError(f"Unsupported model '{model}'. Supported models: {supported}.")

        if input_tokens < 0 or output_tokens < 0:
            raise ValidationError("Token counts must be non-negative integers.")

        input_rate, output_rate = _PRICING_PER_MILLION_TOKENS[model]
        input_cost = (float(input_tokens) / 1_000_000.0) * input_rate
        output_cost = (float(output_tokens) / 1_000_000.0) * output_rate
        self._line_items.append(input_cost + output_cost)

    def total_cost(self) -> float:
        """Return the cumulative estimated dollar cost across tracked events."""
        return float(sum(self._line_items))

