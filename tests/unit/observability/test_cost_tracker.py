from __future__ import annotations

import pytest

from ragway.exceptions import ValidationError
from ragway.observability.cost_tracker import CostTracker


def test_cost_tracker_accumulates_supported_model_costs() -> None:
    """CostTracker should accumulate estimated costs for supported models."""
    tracker = CostTracker()

    tracker.track_tokens("gpt-4o", input_tokens=1000, output_tokens=500)
    tracker.track_tokens("claude-sonnet-4-6", input_tokens=1000, output_tokens=1000)

    expected = (1000 / 1_000_000) * 5.0 + (500 / 1_000_000) * 15.0
    expected += (1000 / 1_000_000) * 3.0 + (1000 / 1_000_000) * 15.0

    assert tracker.total_cost() == pytest.approx(expected)


def test_cost_tracker_rejects_unsupported_model() -> None:
    """CostTracker should raise ValidationError for unknown models."""
    tracker = CostTracker()

    with pytest.raises(ValidationError):
        tracker.track_tokens("unknown-model", input_tokens=1, output_tokens=1)


def test_cost_tracker_rejects_negative_tokens() -> None:
    """CostTracker should raise ValidationError for negative token values."""
    tracker = CostTracker()

    with pytest.raises(ValidationError):
        tracker.track_tokens("gpt-4o", input_tokens=-1, output_tokens=1)

