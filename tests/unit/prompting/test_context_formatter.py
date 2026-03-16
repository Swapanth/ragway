from __future__ import annotations

from ragway.prompting.context_formatter import ContextFormatter
from ragway.schema.node import Node


def test_context_formatter_formats_with_headers() -> None:
    """ContextFormatter should format node content with index headers."""
    formatter = ContextFormatter(include_headers=True)
    nodes = [
        Node(node_id="n1", doc_id="d1", content="First"),
        Node(node_id="n2", doc_id="d1", content="Second"),
    ]

    output = formatter.format(nodes)

    assert "[1] First" in output
    assert "[2] Second" in output


def test_context_formatter_empty_nodes() -> None:
    """ContextFormatter should return empty string for empty nodes list."""
    formatter = ContextFormatter()
    assert formatter.format([]) == ""

