from __future__ import annotations

from ragway.components.context_compression import ContextCompression
from ragway.schema.node import Node


def test_context_compression_truncates_to_token_limit() -> None:
    """ContextCompression should limit output to configured token count."""
    component = ContextCompression(token_limit=5)
    nodes = [
        Node(node_id="n1", doc_id="d1", content="one two three"),
        Node(node_id="n2", doc_id="d1", content="four five six seven"),
    ]

    output = component.compress(nodes)

    assert output == "one two three four five"


def test_context_compression_empty_nodes() -> None:
    """ContextCompression should return empty string for empty nodes."""
    component = ContextCompression(token_limit=5)
    assert component.compress([]) == ""

