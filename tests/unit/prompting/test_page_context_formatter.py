from __future__ import annotations

from ragway.prompting.page_context_formatter import PageContextFormatter
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node


def test_page_context_formatter_includes_page_number() -> None:
    """PageContextFormatter should include page number in formatted output."""
    formatter = PageContextFormatter()
    node = Node(
        node_id="n1",
        doc_id="d1",
        content="Page content",
        metadata=Metadata(attributes={"page": 7}),
    )

    output = formatter.format([node])

    assert "(page 7)" in output


def test_page_context_formatter_unknown_page() -> None:
    """PageContextFormatter should use unknown marker when page is absent."""
    formatter = PageContextFormatter(unknown_page="?")
    node = Node(node_id="n1", doc_id="d1", content="Page content")

    output = formatter.format([node])

    assert "(page ?)" in output

