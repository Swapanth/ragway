from __future__ import annotations

from ragway.prompting.citation_formatter import CitationFormatter
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node


def test_citation_formatter_adds_source_tags() -> None:
    """CitationFormatter should append source citation to each node block."""
    formatter = CitationFormatter()
    nodes = [
        Node(node_id="n1", doc_id="d1", content="Fact one", metadata=Metadata(source="doc-a")),
        Node(node_id="n2", doc_id="d1", content="Fact two", metadata=Metadata(source="doc-b")),
    ]

    output = formatter.format(nodes)

    assert "Fact one [doc-a]" in output
    assert "Fact two [doc-b]" in output


def test_citation_formatter_uses_unknown_when_missing_source() -> None:
    """CitationFormatter should use unknown source placeholder when missing."""
    formatter = CitationFormatter(unknown_source="unknown")
    node = Node(node_id="n1", doc_id="d1", content="Fact")

    output = formatter.format([node])

    assert "[unknown]" in output

