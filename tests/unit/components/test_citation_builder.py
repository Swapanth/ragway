from __future__ import annotations

from ragway.components.citation_builder import CitationBuilder
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node


def test_citation_builder_maps_sentence_to_best_source() -> None:
    """CitationBuilder should map each answer sentence to best matching source."""
    builder = CitationBuilder()
    nodes = [
        Node(node_id="n1", doc_id="d1", content="Paris is the capital of France.", metadata=Metadata(source="wiki-fr")),
        Node(node_id="n2", doc_id="d2", content="Berlin is the capital of Germany.", metadata=Metadata(source="wiki-de")),
    ]

    citations = builder.build("Paris is the capital.", nodes)

    assert citations["Paris is the capital."] == "wiki-fr"

