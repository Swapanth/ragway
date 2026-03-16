from __future__ import annotations

from ragway.chunking.hierarchical_chunker import HierarchicalChunker
from ragway.schema.document import Document


def test_hierarchical_chunker_outputs_parent_and_children() -> None:
    """Hierarchical chunker should create parent nodes and child nodes with links."""
    text = "one two three four five six seven eight nine ten"
    document = Document(doc_id="doc-1", content=text)
    chunker = HierarchicalChunker(parent_chunk_size=5, child_chunk_size=3, child_overlap=1)

    nodes = chunker.chunk(document)

    parent_nodes = [node for node in nodes if node.parent_id is None]
    child_nodes = [node for node in nodes if node.parent_id is not None]

    assert len(parent_nodes) == 2
    assert len(child_nodes) > 0
    assert all(child.parent_id in {parent.node_id for parent in parent_nodes} for child in child_nodes)

