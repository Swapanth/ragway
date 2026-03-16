from __future__ import annotations

import pytest

from ragway.chunking.base_chunker import BaseChunker
from ragway.schema.document import Document
from ragway.schema.node import Node


class _NoImplChunker(BaseChunker):
    pass


class _EchoChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Node]:
        return [
            Node(
                node_id="n1",
                doc_id=document.doc_id,
                content=document.content,
                metadata=document.metadata,
            )
        ]


def test_base_chunker_is_abstract() -> None:
    """The abstract base class must not be instantiated without chunk()."""
    with pytest.raises(TypeError):
        _NoImplChunker()


def test_subclass_can_implement_chunk() -> None:
    """A concrete subclass should return node output from chunk()."""
    chunker = _EchoChunker()
    document = Document(doc_id="doc-1", content="hello world")
    nodes = chunker.chunk(document)

    assert len(nodes) == 1
    assert nodes[0].doc_id == "doc-1"

