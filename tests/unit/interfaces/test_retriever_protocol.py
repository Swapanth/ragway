from __future__ import annotations

import inspect

from ragway.interfaces.retriever_protocol import RetrieverProtocol
from ragway.schema.metadata import Metadata
from ragway.schema.node import Node


class _Retriever:
    async def retrieve(self, query: str, top_k: int = 5) -> list[Node]:
        del query, top_k
        return [
            Node(
                node_id="n1",
                doc_id="d1",
                content="chunk",
                metadata=Metadata(),
            )
        ]


def test_retriever_protocol_runtime_checkable() -> None:
    """A structurally compatible retriever should satisfy RetrieverProtocol."""
    retriever = _Retriever()
    assert isinstance(retriever, RetrieverProtocol)


def test_retriever_protocol_retrieve_is_async() -> None:
    """The protocol retrieve method should be declared as coroutine function."""
    assert inspect.iscoroutinefunction(RetrieverProtocol.retrieve)

