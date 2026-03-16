from __future__ import annotations

import asyncio

import pytest

from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class _NoImplParser(BaseDocumentParser):
    pass


class _EchoParser(BaseDocumentParser):
    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content
        return self._build_document(text, source=source, doc_id=doc_id)


class _NoImplLoader(BaseLoader):
    pass


class _EchoLoader(BaseLoader):
    async def load(self, source: object) -> list[Document]:
        content = str(source)
        return [self.parser.parse(content, source="inline")]


async def test_base_loader_is_abstract() -> None:
    """BaseLoader should not instantiate without load implementation."""
    parser = _EchoParser()
    with pytest.raises(TypeError):
        _NoImplLoader(parser)


async def test_concrete_loader_returns_documents() -> None:
    """Concrete loader should asynchronously produce document list."""
    loader = _EchoLoader(_EchoParser())

    documents = await loader.load("hello")

    assert len(documents) == 1
    assert documents[0].content == "hello"
    assert documents[0].metadata.source == "inline"

