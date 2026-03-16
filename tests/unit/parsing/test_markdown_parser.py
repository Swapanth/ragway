from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from ragway.exceptions import RagError
from ragway.parsing.markdown_parser import MarkdownParser


@dataclass
class _Token:
    type: str
    content: str = ""
    children: list["_Token"] | None = None


class _FakeMarkdownIt:
    def __init__(self, _preset: str) -> None:
        pass

    def parse(self, _text: str) -> list[_Token]:
        return [
            _Token(type="inline", children=[_Token(type="text", content="Heading")]),
            _Token(type="fence", content="print('hello')"),
            _Token(type="inline", children=[_Token(type="text", content="Paragraph")]),
        ]


def test_markdown_parser_extracts_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """MarkdownParser should collect text from markdown-it tokens."""
    parser = MarkdownParser()
    fake_module = types.ModuleType("markdown_it")
    fake_module.MarkdownIt = _FakeMarkdownIt
    monkeypatch.setitem(sys.modules, "markdown_it", fake_module)

    document = parser.parse("# Heading", source="notes.md", doc_id="doc-markdown")

    assert document.doc_id == "doc-markdown"
    assert document.content == "Heading\nprint('hello')\nParagraph"
    assert document.metadata.source == "notes.md"


def test_markdown_parser_raises_without_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """MarkdownParser should raise RagError when markdown-it-py is unavailable."""
    parser = MarkdownParser()
    fake_module = types.ModuleType("markdown_it")
    monkeypatch.setitem(sys.modules, "markdown_it", fake_module)

    with pytest.raises(RagError):
        parser.parse("# Missing backend")


def test_markdown_parser_raises_on_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """MarkdownParser should reject markdown that yields no text tokens."""

    class _EmptyMarkdownIt:
        def __init__(self, _preset: str) -> None:
            pass

        def parse(self, _text: str) -> list[_Token]:
            return [_Token(type="inline", children=[])]

    parser = MarkdownParser()
    fake_module = types.ModuleType("markdown_it")
    fake_module.MarkdownIt = _EmptyMarkdownIt
    monkeypatch.setitem(sys.modules, "markdown_it", fake_module)

    with pytest.raises(RagError):
        parser.parse("# Empty")

