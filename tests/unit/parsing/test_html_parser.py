from __future__ import annotations

import sys
import types

import pytest

from ragway.exceptions import RagError
from ragway.parsing.html_parser import HTMLParser


class _FakeSoup:
    def __init__(self, html: str, parser: str) -> None:
        self._html = html
        self._parser = parser

    def get_text(self, _sep: str, strip: bool) -> str:
        assert self._parser == "html.parser"
        return "Title\nBody" if "<h1>" in self._html else ""


def test_html_parser_extracts_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTMLParser should extract plain text with BeautifulSoup."""
    parser = HTMLParser()
    fake_bs4 = types.ModuleType("bs4")
    fake_bs4.BeautifulSoup = _FakeSoup
    monkeypatch.setitem(sys.modules, "bs4", fake_bs4)

    document = parser.parse("<h1>Title</h1><p>Body</p>", source="page.html", doc_id="doc-html")

    assert document.doc_id == "doc-html"
    assert document.content == "Title\nBody"
    assert document.metadata.source == "page.html"


def test_html_parser_raises_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTMLParser should raise RagError when bs4 import fails."""
    parser = HTMLParser()
    fake_bs4 = types.ModuleType("bs4")
    monkeypatch.setitem(sys.modules, "bs4", fake_bs4)

    with pytest.raises(RagError):
        parser.parse("<p>hello</p>")


def test_html_parser_raises_on_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTMLParser should reject HTML that yields no text."""
    parser = HTMLParser()

    class _EmptySoup:
        def __init__(self, _html: str, _parser: str) -> None:
            pass

        def get_text(self, _sep: str, strip: bool) -> str:
            return ""

    fake_bs4 = types.ModuleType("bs4")
    fake_bs4.BeautifulSoup = _EmptySoup
    monkeypatch.setitem(sys.modules, "bs4", fake_bs4)

    with pytest.raises(RagError):
        parser.parse("<div></div>")

