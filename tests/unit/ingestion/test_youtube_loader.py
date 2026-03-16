from __future__ import annotations

import asyncio
import sys
import types

import pytest

from ragway.exceptions import RagError
from ragway.ingestion.youtube_loader import YouTubeLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.schema.document import Document


class _TextParser(BaseDocumentParser):
    def parse(
        self,
        raw_content: str | bytes,
        *,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content
        return self._build_document(text, source=source, doc_id=doc_id)


async def test_youtube_loader_loads_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """YouTubeLoader should fetch and parse transcript text."""
    fake_module = types.ModuleType("youtube_transcript_api")

    class _FakeApi:
        @staticmethod
        def get_transcript(video_id: str) -> list[dict[str, str]]:
            assert video_id == "abc123"
            return [{"text": "hello"}, {"text": "world"}]

    fake_module.YouTubeTranscriptApi = _FakeApi
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)

    loader = YouTubeLoader(parser=_TextParser())
    docs = await loader.load("https://www.youtube.com/watch?v=abc123")

    assert len(docs) == 1
    assert docs[0].doc_id == "youtube-abc123"
    assert docs[0].content == "hello\nworld"


async def test_youtube_loader_rejects_empty_source() -> None:
    """YouTubeLoader should reject empty source values."""
    loader = YouTubeLoader(parser=_TextParser())
    with pytest.raises(RagError):
        await loader.load("   ")


async def test_youtube_loader_rejects_non_string_source() -> None:
    """YouTubeLoader should reject non-string sources."""
    loader = YouTubeLoader(parser=_TextParser())
    with pytest.raises(RagError):
        await loader.load(123)


async def test_youtube_loader_extracts_short_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """YouTube short URLs should parse the path segment as video id."""
    fake_module = types.ModuleType("youtube_transcript_api")

    class _FakeApi:
        @staticmethod
        def get_transcript(video_id: str) -> list[dict[str, str]]:
            assert video_id == "xyz789"
            return [{"text": "hello"}]

    fake_module.YouTubeTranscriptApi = _FakeApi
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)

    loader = YouTubeLoader(parser=_TextParser())
    docs = await loader.load("https://youtu.be/xyz789")
    assert docs[0].doc_id == "youtube-xyz789"


async def test_youtube_loader_invalid_url_raises() -> None:
    """YouTube watch URLs without `v` parameter should raise."""
    loader = YouTubeLoader(parser=_TextParser())
    with pytest.raises(RagError, match="Could not extract YouTube video id"):
        await loader.load("https://www.youtube.com/watch?x=1")


async def test_youtube_loader_missing_dependency_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing youtube-transcript-api dependency should raise RagError."""
    monkeypatch.delitem(sys.modules, "youtube_transcript_api", raising=False)

    import builtins

    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "youtube_transcript_api":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    loader = YouTubeLoader(parser=_TextParser())
    with pytest.raises(RagError, match="youtube-transcript-api is required"):
        await loader.load("abc123")


async def test_youtube_loader_empty_transcript_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty transcript payload should raise RagError."""
    fake_module = types.ModuleType("youtube_transcript_api")

    class _FakeApi:
        @staticmethod
        def get_transcript(video_id: str) -> list[dict[str, str]]:
            del video_id
            return [{"text": "   "}]

    fake_module.YouTubeTranscriptApi = _FakeApi
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)

    loader = YouTubeLoader(parser=_TextParser())
    with pytest.raises(RagError, match="No transcript content found"):
        await loader.load("abc123")
