"""Asynchronous loader for YouTube transcripts."""

from __future__ import annotations

import asyncio
import importlib
from urllib.parse import parse_qs, urlparse

from ragway.exceptions import RagError
from ragway.ingestion.base_loader import BaseLoader
from ragway.parsing.document_parser import BaseDocumentParser
from ragway.parsing.markdown_parser import MarkdownParser
from ragway.schema.document import Document


class YouTubeLoader(BaseLoader):
    """Load YouTube transcript text and parse it into a document."""

    def __init__(self, parser: BaseDocumentParser | None = None) -> None:
        """Initialize loader with optional custom parser."""
        super().__init__(parser or MarkdownParser())

    async def load(self, source: object) -> list[Document]:
        """Load a YouTube transcript from a URL or video id."""
        if not isinstance(source, str):
            raise RagError("YouTubeLoader source must be a YouTube URL or video id string")

        video_id = self._extract_video_id(source)
        transcript_text = await self._fetch_transcript(video_id)
        document = self.parser.parse(
            transcript_text,
            source=source,
            doc_id=f"youtube-{video_id}",
        )
        return [document]

    def _extract_video_id(self, source: str) -> str:
        """Extract YouTube video id from URL or pass through raw id."""
        text = source.strip()
        if not text:
            raise RagError("YouTube source must be non-empty")

        if "youtube.com" in text or "youtu.be" in text:
            parsed = urlparse(text)
            if "youtu.be" in parsed.netloc:
                candidate = parsed.path.strip("/")
                if candidate:
                    return candidate
            query = parse_qs(parsed.query)
            values = query.get("v", [])
            if values and values[0].strip():
                return values[0].strip()
            raise RagError(f"Could not extract YouTube video id from URL: {source}")

        return text

    async def _fetch_transcript(self, video_id: str) -> str:
        """Fetch transcript text for a video id."""
        try:
            transcript_module = importlib.import_module("youtube_transcript_api")
            transcript_api = getattr(transcript_module, "YouTubeTranscriptApi")
        except (ImportError, AttributeError) as exc:
            raise RagError(f"youtube-transcript-api is required for YouTube loading: {exc}") from exc

        def _fetch_sync() -> str:
            chunks = transcript_api.get_transcript(video_id)
            texts = [str(item.get("text", "")).strip() for item in chunks if str(item.get("text", "")).strip()]
            return "\n".join(texts).strip()

        try:
            transcript = await asyncio.to_thread(_fetch_sync)
        except Exception as exc:
            raise RagError(f"Failed to fetch YouTube transcript for {video_id}: {exc}") from exc

        if not transcript:
            raise RagError(f"No transcript content found for YouTube video {video_id}")
        return transcript
