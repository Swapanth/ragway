"""Public package API for ragway."""

from __future__ import annotations

from ragway.rag import RAG
from ragway.schema.document import Document
from ragway.schema.node import Node

__version__ = "0.1.0"
__author__ = "your name"
__description__ = "The way to build RAG"
__all__ = ["RAG", "Document", "Node"]

