"""Data models for the Clinical RAG system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """A chunk of text retrieved from the vector database."""

    text: str
    source: str
    chunk_index: int
    score: float
