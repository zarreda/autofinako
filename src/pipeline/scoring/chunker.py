"""Split earnings call transcripts into chunks for LLM processing."""

from __future__ import annotations

MIN_CHUNK_LENGTH = 50


def chunk_transcript(
    text: str,
    chunk_size: int = 30,
    overlap: int = 0,
) -> list[str]:
    """Split *text* into sentence-based chunks.

    Parameters
    ----------
    text:
        Raw transcript text.
    chunk_size:
        Number of sentences per chunk.
    overlap:
        Number of overlapping sentences between consecutive chunks.

    Returns
    -------
    list[str]
        Non-empty chunks of at least ``MIN_CHUNK_LENGTH`` characters.
    """
    sentences = text.split(". ")
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + chunk_size]
        chunk_text = ". ".join(chunk_sentences)
        if chunk_text and len(chunk_text.strip()) > MIN_CHUNK_LENGTH:
            chunks.append(chunk_text)

    return chunks
