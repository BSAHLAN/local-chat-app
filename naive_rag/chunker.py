"""Split text into fixed-size overlapping character chunks."""
from __future__ import annotations


def chunk_text(
    text: str, chunk_size: int = 800, chunk_overlap: int = 150
) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text or not text.strip():
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start:start + chunk_size]
        if piece.strip():
            chunks.append(piece)
        start += step
    return chunks
