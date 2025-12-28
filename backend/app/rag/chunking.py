from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text(*, text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(length, start + chunk_size)
        chunk = normalized[start:end]
        if chunk.strip():
            chunks.append(TextChunk(text=chunk.strip(), start=start, end=end))
        if end >= length:
            break
        start = end - chunk_overlap

    return chunks

