"""Sentence-aware text chunking with a small overlap between chunks."""
import re


def _split_units(text: str) -> list[str]:
    """Split on blank lines and sentence endings."""
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """Pack sentences into chunks of roughly `size` characters.

    The last few sentences of each chunk (up to `overlap` characters) are repeated
    at the start of the next one so an answer that spans a boundary isn't lost.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in _split_units(text):
        if current and length + len(unit) + 1 > size:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current, length = carry, carried
        current.append(unit)
        length += len(unit) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks
