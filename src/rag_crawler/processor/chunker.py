"""Text chunking for RAG consumption using recursive splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Approximate ratio: 1 whitespace-delimited word ~ 1.3 tokens.
_TOKENS_PER_WORD = 1.3


@dataclass
class Chunk:
    """A chunk of text with positional metadata."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int


def _estimate_tokens(text: str) -> float:
    """Approximate token count by splitting on whitespace."""
    return len(text.split()) * _TOKENS_PER_WORD


def _split_by_separators(text: str, separators: list[str]) -> list[str]:
    """Split *text* by the first separator that produces >1 segment.

    Falls back to character-level splitting if no separator works.
    """
    for sep in separators:
        if sep == "":
            # Character-level: return list of single chars
            return list(text)
        parts = re.split(re.escape(sep), text) if sep != "\n" else text.split(sep)
        # Keep the separator attached to the preceding segment for readability
        segments: list[str] = []
        for i, part in enumerate(parts):
            if i > 0:
                segments.append(sep + part)
            else:
                segments.append(part)
        segments = [s for s in segments if s.strip()]
        if len(segments) > 1:
            return segments
    # Nothing split the text; return it as-is.
    return [text] if text.strip() else []


def _recursive_split(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    """Recursively split *text* until each piece fits within *chunk_size* tokens."""
    if _estimate_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    segments = _split_by_separators(text, separators)
    if len(segments) <= 1 and _estimate_tokens(text) > chunk_size:
        # Could not split further with current separators; try next level
        if len(separators) > 1:
            return _recursive_split(text, chunk_size, separators[1:])
        # Hard truncate by words as last resort
        words = text.split()
        max_words = max(1, int(chunk_size / _TOKENS_PER_WORD))
        return [" ".join(words[:max_words])]

    result: list[str] = []
    for seg in segments:
        if _estimate_tokens(seg) <= chunk_size:
            result.append(seg)
        else:
            # Recurse with remaining separators
            next_seps = separators[1:] if len(separators) > 1 else separators
            result.extend(_recursive_split(seg, chunk_size, next_seps))
    return result


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split *text* into chunks suitable for RAG embedding.

    Uses recursive splitting: headers -> paragraphs -> sentences -> characters.
    Adjacent chunks share *chunk_overlap* tokens of context.

    Args:
        text: The full text to chunk.
        chunk_size: Target maximum tokens per chunk.
        chunk_overlap: Number of overlap tokens between consecutive chunks.

    Returns:
        Ordered list of ``Chunk`` objects with positional metadata.
    """
    if not text or not text.strip():
        return []

    separators = [
        "\n## ",   # h2 headers
        "\n### ",  # h3 headers
        "\n\n",    # paragraphs
        ". ",      # sentences
        "\n",      # lines
        " ",       # words
    ]

    raw_pieces = _recursive_split(text, chunk_size, separators)

    # Merge small adjacent pieces that together still fit within chunk_size.
    merged: list[str] = []
    buffer = ""
    for piece in raw_pieces:
        candidate = (buffer + piece) if buffer else piece
        if _estimate_tokens(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = piece
    if buffer:
        merged.append(buffer)

    if not merged:
        return []

    # Build Chunk objects with overlap.
    chunks: list[Chunk] = []
    # Track character positions in original text.
    search_start = 0

    for i, segment in enumerate(merged):
        stripped = segment.strip()
        if not stripped:
            continue

        # Compute overlap prefix from previous chunk.
        overlap_text = ""
        if i > 0 and chunk_overlap > 0 and chunks:
            prev_words = chunks[-1].text.split()
            overlap_word_count = max(1, int(chunk_overlap / _TOKENS_PER_WORD))
            overlap_words = prev_words[-overlap_word_count:]
            overlap_text = " ".join(overlap_words) + " "

        chunk_text_str = overlap_text + stripped if overlap_text else stripped

        # Find position of this segment in the original text.
        pos = text.find(stripped, search_start)
        if pos == -1:
            pos = search_start
        start_char = pos
        end_char = pos + len(stripped)
        search_start = end_char

        chunks.append(Chunk(
            text=chunk_text_str,
            chunk_index=len(chunks),
            start_char=start_char,
            end_char=end_char,
        ))

    return chunks
