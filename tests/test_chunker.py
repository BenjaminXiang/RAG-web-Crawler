"""Tests for rag_crawler.processor.chunker module."""

import pytest

from rag_crawler.processor.chunker import Chunk, chunk_text


class TestChunkTextShort:
    """Short text that fits in one chunk."""

    def test_no_chunking_needed(self):
        text = "This is a short document."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    def test_single_chunk_metadata(self):
        text = "Hello world, this is a test."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)


class TestChunkTextLong:
    """Long text that needs to be split."""

    def test_splits_into_multiple_chunks(self):
        # Create text that is definitely larger than chunk_size=5 tokens
        # Each word ~ 1.3 tokens, so 100 words ~ 130 tokens
        paragraphs = []
        for i in range(20):
            paragraphs.append(f"Paragraph {i}: " + " ".join(f"word{j}" for j in range(30)))
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
        assert len(chunks) > 1

    def test_chunks_are_ordered(self):
        paragraphs = [f"Section {i}\n\n" + " ".join(["content"] * 40) for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=0)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_all_content_covered(self):
        sentences = [f"Sentence number {i} with some padding words here." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=0)
        # Every chunk should have content
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0


class TestChunkOverlap:
    """Adjacent chunks share overlap tokens."""

    def test_overlap_present(self):
        # Build text large enough to produce multiple chunks
        paragraphs = []
        for i in range(15):
            paragraphs.append(f"Topic {i}: " + " ".join(f"detail{j}" for j in range(25)))
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        assert len(chunks) >= 2

        # The second chunk should contain words from the end of the first chunk
        if len(chunks) >= 2:
            first_words = chunks[0].text.split()
            second_text = chunks[1].text
            # Some tail words of chunk 0 should appear at the start of chunk 1
            tail_words = first_words[-5:]
            overlap_found = any(w in second_text for w in tail_words)
            assert overlap_found, "Expected overlap words from chunk 0 in chunk 1"

    def test_zero_overlap(self):
        paragraphs = [" ".join(f"word{j}" for j in range(30)) for _ in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=0)
        assert len(chunks) >= 2


class TestChunkEmpty:
    """Empty text produces no chunks."""

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  \t  ") == []

    def test_none_like(self):
        assert chunk_text("") == []


class TestChunkMetadata:
    """Chunks carry correct positional metadata."""

    def test_index_sequential(self):
        text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=0)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_start_end_char_valid(self):
        text = "Hello world. This is a test. More text follows here."
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=0)
        for chunk in chunks:
            assert isinstance(chunk.start_char, int)
            assert isinstance(chunk.end_char, int)
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.end_char <= len(text)

    def test_chunk_dataclass_fields(self):
        text = "Some test content for chunking purposes with enough words."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        c = chunks[0]
        assert hasattr(c, "text")
        assert hasattr(c, "chunk_index")
        assert hasattr(c, "start_char")
        assert hasattr(c, "end_char")
        assert isinstance(c, Chunk)
