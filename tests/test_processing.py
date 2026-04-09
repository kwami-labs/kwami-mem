"""Tests for kwami_mem.processing (chunker + extractor)."""

from kwami_mem.processing.chunker import TextChunker
from kwami_mem.processing.extractor import MetadataExtractor
from kwami_mem.utils.hashing import content_hash


class TestTextChunker:
    """Tests for TextChunker."""

    def test_short_text_no_split(self):
        chunker = TextChunker(chunk_size=500)
        result = chunker.chunk("Hello, this is a short sentence.")
        assert len(result) == 1
        assert result[0] == "Hello, this is a short sentence."

    def test_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_long_text_splits(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        text = ". ".join([f"Sentence number {i} with some content" for i in range(20)])
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # Each chunk should be within bounds
        for chunk in chunks:
            assert len(chunk) >= 10

    def test_overlap_exists(self):
        chunker = TextChunker(chunk_size=80, chunk_overlap=30, min_chunk_size=10)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = chunker.chunk(text)
        if len(chunks) >= 2:
            # Check that there's some overlap (shared text between consecutive chunks)
            for i in range(len(chunks) - 1):
                words_a = set(chunks[i].split())
                words_b = set(chunks[i + 1].split())
                # There should be at least some shared words from overlap
                assert len(words_a & words_b) > 0

    def test_sentence_boundaries_respected(self):
        chunker = TextChunker(chunk_size=60, chunk_overlap=10, min_chunk_size=5)
        text = "Hello there. How are you? I am fine."
        chunks = chunker.chunk(text)
        for chunk in chunks:
            # No chunk should start or end mid-word
            assert not chunk.startswith(" ")


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    def test_extract_topics(self, extractor: MetadataExtractor):
        text = "Python programming language is great for machine learning and data science"
        result = extractor.extract(text)
        assert "topics" in result
        assert isinstance(result["topics"], list)
        assert len(result["topics"]) > 0
        # "python" or "programming" should be in topics
        topics_lower = [t.lower() for t in result["topics"]]
        assert any(t in topics_lower for t in ["python", "programming", "machine", "learning"])

    def test_extract_entities(self, extractor: MetadataExtractor):
        text = "Albert Einstein developed the theory of relativity at Princeton University"
        result = extractor.extract(text)
        assert "entities" in result
        entities = result["entities"]
        entity_texts = [e.lower() for e in entities]
        assert any("albert" in e for e in entity_texts) or any("einstein" in e for e in entity_texts)

    def test_empty_text(self, extractor: MetadataExtractor):
        result = extractor.extract("")
        assert result == {"topics": [], "entities": []}

    def test_caps_at_max(self, extractor: MetadataExtractor):
        # Generate text with many unique words
        text = " ".join([f"Unique{i} topic{i}" for i in range(50)])
        result = extractor.extract(text)
        assert len(result["topics"]) <= 10
        assert len(result["entities"]) <= 10


class TestContentHash:
    """Tests for content hashing."""

    def test_deterministic(self):
        h1 = content_hash("hello", "conv-1", 0)
        h2 = content_hash("hello", "conv-1", 0)
        assert h1 == h2

    def test_different_content(self):
        h1 = content_hash("hello", "conv-1", 0)
        h2 = content_hash("world", "conv-1", 0)
        assert h1 != h2

    def test_different_conversation(self):
        h1 = content_hash("hello", "conv-1", 0)
        h2 = content_hash("hello", "conv-2", 0)
        assert h1 != h2

    def test_different_turn(self):
        h1 = content_hash("hello", "conv-1", 0)
        h2 = content_hash("hello", "conv-1", 1)
        assert h1 != h2

    def test_returns_hex_string(self):
        h = content_hash("test", "conv", 0)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
        int(h, 16)  # Should be valid hex
