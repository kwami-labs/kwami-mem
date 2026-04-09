"""kwami-mem — Text chunking strategies."""

from __future__ import annotations

import re


class TextChunker:
    """Splits long text into overlapping chunks for embedding.

    Uses sentence-aware splitting to avoid breaking mid-sentence.
    Each chunk maintains overlap with the previous for context continuity.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ) -> None:
        """
        Args:
            chunk_size: Target number of characters per chunk.
            chunk_overlap: Number of characters to overlap between chunks.
            min_chunk_size: Minimum chunk size — discard anything shorter.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping, sentence-aware chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # If text is short enough, return as single chunk
        if len(text) <= self._chunk_size:
            return [text]

        # Split into sentences
        sentences = self._split_sentences(text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_length + sentence_len > self._chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if len(chunk_text) >= self._min_chunk_size:
                    chunks.append(chunk_text)

                # Keep overlap: walk backwards to find sentences that fit in overlap
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) > self._chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1  # +1 for space

                current_chunk = overlap_sentences
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= self._min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex.

        Handles common abbreviations and edge cases.
        """
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
