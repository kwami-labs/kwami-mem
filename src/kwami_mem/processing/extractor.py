"""kwami-mem — Lightweight metadata extraction."""

from __future__ import annotations

import re
from typing import Any


class MetadataExtractor:
    """Extracts topics and entities from text without LLM calls.

    Uses lightweight keyword/pattern-based extraction for efficiency.
    The extracted metadata is used as payload indexes in Qdrant for
    filtered retrieval.
    """

    # Common stop words to exclude from topic extraction
    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "don", "now", "and", "but", "or", "if", "about", "this",
        "that", "it", "its", "i", "me", "my", "we", "our", "you", "your",
        "he", "she", "him", "her", "his", "they", "them", "their", "what",
        "which", "who", "whom", "up", "also", "like", "well", "back", "even",
        "still", "way", "take", "come", "go", "make", "get", "know", "think",
        "say", "see", "want", "look", "use", "find", "give", "tell",
    })

    def extract(self, text: str) -> dict[str, Any]:
        """Extract metadata from text content.

        Args:
            text: The text to analyze.

        Returns:
            Dict with 'topics' (list[str]) and 'entities' (list[str]).
        """
        if not text or not text.strip():
            return {"topics": [], "entities": []}

        topics = self._extract_topics(text)
        entities = self._extract_entities(text)

        return {
            "topics": topics[:10],  # Cap at 10 topics
            "entities": entities[:10],  # Cap at 10 entities
        }

    def _extract_topics(self, text: str, max_topics: int = 10) -> list[str]:
        """Extract topic keywords via word frequency analysis.

        Filters stop words, short words, and returns the most frequent
        significant words as topic tags.
        """
        # Lowercase and extract words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Filter stop words
        significant = [w for w in words if w not in self._STOP_WORDS]

        # Count frequency
        freq: dict[str, int] = {}
        for word in significant:
            freq[word] = freq.get(word, 0) + 1

        # Sort by frequency and return top topics
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_topics]]

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities using capitalization patterns.

        Simple heuristic: consecutive capitalized words (not at sentence start)
        are likely named entities.
        """
        # Find sequences of capitalized words
        # This pattern matches 1+ consecutive capitalized words
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        matches = re.findall(pattern, text)

        # Deduplicate while preserving order
        seen: set[str] = set()
        entities: list[str] = []
        for match in matches:
            normalized = match.strip()
            if normalized.lower() not in seen and len(normalized) > 1:
                seen.add(normalized.lower())
                entities.append(normalized)

        return entities
