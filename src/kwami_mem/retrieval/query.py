"""kwami-mem — Query processing and expansion."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any


class QueryProcessor:
    """Processes natural language queries before retrieval.

    Handles:
    - Extracting temporal references ("last week", "yesterday")
    - Building metadata filters from natural language
    - Cleaning and normalizing queries for better embedding
    """

    # Temporal patterns mapped to timedelta offsets
    _TEMPORAL_PATTERNS: list[tuple[re.Pattern, timedelta]] = [
        (re.compile(r"\byesterday\b", re.IGNORECASE), timedelta(days=1)),
        (re.compile(r"\btoday\b", re.IGNORECASE), timedelta(days=0)),
        (re.compile(r"\blast\s+week\b", re.IGNORECASE), timedelta(weeks=1)),
        (re.compile(r"\blast\s+month\b", re.IGNORECASE), timedelta(days=30)),
        (re.compile(r"\b(\d+)\s+days?\s+ago\b", re.IGNORECASE), None),  # dynamic
        (re.compile(r"\b(\d+)\s+hours?\s+ago\b", re.IGNORECASE), None),  # dynamic
        (re.compile(r"\brecently\b", re.IGNORECASE), timedelta(days=3)),
        (re.compile(r"\bthis\s+week\b", re.IGNORECASE), timedelta(days=7)),
        (re.compile(r"\bthis\s+month\b", re.IGNORECASE), timedelta(days=30)),
    ]

    def process(self, query: str) -> tuple[str, dict[str, Any]]:
        """Process a query and extract metadata filters.

        Args:
            query: Natural language search query.

        Returns:
            Tuple of (cleaned_query, extracted_filters).
            Filters may include 'time_after' (ISO datetime string).
        """
        filters: dict[str, Any] = {}
        cleaned = query

        # Extract temporal references
        time_after = self._extract_temporal(query)
        if time_after:
            filters["time_after"] = time_after.isoformat()

        # Extract role filter hints
        role = self._extract_role_hint(query)
        if role:
            filters["role"] = role

        # Clean the query: remove temporal phrases for better embedding
        cleaned = self._clean_temporal(cleaned)
        cleaned = cleaned.strip()

        # Fallback to original if cleaning emptied the query
        if not cleaned:
            cleaned = query.strip()

        return cleaned, filters

    def _extract_temporal(self, query: str) -> datetime | None:
        """Extract a temporal lower bound from the query."""
        now = datetime.now(timezone.utc)

        for pattern, delta in self._TEMPORAL_PATTERNS:
            match = pattern.search(query)
            if match:
                if delta is not None:
                    return now - delta

                # Dynamic patterns (N days/hours ago)
                groups = match.groups()
                if groups:
                    n = int(groups[0])
                    if "hour" in pattern.pattern:
                        return now - timedelta(hours=n)
                    return now - timedelta(days=n)

        return None

    def _extract_role_hint(self, query: str) -> str | None:
        """Detect if the query is specifically about user or assistant messages."""
        lower = query.lower()

        user_phrases = [
            "i said", "i told", "i asked", "i mentioned", "my message",
            "i say", "i tell", "i ask", "i mention",
            "did i say", "did i tell", "did i ask",
        ]
        if any(phrase in lower for phrase in user_phrases):
            return "user"

        assistant_phrases = [
            "you said", "you told", "you answered", "your response", "you mentioned",
            "you say", "you tell", "you answer", "you mention",
            "did you say", "did you tell",
        ]
        if any(phrase in lower for phrase in assistant_phrases):
            return "assistant"

        return None

    def _clean_temporal(self, query: str) -> str:
        """Remove temporal phrases from query for cleaner embedding."""
        cleaned = query
        for pattern, _ in self._TEMPORAL_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned
