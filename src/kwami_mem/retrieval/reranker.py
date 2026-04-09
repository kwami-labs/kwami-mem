"""kwami-mem — Result reranking logic."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from kwami_mem.models import SearchResult


class Reranker:
    """Reranks search results by blending multiple relevance signals.

    Combines:
    - Semantic similarity (vector distance score)
    - Temporal recency (more recent = higher weight)
    - Conversation relevance (same conversation = boost)
    - Role weighting (configurable)
    """

    def __init__(
        self,
        semantic_weight: float = 0.6,
        recency_weight: float = 0.25,
        context_weight: float = 0.15,
        recency_half_life_days: float = 7.0,
    ) -> None:
        """
        Args:
            semantic_weight: Weight for vector similarity score (0-1).
            recency_weight: Weight for temporal recency (0-1).
            context_weight: Weight for conversational context boost (0-1).
            recency_half_life_days: Number of days until recency score halves.
        """
        total = semantic_weight + recency_weight + context_weight
        self._semantic_w = semantic_weight / total
        self._recency_w = recency_weight / total
        self._context_w = context_weight / total
        self._half_life = recency_half_life_days

    def rerank(
        self,
        results: list[SearchResult],
        *,
        current_conversation_id: str | None = None,
    ) -> list[SearchResult]:
        """Rerank results by combined score.

        Args:
            results: Search results from the vector database.
            current_conversation_id: If provided, boost results from this conversation.

        Returns:
            Results sorted by combined_score (descending).
        """
        if not results:
            return []

        now = datetime.now(timezone.utc)

        for result in results:
            semantic_score = max(0.0, min(1.0, result.score))
            recency_score = self._compute_recency(result, now)
            context_score = self._compute_context(result, current_conversation_id)

            result.combined_score = (
                self._semantic_w * semantic_score
                + self._recency_w * recency_score
                + self._context_w * context_score
            )

        # Sort by combined score, descending
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results

    def _compute_recency(self, result: SearchResult, now: datetime) -> float:
        """Exponential decay based on age of the memory.

        Score = 0.5 ^ (age_days / half_life_days)
        - 0 days old → 1.0
        - half_life days old → 0.5
        - 2 * half_life days old → 0.25
        """
        timestamp = result.entry.timestamp
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                return 0.5  # Unknown age → neutral score

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
        return math.pow(0.5, age_days / self._half_life)

    def _compute_context(
        self,
        result: SearchResult,
        current_conversation_id: str | None,
    ) -> float:
        """Boost score if result is from the current conversation."""
        if not current_conversation_id:
            return 0.5  # Neutral

        if result.entry.conversation_id == current_conversation_id:
            return 1.0  # Full boost
        return 0.0  # Different conversation
