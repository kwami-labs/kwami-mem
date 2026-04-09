"""kwami-mem — Abstract storage backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kwami_mem.models import MemoryEntry, SearchResult


class StorageBackend(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage (create collections, indexes, etc.)."""

    @abstractmethod
    async def upsert(
        self,
        entries: list[MemoryEntry],
        vectors: list[list[float]],
    ) -> None:
        """Store memory entries with their embedding vectors.

        Args:
            entries: Memory entries to store.
            vectors: Corresponding embedding vectors.
        """

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar memories.

        Args:
            query_vector: The query embedding vector.
            limit: Maximum number of results.
            filters: Metadata filters to apply.
            score_threshold: Minimum score threshold.

        Returns:
            List of search results ranked by similarity.
        """

    @abstractmethod
    async def get_by_conversation(
        self,
        conversation_id: str,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Retrieve all entries from a specific conversation.

        Args:
            conversation_id: The conversation to retrieve.
            user_id: Optional user filter.
            limit: Maximum number of entries.

        Returns:
            List of memory entries, ordered by turn_index.
        """

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> int:
        """Delete all entries in a conversation. Returns count of deleted entries."""

    @abstractmethod
    async def count(self, *, user_id: str | None = None) -> int:
        """Return total number of stored memory entries."""

    @abstractmethod
    async def exists(self, content_hash: str) -> bool:
        """Check if a memory with this content hash already exists."""
