"""kwami-mem — Episodic memory (conversation log stored in Qdrant)."""

from __future__ import annotations

from kwami_mem.embedding.base import EmbeddingProvider
from kwami_mem.models import (
    MemoryEntry,
    MemoryType,
    Modality,
    Role,
    SearchResult,
)
from kwami_mem.processing.extractor import MetadataExtractor
from kwami_mem.storage.base import StorageBackend
from kwami_mem.utils.hashing import content_hash


class EpisodicMemory:
    """Stores every conversation turn as a vector in Qdrant.

    Preserves full conversation history with metadata for temporal
    and contextual retrieval: "What did we discuss last Tuesday?"
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider,
        extractor: MetadataExtractor,
        user_id: str = "default",
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._extractor = extractor
        self._user_id = user_id

    async def store(
        self,
        content: str,
        role: Role,
        conversation_id: str,
        turn_index: int,
        *,
        modality: Modality = Modality.TEXT,
    ) -> MemoryEntry:
        """Store a conversation turn as an episodic memory.

        Args:
            content: Message text content.
            role: Who produced this content.
            conversation_id: Conversation thread identifier.
            turn_index: Position in the conversation.
            modality: Content modality type.

        Returns:
            The stored MemoryEntry.
        """
        # Deduplicate
        c_hash = content_hash(content, conversation_id, turn_index)
        if await self._storage.exists(c_hash):
            # Return existing — no duplicate storage
            return MemoryEntry(
                content=content,
                role=role,
                conversation_id=conversation_id,
                user_id=self._user_id,
                turn_index=turn_index,
                modality=modality,
                content_hash=c_hash,
            )

        # Extract metadata
        metadata = self._extractor.extract(content)

        entry = MemoryEntry(
            content=content,
            role=role,
            conversation_id=conversation_id,
            user_id=self._user_id,
            memory_type=MemoryType.EPISODIC,
            modality=modality,
            turn_index=turn_index,
            content_hash=c_hash,
            metadata=metadata,
        )

        # Embed and store
        vector = await self._embedder.embed_text(content, task_type="RETRIEVAL_DOCUMENT")
        await self._storage.upsert([entry], [vector])

        return entry

    async def store_multimodal(
        self,
        vector: list[float],
        content_description: str,
        role: Role,
        conversation_id: str,
        turn_index: int,
        modality: Modality,
    ) -> MemoryEntry:
        """Store a pre-embedded multimodal memory (image, audio, PDF).

        Args:
            vector: Pre-computed embedding vector.
            content_description: Text description of the content.
            role: Who produced this content.
            conversation_id: Conversation thread identifier.
            turn_index: Position in the conversation.
            modality: Content modality type.

        Returns:
            The stored MemoryEntry.
        """
        c_hash = content_hash(content_description, conversation_id, turn_index)

        entry = MemoryEntry(
            content=content_description,
            role=role,
            conversation_id=conversation_id,
            user_id=self._user_id,
            memory_type=MemoryType.EPISODIC,
            modality=modality,
            turn_index=turn_index,
            content_hash=c_hash,
        )

        await self._storage.upsert([entry], [vector])
        return entry

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        conversation_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search episodic memories by semantic similarity.

        Args:
            query: Natural language search query.
            limit: Maximum number of results.
            conversation_id: Optional filter to a specific conversation.
            score_threshold: Minimum similarity score.

        Returns:
            List of matching SearchResult entries.
        """
        query_vector = await self._embedder.embed_text(
            query, task_type="RETRIEVAL_QUERY"
        )

        filters = {
            "user_id": self._user_id,
            "memory_type": MemoryType.EPISODIC.value,
        }
        if conversation_id:
            filters["conversation_id"] = conversation_id

        return await self._storage.search(
            query_vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )

    async def get_conversation(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Retrieve full conversation history, ordered by turn index."""
        return await self._storage.get_by_conversation(
            conversation_id, user_id=self._user_id, limit=limit
        )
