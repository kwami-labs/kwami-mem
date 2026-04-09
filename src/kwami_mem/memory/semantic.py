"""kwami-mem — Semantic memory (facts, entities, preferences)."""

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


class SemanticMemory:
    """Stores higher-level facts, entities, and user preferences.

    Acts as the agent's knowledge base about the user — extracted
    from conversations and stored for long-term retrieval.
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

    async def store_fact(
        self,
        fact: str,
        *,
        source_conversation_id: str = "",
        topics: list[str] | None = None,
        entities: list[str] | None = None,
    ) -> MemoryEntry:
        """Store a factual memory.

        Args:
            fact: The fact or piece of knowledge to store.
            source_conversation_id: Where this fact was extracted from.
            topics: Topic tags for this fact.
            entities: Named entities mentioned.

        Returns:
            The stored MemoryEntry.
        """
        c_hash = content_hash(fact, "semantic", 0)

        # If this exact fact already exists, skip
        if await self._storage.exists(c_hash):
            return MemoryEntry(
                content=fact,
                role=Role.SYSTEM,
                conversation_id=source_conversation_id,
                user_id=self._user_id,
                memory_type=MemoryType.SEMANTIC,
                content_hash=c_hash,
            )

        # Extract additional metadata if not provided
        extracted = self._extractor.extract(fact)
        if topics:
            extracted["topics"] = topics
        if entities:
            extracted["entities"] = entities
        extracted["source_conversation_id"] = source_conversation_id

        entry = MemoryEntry(
            content=fact,
            role=Role.SYSTEM,
            conversation_id=source_conversation_id,
            user_id=self._user_id,
            memory_type=MemoryType.SEMANTIC,
            modality=Modality.TEXT,
            content_hash=c_hash,
            metadata=extracted,
        )

        vector = await self._embedder.embed_text(fact, task_type="RETRIEVAL_DOCUMENT")
        await self._storage.upsert([entry], [vector])

        return entry

    async def extract_and_store(
        self,
        conversation_content: str,
        conversation_id: str,
    ) -> list[MemoryEntry]:
        """Extract facts from conversation content and store them.

        This is a lightweight extraction — pulls out key statements
        and stores them as semantic memories.

        Args:
            conversation_content: Full text of the conversation.
            conversation_id: Source conversation identifier.

        Returns:
            List of stored MemoryEntry facts.
        """
        metadata = self._extractor.extract(conversation_content)
        entries: list[MemoryEntry] = []

        # Store the conversation summary as a semantic fact
        if len(conversation_content) > 50:
            # For substantial content, store as a semantic memory
            entry = await self.store_fact(
                conversation_content[:500],  # Store first 500 chars as summary
                source_conversation_id=conversation_id,
                topics=metadata.get("topics", []),
                entities=metadata.get("entities", []),
            )
            entries.append(entry)

        return entries

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        topics: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search semantic memories.

        Args:
            query: Natural language search query.
            limit: Maximum number of results.
            topics: Optional topic filter.
            score_threshold: Minimum similarity score.

        Returns:
            List of matching SearchResult entries.
        """
        query_vector = await self._embedder.embed_text(
            query, task_type="RETRIEVAL_QUERY"
        )

        filters: dict[str, str] = {
            "user_id": self._user_id,
            "memory_type": MemoryType.SEMANTIC.value,
        }

        return await self._storage.search(
            query_vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )
