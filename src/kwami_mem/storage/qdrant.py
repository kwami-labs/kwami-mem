"""kwami-mem — Qdrant storage backend."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    ScrollOrder,
    VectorParams,
)

from kwami_mem.models import (
    MemoryEntry,
    MemoryType,
    Modality,
    Role,
    SearchResult,
)
from kwami_mem.storage.base import StorageBackend


class QdrantStorage(StorageBackend):
    """Qdrant vector storage backend.

    Supports both in-memory mode (development) and remote server (production).
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "kwami_memory",
        vector_size: int = 768,
    ) -> None:
        self._collection_name = collection_name
        self._vector_size = vector_size

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            # In-memory mode for development / testing
            self._client = QdrantClient(":memory:")

    async def _run_sync(self, fn, *args, **kwargs):
        """Run synchronous Qdrant client calls in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

    async def initialize(self) -> None:
        """Create collection and payload indexes if they don't exist."""
        collections = await self._run_sync(self._client.get_collections)
        existing = {c.name for c in collections.collections}

        if self._collection_name not in existing:
            await self._run_sync(
                self._client.create_collection,
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

        # Create payload indexes for efficient filtering
        indexed_fields = {
            "conversation_id": PayloadSchemaType.KEYWORD,
            "user_id": PayloadSchemaType.KEYWORD,
            "role": PayloadSchemaType.KEYWORD,
            "memory_type": PayloadSchemaType.KEYWORD,
            "modality": PayloadSchemaType.KEYWORD,
            "timestamp": PayloadSchemaType.KEYWORD,
            "content_hash": PayloadSchemaType.KEYWORD,
        }
        for field_name, schema_type in indexed_fields.items():
            try:
                await self._run_sync(
                    self._client.create_payload_index,
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                # Index may already exist — safe to ignore
                pass

    async def upsert(
        self,
        entries: list[MemoryEntry],
        vectors: list[list[float]],
    ) -> None:
        """Store memory entries with their embedding vectors."""
        if not entries:
            return

        points = [
            PointStruct(
                id=entry.id,
                vector=vector,
                payload=entry.to_payload(),
            )
            for entry, vector in zip(entries, vectors)
        ]

        await self._run_sync(
            self._client.upsert,
            collection_name=self._collection_name,
            points=points,
        )

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar memories with optional metadata filtering."""
        query_filter = self._build_filter(filters) if filters else None

        results = await self._run_sync(
            self._client.query_points,
            collection_name=self._collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            SearchResult(
                entry=self._point_to_entry(hit.id, hit.payload),
                score=hit.score,
            )
            for hit in results.points
        ]

    async def get_by_conversation(
        self,
        conversation_id: str,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Retrieve all entries from a specific conversation, ordered by turn_index."""
        conditions = [
            FieldCondition(
                key="conversation_id", match=MatchValue(value=conversation_id)
            )
        ]
        if user_id:
            conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )

        results = await self._run_sync(
            self._client.scroll,
            collection_name=self._collection_name,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True,
            order_by="turn_index",
        )

        points = results[0]  # scroll returns (points, next_offset)
        entries = [self._point_to_entry(p.id, p.payload) for p in points]
        return sorted(entries, key=lambda e: e.turn_index)

    async def delete_conversation(self, conversation_id: str) -> int:
        """Delete all entries in a conversation."""
        # Get count first
        entries = await self.get_by_conversation(conversation_id)
        count = len(entries)

        if count > 0:
            await self._run_sync(
                self._client.delete,
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="conversation_id",
                            match=MatchValue(value=conversation_id),
                        )
                    ]
                ),
            )

        return count

    async def count(self, *, user_id: str | None = None) -> int:
        """Return total number of stored memory entries."""
        if user_id:
            result = await self._run_sync(
                self._client.count,
                collection_name=self._collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id", match=MatchValue(value=user_id)
                        )
                    ]
                ),
            )
        else:
            result = await self._run_sync(
                self._client.count,
                collection_name=self._collection_name,
            )
        return result.count

    async def exists(self, content_hash: str) -> bool:
        """Check if a memory with this content hash already exists."""
        results = await self._run_sync(
            self._client.scroll,
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="content_hash", match=MatchValue(value=content_hash)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
        )
        return len(results[0]) > 0

    # --- Private helpers ---

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """Convert a simple dict of field=value pairs into a Qdrant Filter."""
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
            if value is not None
        ]
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _point_to_entry(point_id: str | int, payload: dict[str, Any]) -> MemoryEntry:
        """Convert a Qdrant point payload back into a MemoryEntry."""
        # Extract known fields, rest goes into metadata
        known_keys = {
            "content", "role", "conversation_id", "user_id",
            "memory_type", "modality", "turn_index", "timestamp",
            "content_hash",
        }
        extra_metadata = {
            k: v for k, v in payload.items() if k not in known_keys
        }

        return MemoryEntry(
            id=str(point_id),
            content=payload.get("content", ""),
            role=Role(payload.get("role", "user")),
            conversation_id=payload.get("conversation_id", ""),
            user_id=payload.get("user_id", "default"),
            memory_type=MemoryType(payload.get("memory_type", "episodic")),
            modality=Modality(payload.get("modality", "text")),
            turn_index=payload.get("turn_index", 0),
            timestamp=payload.get("timestamp", ""),
            content_hash=payload.get("content_hash", ""),
            metadata=extra_metadata,
        )
