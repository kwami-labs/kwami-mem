"""Tests for kwami_mem.storage.qdrant."""

import pytest

from kwami_mem.models import MemoryEntry, MemoryType, Modality, Role
from kwami_mem.storage.qdrant import QdrantStorage


@pytest.fixture
async def storage() -> QdrantStorage:
    """Fresh in-memory Qdrant storage for each test."""
    s = QdrantStorage(url=None, collection_name="test_storage", vector_size=768)
    await s.initialize()
    return s


@pytest.fixture
def sample_entry() -> MemoryEntry:
    return MemoryEntry(
        content="Paris is the capital of France",
        role=Role.ASSISTANT,
        conversation_id="conv-1",
        user_id="user-1",
        memory_type=MemoryType.EPISODIC,
        modality=Modality.TEXT,
        turn_index=0,
        content_hash="abc123",
    )


@pytest.fixture
def sample_vector() -> list[float]:
    return [0.5] * 768


class TestQdrantStorage:
    """Tests for QdrantStorage."""

    async def test_initialize_creates_collection(self):
        storage = QdrantStorage(url=None, collection_name="init_test", vector_size=768)
        await storage.initialize()
        count = await storage.count()
        assert count == 0

    async def test_upsert_and_count(
        self, storage: QdrantStorage, sample_entry: MemoryEntry, sample_vector: list[float]
    ):
        await storage.upsert([sample_entry], [sample_vector])
        count = await storage.count()
        assert count == 1

    async def test_upsert_empty_list(self, storage: QdrantStorage):
        await storage.upsert([], [])
        count = await storage.count()
        assert count == 0

    async def test_search_returns_results(
        self, storage: QdrantStorage, sample_entry: MemoryEntry, sample_vector: list[float]
    ):
        await storage.upsert([sample_entry], [sample_vector])
        results = await storage.search(sample_vector, limit=5)
        assert len(results) == 1
        assert results[0].entry.content == "Paris is the capital of France"
        assert results[0].score > 0

    async def test_search_with_filters(
        self, storage: QdrantStorage, sample_entry: MemoryEntry, sample_vector: list[float]
    ):
        await storage.upsert([sample_entry], [sample_vector])

        # Should find with correct filter
        results = await storage.search(
            sample_vector, limit=5, filters={"user_id": "user-1"}
        )
        assert len(results) == 1

        # Should not find with wrong filter
        results = await storage.search(
            sample_vector, limit=5, filters={"user_id": "user-999"}
        )
        assert len(results) == 0

    async def test_get_by_conversation(
        self, storage: QdrantStorage, sample_vector: list[float]
    ):
        entries = [
            MemoryEntry(
                content=f"Message {i}",
                role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
                conversation_id="conv-order",
                user_id="user-1",
                turn_index=i,
                content_hash=f"hash-{i}",
            )
            for i in range(5)
        ]
        vectors = [sample_vector] * 5
        await storage.upsert(entries, vectors)

        result = await storage.get_by_conversation("conv-order", user_id="user-1")
        assert len(result) == 5
        # Should be ordered by turn_index
        for i, entry in enumerate(result):
            assert entry.turn_index == i

    async def test_delete_conversation(
        self, storage: QdrantStorage, sample_entry: MemoryEntry, sample_vector: list[float]
    ):
        await storage.upsert([sample_entry], [sample_vector])
        assert await storage.count() == 1

        deleted = await storage.delete_conversation("conv-1")
        assert deleted == 1
        assert await storage.count() == 0

    async def test_exists(
        self, storage: QdrantStorage, sample_entry: MemoryEntry, sample_vector: list[float]
    ):
        assert await storage.exists("abc123") is False
        await storage.upsert([sample_entry], [sample_vector])
        assert await storage.exists("abc123") is True

    async def test_count_with_user_filter(
        self, storage: QdrantStorage, sample_vector: list[float]
    ):
        entry1 = MemoryEntry(
            content="User 1 message", role=Role.USER,
            conversation_id="c1", user_id="user-1", content_hash="h1",
        )
        entry2 = MemoryEntry(
            content="User 2 message", role=Role.USER,
            conversation_id="c2", user_id="user-2", content_hash="h2",
        )
        await storage.upsert([entry1, entry2], [sample_vector, sample_vector])

        assert await storage.count() == 2
        assert await storage.count(user_id="user-1") == 1
        assert await storage.count(user_id="user-2") == 1
