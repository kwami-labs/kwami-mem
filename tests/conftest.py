"""Shared test fixtures for kwami-mem."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kwami_mem.config import KwamiSettings
from kwami_mem.embedding.gemini import GeminiEmbeddingProvider
from kwami_mem.memory.episodic import EpisodicMemory
from kwami_mem.memory.semantic import SemanticMemory
from kwami_mem.memory.working import WorkingMemory
from kwami_mem.models import MemoryEntry, MemoryType, Modality, Role, SearchResult
from kwami_mem.processing.extractor import MetadataExtractor
from kwami_mem.retrieval.query import QueryProcessor
from kwami_mem.retrieval.reranker import Reranker
from kwami_mem.retrieval.retriever import MemoryRetriever
from kwami_mem.storage.qdrant import QdrantStorage


@pytest.fixture
def fake_vector() -> list[float]:
    """A fake 768-dim embedding vector for testing."""
    return [0.1] * 768


@pytest.fixture
def mock_embedder(fake_vector: list[float]) -> MagicMock:
    """Mock embedding provider that returns deterministic vectors."""
    embedder = MagicMock(spec=GeminiEmbeddingProvider)
    embedder.embed_text = AsyncMock(return_value=fake_vector)
    embedder.embed_texts = AsyncMock(return_value=[fake_vector])
    embedder.embed_image = AsyncMock(return_value=fake_vector)
    embedder.embed_audio = AsyncMock(return_value=fake_vector)
    embedder.embed_pdf = AsyncMock(return_value=[fake_vector, fake_vector])
    return embedder


@pytest.fixture
def qdrant_storage() -> QdrantStorage:
    """In-memory Qdrant storage for testing."""
    return QdrantStorage(
        url=None,
        collection_name="test_memory",
        vector_size=768,
    )


@pytest.fixture
def extractor() -> MetadataExtractor:
    """Metadata extractor instance."""
    return MetadataExtractor()


@pytest.fixture
def working_memory() -> WorkingMemory:
    """Working memory with small window for testing."""
    return WorkingMemory(max_turns=5)


@pytest.fixture
def query_processor() -> QueryProcessor:
    """Query processor instance."""
    return QueryProcessor()


@pytest.fixture
def reranker() -> Reranker:
    """Reranker instance."""
    return Reranker()


@pytest.fixture
async def initialized_storage(qdrant_storage: QdrantStorage) -> QdrantStorage:
    """Qdrant storage with collection already created."""
    await qdrant_storage.initialize()
    return qdrant_storage


@pytest.fixture
def episodic_memory(
    initialized_storage: QdrantStorage,
    mock_embedder: MagicMock,
    extractor: MetadataExtractor,
) -> EpisodicMemory:
    """Episodic memory wired to in-memory Qdrant + mock embedder."""
    return EpisodicMemory(
        storage=initialized_storage,
        embedder=mock_embedder,
        extractor=extractor,
        user_id="test-user",
    )


@pytest.fixture
def semantic_memory(
    initialized_storage: QdrantStorage,
    mock_embedder: MagicMock,
    extractor: MetadataExtractor,
) -> SemanticMemory:
    """Semantic memory wired to in-memory Qdrant + mock embedder."""
    return SemanticMemory(
        storage=initialized_storage,
        embedder=mock_embedder,
        extractor=extractor,
        user_id="test-user",
    )


@pytest.fixture
def retriever(
    working_memory: WorkingMemory,
    episodic_memory: EpisodicMemory,
    semantic_memory: SemanticMemory,
    query_processor: QueryProcessor,
    reranker: Reranker,
) -> MemoryRetriever:
    """Full retriever with all memory layers."""
    return MemoryRetriever(
        working=working_memory,
        episodic=episodic_memory,
        semantic=semantic_memory,
        query_processor=query_processor,
        reranker=reranker,
    )


def make_entry(
    content: str = "test content",
    role: Role = Role.USER,
    conversation_id: str = "conv-1",
    **kwargs,
) -> MemoryEntry:
    """Helper to create a MemoryEntry with defaults."""
    return MemoryEntry(
        content=content,
        role=role,
        conversation_id=conversation_id,
        **kwargs,
    )


def make_search_result(
    content: str = "test content",
    score: float = 0.9,
    **kwargs,
) -> SearchResult:
    """Helper to create a SearchResult with defaults."""
    return SearchResult(
        entry=make_entry(content=content, **kwargs),
        score=score,
    )
