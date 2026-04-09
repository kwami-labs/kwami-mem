"""Tests for kwami_mem.client (KwamiMemory main class)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kwami_mem.client import KwamiMemory
from kwami_mem.models import MemoryType, Role


@pytest.fixture
async def memory() -> KwamiMemory:
    """KwamiMemory instance backed by in-memory Qdrant + mock Gemini."""
    with patch("kwami_mem.client.GeminiEmbeddingProvider") as MockEmbed:
        mock_instance = MagicMock()
        fake_vector = [0.1] * 768
        mock_instance.embed_text = AsyncMock(return_value=fake_vector)
        mock_instance.embed_texts = AsyncMock(return_value=[fake_vector])
        mock_instance.embed_image = AsyncMock(return_value=fake_vector)
        mock_instance.embed_audio = AsyncMock(return_value=fake_vector)
        mock_instance.embed_pdf = AsyncMock(return_value=[fake_vector])
        MockEmbed.return_value = mock_instance

        mem = KwamiMemory(
            gemini_api_key="test-key",
            qdrant_url=None,  # In-memory mode
            collection_name="test_client",
            embedding_dimensions=768,
            user_id="test-user",
        )
        await mem.initialize()
        yield mem


class TestKwamiMemory:
    """Integration tests for the main KwamiMemory client."""

    async def test_add_and_count(self, memory: KwamiMemory):
        await memory.add("user", "Hello!", conversation_id="conv-1")
        await memory.add("assistant", "Hi there!", conversation_id="conv-1")
        count = await memory.count()
        assert count == 2

    async def test_add_auto_generates_conversation_id(self, memory: KwamiMemory):
        entry = await memory.add("user", "Test message")
        assert entry.conversation_id != ""
        assert len(entry.conversation_id) > 0

    async def test_search_returns_results(self, memory: KwamiMemory):
        await memory.add("user", "Python is my favorite language", conversation_id="c1")
        results = await memory.search("Python")
        assert len(results) >= 1

    async def test_get_history(self, memory: KwamiMemory):
        await memory.add("user", "Hello", conversation_id="conv-hist")
        await memory.add("assistant", "Hi", conversation_id="conv-hist")

        history = await memory.get_history("conv-hist")
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi"

    async def test_get_history_last_n(self, memory: KwamiMemory):
        for i in range(5):
            await memory.add("user", f"Message {i}", conversation_id="conv-n")

        history = await memory.get_history("conv-n", last_n=2)
        assert len(history) == 2

    async def test_get_full_history(self, memory: KwamiMemory):
        await memory.add("user", "Msg 1", conversation_id="conv-full")
        await memory.add("assistant", "Reply 1", conversation_id="conv-full")

        full = await memory.get_full_history("conv-full")
        assert len(full) == 2
        assert full[0].turn_index == 0
        assert full[1].turn_index == 1

    async def test_get_context(self, memory: KwamiMemory):
        await memory.add("user", "I love Python programming", conversation_id="ctx-1")
        await memory.add("assistant", "Python is great indeed!", conversation_id="ctx-1")

        context = await memory.get_context("Python", conversation_id="ctx-1")
        assert len(context.working_memory) >= 1
        prompt = context.to_prompt_string()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    async def test_delete_conversation(self, memory: KwamiMemory):
        await memory.add("user", "To be deleted", conversation_id="del-1")
        assert await memory.count() >= 1

        deleted = await memory.delete_conversation("del-1")
        assert deleted >= 1

        # Working memory should also be cleared
        history = await memory.get_history("del-1")
        assert history == []

    async def test_add_fact(self, memory: KwamiMemory):
        entry = await memory.add_fact(
            "User prefers dark mode",
            topics=["preferences", "ui"],
        )
        assert entry.memory_type == MemoryType.SEMANTIC
        assert entry.content == "User prefers dark mode"

    async def test_not_initialized_raises(self):
        with patch("kwami_mem.client.GeminiEmbeddingProvider"):
            mem = KwamiMemory(gemini_api_key="test-key")
            with pytest.raises(RuntimeError, match="not initialized"):
                await mem.add("user", "test")

    async def test_user_id(self, memory: KwamiMemory):
        assert memory.user_id == "test-user"

    async def test_settings_accessible(self, memory: KwamiMemory):
        assert memory.settings.gemini_api_key == "test-key"
        assert memory.settings.collection_name == "test_client"
