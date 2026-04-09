"""Tests for kwami_mem.memory (working, episodic, semantic)."""

import pytest

from kwami_mem.models import Modality, Role
from kwami_mem.memory.working import WorkingMemory
from kwami_mem.memory.episodic import EpisodicMemory
from kwami_mem.memory.semantic import SemanticMemory


class TestWorkingMemory:
    """Tests for WorkingMemory."""

    def test_add_and_get(self, working_memory: WorkingMemory):
        working_memory.add("conv-1", Role.USER, "Hello!")
        working_memory.add("conv-1", Role.ASSISTANT, "Hi there!")

        turns = working_memory.get("conv-1")
        assert len(turns) == 2
        assert turns[0].role == Role.USER
        assert turns[0].content == "Hello!"
        assert turns[1].role == Role.ASSISTANT

    def test_sliding_window_eviction(self):
        wm = WorkingMemory(max_turns=3)
        for i in range(5):
            wm.add("conv-1", Role.USER, f"Message {i}")

        turns = wm.get("conv-1")
        assert len(turns) == 3
        # Should have the last 3 messages
        assert turns[0].content == "Message 2"
        assert turns[2].content == "Message 4"

    def test_per_conversation_isolation(self, working_memory: WorkingMemory):
        working_memory.add("conv-1", Role.USER, "Hello conv 1")
        working_memory.add("conv-2", Role.USER, "Hello conv 2")

        assert len(working_memory.get("conv-1")) == 1
        assert len(working_memory.get("conv-2")) == 1

    def test_get_last_n(self, working_memory: WorkingMemory):
        for i in range(5):
            working_memory.add("conv-1", Role.USER, f"Msg {i}")

        turns = working_memory.get("conv-1", last_n=2)
        assert len(turns) == 2
        assert turns[0].content == "Msg 3"
        assert turns[1].content == "Msg 4"

    def test_turn_counter_increments(self, working_memory: WorkingMemory):
        working_memory.add("conv-1", Role.USER, "First")
        working_memory.add("conv-1", Role.ASSISTANT, "Second")

        assert working_memory.get_turn_count("conv-1") == 2
        turns = working_memory.get("conv-1")
        assert turns[0].turn_index == 0
        assert turns[1].turn_index == 1

    def test_clear(self, working_memory: WorkingMemory):
        working_memory.add("conv-1", Role.USER, "Hello")
        working_memory.clear("conv-1")
        assert working_memory.get("conv-1") == []
        assert working_memory.get_turn_count("conv-1") == 0

    def test_active_conversations(self, working_memory: WorkingMemory):
        working_memory.add("conv-1", Role.USER, "A")
        working_memory.add("conv-2", Role.USER, "B")
        assert set(working_memory.active_conversations) == {"conv-1", "conv-2"}

    def test_modality_stored(self, working_memory: WorkingMemory):
        working_memory.add("conv-1", Role.USER, "Image", modality=Modality.IMAGE)
        turns = working_memory.get("conv-1")
        assert turns[0].modality == Modality.IMAGE

    def test_empty_conversation(self, working_memory: WorkingMemory):
        turns = working_memory.get("nonexistent")
        assert turns == []

    def test_clear_all(self, working_memory: WorkingMemory):
        working_memory.add("conv-1", Role.USER, "A")
        working_memory.add("conv-2", Role.USER, "B")
        working_memory.clear_all()
        assert working_memory.active_conversations == []


class TestEpisodicMemory:
    """Tests for EpisodicMemory (uses mock embedder + in-memory Qdrant)."""

    async def test_store_and_search(self, episodic_memory: EpisodicMemory):
        await episodic_memory.store(
            content="Paris is the capital of France",
            role=Role.ASSISTANT,
            conversation_id="conv-1",
            turn_index=0,
        )

        results = await episodic_memory.search("capital of France", limit=5)
        assert len(results) == 1
        assert "Paris" in results[0].entry.content

    async def test_deduplication(self, episodic_memory: EpisodicMemory):
        # Store same content twice
        await episodic_memory.store(
            content="Hello world",
            role=Role.USER,
            conversation_id="conv-1",
            turn_index=0,
        )
        await episodic_memory.store(
            content="Hello world",
            role=Role.USER,
            conversation_id="conv-1",
            turn_index=0,
        )

        results = await episodic_memory.search("Hello", limit=10)
        assert len(results) == 1

    async def test_get_conversation(self, episodic_memory: EpisodicMemory):
        await episodic_memory.store("Msg 0", Role.USER, "conv-order", 0)
        await episodic_memory.store("Msg 1", Role.ASSISTANT, "conv-order", 1)
        await episodic_memory.store("Msg 2", Role.USER, "conv-order", 2)

        history = await episodic_memory.get_conversation("conv-order")
        assert len(history) == 3
        assert history[0].turn_index == 0
        assert history[2].turn_index == 2


class TestSemanticMemory:
    """Tests for SemanticMemory (uses mock embedder + in-memory Qdrant)."""

    async def test_store_fact_and_search(self, semantic_memory: SemanticMemory):
        await semantic_memory.store_fact(
            "The user prefers dark mode interfaces",
            topics=["preferences", "ui"],
        )

        results = await semantic_memory.search("dark mode", limit=5)
        assert len(results) == 1
        assert "dark mode" in results[0].entry.content

    async def test_fact_deduplication(self, semantic_memory: SemanticMemory):
        await semantic_memory.store_fact("User likes Python")
        await semantic_memory.store_fact("User likes Python")

        results = await semantic_memory.search("Python", limit=10)
        assert len(results) == 1
