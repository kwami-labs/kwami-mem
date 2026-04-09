"""Tests for kwami_mem.retrieval (query processor, reranker, retriever)."""

from datetime import datetime, timedelta, timezone

import pytest

from kwami_mem.models import MemoryEntry, MemoryType, Role, SearchResult
from kwami_mem.retrieval.query import QueryProcessor
from kwami_mem.retrieval.reranker import Reranker
from kwami_mem.retrieval.retriever import MemoryRetriever
from tests.conftest import make_search_result


class TestQueryProcessor:
    """Tests for QueryProcessor."""

    def test_basic_query_passthrough(self, query_processor: QueryProcessor):
        cleaned, filters = query_processor.process("What is Python?")
        assert cleaned == "What is Python?"
        assert filters == {}

    def test_yesterday_extraction(self, query_processor: QueryProcessor):
        cleaned, filters = query_processor.process("What did we discuss yesterday?")
        assert "time_after" in filters
        assert "yesterday" not in cleaned.lower()

    def test_last_week_extraction(self, query_processor: QueryProcessor):
        _, filters = query_processor.process("Show me conversations from last week")
        assert "time_after" in filters

    def test_n_days_ago(self, query_processor: QueryProcessor):
        _, filters = query_processor.process("What happened 3 days ago?")
        assert "time_after" in filters

    def test_role_hint_user(self, query_processor: QueryProcessor):
        _, filters = query_processor.process("What did I say about Python?")
        assert filters.get("role") == "user"

    def test_role_hint_assistant(self, query_processor: QueryProcessor):
        _, filters = query_processor.process("What did you say about Python?")
        assert filters.get("role") == "assistant"

    def test_recently_extraction(self, query_processor: QueryProcessor):
        _, filters = query_processor.process("What did we discuss recently?")
        assert "time_after" in filters

    def test_no_empty_query_after_cleaning(self, query_processor: QueryProcessor):
        cleaned, _ = query_processor.process("yesterday")
        assert len(cleaned.strip()) > 0


class TestReranker:
    """Tests for Reranker."""

    def test_basic_reranking(self, reranker: Reranker):
        results = [
            make_search_result("low score", score=0.3),
            make_search_result("high score", score=0.9),
        ]
        reranked = reranker.rerank(results)
        assert reranked[0].entry.content == "high score"

    def test_recency_boost(self, reranker: Reranker):
        now = datetime.now(timezone.utc)
        old_entry = MemoryEntry(
            content="old", role=Role.USER, conversation_id="c1",
            timestamp=now - timedelta(days=30),
        )
        new_entry = MemoryEntry(
            content="new", role=Role.USER, conversation_id="c1",
            timestamp=now,
        )

        results = [
            SearchResult(entry=old_entry, score=0.8),
            SearchResult(entry=new_entry, score=0.75),
        ]
        reranked = reranker.rerank(results)
        # The newer entry should have a higher combined score
        # despite the slightly lower semantic score
        assert reranked[0].entry.content == "new"

    def test_context_boost(self, reranker: Reranker):
        results = [
            make_search_result("other conv", score=0.85, conversation_id="other"),
            make_search_result("same conv", score=0.80, conversation_id="current"),
        ]
        reranked = reranker.rerank(results, current_conversation_id="current")
        # Same conversation should get boosted
        assert reranked[0].entry.content == "same conv"

    def test_empty_results(self, reranker: Reranker):
        assert reranker.rerank([]) == []

    def test_combined_score_set(self, reranker: Reranker):
        results = [make_search_result("test", score=0.8)]
        reranked = reranker.rerank(results)
        assert reranked[0].combined_score > 0


class TestMemoryRetriever:
    """Tests for MemoryRetriever (integration with mock embedder)."""

    async def test_search_returns_results(self, retriever: MemoryRetriever, episodic_memory):
        # First store something
        await episodic_memory.store("Python is great", Role.USER, "conv-1", 0)

        results = await retriever.search("Python", limit=5)
        assert len(results) >= 1

    async def test_get_context_structure(self, retriever: MemoryRetriever):
        # Add to working memory
        retriever._working.add("conv-1", Role.USER, "Hello")
        retriever._working.add("conv-1", Role.ASSISTANT, "Hi there")

        context = await retriever.get_context("Hello", conversation_id="conv-1")
        assert len(context.working_memory) == 2
        assert isinstance(context.to_prompt_string(), str)

    async def test_get_context_empty(self, retriever: MemoryRetriever):
        context = await retriever.get_context("anything")
        assert context.working_memory == []
        assert isinstance(context.to_prompt_string(), str)
