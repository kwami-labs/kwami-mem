"""kwami-mem — Main retrieval orchestrator."""

from __future__ import annotations

from kwami_mem.memory.episodic import EpisodicMemory
from kwami_mem.memory.semantic import SemanticMemory
from kwami_mem.memory.working import WorkingMemory
from kwami_mem.models import MemoryContext, SearchResult
from kwami_mem.retrieval.query import QueryProcessor
from kwami_mem.retrieval.reranker import Reranker


class MemoryRetriever:
    """Orchestrates retrieval across all three memory layers.

    Combines working memory (in-memory), episodic memory (conversation log),
    and semantic memory (facts/entities) into a single MemoryContext
    ready for LLM prompt injection.
    """

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        query_processor: QueryProcessor | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._working = working
        self._episodic = episodic
        self._semantic = semantic
        self._query_processor = query_processor or QueryProcessor()
        self._reranker = reranker or Reranker()

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        conversation_id: str | None = None,
        include_semantic: bool = True,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search across all long-term memory layers.

        Args:
            query: Natural language search query.
            limit: Maximum total results to return.
            conversation_id: Optional filter to a specific conversation.
            include_semantic: Whether to include semantic memory results.
            score_threshold: Minimum similarity score threshold.

        Returns:
            Reranked list of SearchResult entries.
        """
        # Process query for metadata filters
        cleaned_query, query_filters = self._query_processor.process(query)

        # Search episodic memory
        episodic_results = await self._episodic.search(
            cleaned_query,
            limit=limit,
            conversation_id=conversation_id or query_filters.get("conversation_id"),
            score_threshold=score_threshold,
        )

        # Search semantic memory
        semantic_results: list[SearchResult] = []
        if include_semantic:
            semantic_results = await self._semantic.search(
                cleaned_query,
                limit=max(3, limit // 3),  # Semantic gets ~1/3 of the budget
                score_threshold=score_threshold,
            )

        # Merge and deduplicate
        all_results = self._merge_results(episodic_results, semantic_results)

        # Rerank with multi-signal scoring
        reranked = self._reranker.rerank(
            all_results,
            current_conversation_id=conversation_id,
        )

        return reranked[:limit]

    async def get_context(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        working_memory_turns: int | None = None,
        episodic_limit: int = 5,
        semantic_limit: int = 3,
    ) -> MemoryContext:
        """Build a full MemoryContext combining all three layers.

        This is the primary method for agents — it returns everything
        the LLM needs to "remember" in a single structured object.

        Args:
            query: The current user query (used for retrieval).
            conversation_id: Current conversation ID.
            working_memory_turns: Number of recent turns to include.
            episodic_limit: Max episodic memories to retrieve.
            semantic_limit: Max semantic memories to retrieve.

        Returns:
            MemoryContext ready for prompt injection.
        """
        context = MemoryContext()

        # Layer 1: Working memory (instant, in-memory)
        if conversation_id:
            context.working_memory = self._working.get(
                conversation_id, last_n=working_memory_turns
            )

        # Layer 2 & 3: Long-term retrieval
        cleaned_query, _ = self._query_processor.process(query)

        # Episodic memories
        episodic_results = await self._episodic.search(
            cleaned_query, limit=episodic_limit
        )
        context.episodic_memories = self._reranker.rerank(
            episodic_results, current_conversation_id=conversation_id
        )

        # Semantic memories
        semantic_results = await self._semantic.search(
            cleaned_query, limit=semantic_limit
        )
        context.semantic_memories = self._reranker.rerank(
            semantic_results, current_conversation_id=conversation_id
        )

        return context

    @staticmethod
    def _merge_results(
        *result_lists: list[SearchResult],
    ) -> list[SearchResult]:
        """Merge and deduplicate results from multiple sources."""
        seen_ids: set[str] = set()
        merged: list[SearchResult] = []

        for results in result_lists:
            for result in results:
                if result.entry.id not in seen_ids:
                    seen_ids.add(result.entry.id)
                    merged.append(result)

        return merged
