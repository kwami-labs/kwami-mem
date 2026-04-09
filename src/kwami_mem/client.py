"""kwami-mem — Main KwamiMemory client."""

from __future__ import annotations

import uuid
from pathlib import Path

from kwami_mem.config import KwamiSettings
from kwami_mem.embedding.gemini import GeminiEmbeddingProvider
from kwami_mem.memory.episodic import EpisodicMemory
from kwami_mem.memory.semantic import SemanticMemory
from kwami_mem.memory.working import WorkingMemory
from kwami_mem.models import (
    ConversationTurn,
    MemoryContext,
    MemoryEntry,
    Modality,
    Role,
    SearchResult,
)
from kwami_mem.processing.extractor import MetadataExtractor
from kwami_mem.retrieval.retriever import MemoryRetriever
from kwami_mem.storage.qdrant import QdrantStorage


class KwamiMemory:
    """Long-term memory management for LLMs.

    Provides a simple API to store and retrieve multimodal conversation
    memories using Gemini Embedding 2 and Qdrant vector database.

    Usage::

        from kwami_mem import KwamiMemory

        mem = KwamiMemory()  # reads config from env vars
        await mem.initialize()

        # Store a conversation turn
        await mem.add("user", "What's the capital of France?", conversation_id="abc")
        await mem.add("assistant", "Paris is the capital of France.", conversation_id="abc")

        # Search memories
        results = await mem.search("What did we discuss about France?")

        # Get full context for LLM prompt injection
        context = await mem.get_context("Tell me about France", conversation_id="abc")
        print(context.to_prompt_string())
    """

    def __init__(
        self,
        *,
        gemini_api_key: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        working_memory_size: int | None = None,
        user_id: str | None = None,
    ) -> None:
        """Initialize KwamiMemory.

        All parameters are optional — values are read from environment
        variables if not provided. See .env.example for details.

        Args:
            gemini_api_key: Google Gemini API key.
            qdrant_url: Qdrant server URL. None = in-memory mode.
            qdrant_api_key: Qdrant API key for authenticated clusters.
            collection_name: Qdrant collection name.
            embedding_model: Gemini embedding model identifier.
            embedding_dimensions: Embedding output dimensionality.
            working_memory_size: Number of recent turns in working memory.
            user_id: User identifier for multi-user isolation.
        """
        # Build settings — constructor args override env vars
        overrides: dict = {}
        if gemini_api_key is not None:
            overrides["GEMINI_API_KEY"] = gemini_api_key
        if qdrant_url is not None:
            overrides["QDRANT_URL"] = qdrant_url
        if qdrant_api_key is not None:
            overrides["QDRANT_API_KEY"] = qdrant_api_key
        if collection_name is not None:
            overrides["KWAMI_COLLECTION"] = collection_name
        if embedding_model is not None:
            overrides["KWAMI_EMBEDDING_MODEL"] = embedding_model
        if embedding_dimensions is not None:
            overrides["KWAMI_EMBEDDING_DIMS"] = str(embedding_dimensions)
        if working_memory_size is not None:
            overrides["KWAMI_WORKING_MEMORY"] = str(working_memory_size)
        if user_id is not None:
            overrides["KWAMI_USER_ID"] = user_id

        self._settings = KwamiSettings(**overrides) if overrides else KwamiSettings()
        self._initialized = False

        # Components (created in __init__, initialized in initialize())
        self._embedder = GeminiEmbeddingProvider(
            api_key=self._settings.gemini_api_key,
            model=self._settings.embedding_model,
            dimensions=self._settings.embedding_dimensions,
        )
        self._storage = QdrantStorage(
            url=self._settings.qdrant_url,
            api_key=self._settings.qdrant_api_key,
            collection_name=self._settings.collection_name,
            vector_size=self._settings.embedding_dimensions,
        )
        self._extractor = MetadataExtractor()
        self._working = WorkingMemory(max_turns=self._settings.working_memory_size)
        self._episodic = EpisodicMemory(
            storage=self._storage,
            embedder=self._embedder,
            extractor=self._extractor,
            user_id=self._settings.user_id,
        )
        self._semantic = SemanticMemory(
            storage=self._storage,
            embedder=self._embedder,
            extractor=self._extractor,
            user_id=self._settings.user_id,
        )
        self._retriever = MemoryRetriever(
            working=self._working,
            episodic=self._episodic,
            semantic=self._semantic,
        )

    async def initialize(self) -> None:
        """Initialize storage backend (create collection, indexes).

        Must be called once before using add/search/get_context.
        """
        await self._storage.initialize()
        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Raise if initialize() hasn't been called."""
        if not self._initialized:
            raise RuntimeError(
                "KwamiMemory not initialized. Call `await mem.initialize()` first."
            )

    # ──────────────────────────────────────────────
    # Store
    # ──────────────────────────────────────────────

    async def add(
        self,
        role: str,
        content: str,
        *,
        conversation_id: str | None = None,
        store_semantic: bool = True,
    ) -> MemoryEntry:
        """Store a conversation message in memory.

        This is the primary storage method. It:
        1. Adds to working memory (in-memory buffer)
        2. Embeds and stores in episodic memory (Qdrant)
        3. Optionally extracts facts for semantic memory

        Args:
            role: Who said this — "user", "assistant", "system", or "tool".
            content: The message text content.
            conversation_id: Conversation thread ID. Auto-generated if None.
            store_semantic: Whether to extract and store semantic facts.

        Returns:
            The stored MemoryEntry.
        """
        self._ensure_initialized()

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        parsed_role = Role(role)

        # Layer 1: Working memory
        turn = self._working.add(conversation_id, parsed_role, content)

        # Layer 2: Episodic memory
        entry = await self._episodic.store(
            content=content,
            role=parsed_role,
            conversation_id=conversation_id,
            turn_index=turn.turn_index,
        )

        # Layer 3: Semantic memory (extract facts from user messages)
        if store_semantic and parsed_role == Role.USER and len(content) > 30:
            await self._semantic.extract_and_store(content, conversation_id)

        return entry

    async def add_image(
        self,
        image_path: str | Path,
        *,
        role: str = "user",
        conversation_id: str | None = None,
        description: str = "",
    ) -> MemoryEntry:
        """Store an image in memory.

        Args:
            image_path: Path to the image file (PNG, JPEG).
            role: Who sent this image.
            conversation_id: Conversation thread ID.
            description: Optional text description of the image.

        Returns:
            The stored MemoryEntry.
        """
        self._ensure_initialized()

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        parsed_role = Role(role)
        content_desc = description or f"[Image: {Path(image_path).name}]"

        # Working memory
        turn = self._working.add(
            conversation_id, parsed_role, content_desc, modality=Modality.IMAGE
        )

        # Embed image
        vector = await self._embedder.embed_image(image_path)

        # Store in episodic memory
        entry = await self._episodic.store_multimodal(
            vector=vector,
            content_description=content_desc,
            role=parsed_role,
            conversation_id=conversation_id,
            turn_index=turn.turn_index,
            modality=Modality.IMAGE,
        )

        return entry

    async def add_audio(
        self,
        audio_path: str | Path,
        *,
        role: str = "user",
        conversation_id: str | None = None,
        description: str = "",
    ) -> MemoryEntry:
        """Store an audio file in memory.

        Args:
            audio_path: Path to the audio file (MP3, WAV).
            role: Who sent this audio.
            conversation_id: Conversation thread ID.
            description: Optional text description of the audio.

        Returns:
            The stored MemoryEntry.
        """
        self._ensure_initialized()

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        parsed_role = Role(role)
        content_desc = description or f"[Audio: {Path(audio_path).name}]"

        turn = self._working.add(
            conversation_id, parsed_role, content_desc, modality=Modality.AUDIO
        )

        vector = await self._embedder.embed_audio(audio_path)

        entry = await self._episodic.store_multimodal(
            vector=vector,
            content_description=content_desc,
            role=parsed_role,
            conversation_id=conversation_id,
            turn_index=turn.turn_index,
            modality=Modality.AUDIO,
        )

        return entry

    async def add_pdf(
        self,
        pdf_path: str | Path,
        *,
        role: str = "user",
        conversation_id: str | None = None,
        description: str = "",
    ) -> list[MemoryEntry]:
        """Store a PDF document in memory (one entry per page).

        Args:
            pdf_path: Path to the PDF file.
            role: Who sent this PDF.
            conversation_id: Conversation thread ID.
            description: Optional text description of the PDF.

        Returns:
            List of MemoryEntry instances (one per page).
        """
        self._ensure_initialized()

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        parsed_role = Role(role)
        base_desc = description or f"[PDF: {Path(pdf_path).name}]"

        # Embed PDF (returns one vector per page)
        page_vectors = await self._embedder.embed_pdf(pdf_path)

        entries: list[MemoryEntry] = []
        for page_idx, vector in enumerate(page_vectors):
            page_desc = f"{base_desc} (page {page_idx + 1}/{len(page_vectors)})"

            turn = self._working.add(
                conversation_id, parsed_role, page_desc, modality=Modality.PDF
            )

            entry = await self._episodic.store_multimodal(
                vector=vector,
                content_description=page_desc,
                role=parsed_role,
                conversation_id=conversation_id,
                turn_index=turn.turn_index,
                modality=Modality.PDF,
            )
            entries.append(entry)

        return entries

    async def add_fact(
        self,
        fact: str,
        *,
        source_conversation_id: str = "",
        topics: list[str] | None = None,
        entities: list[str] | None = None,
    ) -> MemoryEntry:
        """Manually store a fact in semantic memory.

        Use this for explicitly teaching the agent something.

        Args:
            fact: The fact or piece of knowledge.
            source_conversation_id: Where this fact came from.
            topics: Topic tags.
            entities: Named entities.

        Returns:
            The stored MemoryEntry.
        """
        self._ensure_initialized()

        return await self._semantic.store_fact(
            fact,
            source_conversation_id=source_conversation_id,
            topics=topics,
            entities=entities,
        )

    # ──────────────────────────────────────────────
    # Retrieve
    # ──────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        conversation_id: str | None = None,
        include_semantic: bool = True,
    ) -> list[SearchResult]:
        """Search all long-term memories.

        Args:
            query: Natural language search query.
            limit: Maximum results. None = use default from config.
            conversation_id: Optional filter to a specific conversation.
            include_semantic: Whether to include semantic memory results.

        Returns:
            Reranked list of relevant memories.
        """
        self._ensure_initialized()

        return await self._retriever.search(
            query,
            limit=limit or self._settings.default_search_limit,
            conversation_id=conversation_id,
            include_semantic=include_semantic,
        )

    async def get_context(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        working_memory_turns: int | None = None,
        episodic_limit: int = 5,
        semantic_limit: int = 3,
    ) -> MemoryContext:
        """Get full memory context for LLM prompt injection.

        Combines working memory, episodic memories, and semantic facts
        into a single context object with a `.to_prompt_string()` method.

        Args:
            query: The current user query.
            conversation_id: Current conversation ID.
            working_memory_turns: Recent turns to include (None = all).
            episodic_limit: Max episodic memories to retrieve.
            semantic_limit: Max semantic memories to retrieve.

        Returns:
            MemoryContext with all relevant memories.
        """
        self._ensure_initialized()

        return await self._retriever.get_context(
            query,
            conversation_id=conversation_id,
            working_memory_turns=working_memory_turns,
            episodic_limit=episodic_limit,
            semantic_limit=semantic_limit,
        )

    async def get_history(
        self,
        conversation_id: str,
        *,
        last_n: int | None = None,
    ) -> list[ConversationTurn]:
        """Get conversation history from working memory.

        Args:
            conversation_id: The conversation to retrieve.
            last_n: Number of most recent turns. None = all in buffer.

        Returns:
            List of ConversationTurn objects.
        """
        return self._working.get(conversation_id, last_n=last_n)

    async def get_full_history(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Get full conversation history from long-term storage.

        Unlike get_history which only returns working memory,
        this retrieves all stored turns from Qdrant.

        Args:
            conversation_id: The conversation to retrieve.
            limit: Maximum number of entries.

        Returns:
            List of MemoryEntry objects ordered by turn_index.
        """
        self._ensure_initialized()

        return await self._episodic.get_conversation(
            conversation_id, limit=limit
        )

    # ──────────────────────────────────────────────
    # Management
    # ──────────────────────────────────────────────

    async def delete_conversation(self, conversation_id: str) -> int:
        """Delete all memories from a conversation.

        Args:
            conversation_id: The conversation to delete.

        Returns:
            Number of entries deleted.
        """
        self._ensure_initialized()

        self._working.clear(conversation_id)
        return await self._storage.delete_conversation(conversation_id)

    async def count(self) -> int:
        """Return total number of stored memories for the current user."""
        self._ensure_initialized()
        return await self._storage.count(user_id=self._settings.user_id)

    @property
    def settings(self) -> KwamiSettings:
        """Access current configuration."""
        return self._settings

    @property
    def user_id(self) -> str:
        """Current user ID."""
        return self._settings.user_id
