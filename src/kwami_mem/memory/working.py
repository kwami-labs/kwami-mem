"""kwami-mem — Working memory (in-memory sliding window)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from kwami_mem.models import ConversationTurn, Modality, Role


class WorkingMemory:
    """In-memory sliding window of recent conversation turns.

    Maintains per-conversation buffers with a configurable maximum size.
    When the window is exceeded, the oldest turns are evicted.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._buffers: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._turn_counters: dict[str, int] = defaultdict(int)

    def add(
        self,
        conversation_id: str,
        role: Role,
        content: str,
        *,
        modality: Modality = Modality.TEXT,
    ) -> ConversationTurn:
        """Add a new turn to working memory.

        Args:
            conversation_id: Conversation thread identifier.
            role: Who produced this content.
            content: Message content.
            modality: Content modality type.

        Returns:
            The created ConversationTurn.
        """
        turn_index = self._turn_counters[conversation_id]
        self._turn_counters[conversation_id] += 1

        turn = ConversationTurn(
            role=role,
            content=content,
            modality=modality,
            timestamp=datetime.now(timezone.utc),
            turn_index=turn_index,
        )

        buffer = self._buffers[conversation_id]
        buffer.append(turn)

        # Evict oldest if over limit
        if len(buffer) > self._max_turns:
            self._buffers[conversation_id] = buffer[-self._max_turns :]

        return turn

    def get(self, conversation_id: str, *, last_n: int | None = None) -> list[ConversationTurn]:
        """Get recent turns from a conversation.

        Args:
            conversation_id: Conversation to retrieve.
            last_n: Number of most recent turns. None = all in window.

        Returns:
            List of conversation turns, oldest first.
        """
        buffer = self._buffers.get(conversation_id, [])
        if last_n is not None:
            return buffer[-last_n:]
        return list(buffer)

    def get_turn_count(self, conversation_id: str) -> int:
        """Get the total turn count for a conversation (including evicted turns)."""
        return self._turn_counters.get(conversation_id, 0)

    def clear(self, conversation_id: str) -> None:
        """Clear all working memory for a conversation."""
        self._buffers.pop(conversation_id, None)
        self._turn_counters.pop(conversation_id, None)

    def clear_all(self) -> None:
        """Clear all working memory for all conversations."""
        self._buffers.clear()
        self._turn_counters.clear()

    @property
    def active_conversations(self) -> list[str]:
        """List conversation IDs with active working memory."""
        return list(self._buffers.keys())
