"""kwami-mem — Content hashing for deduplication."""

from __future__ import annotations

import hashlib


def content_hash(content: str, conversation_id: str, turn_index: int) -> str:
    """Generate a deterministic SHA-256 hash for deduplication.

    Combines content, conversation ID, and turn index to uniquely
    identify a memory entry. This prevents storing the same message
    twice (e.g., on retry or reconnection).

    Args:
        content: The message content.
        conversation_id: Conversation thread identifier.
        turn_index: Position within the conversation.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = f"{conversation_id}:{turn_index}:{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
