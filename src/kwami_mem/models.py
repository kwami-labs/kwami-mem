"""kwami-mem — Data models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Conversation participant role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Modality(str, Enum):
    """Content modality type."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PDF = "pdf"
    VIDEO = "video"


class MemoryType(str, Enum):
    """Classification of the memory layer."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryEntry(BaseModel):
    """A single memory point stored in the vector database."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="Text content or description of the memory")
    role: Role = Field(..., description="Who produced this content")
    conversation_id: str = Field(..., description="Conversation thread identifier")
    user_id: str = Field(default="default", description="User who owns this memory")
    memory_type: MemoryType = Field(
        default=MemoryType.EPISODIC, description="Episodic or semantic"
    )
    modality: Modality = Field(default=Modality.TEXT, description="Content modality")
    turn_index: int = Field(default=0, description="Position within the conversation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this memory was created",
    )
    content_hash: str = Field(default="", description="SHA-256 hash for deduplication")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (topics, entities, etc.)"
    )

    def to_payload(self) -> dict[str, Any]:
        """Convert to Qdrant payload dict."""
        return {
            "content": self.content,
            "role": self.role.value,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "modality": self.modality.value,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp.isoformat(),
            "content_hash": self.content_hash,
            **self.metadata,
        }


class SearchResult(BaseModel):
    """A retrieved memory with relevance score."""

    entry: MemoryEntry
    score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    combined_score: float = Field(
        default=0.0,
        description="Final score after reranking (blends semantic + recency + context)",
    )


class ConversationTurn(BaseModel):
    """A single conversation message for working memory."""

    role: Role
    content: str
    modality: Modality = Modality.TEXT
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_index: int = 0


class MemoryContext(BaseModel):
    """Combined context from all memory layers, ready for prompt injection."""

    working_memory: list[ConversationTurn] = Field(
        default_factory=list, description="Recent conversation turns"
    )
    episodic_memories: list[SearchResult] = Field(
        default_factory=list, description="Retrieved past conversation fragments"
    )
    semantic_memories: list[SearchResult] = Field(
        default_factory=list, description="Retrieved facts and knowledge"
    )

    def to_prompt_string(self) -> str:
        """Format all memories into a string suitable for LLM prompt injection."""
        sections: list[str] = []

        if self.semantic_memories:
            facts = "\n".join(
                f"- {m.entry.content}" for m in self.semantic_memories
            )
            sections.append(f"## Known Facts & Preferences\n{facts}")

        if self.episodic_memories:
            episodes = "\n".join(
                f"- [{m.entry.role.value}] ({m.entry.timestamp:%Y-%m-%d}): {m.entry.content}"
                for m in self.episodic_memories
            )
            sections.append(f"## Relevant Past Conversations\n{episodes}")

        if self.working_memory:
            recent = "\n".join(
                f"- [{t.role.value}]: {t.content}" for t in self.working_memory
            )
            sections.append(f"## Current Conversation\n{recent}")

        return "\n\n".join(sections)
