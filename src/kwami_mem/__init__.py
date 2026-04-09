"""kwami-mem — Long-term memory management for LLMs.

Store and retrieve multimodal conversation memories with high accuracy
using Gemini Embedding 2 and Qdrant vector database.

Usage::

    from kwami_mem import KwamiMemory

    mem = KwamiMemory()
    await mem.initialize()

    await mem.add("user", "Hello!", conversation_id="conv-1")
    await mem.add("assistant", "Hi there!", conversation_id="conv-1")

    results = await mem.search("greeting")
    context = await mem.get_context("What did we talk about?", conversation_id="conv-1")
"""

from kwami_mem.client import KwamiMemory
from kwami_mem.config import KwamiSettings
from kwami_mem.models import (
    ConversationTurn,
    MemoryContext,
    MemoryEntry,
    MemoryType,
    Modality,
    Role,
    SearchResult,
)

__version__ = "0.1.0"

__all__ = [
    "KwamiMemory",
    "KwamiSettings",
    "MemoryEntry",
    "SearchResult",
    "ConversationTurn",
    "MemoryContext",
    "MemoryType",
    "Modality",
    "Role",
]
