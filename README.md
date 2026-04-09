# kwami-mem

> Long-term memory management for LLMs — multimodal embedding & intelligent retrieval powered by Gemini and Qdrant.

**kwami-mem** gives your AI agents perfect memory. It stores all conversations (text, images, audio, PDFs) as embeddings in a vector database and retrieves the most relevant context when needed.

## Features

- 🧠 **Three-layer memory** — Working (in-memory), Episodic (conversation log), Semantic (facts & preferences)
- 🎨 **Multimodal** — Embed text, images, audio, and PDFs via Gemini Embedding 2's unified vector space
- 🔍 **Intelligent retrieval** — Hybrid search with temporal awareness, conversation context boosting, and multi-signal reranking
- 🚀 **Simple API** — Dead-simple async interface: `add()`, `search()`, `get_context()`
- 📦 **Zero server setup** — In-memory Qdrant for development, remote server for production
- 🔒 **Multi-user isolation** — Built-in user scoping for multi-tenant agents
- ⚡ **Deduplication** — Content hashing prevents storing duplicates

## Installation

```bash
pip install kwami-mem
```

Or with uv:
```bash
uv add kwami-mem
```

## Quick Start

```python
import asyncio
from kwami_mem import KwamiMemory

async def main():
    # Initialize (reads GEMINI_API_KEY from environment)
    mem = KwamiMemory()
    await mem.initialize()

    # Store conversation messages
    await mem.add("user", "I'm working on a Python ML project", conversation_id="conv-1")
    await mem.add("assistant", "I can help with that! What framework are you using?", conversation_id="conv-1")
    await mem.add("user", "I prefer PyTorch over TensorFlow", conversation_id="conv-1")

    # Search memories
    results = await mem.search("What ML framework does the user prefer?")
    for r in results:
        print(f"[{r.score:.2f}] {r.entry.content}")

    # Get full context for LLM prompt injection
    context = await mem.get_context(
        "Help me with my project",
        conversation_id="conv-1"
    )
    print(context.to_prompt_string())

asyncio.run(main())
```

## Multimodal Memory

```python
# Store images
await mem.add_image("screenshot.png", conversation_id="conv-1")

# Store audio
await mem.add_audio("voice_note.mp3", conversation_id="conv-1")

# Store PDFs (one embedding per page)
await mem.add_pdf("research_paper.pdf", conversation_id="conv-1")

# All modalities are searchable with text queries!
results = await mem.search("What was in that screenshot?")
```

## Semantic Facts

```python
# Manually teach the agent facts
await mem.add_fact(
    "User prefers dark mode and uses VS Code",
    topics=["preferences", "tools"]
)

# Facts are automatically retrieved alongside conversation context
context = await mem.get_context("Set up the IDE")
# → includes the dark mode preference in context.semantic_memories
```

## Configuration

All settings can be passed via constructor or environment variables:

```python
mem = KwamiMemory(
    gemini_api_key="your-key",        # or GEMINI_API_KEY
    qdrant_url="http://localhost:6333", # or QDRANT_URL (None = in-memory)
    collection_name="my_agent_memory", # or KWAMI_COLLECTION
    embedding_dimensions=768,          # or KWAMI_EMBEDDING_DIMS
    working_memory_size=20,            # or KWAMI_WORKING_MEMORY
    user_id="user-123",                # or KWAMI_USER_ID
)
```

See [`.env.example`](.env.example) for all available options.

## Architecture

```
┌─────────────────────────────────────────────┐
│              KwamiMemory Client             │
├──────────┬──────────┬───────────────────────┤
│ Working  │ Episodic │ Semantic              │
│ Memory   │ Memory   │ Memory                │
│ (RAM)    │ (Qdrant) │ (Qdrant)              │
├──────────┴──────────┴───────────────────────┤
│          Retrieval Pipeline                 │
│  Query Processing → Search → Reranking     │
├─────────────────────────────────────────────┤
│          Gemini Embedding 2                 │
│  Text · Image · Audio · PDF → Vectors      │
├─────────────────────────────────────────────┤
│          Qdrant Vector Database             │
│  Cosine similarity · Payload indexes        │
└─────────────────────────────────────────────┘
```

## Development

```bash
# Clone and install
git clone https://github.com/kwami-io/kwami-mem.git
cd kwami-mem
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/
```

## License

MIT
