"""kwami-mem — Embedding pipeline."""

from kwami_mem.embedding.base import EmbeddingProvider
from kwami_mem.embedding.gemini import GeminiEmbeddingProvider
from kwami_mem.embedding.local import LocalEmbeddingProvider

__all__ = ["EmbeddingProvider", "GeminiEmbeddingProvider", "LocalEmbeddingProvider"]
