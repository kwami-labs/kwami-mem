"""kwami-mem — Embedding pipeline."""

from kwami_mem.embedding.base import EmbeddingProvider
from kwami_mem.embedding.gemini import GeminiEmbeddingProvider

__all__ = ["EmbeddingProvider", "GeminiEmbeddingProvider"]
