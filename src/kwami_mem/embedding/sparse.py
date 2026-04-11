"""Sparse embedding provider using fastembed for BM25/SPLADE.

This provider handles lightweight sparse vectors that are ideal for exact
keyword matching, to be combined with semantic dense vectors in a Hybrid Search.
"""

from typing import Any

from kwami_mem.embedding.base import EmbeddingProvider


class SparseEmbeddingProvider(EmbeddingProvider):
    """Generates sparse vectors (e.g. BM25) for texts using fastembed."""

    def __init__(self, model: str = "Qdrant/bm25") -> None:
        """Initialize the sparse embedding model.
        
        Args:
            model: The fastembed sparse model identifier. Default is Qdrant/bm25.
        """
        try:
            from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
        except ImportError as e:
            raise ImportError(
                "The 'fastembed' package is required for SparseEmbeddingProvider. "
                "Install it with: pip install fastembed"
            ) from e

        self._model_name = model
        self._embedder = SparseTextEmbedding(model_name=model)

    async def embed_text(self, text: str, task_type: str | None = None) -> Any:
        """Embed a single text string into a sparse vector."""
        # fastembed returns an iterator, we just extract the first (and only) result.
        result = list(self._embedder.embed([text]))[0]
        # Qdrant accepts SparseVector models, or dictionaries.
        # fastembed SparseEmbedding objects have .indices and .values
        return {"indices": result.indices.tolist(), "values": result.values.tolist()}

    async def embed_texts(self, texts: list[str], task_type: str | None = None) -> list[Any]:
        """Embed a batch of text strings into sparse vectors."""
        results = list(self._embedder.embed(texts))
        return [
            {"indices": r.indices.tolist(), "values": r.values.tolist()}
            for r in results
        ]

    async def embed_image(self, image_path: str | Any) -> list[float]:
        """Not supported for sparse embeddings."""
        raise NotImplementedError("Sparse embeddings are only supported for text.")

    async def embed_audio(self, audio_path: str | Any) -> list[float]:
        """Not supported for sparse embeddings."""
        raise NotImplementedError("Sparse embeddings are only supported for text.")

    async def embed_pdf(self, pdf_path: str | Any) -> list[list[float]]:
        """Not supported for sparse embeddings."""
        raise NotImplementedError("Sparse embeddings are only supported for text.")
