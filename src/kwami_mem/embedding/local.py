"""Local embedding provider using fastembed.

This provider handles dense vectors entirely locally (CPU/GPU) via ONNX,
removing the dependency on external cloud embedding APIs like Gemini.
"""

from typing import Any

from kwami_mem.embedding.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """Generates dense vectors for texts locally using fastembed."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dimensions: int = 384) -> None:
        """Initialize the local embedding model.
        
        Args:
            model: The fastembed text model identifier.
            dimensions: The expected vector dimensions of the model.
        """
        try:
            from fastembed.text.text_embedding import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "The 'fastembed' package is required for LocalEmbeddingProvider. "
                "Install it with: pip install fastembed"
            ) from e

        self._model_name = model
        self.dimensions = dimensions
        self._embedder = TextEmbedding(model_name=model)

    async def embed_text(self, text: str, task_type: str | None = None) -> list[float]:
        """Embed a single text string into a dense vector."""
        result = list(self._embedder.embed([text]))[0]
        return result.tolist()

    async def embed_texts(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        """Embed a batch of text strings into dense vectors."""
        results = list(self._embedder.embed(texts))
        return [r.tolist() for r in results]

    async def embed_image(self, image_path: str | Any) -> list[float]:
        raise NotImplementedError("Local embedding currently only supported for text.")

    async def embed_audio(self, audio_path: str | Any) -> list[float]:
        raise NotImplementedError("Local embedding currently only supported for text.")

    async def embed_pdf(self, pdf_path: str | Any) -> list[list[float]]:
        raise NotImplementedError("Local embedding currently only supported for text.")
