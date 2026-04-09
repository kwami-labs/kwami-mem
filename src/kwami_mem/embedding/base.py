"""kwami-mem — Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(
        self,
        text: str,
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text content to embed.
            task_type: Optimization hint — "RETRIEVAL_DOCUMENT" for storage,
                       "RETRIEVAL_QUERY" for search queries.

        Returns:
            A list of floats representing the embedding vector.
        """

    @abstractmethod
    async def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Embed multiple text strings in batch.

        Args:
            texts: List of text content to embed.
            task_type: Optimization hint for the embedding model.

        Returns:
            A list of embedding vectors.
        """

    @abstractmethod
    async def embed_image(self, image_path: str | Path) -> list[float]:
        """Embed an image file.

        Args:
            image_path: Path to the image file (PNG, JPEG).

        Returns:
            A list of floats representing the embedding vector.
        """

    @abstractmethod
    async def embed_audio(self, audio_path: str | Path) -> list[float]:
        """Embed an audio file.

        Args:
            audio_path: Path to the audio file (MP3, WAV).

        Returns:
            A list of floats representing the embedding vector.
        """

    @abstractmethod
    async def embed_pdf(self, pdf_path: str | Path) -> list[list[float]]:
        """Embed a PDF file (one embedding per page).

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A list of embedding vectors (one per page).
        """
