"""kwami-mem — Gemini Embedding 2 provider."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from functools import partial
from pathlib import Path

from google import genai
from google.genai import types

from kwami_mem.embedding.base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Google Gemini Embedding 2.

    Supports text, image, audio, and PDF content via the unified
    multimodal embedding space.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2-preview",
        dimensions: int = 768,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    def _make_config(self, task_type: str = "RETRIEVAL_DOCUMENT") -> types.EmbedContentConfig:
        """Build embedding config with task type and output dimensions."""
        return types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimensions,
        )

    async def _run_sync(self, fn, *args, **kwargs):
        """Run a synchronous SDK call in a thread pool to keep async."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

    # --- Text ---

    async def embed_text(
        self,
        text: str,
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[float]:
        """Embed a single text string."""
        response = await self._run_sync(
            self._client.models.embed_content,
            model=self._model,
            contents=text,
            config=self._make_config(task_type),
        )
        return list(response.embeddings[0].values)

    async def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Embed multiple texts in batch."""
        response = await self._run_sync(
            self._client.models.embed_content,
            model=self._model,
            contents=texts,
            config=self._make_config(task_type),
        )
        return [list(e.values) for e in response.embeddings]

    # --- Image ---

    async def embed_image(self, image_path: str | Path) -> list[float]:
        """Embed an image file (PNG, JPEG)."""
        path = Path(image_path)
        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        image_data = path.read_bytes()
        b64 = base64.standard_b64encode(image_data).decode("utf-8")

        content = types.Content(
            parts=[types.Part(inline_data=types.Blob(data=b64, mime_type=mime_type))]
        )
        response = await self._run_sync(
            self._client.models.embed_content,
            model=self._model,
            contents=content,
            config=self._make_config("RETRIEVAL_DOCUMENT"),
        )
        return list(response.embeddings[0].values)

    # --- Audio ---

    async def embed_audio(self, audio_path: str | Path) -> list[float]:
        """Embed an audio file (MP3, WAV)."""
        path = Path(audio_path)
        mime_type = mimetypes.guess_type(str(path))[0] or "audio/mpeg"
        audio_data = path.read_bytes()
        b64 = base64.standard_b64encode(audio_data).decode("utf-8")

        content = types.Content(
            parts=[types.Part(inline_data=types.Blob(data=b64, mime_type=mime_type))]
        )
        response = await self._run_sync(
            self._client.models.embed_content,
            model=self._model,
            contents=content,
            config=self._make_config("RETRIEVAL_DOCUMENT"),
        )
        return list(response.embeddings[0].values)

    # --- PDF ---

    async def embed_pdf(self, pdf_path: str | Path) -> list[list[float]]:
        """Embed a PDF file. Returns one embedding per page (up to 6 pages per request)."""
        path = Path(pdf_path)
        pdf_data = path.read_bytes()
        b64 = base64.standard_b64encode(pdf_data).decode("utf-8")

        content = types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=b64, mime_type="application/pdf")
                )
            ]
        )
        response = await self._run_sync(
            self._client.models.embed_content,
            model=self._model,
            contents=content,
            config=self._make_config("RETRIEVAL_DOCUMENT"),
        )
        return [list(e.values) for e in response.embeddings]
