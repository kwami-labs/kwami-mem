"""Tests for kwami_mem.embedding (Gemini provider with mocks)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kwami_mem.embedding.gemini import GeminiEmbeddingProvider


class TestGeminiEmbeddingProvider:
    """Tests for GeminiEmbeddingProvider using mock embedder from conftest."""

    async def test_embed_text_called(self, mock_embedder, fake_vector):
        result = await mock_embedder.embed_text("Hello world")
        assert result == fake_vector
        mock_embedder.embed_text.assert_awaited_once_with("Hello world")

    async def test_embed_texts_batch(self, mock_embedder, fake_vector):
        result = await mock_embedder.embed_texts(["Hello", "World"])
        assert result == [fake_vector]
        mock_embedder.embed_texts.assert_awaited_once()

    async def test_embed_image_called(self, mock_embedder, fake_vector):
        result = await mock_embedder.embed_image("/tmp/test.jpg")
        assert result == fake_vector

    async def test_embed_audio_called(self, mock_embedder, fake_vector):
        result = await mock_embedder.embed_audio("/tmp/test.mp3")
        assert result == fake_vector

    async def test_embed_pdf_returns_pages(self, mock_embedder, fake_vector):
        result = await mock_embedder.embed_pdf("/tmp/test.pdf")
        assert len(result) == 2  # Mock returns 2 pages
        assert result[0] == fake_vector

    async def test_task_type_respected(self, mock_embedder):
        await mock_embedder.embed_text("query", task_type="RETRIEVAL_QUERY")
        mock_embedder.embed_text.assert_awaited_with(
            "query", task_type="RETRIEVAL_QUERY"
        )
