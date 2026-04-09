"""kwami-mem — Configuration & Settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class KwamiSettings(BaseSettings):
    """Configuration for kwami-mem, loaded from environment variables or constructor."""

    model_config = {"env_prefix": "", "case_sensitive": False}

    # --- Gemini ---
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key (from https://aistudio.google.com/)",
        alias="GEMINI_API_KEY",
    )

    # --- Qdrant ---
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL. None = in-memory mode (dev).",
        alias="QDRANT_URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key for authenticated clusters.",
        alias="QDRANT_API_KEY",
    )

    # --- Collection ---
    collection_name: str = Field(
        default="kwami_memory",
        description="Qdrant collection name for memory storage.",
        alias="KWAMI_COLLECTION",
    )

    # --- Embedding ---
    embedding_model: str = Field(
        default="gemini-embedding-2-preview",
        description="Gemini embedding model identifier.",
        alias="KWAMI_EMBEDDING_MODEL",
    )
    embedding_dimensions: int = Field(
        default=768,
        description="Embedding output dimensionality (MRL: 768, 1536, or 3072).",
        alias="KWAMI_EMBEDDING_DIMS",
    )

    # --- Memory ---
    working_memory_size: int = Field(
        default=20,
        description="Number of recent turns kept in working memory.",
        alias="KWAMI_WORKING_MEMORY",
    )
    default_search_limit: int = Field(
        default=10,
        description="Default number of results returned by search.",
        alias="KWAMI_SEARCH_LIMIT",
    )

    # --- Identity ---
    user_id: str = Field(
        default="default",
        description="User identifier for multi-user isolation.",
        alias="KWAMI_USER_ID",
    )
