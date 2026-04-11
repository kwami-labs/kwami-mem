"""Cross-Encoder re-ranker provider using sentence-transformers.

This module provides re-ranking capabilities to take an expanded set of
candidates (e.g. from a vector search) and re-score them against a query
for high precision.
"""

from kwami_mem.models import SearchResult


class CrossEncoderReranker:
    """Reranks search results against a query using a Cross-Encoder."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialize the cross encoder model.
        
        Args:
            model: HuggingFace cross-encoder identifier.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            ) from e

        self._model_name = model
        self._encoder = CrossEncoder(model)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 10) -> list[SearchResult]:
        """Rescore and rerank the SearchResults against the given query.

        Args:
            query: The question or query string.
            results: A list of candidate SearchResults.
            top_k: Number of results to return.

        Returns:
            The top_k SearchResults sorted by highest relevance.
        """
        if not results:
            return []

        # Build query-document pairs
        pairs = [[query, r.entry.content] for r in results]
        
        # Predict scores
        scores = self._encoder.predict(pairs)

        # Update scores on the result objects
        for r, score in zip(results, scores):
            r.score = float(score)

        # Sort descending by the new scores and truncate
        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
