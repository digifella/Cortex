"""
Embedding adapters to integrate the centralized embedding_service with libraries
that expect a LlamaIndex-style embedding interface.

Option A adapter: a thin wrapper that delegates to embed_query/embed_texts so
ingest uses the exact same vectors as search and async ingest.
"""

from __future__ import annotations

from typing import List

from .embedding_service import embed_query, embed_texts
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


try:
    # Preferred import path in most LlamaIndex versions
    from llama_index.embeddings.base import BaseEmbedding  # type: ignore
except Exception:
    try:
        # Alternative path in some versions
        from llama_index.core.embeddings.base import BaseEmbedding  # type: ignore
    except Exception:
        # Fallback to object to avoid hard dependency; duck-typing should still work
        BaseEmbedding = object  # type: ignore


class EmbeddingServiceAdapter(BaseEmbedding):  # type: ignore
    """Adapter exposing embedding_service via LlamaIndex's embedding interface."""

    def __init__(self, model_name: str | None = None):
        # model_name is optional and informational; embedding_service reads from config
        self.model_name = model_name
        logger.info("EmbeddingServiceAdapter initialized (delegates to embedding_service)")

    # Query/text single embedding
    def get_query_embedding(self, query: str) -> List[float]:  # noqa: D401
        return embed_query(query)

    def get_text_embedding(self, text: str) -> List[float]:  # noqa: D401
        return embed_query(text)

    # Batch embeddings
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:  # noqa: D401
        return embed_texts(texts)

    # Some LlamaIndex versions call this batch method name
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts)

    # Optional utility used by some pipelines (aggregate multi-query)
    def get_agg_embedding_from_queries(self, queries: List[str]) -> List[float]:
        vecs = self.get_text_embeddings(queries)
        if not vecs:
            return []
        # Simple average
        dim = len(vecs[0])
        sums = [0.0] * dim
        for v in vecs:
            for i, val in enumerate(v):
                sums[i] += val
        return [s / len(vecs) for s in sums]

