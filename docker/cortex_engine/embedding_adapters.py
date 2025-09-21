"""
Embedding adapters (Docker copy) to integrate the centralized embedding_service
with libraries that expect a LlamaIndex-style embedding interface.
"""

from __future__ import annotations

from typing import List

from .embedding_service import embed_query, embed_texts
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


try:
    from llama_index.embeddings.base import BaseEmbedding  # type: ignore
except Exception:
    try:
        from llama_index.core.embeddings.base import BaseEmbedding  # type: ignore
    except Exception:
        BaseEmbedding = object  # type: ignore


class EmbeddingServiceAdapter(BaseEmbedding):  # type: ignore
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        logger.info("EmbeddingServiceAdapter initialized (Docker)")

    def get_query_embedding(self, query: str) -> List[float]:
        return embed_query(query)

    def get_text_embedding(self, text: str) -> List[float]:
        return embed_query(text)

    def get_text_embeddings(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        return embed_texts(texts)

    def get_text_embedding_batch(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        return embed_texts(texts)

    def get_agg_embedding_from_queries(self, queries: List[str]) -> List[float]:
        vecs = self.get_text_embeddings(queries)
        if not vecs:
            return []
        dim = len(vecs[0])
        sums = [0.0] * dim
        for v in vecs:
            for i, val in enumerate(v):
                sums[i] += val
        return [s / len(vecs) for s in sums]
