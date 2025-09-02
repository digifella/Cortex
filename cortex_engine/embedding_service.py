"""
Embedding service
Centralized, cached access to the sentence-transformers embedding model used across
ingest and search so we never drift between pipelines.

Uses the model configured in cortex_engine.config.EMBED_MODEL.
"""

from __future__ import annotations

from typing import List, Optional
import threading

from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


_model_lock = threading.Lock()
_model: Optional[SentenceTransformer] = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            logger.info(f"Loading embedding model: {EMBED_MODEL}")
            # Normalize embeddings improves Chroma recall for BGE models
            _model = SentenceTransformer(EMBED_MODEL, device="cpu")
            logger.info("Embedding model loaded")
    return _model


def embed_query(text: str) -> List[float]:
    """Return a single embedding vector for a query string."""
    model = _load_model()
    # BGE models benefit from normalization
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist() if hasattr(vec, 'tolist') else list(vec)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embedding vectors for multiple texts."""
    if not texts:
        return []
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=16)
    if hasattr(vecs, 'tolist'):
        return vecs.tolist()
    return [list(v) for v in vecs]

