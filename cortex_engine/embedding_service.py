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
            # Auto-detect best available device (CUDA GPU > MPS > CPU)
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"🚀 Using NVIDIA GPU for embeddings (CUDA available)")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info(f"🚀 Using Apple Silicon GPU for embeddings (MPS available)")
            else:
                device = "cpu"
                logger.info(f"💻 Using CPU for embeddings (no GPU detected)")

            logger.info(f"Loading embedding model: {EMBED_MODEL} on {device}")
            # Normalize embeddings improves Chroma recall for BGE models
            _model = SentenceTransformer(EMBED_MODEL, device=device)
            logger.info(f"✅ Embedding model loaded on {device}")
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

