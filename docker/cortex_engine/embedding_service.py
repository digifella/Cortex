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
from .utils.performance_monitor import measure
from .utils.gpu_monitor import get_optimal_batch_size, log_gpu_status

logger = get_logger(__name__)


_model_lock = threading.Lock()
_model: Optional[SentenceTransformer] = None
_optimal_batch_size: Optional[int] = None  # Cached optimal batch size


def _load_model() -> SentenceTransformer:
    global _model, _optimal_batch_size
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

            # Calculate optimal batch size for this device
            _optimal_batch_size = get_optimal_batch_size(model_name=EMBED_MODEL, conservative=True)
            log_gpu_status()

    return _model


def get_recommended_batch_size() -> int:
    """
    Get the recommended batch size for the current device.

    This is calculated once when the model is loaded and cached.

    Returns:
        Optimal batch size (4-128 depending on GPU memory)
    """
    global _optimal_batch_size

    # Ensure model is loaded (which calculates batch size)
    if _optimal_batch_size is None:
        _load_model()

    return _optimal_batch_size or 32  # Fallback to default


def embed_query(text: str) -> List[float]:
    """Return a single embedding vector for a query string."""
    model = _load_model()
    # BGE models benefit from normalization
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist() if hasattr(vec, 'tolist') else list(vec)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Return embedding vectors for multiple texts.
    Now uses optimized batch processing for efficiency.
    """
    if not texts:
        return []

    # Use batch processing if more than 1 text
    if len(texts) > 1:
        return embed_texts_batch(texts, batch_size=16)

    # Single text - use direct encoding
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=1)
    if hasattr(vecs, 'tolist'):
        return vecs.tolist()
    return [list(v) for v in vecs]


def embed_texts_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in optimized batches.

    This function utilizes GPU/CPU vectorization by processing texts in batches
    rather than one-at-a-time, resulting in 2-5x speedup for large document sets.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process per batch (default 32, optimal for most GPUs)

    Returns:
        List of embedding vectors (one per text)

    Performance:
        - CPU: ~2x faster than sequential processing
        - GPU: ~5x faster than sequential processing
        - Batch size 32 is optimal for most NVIDIA GPUs (8-16GB VRAM)
    """
    if not texts:
        return []

    # Track performance metrics for the entire batch operation
    with measure("embedding_batch", batch_size=batch_size, doc_count=len(texts)):
        model = _load_model()
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        if len(texts) > batch_size:
            logger.info(f"🔢 Generating embeddings for {len(texts)} texts in {total_batches} batches (size: {batch_size})")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = (i // batch_size) + 1

            if len(texts) > batch_size:
                logger.debug(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

            # Use batch encoding for efficiency
            vecs = model.encode(
                batch,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False
            )

            if hasattr(vecs, 'tolist'):
                all_embeddings.extend(vecs.tolist())
            else:
                all_embeddings.extend([list(v) for v in vecs])

        if len(texts) > batch_size:
            logger.info(f"✅ Embedding generation complete: {len(all_embeddings)} vectors")

        return all_embeddings


def embed_documents_efficient(documents: List[str]) -> List[List[float]]:
    """
    Optimized embedding generation specifically for document ingestion.
    Uses adaptive batch size based on available GPU memory for optimal throughput.

    The batch size is automatically determined based on:
    - GPU memory availability (CUDA)
    - Device type (CUDA/MPS/CPU)
    - Conservative safety margins to avoid OOM errors

    Typical batch sizes:
    - 24GB+ GPU: 128
    - 16GB GPU: 64
    - 8-12GB GPU: 32
    - 4-8GB GPU: 16
    - CPU/MPS: 4-16

    Args:
        documents: List of document texts

    Returns:
        List of embedding vectors

    Usage:
        >>> docs = ["Document 1 text", "Document 2 text", ...]
        >>> embeddings = embed_documents_efficient(docs)
    """
    # Use adaptive batch sizing
    optimal_batch = get_recommended_batch_size()
    return embed_texts_batch(documents, batch_size=optimal_batch)

