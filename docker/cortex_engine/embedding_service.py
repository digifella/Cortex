"""
Embedding service
Centralized, cached access to embedding models used across ingest and search
so we never drift between pipelines.

Supports two embedding backends:
1. SentenceTransformer (BGE/NV-Embed) - Default, text-only
2. Qwen3-VL - Multimodal (text, images, video) when QWEN3_VL_ENABLED=true

Uses adaptive model selection via get_embed_model() which auto-detects
the best model for available hardware (Qwen3-VL > NV-Embed > BGE).
"""

from __future__ import annotations

from typing import List, Optional, Union
from pathlib import Path
import threading
import os

from .config import get_embed_model, QWEN3_VL_ENABLED, QWEN3_VL_MODEL_SIZE
from .utils.logging_utils import get_logger
from .utils.performance_monitor import measure
from .utils.gpu_monitor import get_optimal_batch_size, log_gpu_status

logger = get_logger(__name__)

# Force offline mode for HuggingFace transformers library
# This prevents unnecessary internet checks when model is already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ============================================================================
# Backend Selection
# ============================================================================

def _using_qwen3_vl() -> bool:
    """Check if Qwen3-VL embedding is enabled."""
    return QWEN3_VL_ENABLED


_model_lock = threading.Lock()
_model = None  # SentenceTransformer or None
_qwen3_service = None  # Qwen3VLEmbeddingService or None
_optimal_batch_size: Optional[int] = None  # Cached optimal batch size


def _load_qwen3_vl_service():
    """Load Qwen3-VL embedding service (lazy import to avoid loading if not used)."""
    global _qwen3_service
    if _qwen3_service is not None:
        return _qwen3_service

    with _model_lock:
        if _qwen3_service is None:
            from .qwen3_vl_embedding_service import (
                Qwen3VLEmbeddingService,
                Qwen3VLConfig,
                Qwen3VLModelSize,
            )

            # Determine model size from config
            if QWEN3_VL_MODEL_SIZE == "2B":
                config = Qwen3VLConfig.for_model_size(Qwen3VLModelSize.SMALL)
            elif QWEN3_VL_MODEL_SIZE == "8B":
                config = Qwen3VLConfig.for_model_size(Qwen3VLModelSize.LARGE)
            else:  # auto
                config = Qwen3VLConfig.auto_select()

            _qwen3_service = Qwen3VLEmbeddingService(config)
            logger.info(f"✅ Qwen3-VL embedding service initialized: {config.model_name}")

    return _qwen3_service


def _load_sentence_transformer_model():
    """Load SentenceTransformer model (BGE/NV-Embed)."""
    global _model, _optimal_batch_size

    # Lazy import to avoid loading if using Qwen3-VL
    from sentence_transformers import SentenceTransformer

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

            # Get adaptive model selection
            embed_model = get_embed_model()
            logger.info(f"Loading embedding model: {embed_model} on {device}")
            # Normalize embeddings improves Chroma recall for BGE models
            try:
                # Try loading in offline mode first (HF_HUB_OFFLINE env var is set above)
                _model = SentenceTransformer(embed_model, device=device, trust_remote_code=True)
                logger.info(f"✅ Embedding model loaded from cache on {device} (offline mode)")
            except Exception as offline_error:
                # If offline mode fails, temporarily enable online mode and try downloading
                logger.warning(f"Cached model not found, attempting download (requires internet)")
                logger.debug(f"Offline error details: {offline_error}")
                try:
                    # Temporarily disable offline mode for download
                    os.environ["HF_HUB_OFFLINE"] = "0"
                    os.environ["TRANSFORMERS_OFFLINE"] = "0"
                    _model = SentenceTransformer(embed_model, device=device, trust_remote_code=True)
                    logger.info(f"✅ Embedding model downloaded and loaded on {device}")
                    # Re-enable offline mode after successful download
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                except Exception as download_error:
                    logger.error(f"❌ Failed to load embedding model (offline and online)")
                    logger.error(f"Offline error: {str(offline_error)[:200]}")
                    logger.error(f"Download error: {str(download_error)[:200]}")
                    raise RuntimeError(
                        f"Cannot load embedding model '{embed_model}'. "
                        f"Model not cached locally and internet unavailable. "
                        f"Please rebuild Docker image with internet access to pre-download the model."
                    ) from download_error

            # Calculate optimal batch size for this device
            _optimal_batch_size = get_optimal_batch_size(model_name=embed_model, conservative=True)
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
    """
    Return a single embedding vector for a query string.

    Uses Qwen3-VL if enabled, otherwise SentenceTransformer (BGE/NV-Embed).
    """
    if _using_qwen3_vl():
        service = _load_qwen3_vl_service()
        return service.embed_query(text)

    model = _load_sentence_transformer_model()
    # BGE models benefit from normalization
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist() if hasattr(vec, 'tolist') else list(vec)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Return embedding vectors for multiple texts.

    Uses Qwen3-VL if enabled, otherwise SentenceTransformer with batch processing.
    """
    if not texts:
        return []

    if _using_qwen3_vl():
        service = _load_qwen3_vl_service()
        return service.embed_texts(texts)

    # Use batch processing if more than 1 text
    if len(texts) > 1:
        return embed_texts_batch(texts, batch_size=16)

    # Single text - use direct encoding
    model = _load_sentence_transformer_model()
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

    # Qwen3-VL has its own batch processing
    if _using_qwen3_vl():
        service = _load_qwen3_vl_service()
        return service.embed_texts(texts)

    # Track performance metrics for the entire batch operation
    with measure("embedding_batch", batch_size=batch_size, doc_count=len(texts)):
        model = _load_sentence_transformer_model()
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


# ============================================================================
# Multimodal Embedding Functions (Qwen3-VL only)
# ============================================================================

def embed_image(image_path: Union[str, Path]) -> List[float]:
    """
    Generate embedding for an image file.

    Only available when Qwen3-VL is enabled. Returns vectors in the same
    space as text embeddings, enabling cross-modal search.

    Args:
        image_path: Path to image file (PNG, JPG, etc.)

    Returns:
        Embedding vector as list of floats

    Raises:
        RuntimeError: If Qwen3-VL is not enabled
    """
    if not _using_qwen3_vl():
        raise RuntimeError(
            "Image embedding requires Qwen3-VL. "
            "Enable with QWEN3_VL_ENABLED=true environment variable."
        )

    service = _load_qwen3_vl_service()
    return service.embed_image(image_path)


def embed_multimodal(
    text: Optional[str] = None,
    image_path: Optional[Union[str, Path]] = None
) -> List[float]:
    """
    Generate embedding for mixed text + image content.

    Combines text and visual information into a single embedding vector,
    useful for document pages with both text and figures.

    Args:
        text: Optional text content
        image_path: Optional path to image file

    Returns:
        Embedding vector as list of floats

    Raises:
        RuntimeError: If Qwen3-VL is not enabled
    """
    if not _using_qwen3_vl():
        raise RuntimeError(
            "Multimodal embedding requires Qwen3-VL. "
            "Enable with QWEN3_VL_ENABLED=true environment variable."
        )

    from .qwen3_vl_embedding_service import embed_multimodal as qwen_embed_multimodal
    return qwen_embed_multimodal(text=text, image=image_path)


def embed_images_batch(image_paths: List[Union[str, Path]]) -> List[List[float]]:
    """
    Generate embeddings for multiple images.

    Args:
        image_paths: List of paths to image files

    Returns:
        List of embedding vectors

    Raises:
        RuntimeError: If Qwen3-VL is not enabled
    """
    if not _using_qwen3_vl():
        raise RuntimeError(
            "Image embedding requires Qwen3-VL. "
            "Enable with QWEN3_VL_ENABLED=true environment variable."
        )

    service = _load_qwen3_vl_service()
    return service.embed_images(image_paths)


# ============================================================================
# Utility Functions
# ============================================================================

def get_embedding_info() -> dict:
    """
    Get information about the current embedding backend.

    Returns:
        Dict with embedding service info including model name, dimensions, etc.
    """
    if _using_qwen3_vl():
        service = _load_qwen3_vl_service()
        info = service.get_info()
        info["backend"] = "qwen3_vl"
        info["multimodal"] = True
        return info

    return {
        "backend": "sentence_transformers",
        "model_name": get_embed_model(),
        "multimodal": False,
        "device": "auto-detected on load",
    }


def is_multimodal_enabled() -> bool:
    """Check if multimodal (image) embedding is available."""
    return _using_qwen3_vl()


def get_embedding_dimension() -> int:
    """
    Get the embedding dimension for the current model.

    Returns:
        Embedding dimension (e.g., 768, 1536, 2048, 4096)
    """
    if _using_qwen3_vl():
        service = _load_qwen3_vl_service()
        return service.embedding_dimension

    # For SentenceTransformer models, use known dimensions or probe
    from .utils.embedding_validator import KNOWN_MODEL_DIMENSIONS
    embed_model = get_embed_model()
    if embed_model in KNOWN_MODEL_DIMENSIONS:
        return KNOWN_MODEL_DIMENSIONS[embed_model]

    # Fallback: generate a test embedding to get dimension
    test_vec = embed_query("test")
    return len(test_vec)

