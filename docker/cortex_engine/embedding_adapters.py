"""
Embedding adapters to integrate the centralized embedding_service with libraries
that expect a LlamaIndex-style embedding interface.

Supports both standard text embedding (BGE/NV-Embed) and multimodal embedding
(Qwen3-VL) depending on configuration.

Option A adapter: a thin wrapper that delegates to embed_query/embed_texts so
ingest uses the exact same vectors as search and async ingest.
"""

from __future__ import annotations

from typing import List, Optional, Union
from pathlib import Path

from .embedding_service import (
    embed_query,
    embed_texts,
    embed_image,
    embed_multimodal,
    embed_images_batch,
    is_multimodal_enabled,
    get_embedding_info,
    get_embedding_dimension,
)
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


try:
    # Try new LlamaIndex 0.14+ path first
    from llama_index.core.embeddings import BaseEmbedding  # type: ignore
except Exception:
    try:
        # Alternative path in older versions
        from llama_index.embeddings.base import BaseEmbedding  # type: ignore
    except Exception:
        try:
            from llama_index.core.embeddings.base import BaseEmbedding  # type: ignore
        except Exception:
            # Fallback to object to avoid hard dependency; duck-typing should still work
            BaseEmbedding = object  # type: ignore


class EmbeddingServiceAdapter(BaseEmbedding):  # type: ignore
    """
    Adapter exposing embedding_service via LlamaIndex's embedding interface.

    Supports both text-only (BGE/NV-Embed) and multimodal (Qwen3-VL) embedding
    depending on configuration. When Qwen3-VL is enabled, additional methods
    for image embedding are available.
    """

    # Pydantic field for model name (required by new LlamaIndex BaseEmbedding)
    model_name: str = "embedding-service-adapter"

    def __init__(self, model_name: str | None = None, **kwargs):
        # Call parent init for Pydantic compatibility
        super().__init__(model_name=model_name or "embedding-service-adapter", **kwargs)
        self._multimodal = is_multimodal_enabled()

        if self._multimodal:
            logger.info("EmbeddingServiceAdapter initialized with Qwen3-VL (multimodal)")
        else:
            logger.info("EmbeddingServiceAdapter initialized (delegates to embedding_service)")

    @property
    def embed_dim(self) -> int:
        """Return embedding dimension (LlamaIndex compatibility)."""
        return get_embedding_dimension()

    @property
    def is_multimodal(self) -> bool:
        """Check if multimodal (image) embedding is available."""
        return self._multimodal

    # Abstract methods required by new LlamaIndex BaseEmbedding (0.14+)
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding (internal method for BaseEmbedding)."""
        return embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding (internal method for BaseEmbedding)."""
        return embed_query(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get query embedding (internal method for BaseEmbedding)."""
        # For now, just call sync version - can be made truly async later
        return embed_query(query)

    # Public methods (for backwards compatibility with older LlamaIndex)
    def get_query_embedding(self, query: str) -> List[float]:  # noqa: D401
        return self._get_query_embedding(query)

    def get_text_embedding(self, text: str) -> List[float]:  # noqa: D401
        return self._get_text_embedding(text)

    # Batch embeddings
    def get_text_embeddings(self, texts: List[str], *args, **kwargs) -> List[List[float]]:  # noqa: D401
        # Accept optional args/kwargs like show_progress used by some LlamaIndex versions
        return embed_texts(texts)

    # Some LlamaIndex versions call this batch method name
    def get_text_embedding_batch(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        # Accept optional args/kwargs like show_progress
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

    # -------------------------------------------------------------------------
    # Multimodal Methods (Qwen3-VL only)
    # -------------------------------------------------------------------------

    def get_image_embedding(self, image_path: Union[str, Path]) -> List[float]:
        """
        Generate embedding for an image (Qwen3-VL only).

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector in same space as text embeddings

        Raises:
            RuntimeError: If Qwen3-VL is not enabled
        """
        return embed_image(image_path)

    def get_image_embeddings(self, image_paths: List[Union[str, Path]]) -> List[List[float]]:
        """
        Generate embeddings for multiple images (Qwen3-VL only).

        Args:
            image_paths: List of paths to image files

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If Qwen3-VL is not enabled
        """
        return embed_images_batch(image_paths)

    def get_multimodal_embedding(
        self,
        text: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None
    ) -> List[float]:
        """
        Generate embedding for mixed text + image content (Qwen3-VL only).

        Args:
            text: Optional text content
            image_path: Optional path to image file

        Returns:
            Combined embedding vector

        Raises:
            RuntimeError: If Qwen3-VL is not enabled
        """
        return embed_multimodal(text=text, image_path=image_path)

    def get_info(self) -> dict:
        """Get information about the current embedding backend."""
        return get_embedding_info()
