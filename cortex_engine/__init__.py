"""
Cortex Engine
=============

Core engine components for the Cortex Suite knowledge management system.

Modules:
- embedding_service: Standard text embedding using BGE/NV-Embed models
- qwen3_vl_embedding_service: Multimodal embedding using Qwen3-VL
- qwen3_vl_reranker_service: Neural reranking using Qwen3-VL
- qwen3_vl_llamaindex_adapter: LlamaIndex integration for Qwen3-VL
"""

# Version
__version__ = "3.2.0"

# Lazy imports to avoid loading heavy dependencies on import
def get_embedding_service():
    """Get the standard embedding service."""
    from .embedding_service import embed_query, embed_texts, embed_texts_batch
    return embed_query, embed_texts, embed_texts_batch


def get_qwen3_vl_services():
    """Get Qwen3-VL embedding and reranking services."""
    from .qwen3_vl_embedding_service import (
        Qwen3VLEmbeddingService,
        get_embedding_service as get_qwen3_vl_embedding_service,
        embed_text,
        embed_image,
        embed_multimodal,
    )
    from .qwen3_vl_reranker_service import (
        Qwen3VLRerankerService,
        get_reranker_service,
        rerank_results,
    )
    return {
        "embedding_service": get_qwen3_vl_embedding_service,
        "reranker_service": get_reranker_service,
        "embed_text": embed_text,
        "embed_image": embed_image,
        "embed_multimodal": embed_multimodal,
        "rerank_results": rerank_results,
    }


def get_llamaindex_adapters():
    """Get LlamaIndex-compatible adapters for Qwen3-VL."""
    from .qwen3_vl_llamaindex_adapter import (
        Qwen3VLEmbedding,
        Qwen3VLReranker,
        configure_llamaindex_with_qwen3_vl,
    )
    return {
        "Qwen3VLEmbedding": Qwen3VLEmbedding,
        "Qwen3VLReranker": Qwen3VLReranker,
        "configure": configure_llamaindex_with_qwen3_vl,
    }
