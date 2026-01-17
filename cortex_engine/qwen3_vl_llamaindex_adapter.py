"""
Qwen3-VL LlamaIndex Adapter
============================

Version: 1.0.0
Date: 2026-01-17

Provides LlamaIndex-compatible embedding and node postprocessor classes
for Qwen3-VL models, enabling drop-in replacement for existing LlamaIndex
RAG pipelines.

Usage:
    from cortex_engine.qwen3_vl_llamaindex_adapter import (
        Qwen3VLEmbedding,
        Qwen3VLReranker
    )

    # As embedding model for VectorStoreIndex
    embed_model = Qwen3VLEmbedding()
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model
    )

    # As node postprocessor for reranking
    reranker = Qwen3VLReranker(top_n=5)
    query_engine = index.as_query_engine(
        node_postprocessors=[reranker]
    )
"""

from __future__ import annotations

from typing import List, Optional, Any, Dict, Sequence
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from .qwen3_vl_embedding_service import (
    Qwen3VLEmbeddingService,
    Qwen3VLConfig,
    EmbeddingInput,
    embed_batch,
    get_embedding_service,
)
from .qwen3_vl_reranker_service import (
    Qwen3VLRerankerService,
    Qwen3VLRerankerConfig,
    Document,
    rerank_documents,
    get_reranker_service,
)
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# LlamaIndex Embedding Adapter
# ============================================================================

class Qwen3VLEmbedding(BaseEmbedding):
    """
    LlamaIndex-compatible embedding model using Qwen3-VL.

    This adapter allows Qwen3-VL multimodal embeddings to be used
    directly with LlamaIndex's VectorStoreIndex and query engines.

    Attributes:
        model_name: The Qwen3-VL model identifier
        embed_batch_size: Batch size for embedding generation
        mrl_dim: Optional Matryoshka dimension reduction

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> embed_model = Qwen3VLEmbedding(model_size="8B")
        >>> index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    """

    model_name: str = "Qwen/Qwen3-VL-Embedding-8B"
    embed_batch_size: int = 8
    mrl_dim: Optional[int] = None
    _service: Optional[Qwen3VLEmbeddingService] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_size: Optional[str] = None,  # "2B" or "8B" or "auto"
        mrl_dim: Optional[int] = None,
        embed_batch_size: int = 8,
        **kwargs
    ):
        """
        Initialize Qwen3-VL embedding adapter.

        Args:
            model_name: Full model name (e.g., "Qwen/Qwen3-VL-Embedding-8B")
            model_size: Shorthand size ("2B", "8B", or "auto" for auto-selection)
            mrl_dim: Optional dimension reduction (64, 128, 256, 512, 1024, 2048)
            embed_batch_size: Batch size for embedding generation
            **kwargs: Additional arguments passed to BaseEmbedding
        """
        # Determine model name
        if model_name:
            final_model_name = model_name
        elif model_size:
            if model_size == "2B":
                final_model_name = "Qwen/Qwen3-VL-Embedding-2B"
            elif model_size == "8B":
                final_model_name = "Qwen/Qwen3-VL-Embedding-8B"
            else:  # auto
                from .utils.smart_model_selector import get_optimal_qwen3_vl_embedding_model
                final_model_name = get_optimal_qwen3_vl_embedding_model()
        else:
            # Default to auto-selection
            from .utils.smart_model_selector import get_optimal_qwen3_vl_embedding_model
            final_model_name = get_optimal_qwen3_vl_embedding_model()

        super().__init__(
            model_name=final_model_name,
            embed_batch_size=embed_batch_size,
            **kwargs
        )

        self.mrl_dim = mrl_dim
        self._config = Qwen3VLConfig(
            model_name=final_model_name,
            mrl_dim=mrl_dim,
            max_batch_size=embed_batch_size,
        )

        logger.info(f"Initialized Qwen3VLEmbedding: {final_model_name}")
        if mrl_dim:
            logger.info(f"  MRL dimension reduction: {mrl_dim}")

    def _get_service(self) -> Qwen3VLEmbeddingService:
        """Get or create embedding service."""
        if self._service is None:
            self._service = Qwen3VLEmbeddingService(self._config)
        return self._service

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.

        Args:
            query: Query text

        Returns:
            Embedding vector as list of floats
        """
        service = self._get_service()
        return service.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text string.

        Args:
            text: Text content

        Returns:
            Embedding vector as list of floats
        """
        service = self._get_service()
        return service.embed_query(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        service = self._get_service()
        return service.embed_texts(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of query embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of text embedding."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of batch text embedding."""
        return self._get_text_embeddings(texts)

    # -------------------------------------------------------------------------
    # Multimodal Extensions (beyond standard LlamaIndex interface)
    # -------------------------------------------------------------------------

    def embed_image(self, image_path: str) -> List[float]:
        """
        Generate embedding for an image.

        This extends the standard LlamaIndex interface to support
        multimodal embeddings.

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector as list of floats
        """
        service = self._get_service()
        return service.embed_image(image_path)

    def embed_multimodal(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for mixed text + image content.

        Args:
            text: Optional text content
            image_path: Optional image path

        Returns:
            Embedding vector as list of floats
        """
        from .qwen3_vl_embedding_service import embed_multimodal
        return embed_multimodal(text=text, image=image_path, config=self._config)


# ============================================================================
# LlamaIndex Node Postprocessor for Reranking
# ============================================================================

class Qwen3VLReranker(BaseNodePostprocessor):
    """
    LlamaIndex-compatible node postprocessor using Qwen3-VL reranker.

    This adapter enables Qwen3-VL multimodal reranking as a postprocessing
    step in LlamaIndex query engines, significantly improving retrieval precision.

    Attributes:
        top_n: Number of top results to return after reranking
        model: Qwen3-VL reranker model name

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> reranker = Qwen3VLReranker(top_n=5)
        >>> query_engine = index.as_query_engine(
        ...     node_postprocessors=[reranker],
        ...     similarity_top_k=20  # Get more candidates for reranking
        ... )
    """

    top_n: int = 5
    model: str = "Qwen/Qwen3-VL-Reranker-8B"
    _service: Optional[Qwen3VLRerankerService] = None

    def __init__(
        self,
        top_n: int = 5,
        model_name: Optional[str] = None,
        model_size: Optional[str] = None,  # "2B" or "8B" or "auto"
        **kwargs
    ):
        """
        Initialize Qwen3-VL reranker postprocessor.

        Args:
            top_n: Number of top results to return
            model_name: Full model name
            model_size: Shorthand size ("2B", "8B", or "auto")
            **kwargs: Additional arguments
        """
        # Determine model name
        if model_name:
            final_model_name = model_name
        elif model_size:
            if model_size == "2B":
                final_model_name = "Qwen/Qwen3-VL-Reranker-2B"
            elif model_size == "8B":
                final_model_name = "Qwen/Qwen3-VL-Reranker-8B"
            else:  # auto
                from .utils.smart_model_selector import get_optimal_qwen3_vl_reranker_model
                final_model_name = get_optimal_qwen3_vl_reranker_model()
        else:
            from .utils.smart_model_selector import get_optimal_qwen3_vl_reranker_model
            final_model_name = get_optimal_qwen3_vl_reranker_model()

        super().__init__(top_n=top_n, model=final_model_name, **kwargs)

        self._config = Qwen3VLRerankerConfig(
            model_name=final_model_name,
            default_top_k=top_n,
        )

        logger.info(f"Initialized Qwen3VLReranker: {final_model_name}, top_n={top_n}")

    def _get_service(self) -> Qwen3VLRerankerService:
        """Get or create reranker service."""
        if self._service is None:
            self._service = Qwen3VLRerankerService(self._config)
        return self._service

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Rerank nodes by relevance to query.

        Args:
            nodes: List of nodes with scores from initial retrieval
            query_bundle: Query information

        Returns:
            Reranked list of nodes with updated scores
        """
        if not nodes or not query_bundle:
            return nodes

        query_text = query_bundle.query_str

        # Convert nodes to Document objects for reranker
        documents = []
        for i, node_with_score in enumerate(nodes):
            node = node_with_score.node

            # Extract text content
            if isinstance(node, TextNode):
                content = node.get_content()
            else:
                content = str(node)

            # Check for image metadata
            image_path = None
            if hasattr(node, 'metadata'):
                image_path = node.metadata.get('image_path') or node.metadata.get('figure_path')

            doc = Document(
                content=content,
                image_path=image_path,
                original_score=node_with_score.score,
                metadata=node.metadata if hasattr(node, 'metadata') else {}
            )
            documents.append(doc)

        # Rerank
        logger.info(f"Reranking {len(documents)} nodes for query: {query_text[:50]}...")
        reranked = rerank_documents(
            query_text,
            documents,
            top_k=self.top_n,
            config=self._config
        )

        # Convert back to NodeWithScore objects
        result_nodes = []
        for rr in reranked:
            # Find original node
            original_idx = rr.document.original_rank
            if original_idx is not None and original_idx < len(nodes):
                original_node = nodes[original_idx].node

                # Create new NodeWithScore with rerank score
                new_node_with_score = NodeWithScore(
                    node=original_node,
                    score=rr.rerank_score
                )

                # Add rerank metadata
                if hasattr(original_node, 'metadata'):
                    original_node.metadata['rerank_score'] = rr.rerank_score
                    original_node.metadata['original_score'] = rr.original_score
                    original_node.metadata['rank_change'] = rr.rank_change

                result_nodes.append(new_node_with_score)

        logger.info(f"Reranking complete: {len(result_nodes)} nodes returned")
        return result_nodes


# ============================================================================
# Factory Functions
# ============================================================================

def create_qwen3_vl_embed_model(
    model_size: str = "auto",
    mrl_dim: Optional[int] = None
) -> Qwen3VLEmbedding:
    """
    Factory function to create Qwen3-VL embedding model for LlamaIndex.

    Args:
        model_size: "2B", "8B", or "auto" for auto-selection
        mrl_dim: Optional dimension reduction

    Returns:
        Qwen3VLEmbedding instance
    """
    return Qwen3VLEmbedding(model_size=model_size, mrl_dim=mrl_dim)


def create_qwen3_vl_reranker(
    top_n: int = 5,
    model_size: str = "auto"
) -> Qwen3VLReranker:
    """
    Factory function to create Qwen3-VL reranker for LlamaIndex.

    Args:
        top_n: Number of top results to return
        model_size: "2B", "8B", or "auto" for auto-selection

    Returns:
        Qwen3VLReranker instance
    """
    return Qwen3VLReranker(top_n=top_n, model_size=model_size)


# ============================================================================
# Integration Helper
# ============================================================================

def configure_llamaindex_with_qwen3_vl(
    use_reranker: bool = True,
    embedding_size: str = "auto",
    reranker_size: str = "auto",
    mrl_dim: Optional[int] = None,
    reranker_top_n: int = 5,
    similarity_top_k: int = 20
) -> Dict[str, Any]:
    """
    Configure LlamaIndex with Qwen3-VL embedding and reranking.

    Returns a configuration dict that can be used with VectorStoreIndex
    and query engines.

    Args:
        use_reranker: Whether to enable reranking
        embedding_size: Embedding model size ("2B", "8B", "auto")
        reranker_size: Reranker model size ("2B", "8B", "auto")
        mrl_dim: Optional MRL dimension reduction
        reranker_top_n: Number of results after reranking
        similarity_top_k: Number of candidates for reranking

    Returns:
        Dict with configuration:
        {
            "embed_model": Qwen3VLEmbedding,
            "node_postprocessors": [Qwen3VLReranker] if use_reranker else [],
            "similarity_top_k": int,
            "info": {...}
        }

    Example:
        >>> config = configure_llamaindex_with_qwen3_vl()
        >>> index = VectorStoreIndex.from_documents(
        ...     documents,
        ...     embed_model=config["embed_model"]
        ... )
        >>> query_engine = index.as_query_engine(
        ...     node_postprocessors=config["node_postprocessors"],
        ...     similarity_top_k=config["similarity_top_k"]
        ... )
    """
    # Create embedding model
    embed_model = create_qwen3_vl_embed_model(
        model_size=embedding_size,
        mrl_dim=mrl_dim
    )

    # Create reranker if enabled
    postprocessors = []
    if use_reranker:
        reranker = create_qwen3_vl_reranker(
            top_n=reranker_top_n,
            model_size=reranker_size
        )
        postprocessors.append(reranker)

    # Build info dict
    info = {
        "embedding_model": embed_model.model_name,
        "embedding_dim": embed_model._config.embedding_dim,
        "mrl_dim": mrl_dim,
        "reranker_enabled": use_reranker,
        "reranker_model": postprocessors[0].model if postprocessors else None,
        "reranker_top_n": reranker_top_n if use_reranker else None,
        "similarity_top_k": similarity_top_k,
    }

    logger.info(f"Configured LlamaIndex with Qwen3-VL:")
    logger.info(f"  Embedding: {info['embedding_model']}")
    if use_reranker:
        logger.info(f"  Reranker: {info['reranker_model']} (top_n={reranker_top_n})")
    logger.info(f"  Retrieval: top_k={similarity_top_k}")

    return {
        "embed_model": embed_model,
        "node_postprocessors": postprocessors,
        "similarity_top_k": similarity_top_k,
        "info": info
    }
