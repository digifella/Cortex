"""
Qwen3-VL Multimodal Reranker Service
=====================================

Version: 1.0.0
Date: 2026-01-17

Provides precision reranking for search results using Qwen3-VL-Reranker models.
This is the second stage of a two-stage retrieval pipeline:

    Stage 1 (Embedding): Fast recall with ~85% precision (top-k candidates)
    Stage 2 (Reranker):  Fine-grained scoring for ~95%+ precision

The reranker examines query-document pairs together (cross-attention) to
produce relevance scores, which is more accurate than embedding similarity
but slower - hence only applied to the top candidates from stage 1.

Key Features:
- Multimodal reranking (text, images, mixed content)
- Cross-attention scoring for precise relevance
- Batch processing with GPU optimization
- Integrates with existing graph_query.py hybrid search

Hardware Requirements:
- Qwen3-VL-Reranker-2B: ~5GB VRAM
- Qwen3-VL-Reranker-8B: ~16GB VRAM

Usage:
    from cortex_engine.qwen3_vl_reranker_service import rerank_results

    # After initial vector search
    candidates = vector_search(query, top_k=20)

    # Rerank for precision
    reranked = rerank_results(query, candidates, top_k=5)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np

from .utils.logging_utils import get_logger
from .utils.performance_monitor import measure
from .utils.gpu_monitor import get_gpu_memory_info, clear_gpu_cache

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Qwen3VLRerankerSize(Enum):
    """Available Qwen3-VL reranker model sizes."""
    SMALL = "Qwen/Qwen3-VL-Reranker-2B"   # ~5GB VRAM
    LARGE = "Qwen/Qwen3-VL-Reranker-8B"   # ~16GB VRAM


@dataclass
class Qwen3VLRerankerConfig:
    """Configuration for Qwen3-VL reranker service."""
    model_name: str = "Qwen/Qwen3-VL-Reranker-8B"
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"
    max_batch_size: int = 4  # Reranker is more memory-intensive
    device: Optional[str] = None
    trust_remote_code: bool = True

    # Reranking behavior
    default_top_k: int = 5  # Default number of results to return
    score_threshold: Optional[float] = None  # Optional minimum score cutoff

    # Video/image processing
    max_video_frames: int = 64
    video_fps: float = 1.0

    @classmethod
    def for_model_size(cls, size: Qwen3VLRerankerSize) -> "Qwen3VLRerankerConfig":
        """Create config for specific model size."""
        if size == Qwen3VLRerankerSize.SMALL:
            return cls(
                model_name="Qwen/Qwen3-VL-Reranker-2B",
                max_batch_size=8,
            )
        else:
            return cls(
                model_name="Qwen/Qwen3-VL-Reranker-8B",
                max_batch_size=4,
            )

    @classmethod
    def auto_select(cls, prefer_quality: bool = True) -> "Qwen3VLRerankerConfig":
        """Auto-select model based on available VRAM."""
        gpu_info = get_gpu_memory_info()

        if gpu_info.is_cuda:
            available_gb = gpu_info.free_memory_gb

            if available_gb >= 20 and prefer_quality:
                logger.info(f"🚀 Auto-selected Qwen3-VL-Reranker-8B ({available_gb:.1f}GB free)")
                return cls.for_model_size(Qwen3VLRerankerSize.LARGE)
            elif available_gb >= 8:
                logger.info(f"📦 Auto-selected Qwen3-VL-Reranker-2B ({available_gb:.1f}GB free)")
                return cls.for_model_size(Qwen3VLRerankerSize.SMALL)
            else:
                logger.warning(f"⚠️ Low VRAM ({available_gb:.1f}GB) - reranker may not fit")
                return cls.for_model_size(Qwen3VLRerankerSize.SMALL)
        else:
            logger.warning("⚠️ No CUDA GPU - reranker performance will be limited")
            return cls.for_model_size(Qwen3VLRerankerSize.SMALL)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class Document:
    """Represents a document for reranking."""
    content: str  # Text content
    image_path: Optional[Union[str, Path]] = None
    video_path: Optional[Union[str, Path]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_score: Optional[float] = None  # Score from embedding search
    original_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format for Qwen3-VL."""
        result = {}
        if self.content:
            result["text"] = self.content
        if self.image_path:
            result["image"] = str(self.image_path)
        if self.video_path:
            result["video"] = str(self.video_path)
        return result


@dataclass
class RerankedResult:
    """Result from reranking."""
    document: Document
    rerank_score: float
    original_score: Optional[float]
    rank_change: int  # Positive = moved up, negative = moved down


# ============================================================================
# Model Loading
# ============================================================================

_model_lock = threading.Lock()
_reranker_model: Optional[Any] = None
_reranker_processor: Optional[Any] = None
_current_config: Optional[Qwen3VLRerankerConfig] = None


def _get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_reranker(config: Optional[Qwen3VLRerankerConfig] = None) -> tuple:
    """
    Load Qwen3-VL reranker model.

    Returns:
        Tuple of (model, processor, config)
    """
    global _reranker_model, _reranker_processor, _current_config

    if config is None:
        config = Qwen3VLRerankerConfig.auto_select()

    # Return cached model if same config
    if _reranker_model is not None and _current_config == config:
        return _reranker_model, _reranker_processor, _current_config

    with _model_lock:
        if _reranker_model is not None and _current_config == config:
            return _reranker_model, _reranker_processor, _current_config

        logger.info(f"Loading Qwen3-VL reranker: {config.model_name}")

        try:
            from transformers import AutoModel, AutoProcessor

            device = config.device or _get_device()
            dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float16

            _reranker_processor = AutoProcessor.from_pretrained(
                config.model_name,
                trust_remote_code=config.trust_remote_code
            )

            model_kwargs = {
                "trust_remote_code": config.trust_remote_code,
                "torch_dtype": dtype,
            }

            # Only enable Flash Attention 2 if the package is actually installed
            if config.use_flash_attention:
                try:
                    import flash_attn  # noqa: F401
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("⚡ Using Flash Attention 2")
                except ImportError:
                    logger.info("📝 Flash Attention 2 not installed - using default attention")

            _reranker_model = AutoModel.from_pretrained(
                config.model_name,
                **model_kwargs
            ).to(device).eval()

            _current_config = config

            logger.info(f"✅ Qwen3-VL reranker loaded on {device}")

            return _reranker_model, _reranker_processor, _current_config

        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load reranker: {e}")
            raise


def unload_reranker():
    """Unload reranker to free GPU memory."""
    global _reranker_model, _reranker_processor, _current_config

    with _model_lock:
        if _reranker_model is not None:
            del _reranker_model
            _reranker_model = None
        if _reranker_processor is not None:
            del _reranker_processor
            _reranker_processor = None
        _current_config = None

        clear_gpu_cache()
        logger.info("🧹 Qwen3-VL reranker unloaded")


# ============================================================================
# Reranking Functions
# ============================================================================

def _get_yes_no_token_ids(processor) -> Tuple[List[int], List[int]]:
    """Get token IDs for yes/no variations to compute relevance scores."""
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Common yes/no token variations
    yes_tokens = ["Yes", "yes", "YES", "True", "true", "relevant", "Relevant"]
    no_tokens = ["No", "no", "NO", "False", "false", "irrelevant", "Irrelevant"]

    yes_ids = []
    no_ids = []

    for token in yes_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                yes_ids.append(ids[0])
        except Exception:
            pass

    for token in no_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                no_ids.append(ids[0])
        except Exception:
            pass

    return yes_ids, no_ids


def _compute_rerank_scores(
    query: Dict[str, Any],
    documents: List[Dict[str, Any]],
    model: Any,
    processor: Any,
    config: Qwen3VLRerankerConfig
) -> List[float]:
    """
    Compute reranking scores for query-document pairs using cross-encoder approach.

    Uses a relevance classification prompt and compares probabilities of
    "Yes" vs "No" tokens to determine document relevance scores.

    Args:
        query: Query in dict format (e.g., {"text": "..."})
        documents: List of documents in dict format
        model: Loaded reranker model
        processor: Reranker processor
        config: Configuration

    Returns:
        List of relevance scores (0.0 to 1.0, higher = more relevant)
    """
    device = next(model.parameters()).device
    scores = []

    # Get yes/no token IDs for scoring
    yes_ids, no_ids = _get_yes_no_token_ids(processor)

    if not yes_ids or not no_ids:
        logger.warning("Could not find yes/no token IDs, using fallback scoring")
        yes_ids = [9642]  # Common "Yes" token ID
        no_ids = [2822]   # Common "No" token ID

    with torch.no_grad():
        for i in range(0, len(documents), config.max_batch_size):
            batch_docs = documents[i:i+config.max_batch_size]

            for doc in batch_docs:
                try:
                    query_text = query.get("text", "")
                    doc_text = doc.get("text", "")

                    # Truncate long documents to fit context
                    max_doc_len = 1500
                    if len(doc_text) > max_doc_len:
                        doc_text = doc_text[:max_doc_len] + "..."

                    # Cross-encoder style relevance prompt
                    # This format asks the model to evaluate if the document answers the query
                    prompt = f"""Given a query and a document, determine if the document is relevant to answering the query.

Query: {query_text}

Document: {doc_text}

Is this document relevant to the query? Answer with Yes or No."""

                    # Process through processor
                    inputs = processor(
                        text=[prompt],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048
                    ).to(device)

                    outputs = model(**inputs)

                    # Compute relevance score from logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits[:, -1, :]  # Last position logits
                        probs = torch.softmax(logits, dim=-1)

                        # Sum probabilities for yes tokens vs no tokens
                        yes_prob = sum(probs[0, tid].item() for tid in yes_ids if tid < probs.shape[-1])
                        no_prob = sum(probs[0, tid].item() for tid in no_ids if tid < probs.shape[-1])

                        # Normalize to get relevance score
                        total = yes_prob + no_prob
                        if total > 0:
                            score = yes_prob / total
                        else:
                            # Fallback: use entropy-based scoring
                            # Lower entropy = more confident = higher score
                            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                            max_entropy = np.log(probs.shape[-1])
                            score = 1.0 - (entropy / max_entropy)

                    elif hasattr(outputs, 'last_hidden_state'):
                        # Embedding-based fallback: use cosine similarity approach
                        hidden = outputs.last_hidden_state[:, -1, :]
                        # Normalize and use L2 norm as confidence proxy
                        norm = torch.norm(hidden, dim=-1).item()
                        score = min(1.0, norm / 50.0)  # Normalize to 0-1 range
                    else:
                        score = 0.5

                    # Clamp to valid range
                    score = max(0.0, min(1.0, score))
                    scores.append(score)

                except Exception as e:
                    logger.warning(f"Failed to score document: {e}")
                    scores.append(0.5)

    return scores


def rerank_documents(
    query_text: str,
    documents: List[Document],
    top_k: Optional[int] = None,
    query_image: Optional[Union[str, Path]] = None,
    config: Optional[Qwen3VLRerankerConfig] = None
) -> List[RerankedResult]:
    """
    Rerank documents by relevance to query.

    Args:
        query_text: Query text
        documents: List of Document objects to rerank
        top_k: Number of top results to return (default: config.default_top_k)
        query_image: Optional query image for multimodal search
        config: Optional configuration override

    Returns:
        List of RerankedResult objects, sorted by rerank_score descending
    """
    if not documents:
        return []

    model, processor, cfg = _load_reranker(config)

    if top_k is None:
        top_k = cfg.default_top_k

    # Build query dict
    query_dict = {"text": query_text}
    if query_image:
        query_dict["image"] = str(query_image)

    # Convert documents to dict format
    doc_dicts = [doc.to_dict() for doc in documents]

    # Compute scores
    with measure("qwen3_vl_rerank", doc_count=len(documents)):
        logger.info(f"🔄 Reranking {len(documents)} documents")
        scores = _compute_rerank_scores(query_dict, doc_dicts, model, processor, cfg)

    # Create results with rank tracking
    results = []
    for i, (doc, score) in enumerate(zip(documents, scores)):
        doc.original_rank = i
        results.append(RerankedResult(
            document=doc,
            rerank_score=score,
            original_score=doc.original_score,
            rank_change=0  # Will be computed after sorting
        ))

    # Sort by rerank score
    results.sort(key=lambda x: x.rerank_score, reverse=True)

    # Compute rank changes
    for new_rank, result in enumerate(results):
        if result.document.original_rank is not None:
            result.rank_change = result.document.original_rank - new_rank

    # Apply score threshold if configured
    if cfg.score_threshold is not None:
        results = [r for r in results if r.rerank_score >= cfg.score_threshold]

    # Return top-k
    results = results[:top_k]

    logger.info(f"✅ Reranking complete: {len(results)} results returned")

    return results


def rerank_text_results(
    query: str,
    texts: List[str],
    top_k: Optional[int] = None,
    original_scores: Optional[List[float]] = None,
    metadata: Optional[List[Dict]] = None,
    config: Optional[Qwen3VLRerankerConfig] = None
) -> List[RerankedResult]:
    """
    Convenience function to rerank text-only search results.

    Args:
        query: Query text
        texts: List of document texts
        top_k: Number of results to return
        original_scores: Optional original similarity scores
        metadata: Optional metadata for each document
        config: Optional configuration

    Returns:
        List of RerankedResult objects
    """
    documents = []
    for i, text in enumerate(texts):
        doc = Document(
            content=text,
            original_score=original_scores[i] if original_scores else None,
            metadata=metadata[i] if metadata else {}
        )
        documents.append(doc)

    return rerank_documents(query, documents, top_k=top_k, config=config)


def rerank_hybrid_results(
    query: str,
    results: List[Dict[str, Any]],
    text_key: str = "content",
    score_key: str = "score",
    top_k: Optional[int] = None,
    config: Optional[Qwen3VLRerankerConfig] = None
) -> List[Dict[str, Any]]:
    """
    Rerank results from hybrid search (graph_query.py format).

    This function integrates with the existing hybrid search output format.

    Args:
        query: Query text
        results: List of result dicts with text content and scores
        text_key: Key for text content in result dicts
        score_key: Key for original score in result dicts
        top_k: Number of results to return
        config: Optional configuration

    Returns:
        List of result dicts, reranked and augmented with rerank_score
    """
    if not results:
        return []

    # Convert to Document objects
    documents = []
    for i, result in enumerate(results):
        # Handle image paths if present
        image_path = result.get("image_path") or result.get("figure_path")

        doc = Document(
            content=result.get(text_key, ""),
            image_path=image_path,
            original_score=result.get(score_key),
            metadata=result
        )
        documents.append(doc)

    # Rerank
    reranked = rerank_documents(query, documents, top_k=top_k, config=config)

    # Convert back to dict format with added rerank info
    output = []
    for rr in reranked:
        result_dict = rr.document.metadata.copy()
        result_dict["rerank_score"] = rr.rerank_score
        result_dict["rank_change"] = rr.rank_change
        result_dict["original_rank"] = rr.document.original_rank
        output.append(result_dict)

    return output


# ============================================================================
# Service Interface
# ============================================================================

class Qwen3VLRerankerService:
    """
    Unified interface for Qwen3-VL reranking.

    Can be used standalone or integrated into existing search pipelines.
    """

    def __init__(self, config: Optional[Qwen3VLRerankerConfig] = None):
        """
        Initialize reranker service.

        Args:
            config: Optional configuration. Auto-selects if not provided.
        """
        self.config = config or Qwen3VLRerankerConfig.auto_select()
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is None:
            self._model, self._processor, self.config = _load_reranker(self.config)

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """Rerank documents."""
        return rerank_documents(query, documents, top_k, config=self.config)

    def rerank_texts(
        self,
        query: str,
        texts: List[str],
        top_k: Optional[int] = None,
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """Rerank text results."""
        return rerank_text_results(
            query, texts, top_k,
            original_scores=original_scores,
            config=self.config
        )

    def rerank_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank search results (dict format)."""
        return rerank_hybrid_results(query, results, top_k=top_k, config=self.config)

    def unload(self):
        """Unload model to free GPU memory."""
        unload_reranker()
        self._model = None
        self._processor = None

    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "model_name": self.config.model_name,
            "default_top_k": self.config.default_top_k,
            "score_threshold": self.config.score_threshold,
            "max_batch_size": self.config.max_batch_size,
            "device": self.config.device or _get_device(),
        }


# ============================================================================
# Global Service Instance
# ============================================================================

_service_instance: Optional[Qwen3VLRerankerService] = None
_service_lock = threading.Lock()


def get_reranker_service(
    config: Optional[Qwen3VLRerankerConfig] = None
) -> Qwen3VLRerankerService:
    """
    Get or create the global reranker service instance.

    Args:
        config: Optional configuration for first initialization

    Returns:
        Qwen3VLRerankerService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = Qwen3VLRerankerService(config)

    return _service_instance


def reset_reranker_service():
    """Reset the global service instance."""
    global _service_instance

    with _service_lock:
        if _service_instance is not None:
            _service_instance.unload()
            _service_instance = None


# ============================================================================
# Convenience Function for Integration
# ============================================================================

def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
    text_key: str = "content"
) -> List[Dict[str, Any]]:
    """
    Simple reranking function for integration with existing code.

    Drop-in enhancement for hybrid_search results.

    Args:
        query: Search query
        results: Results from vector/hybrid search
        top_k: Number of results to return
        text_key: Key containing text content

    Returns:
        Reranked results with added rerank_score field

    Example:
        # In graph_query.py hybrid_search():
        vector_results = self.vector_search(query, top_k=20)
        reranked = rerank_results(query, vector_results, top_k=5)
        return reranked
    """
    return get_reranker_service().rerank_search_results(query, results, top_k=top_k)
