"""
Embedding Model Validator
==========================

Validates embedding model consistency for ChromaDB collections.

This module prevents the critical data corruption issue that occurs when
different embedding models are used for the same ChromaDB collection.

Key Functions:
- validate_embedding_compatibility(): Check if current model matches collection
- get_embedding_dimension(): Get dimension for a model name
- EmbeddingModelMismatchError: Exception raised on validation failures
"""

from typing import Optional, Dict
from ..config import EMBED_MODEL
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingModelMismatchError(Exception):
    """Raised when embedding model doesn't match collection's model."""

    def __init__(self, message: str, current_model: str, expected_model: str):
        super().__init__(message)
        self.current_model = current_model
        self.expected_model = expected_model


# Known embedding model dimensions
# This is a lookup table for common models to avoid loading them just to check dimension
KNOWN_MODEL_DIMENSIONS = {
    # NVIDIA models
    "nvidia/NV-Embed-v2": 1536,

    # BGE models
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-small-en-v1.5": 384,

    # Sentence Transformers
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,

    # Qwen3-VL Multimodal Embedding models
    "Qwen/Qwen3-VL-Embedding-2B": 2048,
    "Qwen/Qwen3-VL-Embedding-8B": 4096,

    # Qwen3 Text Embedding models
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,

    # GTE-Qwen2 models (predecessor)
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": 1536,
    "Alibaba-NLP/gte-Qwen2-7B-instruct": 3584,
}


def get_embedding_dimension(model_name: str) -> int:
    """
    Get the embedding dimension for a model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Embedding vector dimension

    Raises:
        RuntimeError: If dimension cannot be determined
    """
    # Check known models first (fast path)
    if model_name in KNOWN_MODEL_DIMENSIONS:
        return KNOWN_MODEL_DIMENSIONS[model_name]

    # Fall back to actually loading the model and testing
    try:
        from ..embedding_service import embed_query
        logger.info(f"Detecting dimension for unknown model: {model_name}")
        test_embedding = embed_query("test")
        dimension = len(test_embedding)
        logger.info(f"✅ Detected dimension for {model_name}: {dimension}")
        return dimension
    except Exception as e:
        logger.error(f"❌ Failed to detect embedding dimension for {model_name}: {e}")
        raise RuntimeError(f"Cannot determine embedding dimension for model '{model_name}'") from e


def validate_embedding_compatibility(
    collection_metadata: Dict,
    current_model: Optional[str] = None,
    strict: bool = True
) -> Dict:
    """
    Validate that the current embedding model is compatible with a collection.

    Args:
        collection_metadata: Metadata dict from WorkingCollectionManager or ChromaDB
        current_model: Model to validate (defaults to config.EMBED_MODEL)
        strict: If True, raise exception on mismatch. If False, return validation result.

    Returns:
        Dict with validation results:
        {
            "compatible": bool,
            "current_model": str,
            "current_dimension": int,
            "stored_model": str or None,
            "stored_dimension": int or None,
            "warnings": list,
            "errors": list
        }

    Raises:
        EmbeddingModelMismatchError: If strict=True and models don't match
    """
    if current_model is None:
        current_model = EMBED_MODEL

    result = {
        "compatible": True,
        "current_model": current_model,
        "current_dimension": None,
        "stored_model": collection_metadata.get("embedding_model"),
        "stored_dimension": collection_metadata.get("embedding_dimension"),
        "warnings": [],
        "errors": []
    }

    # Get current model dimension
    try:
        result["current_dimension"] = get_embedding_dimension(current_model)
    except Exception as e:
        result["compatible"] = False
        result["errors"].append(f"Failed to determine current model dimension: {e}")
        if strict:
            raise EmbeddingModelMismatchError(
                f"Cannot validate embedding compatibility: {e}",
                current_model,
                result["stored_model"] or "unknown"
            )
        return result

    # If no metadata stored, this is a new collection or legacy collection
    if not result["stored_model"] and not result["stored_dimension"]:
        result["warnings"].append(
            "⚠️ Collection has no embedding model metadata. "
            "This may be a legacy collection created before metadata tracking was implemented."
        )
        # Not necessarily incompatible, but we can't verify
        return result

    # Check model name mismatch
    if result["stored_model"] and result["stored_model"] != current_model:
        error_msg = (
            f"❌ Embedding model mismatch!\n"
            f"   Collection uses: {result['stored_model']}\n"
            f"   Current system uses: {current_model}\n"
            f"   This will produce incorrect search results."
        )
        result["compatible"] = False
        result["errors"].append(error_msg)

        if strict:
            raise EmbeddingModelMismatchError(
                error_msg,
                current_model,
                result["stored_model"]
            )

    # Check dimension mismatch
    if result["stored_dimension"] and result["stored_dimension"] != result["current_dimension"]:
        error_msg = (
            f"❌ Embedding dimension mismatch!\n"
            f"   Collection uses: {result['stored_dimension']}-dimensional embeddings\n"
            f"   Current model produces: {result['current_dimension']}-dimensional embeddings\n"
            f"   This indicates different embedding models were used."
        )
        result["compatible"] = False
        result["errors"].append(error_msg)

        if strict:
            raise EmbeddingModelMismatchError(
                error_msg,
                current_model,
                result["stored_model"] or f"{result['stored_dimension']}D-model"
            )

    # All checks passed
    if result["compatible"] and not result["warnings"]:
        logger.debug(f"✅ Embedding model validation passed for {current_model}")

    return result


def get_validation_summary(validation_result: Dict) -> str:
    """
    Get a human-readable summary of validation results.

    Args:
        validation_result: Result from validate_embedding_compatibility()

    Returns:
        Formatted string summary
    """
    lines = []

    if validation_result["compatible"]:
        lines.append("✅ Embedding Model: Compatible")
    else:
        lines.append("❌ Embedding Model: INCOMPATIBLE")

    lines.append(f"   Current: {validation_result['current_model']} ({validation_result['current_dimension']}D)")

    if validation_result["stored_model"]:
        lines.append(f"   Stored:  {validation_result['stored_model']} ({validation_result['stored_dimension']}D)")
    else:
        lines.append("   Stored:  No metadata")

    if validation_result["errors"]:
        lines.append("\n".join(validation_result["errors"]))

    if validation_result["warnings"]:
        lines.append("\n".join(validation_result["warnings"]))

    return "\n".join(lines)
