"""
Embedding Model Validator
==========================

Validates embedding model consistency for ChromaDB collections.

This module prevents the critical data corruption issue that occurs when
different embedding models are used for the same ChromaDB collection.

CRITICAL: Using an embedding model with different dimensions than the existing
database WILL corrupt the database or cause errors. This module provides
validation to prevent such mismatches.

Key Functions:
- validate_embedding_compatibility(): Check if current model matches collection
- get_embedding_dimension(): Get dimension for a model name
- get_database_embedding_dimension(): Detect dimension from existing ChromaDB
- get_compatible_model_sizes(): Get Qwen3-VL sizes compatible with existing DB
- EmbeddingModelMismatchError: Exception raised on validation failures
"""

import os
from typing import Optional, Dict, List, Tuple
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


# ============================================================================
# Database Dimension Detection
# ============================================================================

def get_database_embedding_dimension(db_path: str) -> Optional[int]:
    """
    Detect the embedding dimension from an existing ChromaDB database.

    This queries the actual ChromaDB collection to get a sample embedding
    and determine its dimension. This is the most reliable way to detect
    the database's embedding dimension.

    Args:
        db_path: Path to the AI database directory (parent of knowledge_hub_db)

    Returns:
        Embedding dimension (e.g., 768, 1536, 2048, 4096) or None if:
        - Database doesn't exist
        - Database is empty
        - Error accessing database
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    try:
        # Handle Docker/WSL path conversion
        from ..utils import convert_to_docker_mount_path
        safe_path = convert_to_docker_mount_path(db_path)
        chroma_db_path = os.path.join(safe_path, "knowledge_hub_db")

        if not os.path.isdir(chroma_db_path):
            logger.debug(f"ChromaDB not found at {chroma_db_path} - new database")
            return None

        # Connect to ChromaDB
        settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=chroma_db_path, settings=settings)

        # Get the main collection
        from ..config import COLLECTION_NAME
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            logger.debug(f"Collection '{COLLECTION_NAME}' not found - new database")
            return None

        # Check if collection has any embeddings
        count = collection.count()
        if count == 0:
            logger.debug("Collection is empty - no dimension to detect")
            return None

        # Get a sample embedding to determine dimension
        # We only need 1 result with embeddings included
        results = collection.get(
            limit=1,
            include=["embeddings"]
        )

        if results and results.get("embeddings") and len(results["embeddings"]) > 0:
            sample_embedding = results["embeddings"][0]
            dimension = len(sample_embedding)
            logger.info(f"Detected database embedding dimension: {dimension}")
            return dimension

        logger.warning("Could not retrieve sample embedding from database")
        return None

    except Exception as e:
        logger.warning(f"Error detecting database embedding dimension: {e}")
        return None


def get_compatible_qwen3vl_sizes(db_path: str) -> Dict[str, any]:
    """
    Get Qwen3-VL model sizes that are compatible with the existing database.

    This is critical for preventing dimension mismatch errors. If a database
    exists with 2048-dim embeddings, only the 2B model can be used. If it has
    4096-dim embeddings, only the 8B model can be used.

    Args:
        db_path: Path to the AI database directory

    Returns:
        Dict with:
        - 'database_dimension': Detected dimension or None for new database
        - 'is_new_database': True if no existing embeddings
        - 'compatible_sizes': List of compatible size strings ('2B', '8B')
        - 'incompatible_sizes': List of incompatible size strings
        - 'recommended_size': Best size to use (or None if all incompatible)
        - 'warning_message': User-facing warning if there are restrictions
    """
    # Qwen3-VL model dimensions
    SIZE_DIMENSIONS = {
        "2B": 2048,
        "8B": 4096,
    }

    result = {
        "database_dimension": None,
        "is_new_database": True,
        "compatible_sizes": [],
        "incompatible_sizes": [],
        "recommended_size": None,
        "warning_message": None,
    }

    # Detect existing database dimension
    db_dimension = get_database_embedding_dimension(db_path)
    result["database_dimension"] = db_dimension

    if db_dimension is None:
        # New database - all sizes are compatible
        result["is_new_database"] = True
        result["compatible_sizes"] = list(SIZE_DIMENSIONS.keys())
        result["recommended_size"] = "2B"  # Default to smaller for broader compatibility
        return result

    result["is_new_database"] = False

    # Check which sizes are compatible with the existing dimension
    for size, dim in SIZE_DIMENSIONS.items():
        if dim == db_dimension:
            result["compatible_sizes"].append(size)
        else:
            result["incompatible_sizes"].append(size)

    # Set recommended size
    if result["compatible_sizes"]:
        result["recommended_size"] = result["compatible_sizes"][0]
    else:
        # No compatible sizes - this shouldn't happen with standard Qwen3-VL
        # but could occur with imported databases from other models
        result["warning_message"] = (
            f"Database has {db_dimension}-dimensional embeddings which don't match "
            f"any Qwen3-VL model. You may need to rebuild the database."
        )
        return result

    # Create warning message for incompatible sizes
    if result["incompatible_sizes"]:
        incompatible_str = ", ".join(result["incompatible_sizes"])
        compatible_str = ", ".join(result["compatible_sizes"])
        result["warning_message"] = (
            f"Database uses {db_dimension}D embeddings. "
            f"Only {compatible_str} model(s) are compatible. "
            f"{incompatible_str} model(s) would corrupt the database."
        )

    return result


def validate_model_for_database(
    model_dimension: int,
    db_path: str,
    strict: bool = True
) -> Tuple[bool, str]:
    """
    Validate that a model's embedding dimension matches the existing database.

    Args:
        model_dimension: Dimension of the model to validate
        db_path: Path to the AI database directory
        strict: If True, raise exception on mismatch

    Returns:
        Tuple of (is_compatible, message)

    Raises:
        EmbeddingModelMismatchError: If strict=True and dimensions don't match
    """
    db_dimension = get_database_embedding_dimension(db_path)

    if db_dimension is None:
        # New database - any dimension is fine
        return True, f"New database - will use {model_dimension}D embeddings"

    if db_dimension == model_dimension:
        return True, f"Model dimension ({model_dimension}D) matches database ({db_dimension}D)"

    error_msg = (
        f"DIMENSION MISMATCH: Model produces {model_dimension}D embeddings "
        f"but database uses {db_dimension}D embeddings. "
        f"Using this model would corrupt the database!"
    )

    if strict:
        raise EmbeddingModelMismatchError(
            error_msg,
            f"{model_dimension}D-model",
            f"{db_dimension}D-database"
        )

    return False, error_msg
