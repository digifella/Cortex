#!/usr/bin/env python3
"""
Embedding Model Inspector for Cortex Suite
===========================================

Diagnoses embedding model consistency issues in ChromaDB collections.

This tool checks for:
- Mixed embedding dimensions (indicates multiple embedding models used)
- Current vs. stored embedding model configuration
- Collection health and compatibility

Usage:
    python scripts/embedding_inspector.py
    python scripts/embedding_inspector.py --db-path /custom/path
    python scripts/embedding_inspector.py --detailed
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.config import EMBED_MODEL, DB_PATH
from cortex_engine.embedding_service import embed_query, _load_model
from cortex_engine.utils.logging_utils import get_logger
from cortex_engine.utils.path_utils import convert_windows_to_wsl_path

logger = get_logger(__name__)


# Known embedding model dimensions
KNOWN_MODELS = {
    "nvidia/NV-Embed-v2": 1536,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


def get_embedding_dimension(model_name: str) -> int:
    """
    Get the embedding dimension for a model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Embedding vector dimension
    """
    # Check known models first
    if model_name in KNOWN_MODELS:
        return KNOWN_MODELS[model_name]

    # Try to load model and check dimension
    try:
        logger.info(f"Detecting dimension for model: {model_name}")
        test_embedding = embed_query("test")
        dimension = len(test_embedding)
        logger.info(f"✅ Detected dimension: {dimension}")
        return dimension
    except Exception as e:
        logger.error(f"❌ Failed to detect embedding dimension: {e}")
        raise


def inspect_chroma_collection(db_path: str, detailed: bool = False) -> Dict:
    """
    Inspect a ChromaDB collection for embedding consistency.

    Args:
        db_path: Path to database root
        detailed: If True, perform detailed analysis of all embeddings

    Returns:
        Dictionary with inspection results
    """
    try:
        import chromadb
    except ImportError:
        logger.error("❌ ChromaDB not installed. Run: pip install chromadb")
        sys.exit(1)

    # Resolve database path
    if db_path.startswith("C:\\") or db_path.startswith("D:\\"):
        db_path = convert_windows_to_wsl_path(db_path)

    chroma_db_dir = Path(db_path) / "knowledge_hub_db"

    if not chroma_db_dir.exists():
        return {
            "status": "not_found",
            "message": f"ChromaDB not found at: {chroma_db_dir}",
            "path": str(chroma_db_dir)
        }

    # Initialize ChromaDB client
    try:
        client = chromadb.PersistentClient(path=str(chroma_db_dir))
        collections = client.list_collections()

        if not collections:
            return {
                "status": "empty",
                "message": "No collections found in database",
                "path": str(chroma_db_dir)
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to open ChromaDB: {e}",
            "path": str(chroma_db_dir)
        }

    # Inspect each collection
    results = {
        "status": "ok",
        "db_path": str(chroma_db_dir),
        "current_model": EMBED_MODEL,
        "current_dimension": get_embedding_dimension(EMBED_MODEL),
        "collections": []
    }

    for collection in collections:
        collection_info = inspect_single_collection(
            client,
            collection.name,
            results["current_dimension"],
            detailed
        )
        results["collections"].append(collection_info)

    # Determine overall health
    results["health"] = determine_health(results)

    return results


def inspect_single_collection(
    client,
    collection_name: str,
    expected_dimension: int,
    detailed: bool = False
) -> Dict:
    """
    Inspect a single ChromaDB collection.

    Args:
        client: ChromaDB client
        collection_name: Name of collection to inspect
        expected_dimension: Expected embedding dimension
        detailed: Perform detailed analysis

    Returns:
        Collection inspection results
    """
    try:
        collection = client.get_collection(collection_name)

        # Get collection metadata
        metadata = collection.metadata or {}

        # Get document count
        count = collection.count()

        if count == 0:
            return {
                "name": collection_name,
                "status": "empty",
                "document_count": 0,
                "metadata": metadata
            }

        # Sample embeddings to check dimensions
        sample_size = min(100, count) if detailed else min(10, count)

        result = collection.get(
            limit=sample_size,
            include=["embeddings", "metadatas"]
        )

        embeddings = result.get("embeddings", [])

        if not embeddings:
            return {
                "name": collection_name,
                "status": "no_embeddings",
                "document_count": count,
                "metadata": metadata,
                "warning": "Collection has documents but no embeddings found"
            }

        # Analyze embedding dimensions
        dimensions = [len(emb) for emb in embeddings if emb is not None]

        if not dimensions:
            return {
                "name": collection_name,
                "status": "invalid",
                "document_count": count,
                "metadata": metadata,
                "error": "No valid embeddings found"
            }

        unique_dimensions = set(dimensions)

        # Check for consistency
        if len(unique_dimensions) > 1:
            # CRITICAL: Mixed dimensions detected!
            return {
                "name": collection_name,
                "status": "CRITICAL_MIXED_DIMENSIONS",
                "document_count": count,
                "sampled_count": len(dimensions),
                "unique_dimensions": sorted(unique_dimensions),
                "dimension_distribution": {
                    dim: dimensions.count(dim) for dim in unique_dimensions
                },
                "expected_dimension": expected_dimension,
                "metadata": metadata,
                "error": "⚠️ MIXED EMBEDDING DIMENSIONS DETECTED - Collection is corrupted!"
            }

        actual_dimension = dimensions[0]

        # Check if dimension matches expected
        if actual_dimension != expected_dimension:
            return {
                "name": collection_name,
                "status": "dimension_mismatch",
                "document_count": count,
                "sampled_count": len(dimensions),
                "actual_dimension": actual_dimension,
                "expected_dimension": expected_dimension,
                "metadata": metadata,
                "warning": f"⚠️ Collection uses {actual_dimension}-dim embeddings, but current model produces {expected_dimension}-dim"
            }

        # All good!
        return {
            "name": collection_name,
            "status": "healthy",
            "document_count": count,
            "sampled_count": len(dimensions),
            "dimension": actual_dimension,
            "metadata": metadata
        }

    except Exception as e:
        return {
            "name": collection_name,
            "status": "error",
            "error": str(e)
        }


def determine_health(results: Dict) -> str:
    """
    Determine overall database health based on inspection results.

    Returns:
        "healthy", "warning", or "critical"
    """
    if not results.get("collections"):
        return "empty"

    statuses = [c.get("status") for c in results["collections"]]

    if any("CRITICAL" in s for s in statuses):
        return "critical"

    if any(s in ["dimension_mismatch", "no_embeddings"] for s in statuses):
        return "warning"

    return "healthy"


def print_report(results: Dict):
    """Print formatted inspection report."""

    print("\n" + "="*80)
    print("CORTEX SUITE - EMBEDDING MODEL INSPECTION REPORT")
    print("="*80)

    print(f"\n📍 Database Path: {results.get('db_path', 'N/A')}")
    print(f"🤖 Current Model: {results.get('current_model', 'N/A')}")
    print(f"📊 Current Dimension: {results.get('current_dimension', 'N/A')}")

    health = results.get("health", "unknown")

    if health == "critical":
        print(f"\n🚨 HEALTH STATUS: CRITICAL - IMMEDIATE ACTION REQUIRED")
    elif health == "warning":
        print(f"\n⚠️  HEALTH STATUS: WARNING - Attention needed")
    elif health == "healthy":
        print(f"\n✅ HEALTH STATUS: HEALTHY")
    else:
        print(f"\n❓ HEALTH STATUS: {health.upper()}")

    print("\n" + "-"*80)
    print("COLLECTION DETAILS")
    print("-"*80)

    collections = results.get("collections", [])

    if not collections:
        print("\n📭 No collections found in database")
        return

    for i, coll in enumerate(collections, 1):
        print(f"\n[{i}] Collection: {coll.get('name', 'Unknown')}")
        print(f"    Status: {coll.get('status', 'unknown')}")
        print(f"    Documents: {coll.get('document_count', 0):,}")

        if "dimension" in coll:
            print(f"    Embedding Dimension: {coll['dimension']}")

        if "unique_dimensions" in coll:
            print(f"    ⚠️  MIXED DIMENSIONS: {coll['unique_dimensions']}")
            print(f"    Distribution: {coll.get('dimension_distribution', {})}")

        if "actual_dimension" in coll and "expected_dimension" in coll:
            print(f"    Actual Dimension: {coll['actual_dimension']}")
            print(f"    Expected Dimension: {coll['expected_dimension']}")

        if "warning" in coll:
            print(f"    {coll['warning']}")

        if "error" in coll:
            print(f"    ❌ Error: {coll['error']}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if health == "critical":
        print("""
🚨 CRITICAL ISSUES DETECTED

Your database contains MIXED EMBEDDING DIMENSIONS. This means documents were
embedded using different models, resulting in corrupted vector search.

IMMEDIATE ACTION REQUIRED:
1. Backup your source documents
2. Use the database delete function in Maintenance page
3. Re-ingest all documents with a single embedding model
4. Set CORTEX_EMBED_MODEL environment variable to lock your model choice

DO NOT attempt to use this database for queries - results will be unreliable.
""")
    elif health == "warning":
        print("""
⚠️  WARNINGS DETECTED

Your database embeddings don't match your current embedding model configuration.
This can happen if:
- You changed hardware (added/removed GPU)
- The embedding model was changed in config
- Database was created on a different machine

RECOMMENDED ACTIONS:
1. Decide on ONE embedding model to standardize on
2. Set CORTEX_EMBED_MODEL environment variable to prevent auto-switching
3. Consider re-embedding your database for consistency

Current queries will work, but may not be optimal.
""")
    else:
        print("""
✅ NO ISSUES DETECTED

Your database appears healthy. All embeddings use consistent dimensions
matching your current model configuration.

BEST PRACTICES:
- Set CORTEX_EMBED_MODEL environment variable to lock your model choice
- Avoid changing embedding models without re-embedding existing data
- Run this inspector after major system changes
""")

    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect ChromaDB collections for embedding model consistency"
    )
    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help="Path to database root (default: from config)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Perform detailed analysis (samples more embeddings)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    try:
        results = inspect_chroma_collection(args.db_path, args.detailed)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_report(results)

        # Exit code based on health
        health = results.get("health", "unknown")
        if health == "critical":
            sys.exit(2)
        elif health == "warning":
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Inspection failed: {e}", exc_info=True)
        print(f"\n❌ Inspection failed: {e}\n")
        sys.exit(3)


if __name__ == "__main__":
    main()
