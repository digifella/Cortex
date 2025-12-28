#!/usr/bin/env python3
"""
Embedding Model Migration Utility for Cortex Suite
===================================================

Safely migrates ChromaDB collections from one embedding model to another.

This tool:
- Extracts all documents from existing collection
- Re-embeds them with new embedding model
- Creates new collection with new embeddings
- Preserves all document metadata

Usage:
    python scripts/embedding_migrator.py --source-model nvidia/NV-Embed-v2 --target-model BAAI/bge-base-en-v1.5
    python scripts/embedding_migrator.py --target-model nvidia/NV-Embed-v2 --auto-detect-source
    python scripts/embedding_migrator.py --db-path /custom/path --target-model BAAI/bge-base-en-v1.5
"""

import sys
from pathlib import Path
import argparse
import json
import os
from typing import Dict, List
from datetime import datetime

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.config import DB_PATH, EMBED_MODEL
from cortex_engine.utils.logging_utils import get_logger
from cortex_engine.utils.path_utils import convert_windows_to_wsl_path
from cortex_engine.collection_manager import WorkingCollectionManager

logger = get_logger(__name__)


def migrate_collection(
    db_path: str,
    target_model: str,
    source_model: str = None,
    backup: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    Migrate a ChromaDB collection to a new embedding model.

    Args:
        db_path: Path to database root
        target_model: New embedding model to use
        source_model: Current embedding model (auto-detected if None)
        backup: Whether to backup before migration
        dry_run: If True, only simulate migration

    Returns:
        Dictionary with migration results
    """
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except ImportError:
        logger.error("❌ ChromaDB not installed. Run: pip install chromadb")
        sys.exit(1)

    # Resolve database path
    if db_path.startswith("C:\\") or db_path.startswith("D:\\"):
        db_path = convert_windows_to_wsl_path(db_path)

    chroma_db_dir = Path(db_path) / "knowledge_hub_db"

    if not chroma_db_dir.exists():
        return {
            "status": "error",
            "message": f"ChromaDB not found at: {chroma_db_dir}"
        }

    logger.info("="*80)
    logger.info("EMBEDDING MODEL MIGRATION")
    logger.info("="*80)
    logger.info(f"Database: {chroma_db_dir}")
    logger.info(f"Target Model: {target_model}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("")

    # Initialize ChromaDB client
    try:
        db_settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=str(chroma_db_dir), settings=db_settings)
        collections = client.list_collections()

        if not collections:
            return {
                "status": "error",
                "message": "No collections found in database"
            }

        collection_name = "knowledge_hub"
        collection = client.get_collection(collection_name)
        doc_count = collection.count()

        logger.info(f"📊 Collection '{collection_name}': {doc_count:,} documents")

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to open ChromaDB: {e}"
        }

    # Get current embedding model from metadata
    collection_mgr = WorkingCollectionManager()
    metadata = collection_mgr.get_embedding_model_metadata("default")

    if not source_model:
        source_model = metadata.get("embedding_model") or "Unknown"

    logger.info(f"📥 Source Model: {source_model}")
    logger.info("")

    if source_model == target_model:
        logger.warning("⚠️  Source and target models are the same. No migration needed.")
        return {
            "status": "no_change",
            "message": "Source and target models are identical"
        }

    # Backup if requested
    if backup and not dry_run:
        logger.info("💾 Creating backup...")
        backup_dir = Path(db_path) / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            import shutil
            shutil.copytree(chroma_db_dir, backup_dir / "knowledge_hub_db")
            logger.info(f"✅ Backup created: {backup_dir}")
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return {
                "status": "error",
                "message": f"Backup failed: {e}"
            }

    # Extract all documents
    logger.info("📤 Extracting documents from collection...")
    try:
        all_data = collection.get(
            include=["documents", "metadatas", "ids"]
        )

        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])

        logger.info(f"✅ Extracted {len(documents):,} documents")

        if len(documents) != doc_count:
            logger.warning(f"⚠️  Expected {doc_count} documents, got {len(documents)}")

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to extract documents: {e}"
        }

    if dry_run:
        logger.info("")
        logger.info("🔍 DRY RUN - Migration plan:")
        logger.info(f"   1. Delete collection '{collection_name}'")
        logger.info(f"   2. Create new collection with {target_model}")
        logger.info(f"   3. Re-embed {len(documents):,} documents")
        logger.info(f"   4. Update collection metadata")
        logger.info("")
        logger.info("Run without --dry-run to execute migration")
        return {
            "status": "dry_run",
            "documents_to_migrate": len(documents),
            "source_model": source_model,
            "target_model": target_model
        }

    # Set target model in environment
    logger.info(f"🔧 Configuring target model: {target_model}")
    os.environ["CORTEX_EMBED_MODEL"] = target_model

    # Reload config to pick up new model
    import importlib
    import cortex_engine.config
    importlib.reload(cortex_engine.config)

    # Verify model loaded correctly
    from cortex_engine.config import EMBED_MODEL as CURRENT_MODEL
    if CURRENT_MODEL != target_model:
        logger.warning(f"⚠️  Model mismatch: Expected {target_model}, got {CURRENT_MODEL}")

    # Delete old collection
    logger.info(f"🗑️  Deleting old collection...")
    try:
        client.delete_collection(collection_name)
        logger.info(f"✅ Deleted collection '{collection_name}'")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete collection: {e}"
        }

    # Create new collection
    logger.info(f"➕ Creating new collection with {target_model}...")
    try:
        new_collection = client.create_collection(collection_name)
        logger.info(f"✅ Created new collection")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create collection: {e}"
        }

    # Re-embed and add documents in batches
    logger.info("🔄 Re-embedding documents...")
    from cortex_engine.embedding_service import embed_texts_batch

    batch_size = 32
    total_embedded = 0
    errors = []

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        try:
            # Generate new embeddings
            batch_embeddings = embed_texts_batch(batch_docs, batch_size=batch_size)

            # Add to collection
            new_collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embeddings
            )

            total_embedded += len(batch_docs)
            progress = (total_embedded / len(documents)) * 100
            logger.info(f"   Progress: {total_embedded:,}/{len(documents):,} ({progress:.1f}%)")

        except Exception as e:
            error_msg = f"Batch {i//batch_size + 1} failed: {e}"
            logger.error(f"❌ {error_msg}")
            errors.append(error_msg)

    # Update collection metadata
    from cortex_engine.utils.embedding_validator import get_embedding_dimension

    try:
        target_dimension = get_embedding_dimension(target_model)
        collection_mgr.set_embedding_model_metadata(
            "default",
            target_model,
            target_dimension
        )
        logger.info(f"✅ Updated collection metadata: {target_model} ({target_dimension}D)")
    except Exception as e:
        logger.warning(f"⚠️  Could not update metadata: {e}")

    logger.info("")
    logger.info("="*80)
    logger.info("MIGRATION COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Successfully migrated {total_embedded:,} documents")
    logger.info(f"📊 New embedding model: {target_model}")

    if errors:
        logger.warning(f"⚠️  {len(errors)} errors occurred during migration")
        for error in errors[:5]:  # Show first 5 errors
            logger.warning(f"   - {error}")

    return {
        "status": "success",
        "documents_migrated": total_embedded,
        "source_model": source_model,
        "target_model": target_model,
        "errors": errors
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate ChromaDB collection to a new embedding model"
    )
    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help="Path to database root (default: from config)"
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target embedding model (e.g., nvidia/NV-Embed-v2)"
    )
    parser.add_argument(
        "--source-model",
        help="Source embedding model (auto-detected if not specified)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation (not recommended)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration plan without executing"
    )

    args = parser.parse_args()

    try:
        result = migrate_collection(
            db_path=args.db_path,
            target_model=args.target_model,
            source_model=args.source_model,
            backup=not args.no_backup,
            dry_run=args.dry_run
        )

        if result["status"] == "success":
            sys.exit(0)
        elif result["status"] == "dry_run":
            sys.exit(0)
        elif result["status"] == "no_change":
            sys.exit(0)
        else:
            logger.error(f"Migration failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
