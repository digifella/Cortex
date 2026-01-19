#!/usr/bin/env python3
"""
Verify embeddings and database integrity for Cortex Suite
"""

import os
import sys
os.environ["CORTEX_EMBED_MODEL"] = "BAAI/bge-base-en-v1.5"

from cortex_engine.config import CHROMA_DB_PATH, EMBED_MODEL
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils.embedding_validator import (
    validate_embedding_compatibility,
    get_embedding_dimension
)
import chromadb

print("=" * 70)
print("CORTEX SUITE - DATABASE & EMBEDDING VERIFICATION")
print("=" * 70)

# 1. Check Configuration
print("\n📋 CONFIGURATION")
print("-" * 70)
print(f"Embedding Model: {EMBED_MODEL}")
print(f"Expected Dimension: {get_embedding_dimension(EMBED_MODEL)}D")
print(f"ChromaDB Path: {CHROMA_DB_PATH}")

# 2. Check Collection Metadata
print("\n📁 COLLECTION METADATA (WorkingCollectionManager)")
print("-" * 70)
collection_mgr = WorkingCollectionManager()
collections = collection_mgr.collections

if not collections:
    print("⚠️  No collections found in WorkingCollectionManager")
else:
    for coll_name, coll_data in collections.items():
        print(f"\n  Collection: {coll_name}")
        print(f"    Documents: {len(coll_data.get('doc_ids', []))}")
        print(f"    Created: {coll_data.get('created_at', 'Unknown')}")
        print(f"    Modified: {coll_data.get('modified_at', 'Unknown')}")

        # Check embedding metadata
        metadata = collection_mgr.get_embedding_model_metadata(coll_name)
        if metadata.get('embedding_model'):
            print(f"    Embedding Model: {metadata['embedding_model']}")
            print(f"    Embedding Dimension: {metadata['embedding_dimension']}D")

            # Validate compatibility
            try:
                validation = validate_embedding_compatibility(
                    metadata,
                    current_model=EMBED_MODEL,
                    strict=False
                )
                if validation['compatible']:
                    print(f"    ✅ COMPATIBLE with current system")
                else:
                    print(f"    ❌ INCOMPATIBLE - Model mismatch!")
                    for error in validation['errors']:
                        print(f"       {error}")
            except Exception as e:
                print(f"    ⚠️  Validation error: {e}")
        else:
            print(f"    ⚠️  No embedding metadata stored")

# 3. Check ChromaDB
print("\n💾 CHROMADB VERIFICATION")
print("-" * 70)

try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collections = client.list_collections()

    if not chroma_collections:
        print("⚠️  No collections found in ChromaDB")
    else:
        print(f"Found {len(chroma_collections)} collection(s) in ChromaDB:\n")

        for collection in chroma_collections:
            print(f"  Collection: {collection.name}")
            count = collection.count()
            print(f"    Documents in ChromaDB: {count}")

            if count > 0:
                # Sample a few embeddings to check dimensions
                sample_size = min(5, count)
                results = collection.get(limit=sample_size, include=['embeddings'])

                if results['embeddings']:
                    dimensions = [len(emb) for emb in results['embeddings']]
                    unique_dims = set(dimensions)

                    print(f"    Sampled {sample_size} embeddings:")
                    print(f"      Dimensions found: {unique_dims}")

                    if len(unique_dims) == 1:
                        dim = list(unique_dims)[0]
                        expected_dim = get_embedding_dimension(EMBED_MODEL)
                        if dim == expected_dim:
                            print(f"      ✅ All embeddings are {dim}D (matches {EMBED_MODEL})")
                        else:
                            print(f"      ❌ MISMATCH! Embeddings are {dim}D but expected {expected_dim}D")
                    else:
                        print(f"      ❌ CRITICAL! Mixed embedding dimensions detected!")
                        print(f"         This indicates data corruption from multiple models")
                else:
                    print(f"    ⚠️  Could not retrieve embeddings")
            else:
                print(f"    ⚠️  Collection is empty")

            print()

except Exception as e:
    print(f"❌ Error accessing ChromaDB: {e}")
    import traceback
    traceback.print_exc()

# 4. Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

# Count total documents
total_docs_in_metadata = sum(len(c.get('doc_ids', [])) for c in collections.values())
total_docs_in_chromadb = sum(c.count() for c in chroma_collections) if chroma_collections else 0

print(f"\nTotal Documents (Metadata): {total_docs_in_metadata}")
print(f"Total Documents (ChromaDB): {total_docs_in_chromadb}")

if total_docs_in_metadata == total_docs_in_chromadb:
    print("✅ Document counts match!")
else:
    diff = abs(total_docs_in_metadata - total_docs_in_chromadb)
    print(f"⚠️  MISMATCH: {diff} document(s) difference")
    if total_docs_in_metadata > total_docs_in_chromadb:
        print(f"   {diff} orphaned document(s) in metadata")
    else:
        print(f"   {diff} document(s) in ChromaDB not tracked in metadata")

print("\n" + "=" * 70)
print()
