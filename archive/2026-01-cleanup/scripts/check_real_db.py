#!/usr/bin/env python3
"""Check the REAL ChromaDB at /mnt/f/ai_databases/knowledge_hub_db/"""

import os
os.environ["CORTEX_EMBED_MODEL"] = "BAAI/bge-base-en-v1.5"

import chromadb
from cortex_engine.utils.embedding_validator import get_embedding_dimension
from cortex_engine.config import EMBED_MODEL

REAL_DB_PATH = "/mnt/f/ai_databases/knowledge_hub_db"

print("=" * 70)
print("CHECKING REAL CHROMADB DATABASE")
print("=" * 70)
print(f"\nDatabase Path: {REAL_DB_PATH}")
print(f"Expected Model: {EMBED_MODEL}")
print(f"Expected Dimension: {get_embedding_dimension(EMBED_MODEL)}D")
print()

try:
    client = chromadb.PersistentClient(path=REAL_DB_PATH)
    collections = client.list_collections()

    if not collections:
        print("❌ No collections found")
    else:
        print(f"✅ Found {len(collections)} collection(s):\n")

        for collection in collections:
            print(f"📁 Collection: {collection.name}")
            count = collection.count()
            print(f"   Documents: {count}")

            if count > 0:
                # Sample embeddings
                sample_size = min(10, count)
                results = collection.get(limit=sample_size, include=['embeddings', 'metadatas'])

                if results['embeddings']:
                    dimensions = [len(emb) for emb in results['embeddings']]
                    unique_dims = set(dimensions)

                    print(f"   Sampled {sample_size} embeddings:")
                    print(f"   Unique dimensions: {unique_dims}")

                    if len(unique_dims) == 1:
                        dim = list(unique_dims)[0]
                        expected = get_embedding_dimension(EMBED_MODEL)
                        if dim == expected:
                            print(f"   ✅ All embeddings are {dim}D (matches {EMBED_MODEL})")
                        else:
                            print(f"   ❌ MISMATCH! Embeddings are {dim}D but expected {expected}D")
                    else:
                        print(f"   ❌ MIXED DIMENSIONS! Data corruption detected!")

                # Show sample metadata
                if results['metadatas'] and len(results['metadatas']) > 0:
                    sample_meta = results['metadatas'][0]
                    print(f"   Sample metadata keys: {list(sample_meta.keys())[:5]}")
            else:
                print(f"   ⚠️  Empty collection")

            print()

        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        total_docs = sum(c.count() for c in collections)
        print(f"\nTotal Documents in ChromaDB: {total_docs}")
        print(f"Collections: {', '.join(c.name for c in collections)}")
        print("\n✅ Database is working and contains data!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
