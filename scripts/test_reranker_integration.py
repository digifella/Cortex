#!/usr/bin/env python3
"""
Test Qwen3-VL Reranker End-to-End Integration
==============================================

This script tests the reranker with actual ChromaDB data to verify
the end-to-end integration works correctly.

Usage:
    cd /home/longboardfella/cortex_suite
    source venv/bin/activate
    python scripts/test_reranker_integration.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable reranker for this test
os.environ["QWEN3_VL_RERANKER_ENABLED"] = "true"
os.environ["QWEN3_VL_RERANKER_SIZE"] = "auto"
os.environ["QWEN3_VL_RERANKER_TOP_K"] = "5"
os.environ["QWEN3_VL_RERANKER_CANDIDATES"] = "20"


def test_reranker_loading():
    """Test that the reranker model loads correctly."""
    print("=" * 60)
    print("Test 1: Reranker Model Loading")
    print("=" * 60)

    try:
        from cortex_engine.qwen3_vl_reranker_service import (
            _load_reranker,
            Qwen3VLRerankerConfig
        )

        print("Loading Qwen3-VL Reranker (this may take a moment)...")
        config = Qwen3VLRerankerConfig.auto_select()
        print(f"Selected model: {config.model_name}")

        model, processor, cfg = _load_reranker(config)

        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Device: {next(model.parameters()).device}")
        return True

    except Exception as e:
        print(f"❌ Failed to load reranker: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reranker_scoring():
    """Test reranker with sample data."""
    print("\n" + "=" * 60)
    print("Test 2: Reranker Scoring")
    print("=" * 60)

    try:
        from cortex_engine.qwen3_vl_reranker_service import rerank_text_results

        query = "How do I configure the database connection?"

        # Sample results simulating vector search output
        texts = [
            "The system uses PostgreSQL for data storage.",
            "Database configuration is in config/database.yml",
            "Connection pooling settings can be adjusted in the config file",
            "Run migrations with: python manage.py migrate",
            "The frontend uses React for the user interface",
        ]

        original_scores = [0.85, 0.82, 0.80, 0.78, 0.75]

        print(f"Query: '{query}'")
        print(f"\nInitial results (by embedding similarity):")
        for i, (text, score) in enumerate(zip(texts, original_scores)):
            print(f"  {i+1}. [{score:.2f}] {text}")

        print("\nApplying neural reranking...")
        reranked = rerank_text_results(
            query,
            texts,
            top_k=3,
            original_scores=original_scores
        )

        print(f"\n✅ Reranked results (top 3):")
        for i, result in enumerate(reranked):
            print(f"  {i+1}. [{result.rerank_score:.3f}] {result.document.content}")
            if result.rank_change != 0:
                direction = "↑" if result.rank_change > 0 else "↓"
                print(f"      ({direction} moved {abs(result.rank_change)} positions)")

        return True

    except Exception as e:
        print(f"❌ Reranking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_query_integration():
    """Test reranker integration with graph_query.py."""
    print("\n" + "=" * 60)
    print("Test 3: graph_query.py Integration")
    print("=" * 60)

    try:
        from cortex_engine.graph_query import rerank_search_results
        from cortex_engine.config import QWEN3_VL_RERANKER_ENABLED

        print(f"QWEN3_VL_RERANKER_ENABLED: {QWEN3_VL_RERANKER_ENABLED}")

        # Simulate search results format from direct_chromadb_search
        sample_results = [
            {
                "text": "The system uses PostgreSQL for data storage and retrieval.",
                "score": 0.85,
                "doc_id": "doc_001",
                "file_name": "database_guide.pdf"
            },
            {
                "text": "Database configuration is stored in config/database.yml file.",
                "score": 0.82,
                "doc_id": "doc_002",
                "file_name": "installation.pdf"
            },
            {
                "text": "Frontend uses React framework for user interface components.",
                "score": 0.75,
                "doc_id": "doc_003",
                "file_name": "architecture.pdf"
            },
        ]

        query = "database configuration settings"

        print(f"\nQuery: '{query}'")
        print(f"\nOriginal results:")
        for i, r in enumerate(sample_results):
            print(f"  {i+1}. [{r['score']:.2f}] {r['file_name']}: {r['text'][:50]}...")

        print("\nCalling rerank_search_results()...")
        reranked = rerank_search_results(
            query=query,
            results=sample_results,
            top_k=3,
            text_key="text"
        )

        print(f"\n✅ Reranked results:")
        for i, r in enumerate(reranked):
            rerank_score = r.get('rerank_score', 'N/A')
            rank_change = r.get('rank_change', 0)
            print(f"  {i+1}. [{rerank_score:.3f}] {r['file_name']}: {r['text'][:50]}...")
            if rank_change != 0:
                direction = "↑" if rank_change > 0 else "↓"
                print(f"      ({direction} moved {abs(rank_change)} positions)")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_live_search():
    """Test with actual ChromaDB database."""
    print("\n" + "=" * 60)
    print("Test 4: Live ChromaDB Search with Reranking")
    print("=" * 60)

    try:
        # Get database path from config
        import json
        config_path = Path(__file__).parent.parent / "cortex_config.json"

        if not config_path.exists():
            print("cortex_config.json not found - skipping live test")
            return True

        with open(config_path) as f:
            config = json.load(f)

        db_path = config.get("ai_database_path", "")
        if not db_path:
            print("No database path configured - skipping live test")
            return True

        # Convert Windows path to WSL if needed
        from cortex_engine.utils import convert_windows_to_wsl_path
        db_path = convert_windows_to_wsl_path(db_path)

        chromadb_path = Path(db_path) / "knowledge_hub_db"
        if not chromadb_path.exists():
            print(f"ChromaDB not found at {chromadb_path} - skipping live test")
            return True

        print(f"Using database: {chromadb_path}")

        # Import search function
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(chromadb_path),
            settings=Settings(anonymized_telemetry=False)
        )

        collections = client.list_collections()
        if not collections:
            print("No collections found - skipping live test")
            return True

        collection = collections[0]
        print(f"Collection: {collection.name} ({collection.count()} documents)")

        # Query using the embedding service (matches collection dimensions)
        query = "strategy"
        print(f"\nQuery: '{query}'")

        # Use the configured embedding model to match collection dimensions
        try:
            from cortex_engine.embedding_service import get_embedding_function
            embed_fn = get_embedding_function()
            query_embedding = embed_fn([query])[0]
            print(f"Query embedding dimension: {len(query_embedding)}")

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )
        except Exception as e:
            print(f"Embedding service unavailable ({e}), using text query")
            results = collection.query(
                query_texts=[query],
                n_results=10
            )

        if not results['documents'] or not results['documents'][0]:
            print("No results found")
            return True

        # Format results
        formatted_results = []
        for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
            score = 1 - dist if dist <= 2 else 0
            meta = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
            formatted_results.append({
                "text": doc,
                "score": score,
                "doc_id": meta.get('doc_id', f'doc_{i}'),
                "file_name": meta.get('file_name', 'unknown')
            })

        print(f"\nVector search returned {len(formatted_results)} results")
        print("\nTop 3 before reranking:")
        for i, r in enumerate(formatted_results[:3]):
            print(f"  {i+1}. [{r['score']:.3f}] {r['file_name']}")

        # Apply reranking
        from cortex_engine.graph_query import rerank_search_results

        print("\nApplying neural reranking...")
        reranked = rerank_search_results(
            query=query,
            results=formatted_results,
            top_k=5,
            text_key="text"
        )

        print(f"\n✅ Reranked results (top 5):")
        for i, r in enumerate(reranked):
            rerank_score = r.get('rerank_score', 'N/A')
            original_rank = r.get('original_rank', 'N/A')
            rank_change = r.get('rank_change', 0)
            print(f"  {i+1}. [rerank:{rerank_score:.3f}] {r['file_name']}")
            if rank_change != 0:
                direction = "↑" if rank_change > 0 else "↓"
                print(f"      (was #{original_rank}, moved {direction}{abs(rank_change)})")

        return True

    except Exception as e:
        print(f"❌ Live search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Qwen3-VL Reranker Integration Tests")
    print("=" * 60)
    print(f"QWEN3_VL_RERANKER_ENABLED={os.environ.get('QWEN3_VL_RERANKER_ENABLED', 'not set')}")
    print()

    results = []

    # Test 1: Model loading
    results.append(("Model Loading", test_reranker_loading()))

    if results[-1][1]:
        # Test 2: Basic scoring
        results.append(("Reranker Scoring", test_reranker_scoring()))

        # Test 3: graph_query integration
        results.append(("graph_query Integration", test_graph_query_integration()))

        # Test 4: Live search
        results.append(("Live ChromaDB Search", test_live_search()))
    else:
        print("\n⚠️  Skipping remaining tests due to model loading failure")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1

    print(f"\n{passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Reranker is ready for use.")
        print("\nTo use in Cortex Suite:")
        print("  1. Set environment variables:")
        print("     export QWEN3_VL_RERANKER_ENABLED=true")
        print("  2. Restart Streamlit:")
        print("     streamlit run Cortex_Suite.py")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
