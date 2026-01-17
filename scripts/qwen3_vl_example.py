#!/usr/bin/env python3
"""
Qwen3-VL Integration Example
=============================

This script demonstrates how to use Qwen3-VL multimodal embeddings
and reranking in the Cortex Suite.

Usage:
    python scripts/qwen3_vl_example.py

Before running:
    1. Install dependencies: pip install -r requirements-qwen3-vl.txt
    2. Enable Qwen3-VL: export QWEN3_VL_ENABLED=true
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_hardware():
    """Check hardware compatibility."""
    print("=" * 60)
    print("Hardware Check")
    print("=" * 60)

    from cortex_engine.utils.smart_model_selector import (
        detect_nvidia_gpu,
        get_optimal_qwen3_vl_config
    )

    # Check GPU
    has_nvidia, gpu_info = detect_nvidia_gpu()

    if has_nvidia:
        print(f"GPU: {gpu_info.get('device_name', 'Unknown')}")
        print(f"VRAM: {gpu_info.get('memory_total_gb', 0):.1f}GB")
        print(f"Detection method: {gpu_info.get('method', 'unknown')}")
    else:
        print("No NVIDIA GPU detected")
        print(f"Issues: {gpu_info.get('issues', [])}")
        return False

    # Get optimal config
    print("\nOptimal Qwen3-VL Configuration:")
    config = get_optimal_qwen3_vl_config()
    print(f"  Embedding: {config['embedding_model']}")
    print(f"  Reranker: {config['reranker_model']}")
    print(f"  Can run both: {config['can_run_both']}")
    print(f"  Notes: {config['notes']}")

    return True


def demo_text_embedding():
    """Demonstrate text embedding."""
    print("\n" + "=" * 60)
    print("Text Embedding Demo")
    print("=" * 60)

    from cortex_engine.qwen3_vl_embedding_service import (
        get_embedding_service,
        embed_text
    )

    # Get embedding service (auto-selects model based on hardware)
    service = get_embedding_service()
    print(f"Using model: {service.model_name}")
    print(f"Embedding dimension: {service.embedding_dimension}")

    # Embed some text
    texts = [
        "The quarterly revenue report shows a 15% increase in sales.",
        "A fluffy golden retriever playing on the beach at sunset.",
        "Machine learning models require significant computational resources.",
    ]

    print("\nEmbedding texts...")
    embeddings = service.embed_texts(texts)

    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"\n  Text {i+1}: {text[:50]}...")
        print(f"  Embedding shape: {len(emb)} dimensions")
        print(f"  First 5 values: {emb[:5]}")


def demo_image_embedding():
    """Demonstrate image embedding (requires an image file)."""
    print("\n" + "=" * 60)
    print("Image Embedding Demo")
    print("=" * 60)

    from cortex_engine.qwen3_vl_embedding_service import get_embedding_service

    # Check for sample image
    sample_images = list(Path(".").glob("**/*.png"))[:1]
    sample_images += list(Path(".").glob("**/*.jpg"))[:1]

    if not sample_images:
        print("No sample images found. Skipping image demo.")
        print("To test: place a .png or .jpg file in the project directory.")
        return

    image_path = sample_images[0]
    print(f"Using image: {image_path}")

    service = get_embedding_service()

    print("\nEmbedding image...")
    embedding = service.embed_image(str(image_path))

    print(f"Image embedding shape: {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}")


def demo_cross_modal_search():
    """Demonstrate cross-modal search (text query finding images)."""
    print("\n" + "=" * 60)
    print("Cross-Modal Search Demo")
    print("=" * 60)

    import numpy as np
    from cortex_engine.qwen3_vl_embedding_service import get_embedding_service

    service = get_embedding_service()

    # Simulate a mini database with text and "image" descriptions
    documents = [
        {"type": "text", "content": "Financial report Q3 2025 showing revenue growth"},
        {"type": "text", "content": "Employee handbook section on vacation policies"},
        {"type": "text", "content": "Architecture diagram of the microservices system"},
        {"type": "text", "content": "Bar chart comparing monthly sales across regions"},
    ]

    print("Indexing documents...")
    embeddings = service.embed_texts([d["content"] for d in documents])

    # Query
    query = "show me the sales chart"
    print(f"\nQuery: '{query}'")

    query_embedding = service.embed_query(query)

    # Compute similarities
    query_vec = np.array(query_embedding)
    doc_vecs = np.array(embeddings)

    # Cosine similarity (vectors are normalized)
    similarities = doc_vecs @ query_vec

    # Rank results
    ranked_indices = np.argsort(similarities)[::-1]

    print("\nResults (ranked by similarity):")
    for i, idx in enumerate(ranked_indices):
        print(f"  {i+1}. [{similarities[idx]:.3f}] {documents[idx]['content'][:50]}...")


def demo_reranking():
    """Demonstrate reranking of search results."""
    print("\n" + "=" * 60)
    print("Reranking Demo")
    print("=" * 60)

    from cortex_engine.qwen3_vl_reranker_service import (
        get_reranker_service,
        rerank_text_results
    )

    # Simulate initial search results (from vector search)
    query = "How do I configure the database connection?"

    initial_results = [
        "The system uses PostgreSQL for data storage.",
        "Database configuration is in config/database.yml",
        "Connection pooling settings can be adjusted in the config file",
        "Run migrations with: python manage.py migrate",
        "The frontend uses React for the user interface",
    ]

    original_scores = [0.85, 0.82, 0.80, 0.78, 0.75]

    print(f"Query: '{query}'")
    print(f"\nInitial results (by embedding similarity):")
    for i, (text, score) in enumerate(zip(initial_results, original_scores)):
        print(f"  {i+1}. [{score:.2f}] {text}")

    # Rerank
    print("\nReranking...")
    reranked = rerank_text_results(
        query,
        initial_results,
        top_k=3,
        original_scores=original_scores
    )

    print(f"\nReranked results (top 3):")
    for i, result in enumerate(reranked):
        print(f"  {i+1}. [{result.rerank_score:.3f}] {result.document.content}")
        if result.rank_change != 0:
            direction = "up" if result.rank_change > 0 else "down"
            print(f"      (moved {direction} {abs(result.rank_change)} positions)")


def demo_llamaindex_integration():
    """Demonstrate LlamaIndex integration."""
    print("\n" + "=" * 60)
    print("LlamaIndex Integration Demo")
    print("=" * 60)

    try:
        from cortex_engine.qwen3_vl_llamaindex_adapter import (
            configure_llamaindex_with_qwen3_vl
        )

        print("Configuring LlamaIndex with Qwen3-VL...")
        config = configure_llamaindex_with_qwen3_vl(
            use_reranker=True,
            embedding_size="auto",
            reranker_size="auto",
            reranker_top_n=5,
            similarity_top_k=20
        )

        print("\nConfiguration:")
        for key, value in config["info"].items():
            print(f"  {key}: {value}")

        print("\nUsage example:")
        print("  from llama_index.core import VectorStoreIndex")
        print("  index = VectorStoreIndex.from_documents(")
        print("      documents,")
        print("      embed_model=config['embed_model']")
        print("  )")
        print("  query_engine = index.as_query_engine(")
        print("      node_postprocessors=config['node_postprocessors'],")
        print("      similarity_top_k=config['similarity_top_k']")
        print("  )")

    except ImportError as e:
        print(f"LlamaIndex not installed: {e}")
        print("Install with: pip install llama-index")


def main():
    """Run all demos."""
    print("Qwen3-VL Integration Examples")
    print("=" * 60)

    # Check hardware first
    if not check_hardware():
        print("\nQwen3-VL requires an NVIDIA GPU with CUDA support.")
        return

    # Run demos
    try:
        demo_text_embedding()
    except Exception as e:
        print(f"Text embedding demo failed: {e}")

    try:
        demo_image_embedding()
    except Exception as e:
        print(f"Image embedding demo failed: {e}")

    try:
        demo_cross_modal_search()
    except Exception as e:
        print(f"Cross-modal search demo failed: {e}")

    try:
        demo_reranking()
    except Exception as e:
        print(f"Reranking demo failed: {e}")

    try:
        demo_llamaindex_integration()
    except Exception as e:
        print(f"LlamaIndex demo failed: {e}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo enable Qwen3-VL in Cortex Suite:")
    print("  export QWEN3_VL_ENABLED=true")
    print("  export QWEN3_VL_RERANKER_ENABLED=true")
    print("  streamlit run Cortex_Suite.py")


if __name__ == "__main__":
    main()
