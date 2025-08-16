# ## File: cortex_inspector.py
# Version: 4.0.0 (Utilities Refactor)
# Date: 2025-07-23
# Purpose: A diagnostic tool to inspect the contents of the Cortex knowledge base.
#          - REFACTOR (v4.0.0): Updated to use centralized utilities for path handling,
#            logging, and error handling. Removed code duplication.

import pickle
import argparse
import networkx as nx
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
import os
import sys
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import centralized utilities
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger
from cortex_engine import config

# Set up logging
logger = get_logger(__name__)

def print_header(title: str):
    print("\n" + "="*80); print(f" {title.upper()} ".center(80, "=")); print("="*80)

def inspect_document(graph: nx.DiGraph, collection: chromadb.Collection, doc_path: str):
    posix_path = Path(doc_path).as_posix()
    print_header(f"Inspecting Document: {posix_path}")

    if not graph.has_node(posix_path):
        print(f"\n❌ ERROR: Document not found in the Knowledge Graph.")
        print(f"   Please ensure the path is correct and was included in the last ingestion.")
        return

    # --- Print Graph Metadata ---
    graph_metadata = graph.nodes[posix_path]
    print("\n--- 🔎 Metadata from Knowledge Graph Node ---")
    for key, value in sorted(graph_metadata.items()):
        if key == 'thematic_tags' and isinstance(value, list):
            print(f"  - {key}:")
            for tag in value:
                print(f"    - {tag}")
        else:
            print(f"  - {key}: {value}")

    # --- Print Vector Store Metadata ---
    print("\n--- 🔎 Metadata from a Vector Store Chunk ---")
    try:
        # Fetch one chunk from the vector store associated with this file
        vector_chunks = collection.get(where={"file_path": posix_path}, limit=1, include=["metadatas"])
        if not vector_chunks or not vector_chunks['ids']:
            print("\n-> No associated chunks found in the vector store for this file.")
        else:
            vector_metadata = vector_chunks['metadatas'][0]
            print("(Showing one example chunk to verify metadata was propagated correctly)")
            for key, value in sorted(vector_metadata.items()):
                 if key == 'thematic_tags' and isinstance(value, list):
                    print(f"  - {key}:")
                    for tag in value:
                        print(f"    - {tag}")
                 else:
                    print(f"  - {key}: {value}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not query the vector store. Reason: {e}")


def inspect_stats(graph: nx.DiGraph, collection: chromadb.Collection, db_path: str):
    graph_file_path = os.path.join(db_path, "knowledge_cortex.gpickle")
    image_store_path = os.path.join(db_path, "knowledge_hub_db", "images")
    print_header("Knowledge Base Health Check")
    print("\n--- Knowledge Graph ---")
    print(f"Graph loaded from: {graph_file_path}"); print(f"Total Nodes: {graph.number_of_nodes()}"); print(f"Total Edges: {graph.number_of_edges()}")
    node_types = {}
    for _, data in graph.nodes(data=True): ntype = data.get('node_type', 'Unknown'); node_types[ntype] = node_types.get(ntype, 0) + 1
    print("Node Distribution:"); [print(f"  - {ntype}: {count}") for ntype, count in sorted(node_types.items())]
    print("\n--- Vector Store ---"); print(f"ChromaDB Collection '{collection.name}'"); print(f"Total Embeddings: {collection.count()}")
    print("\n--- Image Store ---")
    if os.path.exists(image_store_path):
        image_count = len([name for name in os.listdir(image_store_path) if os.path.isfile(os.path.join(image_store_path, name))])
        print(f"Image store path: {image_store_path}"); print(f"Total Images Found: {image_count}")
    else: print(f"Image store path not found at: {image_store_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A diagnostic tool for Project Cortex's knowledge base.")
    parser.add_argument("--db-path", required=True, type=str, help="The root directory where the database is stored.")
    parser.add_argument("--stats", action="store_true", help="Display overall stats of the graph and vector store.")
    parser.add_argument("--inspect-doc", type=str, metavar="PATH", help="Inspect all stored metadata for a specific document path.")

    args = parser.parse_args()
    wsl_db_path = convert_windows_to_wsl_path(args.db_path)

    graph_file_path = os.path.join(wsl_db_path, "knowledge_cortex.gpickle")
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")

    graph, collection = None, None

    try:
        with open(graph_file_path, "rb") as f: graph = pickle.load(f)
    except FileNotFoundError:
        graph = nx.DiGraph()
        if not args.inspect_doc:
             print(f"❌ ERROR: Graph file not found at '{graph_file_path}'. Please run an ingestion first."); sys.exit(1)

    # --- FIX: Use consistent ChromaSettings ---
    db_settings = ChromaSettings(anonymized_telemetry=False)
    chroma_client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
    try:
        collection = chroma_client.get_collection(name=config.COLLECTION_NAME)
    except Exception as e:
        print(f"❌ ERROR: Could not connect to ChromaDB collection. Have you run an ingestion? Reason: {e}"); sys.exit(1)

    if args.stats:
        inspect_stats(graph, collection, wsl_db_path)
    elif args.inspect_doc:
        wsl_doc_path = convert_windows_to_wsl_path(args.inspect_doc)
        inspect_document(graph, collection, wsl_doc_path)
    else:
        parser.print_help()