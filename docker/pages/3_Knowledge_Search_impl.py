"""
Minimal Knowledge Search implementation for Docker environments.
Provides direct ChromaDB search using the centralized embedding service.
"""

import os
from pathlib import Path
import streamlit as st
import chromadb
from chromadb.config import Settings as ChromaSettings

from cortex_engine.utils import convert_to_docker_mount_path, get_logger
from cortex_engine.config_manager import ConfigManager
from cortex_engine.config import COLLECTION_NAME
from cortex_engine.embedding_service import embed_query

logger = get_logger(__name__)


def _get_collection(db_path: str):
    container_db = convert_to_docker_mount_path(db_path)
    chroma_db_path = os.path.join(container_db, "knowledge_hub_db")
    db_settings = ChromaSettings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.get_or_create_collection(COLLECTION_NAME)


def main():
    st.set_page_config(page_title="Knowledge Search", layout="wide")
    st.title("🔍 Knowledge Search (Docker)")

    # Config
    cfg = ConfigManager().get_config()
    db_path = cfg.get("ai_database_path", "")
    st.text_input("Knowledge Hub DB Path", value=db_path, disabled=True)

    if not db_path:
        st.warning("Please set the AI Database Path on Knowledge Ingest first.")
        return

    try:
        collection = _get_collection(db_path)
        count = collection.count()
        st.caption(f"Vector store documents: {count}")
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")
        return

    # Query UI
    query = st.text_input("Enter your search query:", placeholder="e.g., strategy and transformation")
    top_k = st.slider("Results", min_value=5, max_value=50, value=20, step=5)

    if st.button("Search", type="primary") and query.strip():
        try:
            q_emb = embed_query(query)
            res = collection.query(query_embeddings=[q_emb], n_results=top_k)
            documents = res.get("documents", [[]])[0]
            metadatas = res.get("metadatas", [[]])[0]
            ids = res.get("ids", [[]])[0]
            distances = res.get("distances", [[]])[0]

            if not documents:
                st.info("No results found.")
                return

            st.subheader("Results")
            for i, (doc, meta, _id, dist) in enumerate(zip(documents, metadatas, ids, distances), start=1):
                with st.container(border=True):
                    st.markdown(f"**{i}.** `{meta.get('file_name', 'Unknown')}` • id=`{_id}` • score={1.0 - dist:.4f}")
                    st.caption(meta.get("doc_posix_path", ""))
                    st.write(doc[:1200] + ("..." if len(doc) > 1200 else ""))
        except Exception as e:
            st.error(f"Search failed: {e}")

