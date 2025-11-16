# ## File: pages/13_Metadata_Management.py
# Purpose: Browse and edit document metadata (thematic_tags, types, outcomes) across the knowledge base.

import os
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings

from cortex_engine.config import COLLECTION_NAME
from cortex_engine.config_manager import ConfigManager
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils import (
    get_logger,
    convert_to_docker_mount_path,
    convert_windows_to_wsl_path,
    resolve_db_root_path,
)

logger = get_logger(__name__)

st.set_page_config(page_title="Metadata & Tag Management", layout="wide")


def normalize_tags(raw) -> List[str]:
    """Normalize thematic_tags to a clean list of strings."""
    if not raw:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]
    if isinstance(raw, list):
        return [str(p).strip() for p in raw if str(p).strip()]
    return []


def load_chroma_collection() -> tuple:
    """Load ChromaDB collection based on user-configured db path."""
    try:
        config = ConfigManager().get_config()
        raw_db = config.get("ai_database_path", "")
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return None, None, None

    # Normalize path for current environment
    resolved_root = resolve_db_root_path(raw_db) or Path(convert_windows_to_wsl_path(raw_db))
    safe_root = convert_to_docker_mount_path(str(resolved_root)) if resolved_root else ""
    chroma_dir = os.path.join(safe_root, "knowledge_hub_db")

    if not chroma_dir or not os.path.isdir(chroma_dir):
        st.error(f"Knowledge base not found at '{chroma_dir}'. Set a valid database path in Knowledge Ingest/Maintenance.")
        return None, None, None

    try:
        settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=chroma_dir, settings=settings)
        collection = client.get_collection(COLLECTION_NAME)
        return collection, chroma_dir, raw_db
    except Exception as e:
        st.error(f"Could not open ChromaDB at {chroma_dir}: {e}")
        return None, None, None


def gather_tags(metadatas: List[Dict[str, Any]]) -> List[str]:
    tags = set()
    for meta in metadatas:
        for tag in normalize_tags(meta.get("thematic_tags")):
            tags.add(tag)
    return sorted(tags)


def update_tags(collection, doc_ids: List[str], metadatas: List[Dict[str, Any]], add_tags: List[str], remove_tags: List[str]) -> int:
    """Apply tag updates to a set of documents."""
    updated = 0
    for doc_id, meta in zip(doc_ids, metadatas):
        existing = normalize_tags(meta.get("thematic_tags"))
        tag_set = set(existing)
        if add_tags:
            tag_set.update(add_tags)
        if remove_tags:
            tag_set.difference_update(remove_tags)
        meta = dict(meta or {})
        meta["thematic_tags"] = ", ".join(sorted(tag_set))
        try:
            collection.update(ids=[doc_id], metadatas=[meta])
            updated += 1
        except Exception as e:
            logger.warning(f"Failed to update tags for {doc_id}: {e}")
    return updated


def render():
    st.title("🔖 13. Metadata & Tag Management")
    st.caption("Browse documents, filter, and bulk-edit thematic tags, types, and outcomes.")

    collection, chroma_dir, raw_db = load_chroma_collection()
    if not collection:
        return

    st.success(f"Connected to knowledge base at `{chroma_dir}` (configured as `{raw_db}`)")

    # Fetch a page of documents
    with st.sidebar:
        st.header("Filters")
        limit = st.number_input("Docs to load", min_value=50, max_value=2000, value=500, step=50)
        offset = st.number_input("Offset", min_value=0, value=0, step=limit)

    try:
        docs = collection.get(limit=limit, offset=offset, include=["metadatas", "documents"])
    except Exception as e:
        st.error(f"Failed to load documents from ChromaDB: {e}")
        return

    metadatas = docs.get("metadatas", [])
    ids = docs.get("ids", [])
    documents = docs.get("documents", [])
    if not metadatas or not ids:
        st.info("No documents found in this range.")
        return

    # Flatten single-list structure
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if documents and isinstance(documents[0], list):
        documents = documents[0]

    all_tags = gather_tags(metadatas)

    with st.sidebar:
        doc_type_filter = st.selectbox(
            "Document Type",
            ["Any", "Project Plan", "Technical Documentation", "Proposal/Quote", "Case Study / Trophy",
             "Final Report", "Draft Report", "Presentation", "Contract/SOW",
             "Meeting Minutes", "Financial Report", "Research Paper", "Email Correspondence", "Image/Diagram", "Other"],
            index=0,
        )
        outcome_filter = st.selectbox("Outcome", ["Any", "Won", "Lost", "Pending", "N/A"], index=0)
        tag_filter = st.multiselect("Thematic Tags", all_tags, key="filter_tags")

    rows = []
    for doc_id, meta, text in zip(ids, metadatas, documents or [""] * len(ids)):
        tags = normalize_tags(meta.get("thematic_tags"))
        if doc_type_filter != "Any" and meta.get("document_type") != doc_type_filter:
            continue
        if outcome_filter != "Any" and meta.get("proposal_outcome") != outcome_filter:
            continue
        if tag_filter and not set(tag_filter).issubset(set(tags)):
            continue
        rows.append({
            "doc_id": doc_id,
            "file_name": meta.get("file_name", meta.get("doc_posix_path", "")),
            "document_type": meta.get("document_type", "Unknown"),
            "proposal_outcome": meta.get("proposal_outcome", "N/A"),
            "thematic_tags": ", ".join(tags),
            "summary": meta.get("summary", text[:200] + "…"),
        })

    if not rows:
        st.warning("No documents match the current filters.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Bulk tag editing")
    choices = [f"{r['file_name']} ({r['doc_id']})" for r in rows]
    selected_labels = st.multiselect("Select documents to edit", choices)
    label_to_id = {f"{r['file_name']} ({r['doc_id']})": r["doc_id"] for r in rows}
    selected_ids = [label_to_id[lbl] for lbl in selected_labels]

    col1, col2 = st.columns(2)
    with col1:
        add_tags_input = st.text_input("Add tags (comma-separated)", key="add_tags_input")
    with col2:
        remove_tags_input = st.text_input("Remove tags (comma-separated)", key="remove_tags_input")

    if st.button("Apply Tag Updates", type="primary", disabled=not selected_ids):
        add_tags = [t.strip() for t in add_tags_input.split(",") if t.strip()]
        remove_tags = [t.strip() for t in remove_tags_input.split(",") if t.strip()]
        # Map selected ids to their metadatas for update
        id_to_meta = {id_: meta for id_, meta in zip(ids, metadatas)}
        metas_to_update = [id_to_meta[i] for i in selected_ids if i in id_to_meta]
        updated = update_tags(collection, selected_ids, metas_to_update, add_tags, remove_tags)
        st.success(f"Updated tags for {updated} document(s).")
        st.experimental_rerun()

    st.markdown("---")
    st.caption("Tip: Use filters to reduce the set, then bulk-add or remove tags across the selected documents.")


if __name__ == "__main__":
    render()
