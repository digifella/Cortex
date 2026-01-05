"""
Metadata & Tag Management - Browse and edit document metadata
Version: v1.0.0
Date: 2026-01-01
Purpose: Browse and bulk-edit thematic tags, document types, and outcomes across the knowledge base
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Metadata & Tag Management",
    page_icon="🔖",
    layout="wide"
)

# Import dependencies
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.ui_components import error_display, render_version_footer
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

# Apply theme IMMEDIATELY
apply_theme()

# ============================================
# HELPER FUNCTIONS
# ============================================

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
        error_display(
            str(e),
            error_type="Configuration Error",
            recovery_suggestion="Check your configuration file in the Maintenance page"
        )
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return None, None, None

    # Normalize path for current environment
    resolved_root = resolve_db_root_path(raw_db) or Path(convert_windows_to_wsl_path(raw_db))
    safe_root = convert_to_docker_mount_path(str(resolved_root)) if resolved_root else ""
    chroma_dir = os.path.join(safe_root, "knowledge_hub_db")

    if not chroma_dir or not os.path.isdir(chroma_dir):
        error_display(
            f"Knowledge base not found at '{chroma_dir}'",
            error_type="Database Not Found",
            recovery_suggestion="Set a valid database path in Knowledge Ingest or Maintenance page"
        )
        return None, None, None

    try:
        settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=chroma_dir, settings=settings)
        collection = client.get_collection(COLLECTION_NAME)
        logger.info(f"Connected to ChromaDB at {chroma_dir}")
        return collection, chroma_dir, raw_db
    except Exception as e:
        error_display(
            str(e),
            error_type="Database Connection Error",
            recovery_suggestion="Ensure the database exists and is not corrupted. Try running a collection check in Maintenance."
        )
        logger.error(f"Could not open ChromaDB at {chroma_dir}: {e}", exc_info=True)
        return None, None, None


def gather_tags(metadatas: List[Dict[str, Any]]) -> List[str]:
    """Extract all unique tags from metadata."""
    tags = set()
    for meta in metadatas:
        for tag in normalize_tags(meta.get("thematic_tags")):
            tags.add(tag)
    return sorted(tags)


def update_tags(collection, doc_ids: List[str], metadatas: List[Dict[str, Any]],
                add_tags: List[str], remove_tags: List[str]) -> int:
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

# ============================================
# PAGE HEADER
# ============================================

st.title("🔖 Metadata & Tag Management")
st.markdown("""
**Browse and edit document metadata** across your knowledge base.
Manage thematic tags, document types, and proposal outcomes with bulk editing capabilities.
""")

st.caption("💡 Use the sidebar (←) to navigate between pages")
st.markdown("---")

# ============================================
# DATABASE CONNECTION
# ============================================

collection, chroma_dir, raw_db = load_chroma_collection()
if not collection:
    st.stop()

st.success(f"✅ Connected to knowledge base at `{chroma_dir}`")
if raw_db != chroma_dir:
    st.caption(f"Configured path: `{raw_db}`")

# ============================================
# SIDEBAR CONFIGURATION
# ============================================

with st.sidebar:
    st.header("⚙️ Configuration")

    section_header("📊", "Loading Options")

    limit = st.number_input(
        "Documents to load",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Number of documents to load at once. Higher values may be slower."
    )

    offset = st.number_input(
        "Offset",
        min_value=0,
        value=0,
        step=limit,
        help="Skip this many documents from the start. Use for pagination."
    )

    st.divider()

# ============================================
# LOAD DOCUMENTS
# ============================================

section_header("📚", "Document Browser", f"Showing documents {offset} to {offset + limit}")

try:
    docs = collection.get(limit=limit, offset=offset, include=["metadatas", "documents"])
    logger.info(f"Loaded {len(docs.get('ids', []))} documents")
except Exception as e:
    error_display(
        str(e),
        error_type="Document Loading Error",
        recovery_suggestion="Try reducing the number of documents to load or check database integrity"
    )
    logger.error(f"Failed to load documents from ChromaDB: {e}", exc_info=True)
    st.stop()

metadatas = docs.get("metadatas", [])
ids = docs.get("ids", [])
documents = docs.get("documents", [])

if not metadatas or not ids:
    st.info("📋 No documents found in this range. Try adjusting the offset.")
    st.stop()

# Flatten single-list structure (ChromaDB sometimes returns nested lists)
if metadatas and isinstance(metadatas[0], list):
    metadatas = metadatas[0]
if ids and isinstance(ids[0], list):
    ids = ids[0]
if documents and isinstance(documents[0], list):
    documents = documents[0]

all_tags = gather_tags(metadatas)

# ============================================
# FILTERS
# ============================================

with st.sidebar:
    section_header("🔍", "Filters")

    doc_type_filter = st.selectbox(
        "Document Type",
        ["Any", "Project Plan", "Technical Documentation", "Proposal/Quote", "Case Study / Trophy",
         "Final Report", "Draft Report", "Presentation", "Contract/SOW",
         "Meeting Minutes", "Financial Report", "Research Paper", "Email Correspondence", "Image/Diagram", "Other"],
        index=0,
        help="Filter documents by type"
    )

    outcome_filter = st.selectbox(
        "Outcome",
        ["Any", "Won", "Lost", "Pending", "N/A"],
        index=0,
        help="Filter by proposal outcome status"
    )

    tag_filter = st.multiselect(
        "Thematic Tags",
        all_tags,
        key="filter_tags",
        help="Filter by tags (documents must have ALL selected tags)"
    )

# ============================================
# FILTER AND DISPLAY DOCUMENTS
# ============================================

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
        "summary": meta.get("summary", text[:200] + "…" if text else ""),
    })

if not rows:
    st.warning("⚠️ No documents match the current filters. Try adjusting your filter settings.")
    st.stop()

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
st.caption(f"Showing {len(rows)} of {len(ids)} loaded documents")

# ============================================
# BULK TAG EDITING
# ============================================

st.markdown("---")
section_header("✏️", "Bulk Tag Editing", "Select documents and add/remove tags")

choices = [f"{r['file_name']} ({r['doc_id']})" for r in rows]
selected_labels = st.multiselect(
    "Select documents to edit",
    choices,
    help="Choose one or more documents to apply tag changes"
)

label_to_id = {f"{r['file_name']} ({r['doc_id']})": r["doc_id"] for r in rows}
selected_ids = [label_to_id[lbl] for lbl in selected_labels]

col1, col2 = st.columns(2)
with col1:
    add_tags_input = st.text_input(
        "Add tags (comma-separated)",
        key="add_tags_input",
        placeholder="e.g., ai, healthcare, research",
        help="Tags to add to selected documents"
    )
with col2:
    remove_tags_input = st.text_input(
        "Remove tags (comma-separated)",
        key="remove_tags_input",
        placeholder="e.g., draft, old",
        help="Tags to remove from selected documents"
    )

if st.button("✅ Apply Tag Updates", type="primary", disabled=not selected_ids, use_container_width=True):
    add_tags = [t.strip() for t in add_tags_input.split(",") if t.strip()]
    remove_tags = [t.strip() for t in remove_tags_input.split(",") if t.strip()]

    if not add_tags and not remove_tags:
        st.warning("⚠️ Please specify tags to add or remove.")
    else:
        try:
            # Map selected ids to their metadatas for update
            id_to_meta = {id_: meta for id_, meta in zip(ids, metadatas)}
            metas_to_update = [id_to_meta[i] for i in selected_ids if i in id_to_meta]

            with st.spinner("Updating tags..."):
                updated = update_tags(collection, selected_ids, metas_to_update, add_tags, remove_tags)

            st.success(f"✅ Updated tags for {updated} document(s).")
            logger.info(f"Bulk tag update: {updated} documents updated")
            st.rerun()

        except Exception as e:
            error_display(
                str(e),
                error_type="Tag Update Error",
                recovery_suggestion="Check that the documents still exist in the database"
            )
            logger.error(f"Failed to update tags: {e}", exc_info=True)

# ============================================
# TIPS
# ============================================

with st.expander("💡 Tips for Tag Management"):
    st.markdown("""
    - **Filter first**: Use the sidebar filters to narrow down documents before selecting
    - **Pagination**: Use offset to browse through large document sets
    - **Bulk operations**: Select multiple documents to apply tag changes efficiently
    - **Tag consistency**: Use consistent tag naming (e.g., lowercase, hyphenated)
    - **Common tags**: Review all existing tags in the filter dropdown for consistency
    """)

# ============================================
# FOOTER
# ============================================

render_version_footer()
