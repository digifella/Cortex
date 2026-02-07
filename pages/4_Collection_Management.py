# ## File: pages/4_Collection_Management.py
# Version: v5.6.0
# Date: 2026-01-26
# Purpose: A UI for managing Working Collections only.
#          Knowledge base maintenance functions moved to Maintenance page (page 13).
#          - FEATURE (v5.2.0): Enhanced tag management with three tabs: Tag Selected, Tag All
#            (entire collection), and Rename Tag (find/replace tags across collection).
#          - REFACTOR (v4.0.0): Removed knowledge maintenance functions, now collections only.

import streamlit as st
import sys
import time
from pathlib import Path
import os
import json
import shutil
import chromadb
from chromadb.config import Settings as ChromaSettings
from collections import OrderedDict
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import centralized utilities
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger, resolve_db_root_path
from cortex_engine.utils import convert_to_docker_mount_path
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.session_state import initialize_app_session_state
from cortex_engine.config import COLLECTION_NAME
from cortex_engine.document_type_manager import get_document_type_manager
from cortex_engine.config_manager import ConfigManager

# Set up logging
logger = get_logger(__name__)


def _normalize_tags(raw) -> list:
    """Normalize thematic_tags to a clean list of strings."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    if isinstance(raw, list):
        return [str(p).strip() for p in raw if str(p).strip()]
    return []


def _safe_collection_archive_name(collection_name: str) -> str:
    """Create a filesystem-safe archive name from collection name."""
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(collection_name))
    cleaned = cleaned.strip("_")
    return cleaned or "collection"


def _parse_iso_datetime(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _detect_active_ingestion(db_path: str, active_window_minutes: int = 30) -> dict:
    """Detect likely active ingestion by checking recent ingestion state artifacts."""
    try:
        resolved = resolve_db_root_path(db_path)
        if resolved:
            safe_db_path = convert_to_docker_mount_path(str(resolved))
        else:
            safe_db_path = convert_windows_to_wsl_path(db_path)
        if not safe_db_path:
            return {"active": False, "signals": []}

        base = Path(safe_db_path)
        now = datetime.now().timestamp()
        active_window_seconds = active_window_minutes * 60
        signals = []

        state_files = [
            base / "batch_state.json",
            base / "staging_ingestion.json",
            base / "ingestion_progress.json",
        ]
        for state_file in state_files:
            if state_file.exists() and state_file.is_file():
                age_seconds = max(0, now - state_file.stat().st_mtime)
                if age_seconds <= active_window_seconds:
                    signals.append(f"recent state file: {state_file.name}")

        progress_dir = base / "ingestion_progress"
        if progress_dir.exists() and progress_dir.is_dir():
            for progress_file in progress_dir.glob("*.json"):
                try:
                    with open(progress_file, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    status = str(payload.get("status", "")).lower()
                    last_update = _parse_iso_datetime(payload.get("last_update"))
                    last_update_ts = last_update.timestamp() if last_update else progress_file.stat().st_mtime
                    age_seconds = max(0, now - last_update_ts)
                    if status == "running" and age_seconds <= active_window_seconds:
                        session_id = payload.get("session_id", progress_file.name)
                        signals.append(f"running progress session: {session_id}")
                except Exception:
                    continue

        return {"active": len(signals) > 0, "signals": signals}
    except Exception as e:
        logger.debug(f"Ingestion activity detection failed: {e}")
        return {"active": False, "signals": []}

@st.cache_resource
def init_chroma_client(db_path):
    if not db_path:
        st.error("Database path not configured. Please configure it in Knowledge Ingest.")
        return None

    resolved = resolve_db_root_path(db_path)
    if resolved:
        safe_path = convert_to_docker_mount_path(str(resolved))
    else:
        safe_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(safe_path, "knowledge_hub_db")
    
    
    if not os.path.isdir(chroma_db_path):
        st.error(f"Database path not found: '{chroma_db_path}'. Please configure it in Knowledge Ingest.")
        return None
        
    try:
        db_settings = ChromaSettings(
            anonymized_telemetry=False
        )
        return chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
    except Exception as e:
        # Check if this is a Docker vs development database schema conflict
        if "collections.config_json_str" in str(e):
            st.warning("🐳 **Development database detected in Docker environment**")
            st.info("📝 **This database was created outside Docker and has schema differences.**\n\n💡 **Options:**\n- Use a separate Docker database path (recommended)\n- Or continue with limited collection features")
            logger.warning(f"Development database schema conflict in Docker: {e}")
            return None
        else:
            st.error(f"Failed to connect to ChromaDB: {e}")
        return None

st.set_page_config(layout="wide", page_title="Cortex Collection Management")
st.title("📚 4. Collection Management")
st.caption("Manage your curated Working Collections. For knowledge base maintenance, see the Maintenance page (page 13).")

# Add info box about knowledge base management
st.info("""
🔧 **Looking for knowledge base management?**
- Database deduplication, backup/restore, and deletion functions are now in the **Maintenance page (page 13)**
- This page is now dedicated exclusively to managing your Working Collections
""")

initialize_app_session_state()

config = ConfigManager().get_config()
db_path = config.get("ai_database_path", "")
st.text_input("Knowledge Hub DB Path", value=db_path, disabled=True, help="Set this on the Knowledge Ingest page.")

if not db_path:
    st.warning("Please set a valid AI Database Path on the 'Knowledge Ingest' page first.")
    st.stop()

ingestion_activity = _detect_active_ingestion(db_path)
if ingestion_activity.get("active"):
    signal_preview = ", ".join(ingestion_activity.get("signals", [])[:3])
    st.warning(
        "⚠️ Active ingestion appears to be running. ZIP export/download is generally safe, "
        "but avoid mutating collection actions (rename/delete/merge/clear) until ingestion completes."
        + (f" Signals: {signal_preview}" if signal_preview else "")
    )

chroma_client = init_chroma_client(db_path)
if not chroma_client: st.stop()

try:
    vector_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    # Quick diagnostics to help verify DB connectivity
    with st.container(border=True):
        st.markdown("### 🔎 Knowledge Base Diagnostics")
        try:
            doc_count = vector_collection.count()
        except Exception as _diag_e:
            doc_count = 0
        st.write(f"📊 Vector store documents: {doc_count}")
        
        # Show available Chroma collections and their counts (helps detect name mismatches)
        try:
            available = chroma_client.list_collections()
            if available:
                cols_info = []
                for c in available:
                    try:
                        c_obj = chroma_client.get_collection(c.name)
                        cols_info.append((c.name, c_obj.count()))
                    except Exception:
                        cols_info.append((c.name, "?"))
                pretty = ", ".join([f"{n} (count={cnt})" for n, cnt in cols_info])
                st.caption(f"Available Chroma collections: {pretty}")
        except Exception:
            pass
        
        # Offer a rescue sync: populate 'default' collection from vector store IDs
        st.caption("If collections appear empty but the vector store has documents, you can sync the 'default' collection from the vector store IDs.")
        if st.button("🔄 Sync 'default' collection from vector store", help="Populate 'default' with all vector store document IDs"):
            try:
                # Chroma 'get' does not accept 'ids' in include; ids are always returned.
                # Request metadatas (cheap) so we can pull ids reliably.
                all_ids = vector_collection.get(include=["metadatas"]).get("ids", [])
                if isinstance(all_ids, list) and all_ids and isinstance(all_ids[0], list):
                    # Flatten if nested
                    flat_ids = [i for sub in all_ids for i in sub]
                else:
                    flat_ids = all_ids if isinstance(all_ids, list) else []
                if not flat_ids:
                    st.warning("No IDs returned from vector store.")
                else:
                    mgr = WorkingCollectionManager()
                    mgr.add_docs_by_id_to_collection("default", flat_ids)
                    st.success(f"✅ Added {len(flat_ids)} IDs to 'default' collection.")
                    st.rerun()
            except Exception as se:
                st.error(f"Sync failed: {se}")
except Exception as e:
    error_msg = str(e)
    
    # Check for specific schema errors
    if "collections.config_json_str" in error_msg or "no such column" in error_msg.lower():
        st.error("🔧 **ChromaDB Schema Error Detected**")
        st.markdown(f"""
        **Error:** `{error_msg}`
        
        **This is a database schema conflict.** This typically happens when:
        - Different ChromaDB versions created incompatible database structures
        - Docker vs non-Docker environments have different schemas
        - Database files are corrupted or partially created
        
        ### 🚀 **Recommended Solution:**
        1. **Go to Maintenance page (page 13)**
        2. **Use the new "Clean Start" function** 
        3. **This will completely reset your database** and fix all schema conflicts
        4. **Then re-ingest your documents** with a fresh, compatible database
        
        The Clean Start function is specifically designed to resolve these ChromaDB schema issues.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔧 Go to Maintenance Page", use_container_width=True, type="primary"):
                st.switch_page("pages/6_Maintenance.py")
        with col2:
            if st.button("📖 Learn More About This Error", use_container_width=True):
                st.info("""
                **Technical Details:**
                ChromaDB schema errors occur when the database structure doesn't match what the application expects.
                The 'collections.config_json_str' column was added in newer ChromaDB versions, but older databases
                don't have this column, causing compatibility issues.
                
                **Why Clean Start works:**
                - Deletes the incompatible database structure completely
                - Forces creation of new database with current ChromaDB version schema  
                - Eliminates all version conflicts and corruption issues
                """)
    else:
        st.error(f"Could not connect to collection '{COLLECTION_NAME}'. Please run an ingestion process first. Error: {e}")
    
    st.stop()

if 'collection_sort_key' not in st.session_state: st.session_state.collection_sort_key = 'modified_at'
if 'collection_sort_order_asc' not in st.session_state: st.session_state.collection_sort_order_asc = False

def set_sort_order(key):
    if st.session_state.collection_sort_key == key: st.session_state.collection_sort_order_asc = not st.session_state.collection_sort_order_asc
    else: st.session_state.collection_sort_key = key; st.session_state.collection_sort_order_asc = True if key == 'name' else False

def render_document_metadata_editor(doc_id, meta, vector_collection, collection_name):
    """Render metadata viewer and editor for a document"""
    import json

    st.markdown("### 📊 Document Metadata")

    # Basic file info (read-only)
    with st.container(border=True):
        st.markdown("**📁 File Information**")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.text_input("Document ID", value=doc_id, disabled=True, key=f"doc_id_{doc_id}")
            st.text_input("File Name", value=meta.get('file_name', 'N/A'), disabled=True, key=f"filename_{doc_id}")
        with info_col2:
            st.text_input("Last Modified", value=meta.get('last_modified_date', 'N/A'), disabled=True, key=f"modified_{doc_id}")
            st.text_input("File Path", value=meta.get('doc_posix_path', 'N/A'), disabled=True, key=f"path_{doc_id}")

    st.markdown("---")

    # Editable rich metadata
    with st.container(border=True):
        st.markdown("**✏️ Editable Metadata** *(Changes will be saved to the database)*")

        # Document Type
        doc_type_options = [
            "Project Plan", "Technical Documentation", "Proposal/Quote", "Case Study / Trophy",
            "Final Report", "Draft Report", "Presentation", "Contract/SOW",
            "Meeting Minutes", "Financial Report", "Research Paper", "Email Correspondence",
            "Image/Diagram", "Other"
        ]
        current_doc_type = meta.get('document_type', 'Other')
        new_doc_type = st.selectbox(
            "Document Type",
            options=doc_type_options,
            index=doc_type_options.index(current_doc_type) if current_doc_type in doc_type_options else doc_type_options.index('Other'),
            key=f"doc_type_edit_{doc_id}"
        )

        # Proposal Outcome
        outcome_options = ["Won", "Lost", "Pending", "N/A"]
        current_outcome = meta.get('proposal_outcome', 'N/A')
        new_outcome = st.selectbox(
            "Proposal Outcome",
            options=outcome_options,
            index=outcome_options.index(current_outcome) if current_outcome in outcome_options else outcome_options.index('N/A'),
            key=f"outcome_edit_{doc_id}",
            help="Only relevant for proposals/quotes"
        )

        # Summary
        current_summary = meta.get('summary', '')
        new_summary = st.text_area(
            "Summary",
            value=current_summary,
            height=100,
            key=f"summary_edit_{doc_id}",
            help="1-3 sentence summary of the document's content and purpose"
        )

        # Thematic Tags
        current_tags = meta.get('thematic_tags', [])
        if isinstance(current_tags, list):
            tags_str = ", ".join(current_tags)
        else:
            tags_str = str(current_tags)

        new_tags_str = st.text_area(
            "Thematic Tags",
            value=tags_str,
            height=60,
            key=f"tags_edit_{doc_id}",
            help="Comma-separated list of 3-5 key themes, topics, or technologies"
        )

        # Save button
        save_col1, save_col2 = st.columns([1, 3])
        with save_col1:
            if st.button("💾 Save Changes", key=f"save_meta_{doc_id}", type="primary", use_container_width=True):
                try:
                    # Parse new tags
                    new_tags = [tag.strip() for tag in new_tags_str.split(',') if tag.strip()]

                    # Update metadata in ChromaDB — preserve all existing fields
                    updated_metadata = dict(meta)
                    updated_metadata['document_type'] = new_doc_type
                    updated_metadata['proposal_outcome'] = new_outcome
                    updated_metadata['summary'] = new_summary
                    updated_metadata['thematic_tags'] = json.dumps(new_tags)

                    # Update in vector collection
                    vector_collection.update(
                        ids=[doc_id],
                        metadatas=[updated_metadata]
                    )

                    st.success("✅ Metadata updated successfully!")
                    logger.info(f"Updated metadata for document {doc_id}")

                    # Close the expander and refresh
                    time.sleep(0.5)
                    st.session_state[f"show_metadata_{doc_id}_{collection_name}"] = False
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Failed to update metadata: {e}")
                    logger.error(f"Metadata update error for {doc_id}: {e}", exc_info=True)

        with save_col2:
            st.caption("Changes are saved immediately to the database when you click Save.")

    st.markdown("---")

    # Additional metadata (read-only)
    with st.container(border=True):
        st.markdown("**🔍 Additional Metadata (Read-Only)**")
        # Extracted entities
        entities = meta.get('extracted_entities', [])
        if entities:
            st.markdown("**Extracted Entities:**")
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                except (json.JSONDecodeError, ValueError):
                    pass
            if isinstance(entities, list) and len(entities) > 0:
                entity_df_data = []
                for ent in entities[:20]:  # Show first 20
                    if isinstance(ent, dict):
                        entity_df_data.append({
                            'Name': ent.get('name', 'N/A'),
                            'Type': ent.get('type', 'N/A'),
                            'Description': ent.get('description', 'N/A')[:50] + '...' if len(ent.get('description', '')) > 50 else ent.get('description', 'N/A')
                        })
                if entity_df_data:
                    st.dataframe(entity_df_data, use_container_width=True)
                if len(entities) > 20:
                    st.caption(f"... and {len(entities) - 20} more entities")
            else:
                st.caption("No entities extracted")
        else:
            st.caption("No entities extracted")

        # Extracted relationships
        relationships = meta.get('extracted_relationships', [])
        if relationships:
            st.markdown("**Extracted Relationships:**")
            if isinstance(relationships, str):
                try:
                    relationships = json.loads(relationships)
                except (json.JSONDecodeError, ValueError):
                    pass
            if isinstance(relationships, list) and len(relationships) > 0:
                rel_df_data = []
                for rel in relationships[:15]:  # Show first 15
                    if isinstance(rel, dict):
                        rel_df_data.append({
                            'Source': rel.get('source', 'N/A'),
                            'Relation': rel.get('relation', 'N/A'),
                            'Target': rel.get('target', 'N/A')
                        })
                if rel_df_data:
                    st.dataframe(rel_df_data, use_container_width=True)
                if len(relationships) > 15:
                    st.caption(f"... and {len(relationships) - 15} more relationships")
            else:
                st.caption("No relationships extracted")
        else:
            st.caption("No relationships extracted")

def display_enhanced_document_list(unique_docs, collection_name, collection_mgr):
    """Display documents with enhanced pagination, sorting, and filtering options"""
    
    # Initialize session state for this collection's document view
    view_key = f"doc_view_{collection_name}"
    if f'{view_key}_page' not in st.session_state:
        st.session_state[f'{view_key}_page'] = 0
    if f'{view_key}_per_page' not in st.session_state:
        st.session_state[f'{view_key}_per_page'] = 10
    if f'{view_key}_sort_by' not in st.session_state:
        st.session_state[f'{view_key}_sort_by'] = 'name'
    if f'{view_key}_sort_asc' not in st.session_state:
        st.session_state[f'{view_key}_sort_asc'] = True
    if f'{view_key}_filter' not in st.session_state:
        st.session_state[f'{view_key}_filter'] = ''
    if f'{view_key}_type_filter' not in st.session_state:
        st.session_state[f'{view_key}_type_filter'] = 'All'
    
    # Document controls section
    with st.container():
        st.markdown("**📋 Document View Controls**")
        
        # Filter and search controls
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.5, 1, 1, 1.5])
        
        with filter_col1:
            search_filter = st.text_input(
                "🔍 Search documents",
                value=st.session_state[f'{view_key}_filter'],
                key=f'{view_key}_filter_input',
                placeholder="Search by filename or path...",
                help="Filter documents by filename or file path"
            )
            if search_filter != st.session_state[f'{view_key}_filter']:
                st.session_state[f'{view_key}_filter'] = search_filter
                st.session_state[f'{view_key}_page'] = 0  # Reset to first page
        
        with filter_col2:
            # Extract document types from metadata
            doc_types = set()
            for meta in unique_docs.values():
                doc_type = meta.get('document_type', 'Unknown')
                if doc_type and doc_type != 'Unknown':
                    doc_types.add(doc_type)
            
            type_options = ['All'] + sorted(list(doc_types))
            type_filter = st.selectbox(
                "📄 Document Type",
                options=type_options,
                index=type_options.index(st.session_state[f'{view_key}_type_filter']) if st.session_state[f'{view_key}_type_filter'] in type_options else 0,
                key=f'{view_key}_type_filter_select',
                help="Filter by document type"
            )
            if type_filter != st.session_state[f'{view_key}_type_filter']:
                st.session_state[f'{view_key}_type_filter'] = type_filter
                st.session_state[f'{view_key}_page'] = 0  # Reset to first page
        
        with filter_col3:
            per_page = st.selectbox(
                "📄 Per Page",
                options=[5, 10, 20, 50, 100],
                index=[5, 10, 20, 50, 100].index(st.session_state[f'{view_key}_per_page']),
                key=f'{view_key}_per_page_select',
                help="Number of documents per page"
            )
            if per_page != st.session_state[f'{view_key}_per_page']:
                st.session_state[f'{view_key}_per_page'] = per_page
                st.session_state[f'{view_key}_page'] = 0  # Reset to first page
                
        with filter_col4:
            # Bulk action controls
            st.markdown("**📦 Bulk Actions**")
            all_available_collections = [name for name in collection_mgr.get_collection_names() if name != collection_name]
            if all_available_collections:
                bulk_target_collection = st.selectbox(
                    "Add all visible to:",
                    options=[""] + all_available_collections,
                    key=f"bulk_add_to_collection_{collection_name}",
                    format_func=lambda x: "Select collection..." if x == "" else x,
                    help="Add all visible documents to another collection"
                )
        
        # Sort controls
        sort_col1, sort_col2 = st.columns([1, 1])
        
        with sort_col1:
            sort_by = st.selectbox(
                "📊 Sort By",
                options=['name', 'path', 'type', 'date_modified', 'size'],
                index=['name', 'path', 'type', 'date_modified', 'size'].index(st.session_state[f'{view_key}_sort_by']),
                key=f'{view_key}_sort_by_select',
                help="Sort documents by different criteria"
            )
            if sort_by != st.session_state[f'{view_key}_sort_by']:
                st.session_state[f'{view_key}_sort_by'] = sort_by
        
        with sort_col2:
            sort_direction = st.radio(
                "Sort Direction",
                options=['Ascending', 'Descending'],
                index=0 if st.session_state[f'{view_key}_sort_asc'] else 1,
                key=f'{view_key}_sort_direction',
                horizontal=True,
                help="Sort order"
            )
            new_sort_asc = (sort_direction == 'Ascending')
            if new_sort_asc != st.session_state[f'{view_key}_sort_asc']:
                st.session_state[f'{view_key}_sort_asc'] = new_sort_asc
    
    st.divider()
    
    # Apply filters
    filtered_docs = {}
    search_term = st.session_state[f'{view_key}_filter'].lower()
    type_filter = st.session_state[f'{view_key}_type_filter']
    
    for doc_id, meta in unique_docs.items():
        file_name = meta.get('file_name', '').lower()
        file_path = meta.get('doc_posix_path', '').lower()
        doc_type = meta.get('document_type', 'Unknown')
        
        # Apply search filter
        if search_term and search_term not in file_name and search_term not in file_path:
            continue
        
        # Apply type filter
        if type_filter != 'All' and doc_type != type_filter:
            continue
        
        filtered_docs[doc_id] = meta
    
    # Sort documents
    sort_key_map = {
        'name': lambda x: x[1].get('file_name', '').lower(),
        'path': lambda x: x[1].get('doc_posix_path', '').lower(),
        'type': lambda x: x[1].get('document_type', 'Unknown').lower(),
        'date_modified': lambda x: x[1].get('last_modified_date', '1970-01-01'),
        'size': lambda x: len(x[1].get('content', ''))  # Approximate size by content length
    }
    
    sort_func = sort_key_map.get(st.session_state[f'{view_key}_sort_by'], sort_key_map['name'])
    sorted_docs = sorted(filtered_docs.items(), key=sort_func, reverse=not st.session_state[f'{view_key}_sort_asc'])
    
    # Pagination
    total_docs = len(sorted_docs)
    per_page = st.session_state[f'{view_key}_per_page']
    total_pages = max(1, (total_docs + per_page - 1) // per_page)
    current_page = st.session_state[f'{view_key}_page']
    
    # Ensure current page is valid
    if current_page >= total_pages:
        current_page = max(0, total_pages - 1)
        st.session_state[f'{view_key}_page'] = current_page
    
    start_idx = current_page * per_page
    end_idx = min(start_idx + per_page, total_docs)
    page_docs = sorted_docs[start_idx:end_idx]
    
    # Display summary and pagination controls
    summary_col1, summary_col2 = st.columns([2, 1])
    
    with summary_col1:
        if total_docs > 0:
            st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {total_docs} documents** (Page {current_page + 1} of {total_pages})")
        else:
            st.markdown("**No documents match your filters**")
    
    with summary_col2:
        if total_pages > 1:
            # Pagination buttons
            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 1])
            
            with nav_col1:
                if st.button("⏮️", key=f'{view_key}_first', disabled=current_page == 0, help="First page"):
                    st.session_state[f'{view_key}_page'] = 0
                    st.rerun()
            
            with nav_col2:
                if st.button("◀️", key=f'{view_key}_prev', disabled=current_page == 0, help="Previous page"):
                    st.session_state[f'{view_key}_page'] = max(0, current_page - 1)
                    st.rerun()
            
            with nav_col3:
                if st.button("▶️", key=f'{view_key}_next', disabled=current_page >= total_pages - 1, help="Next page"):
                    st.session_state[f'{view_key}_page'] = min(total_pages - 1, current_page + 1)
                    st.rerun()
            
            with nav_col4:
                if st.button("⏭️", key=f'{view_key}_last', disabled=current_page >= total_pages - 1, help="Last page"):
                    st.session_state[f'{view_key}_page'] = total_pages - 1
                    st.rerun()
    
    st.divider()
    
    # Display documents
    if page_docs:
        # Column headers
        header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7 = st.columns([2.5, 2, 1.5, 1.5, 0.8, 0.5, 0.7])

        with header_col1:
            st.markdown("**📄 Document Name**")
        with header_col2:
            st.markdown("**📁 Path**")
        with header_col3:
            st.markdown("**📋 Type**")
        with header_col4:
            st.markdown("**📅 Modified**")
        with header_col5:
            st.markdown("**📥 Add To**")
        with header_col6:
            st.markdown("**🗑️**")
        with header_col7:
            st.markdown("**📝 Meta**")

        st.markdown("---")

        # Display each document
        for doc_id, meta in page_docs:
            doc_col1, doc_col2, doc_col3, doc_col4, doc_col5, doc_col6, doc_col7 = st.columns([2.5, 2, 1.5, 1.5, 0.8, 0.5, 0.7])

            with doc_col1:
                file_name = meta.get('file_name', 'N/A')
                st.markdown(f"**{file_name}**")
            
            with doc_col2:
                file_path = meta.get('doc_posix_path', 'N/A')
                # Truncate long paths for display
                display_path = file_path if len(file_path) <= 35 else f"...{file_path[-32:]}"
                st.markdown(f"`{display_path}`")
            
            with doc_col3:
                doc_type = meta.get('document_type', 'Unknown')
                type_emoji = {
                    'Project Plan': '🗂️',
                    'Technical Documentation': '📖',
                    'Proposal/Quote': '💼',
                    'Case Study / Trophy': '🏆',
                    'Final Report': '📊',
                    'Draft Report': '📝',
                    'Presentation': '📺',
                    'Contract/SOW': '📋',
                    'Meeting Minutes': '🗣️',
                    'Financial Report': '💰',
                    'Research Paper': '🔬',
                    'Email Correspondence': '📧',
                    'Image/Diagram': '🖼️',
                    'Other': '📄'
                }.get(doc_type, '📄')
                st.markdown(f"{type_emoji} {doc_type}")
            
            with doc_col4:
                try:
                    modified_date = meta.get('last_modified_date', 'N/A')
                    if modified_date != 'N/A':
                        # Try to parse and format the date
                        from datetime import datetime
                        dt = datetime.fromisoformat(modified_date.replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%Y-%m-%d')
                        st.markdown(formatted_date)
                    else:
                        st.markdown('N/A')
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Error formatting date: {e}")
                    st.markdown('N/A')
                    
            with doc_col5:
                # Add to collection functionality
                available_collections = [name for name in collection_mgr.get_collection_names() if name != collection_name]
                if available_collections:
                    selected_collection = st.selectbox(
                        "Add to:",
                        options=[""] + available_collections,
                        key=f"add_to_collection_{doc_id}_{collection_name}",
                        format_func=lambda x: "Select collection..." if x == "" else x,
                        label_visibility="collapsed"
                    )
                    if selected_collection and st.button("📥", key=f"add_{doc_id}_to_{selected_collection}", help=f"Add '{file_name}' to '{selected_collection}'"):
                        collection_mgr.add_docs_by_id_to_collection(selected_collection, [doc_id])
                        st.toast(f"Added '{file_name}' to '{selected_collection}'.")
                        st.rerun()
            
            with doc_col6:
                if st.button("❌", key=f"remove_{doc_id}_from_{collection_name}_enhanced", help=f"Remove '{file_name}' from collection"):
                    collection_mgr.remove_from_collection(collection_name, [doc_id])
                    st.toast(f"Removed '{file_name}' from '{collection_name}'.")
                    st.rerun()

            with doc_col7:
                # Toggle metadata viewer for this document
                metadata_key = f"show_metadata_{doc_id}_{collection_name}"
                if metadata_key not in st.session_state:
                    st.session_state[metadata_key] = False

                if st.button("📝", key=f"view_meta_{doc_id}_{collection_name}", help=f"View/Edit metadata for '{file_name}'"):
                    st.session_state[metadata_key] = not st.session_state[metadata_key]
                    st.rerun()

            # Show metadata viewer if toggled
            if st.session_state.get(metadata_key, False):
                with st.expander(f"📋 Metadata for: {file_name}", expanded=True):
                    render_document_metadata_editor(doc_id, meta, vector_collection, collection_name)
        
        # Bulk action execution
        if page_docs:
            st.divider()
            bulk_col1, bulk_col2, bulk_col3 = st.columns([2, 1, 2])
            with bulk_col2:
                all_available_collections = [name for name in collection_mgr.get_collection_names() if name != collection_name]
                if all_available_collections:
                    bulk_target = st.session_state.get(f"bulk_add_to_collection_{collection_name}", "")
                    if bulk_target and st.button(f"📥 Add Visible ({len(page_docs)}) to '{bulk_target}'", key=f"execute_bulk_add_{collection_name}", use_container_width=True):
                        doc_ids_to_add = [doc_id for doc_id, meta in page_docs]
                        collection_mgr.add_docs_by_id_to_collection(bulk_target, doc_ids_to_add)
                        st.toast(f"Added {len(doc_ids_to_add)} documents to '{bulk_target}'.")
                        st.rerun()
        
        # Show additional pagination at bottom if needed
        if total_pages > 1:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Page selector
                new_page = st.selectbox(
                    "Jump to page:",
                    options=list(range(1, total_pages + 1)),
                    index=current_page,
                    key=f'{view_key}_page_select',
                    help="Select a page to jump to"
                )
                if new_page - 1 != current_page:
                    st.session_state[f'{view_key}_page'] = new_page - 1
                    st.rerun()
    else:
        if st.session_state[f'{view_key}_filter'] or st.session_state[f'{view_key}_type_filter'] != 'All':
            st.info("No documents match your current filters. Try adjusting the search terms or document type filter.")
        else:
            st.info("This collection is empty.")

# Initialize collection manager with Docker environment error handling
try:
    collection_mgr = WorkingCollectionManager()
    collections_list = list(collection_mgr.collections.values())
    collections_available = True
except Exception as e:
    collections_available = False
    collections_list = []
    
    # Check if this is a Docker environment compatibility issue
    if "collections.config_json_str" in str(e) or "schema" in str(e).lower():
        st.error("🐳 **Collection Management Unavailable in Current Environment**")
        st.warning("**Database Schema Conflict Detected**")
        st.info("""
        📝 **Issue:** This database was created in a different environment (Docker vs non-Docker) and has incompatible schema.
        
        💡 **Solutions:**
        1. **Use separate database path for Docker** (Recommended)
           - Go to Knowledge Search and set path to `/app/data/ai_databases`
           - Run Knowledge Ingest to populate the Docker database
        
        2. **Recreate collections in current environment**
           - Your documents are safe in the knowledge base
           - Collections will need to be recreated from search results
        
        3. **Switch back to original environment** 
           - Use Cortex outside of Docker if database was created there
        """)
        
        # Show some basic info if possible
        st.markdown("### 🔍 Troubleshooting Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔧 Go to Knowledge Search", use_container_width=True):
                st.switch_page("pages/3_Knowledge_Search.py")
        
        with col2:
            if st.button("📊 Check Database Status", use_container_width=True):
                st.switch_page("pages/6_Maintenance.py")
        
        st.stop()  # Stop execution here
    else:
        st.error(f"❌ Failed to initialize Collection Management: {e}")
        st.info("Please check the Maintenance page for database diagnostics.")
        st.stop()

is_reverse = not st.session_state.collection_sort_order_asc
sort_key = st.session_state.collection_sort_key
if sort_key == 'name': collections_list.sort(key=lambda x: x.get('name', '').lower(), reverse=is_reverse)
else: collections_list.sort(key=lambda x: x.get('modified_at', '1970-01-01T00:00:00Z'), reverse=is_reverse)

st.header("Your Collections")

# Collection summary stats
total_collections = len(collections_list)
total_documents = sum(len(collection.get('doc_ids', [])) for collection in collections_list)
non_empty_collections = len([c for c in collections_list if len(c.get('doc_ids', [])) > 0])

# Display summary
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
with summary_col1:
    st.metric("📚 Total Collections", total_collections)
with summary_col2:
    st.metric("📄 Total Documents", total_documents) 
with summary_col3:
    st.metric("📋 Non-Empty Collections", non_empty_collections)
with summary_col4:
    avg_docs = total_documents / max(non_empty_collections, 1)
    st.metric("📊 Avg Docs/Collection", f"{avg_docs:.1f}")

# Collection Management Actions
if total_collections > 1 or (total_collections == 1 and total_documents > 0):
    with st.expander("🧹 Collection Management Tools", expanded=False):
        st.markdown("**Bulk Collection Operations**")
        
        # Get empty collections count
        empty_collections = [c for c in collections_list if len(c.get('doc_ids', [])) == 0]
        empty_count = len(empty_collections)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Clear Empty Collections**")
            st.caption(f"Remove {empty_count} collections with no documents")
            if empty_count > 0:
                if st.button("🗑️ Remove Empty Collections", key="clear_empty"):
                    try:
                        cleared = collection_mgr.clear_empty_collections()
                        if cleared:
                            st.success(f"✅ Removed {len(cleared)} empty collections: {', '.join(cleared.keys())}")
                            st.rerun()
                        else:
                            st.info("No empty collections found to remove.")
                    except Exception as e:
                        st.error(f"Error removing empty collections: {e}")
            else:
                st.info("No empty collections to remove.")
        
        with col2:
            st.markdown("**Clear All Documents**")
            st.caption(f"Remove all documents from all {total_collections} collections")
            if total_documents > 0:
                if st.checkbox("⚠️ I understand this will remove all document references", key="confirm_clear_docs"):
                    if st.button("📝 Clear All Documents", key="clear_all_docs", type="primary"):
                        try:
                            # Clear documents from all collections but keep collections
                            for collection in collections_list:
                                collection_name = collection['name']
                                doc_ids = collection.get('doc_ids', [])
                                if doc_ids:
                                    collection_mgr.collections[collection_name]['doc_ids'] = []
                                    collection_mgr.collections[collection_name]['modified_at'] = datetime.now().isoformat()
                            
                            collection_mgr._save()
                            st.success(f"✅ Cleared documents from all collections ({total_documents} document references removed)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing documents: {e}")
            else:
                st.info("No documents to clear.")
        
        with col3:
            st.markdown("**Reset All Collections**")
            st.caption("Remove all collections except 'default'")
            if total_collections > 1:
                if st.checkbox("⚠️ I understand this will delete all custom collections", key="confirm_clear_all"):
                    if st.button("🚨 Reset Collections", key="clear_all_collections", type="primary"):
                        try:
                            cleared = collection_mgr.clear_all_collections()
                            if cleared:
                                cleared_names = [name for name in cleared.keys() if name != "default"]
                                default_cleared = cleared.get("default", 0)
                                message = f"✅ Reset complete!"
                                if cleared_names:
                                    message += f" Removed {len(cleared_names)} collections: {', '.join(cleared_names)}."
                                if default_cleared > 0:
                                    message += f" Cleared {default_cleared} documents from default collection."
                                st.success(message)
                                st.rerun()
                            else:
                                st.info("No collections to clear.")
                        except Exception as e:
                            st.error(f"Error clearing collections: {e}")
            else:
                st.info("Only default collection exists.")

st.divider()

# Collection filtering
if 'collection_filter' not in st.session_state:
    st.session_state.collection_filter = ''

filter_col1, filter_col2 = st.columns([2, 1])

with filter_col1:
    collection_filter = st.text_input(
        "🔍 Search collections",
        value=st.session_state.collection_filter,
        key="collection_filter_input",
        placeholder="Search by collection name...",
        help="Filter collections by name"
    )
    if collection_filter != st.session_state.collection_filter:
        st.session_state.collection_filter = collection_filter
        st.session_state.collection_page = 0  # Reset to first page

with filter_col2:
    show_empty = st.checkbox(
        "Show empty collections",
        value=True,
        key="show_empty_collections",
        help="Include collections with no documents"
    )

# Apply collection filtering
filtered_collections = []
search_term = st.session_state.collection_filter.lower()

for collection in collections_list:
    collection_name = collection.get('name', '').lower()
    doc_count = len(collection.get('doc_ids', []))
    
    # Apply search filter
    if search_term and search_term not in collection_name:
        continue
    
    # Apply empty collection filter
    if not show_empty and doc_count == 0:
        continue
    
    filtered_collections.append(collection)

# Update collections_list to use filtered results
collections_list = filtered_collections

st.divider()

# Collection header with sorting
c1, c2, c3 = st.columns([4, 1, 2])
with c1:
    name_sort_icon = "🔼" if st.session_state.collection_sort_order_asc else "🔽"
    if st.button(f"Collection Name {name_sort_icon if sort_key == 'name' else ''}", use_container_width=True): set_sort_order('name'); st.rerun()
with c2:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Documents</p>", unsafe_allow_html=True)
with c3:
    mod_sort_icon = "🔼" if st.session_state.collection_sort_order_asc else "🔽"
    if st.button(f"Last Modified {mod_sort_icon if sort_key == 'modified_at' else ''}", use_container_width=True): set_sort_order('modified_at'); st.rerun()
st.markdown("---")

if not collections_list:
    st.info("No collections found. Create one on the 'Knowledge Search' page."); st.stop()

# Collection pagination if there are many collections
COLLECTIONS_PER_PAGE = 10

if 'collection_page' not in st.session_state:
    st.session_state.collection_page = 0

# Calculate pagination for collections
total_collection_pages = max(1, (len(collections_list) + COLLECTIONS_PER_PAGE - 1) // COLLECTIONS_PER_PAGE)
current_collection_page = st.session_state.collection_page

# Ensure current page is valid
if current_collection_page >= total_collection_pages:
    current_collection_page = max(0, total_collection_pages - 1)
    st.session_state.collection_page = current_collection_page

# Get collections for current page
start_col_idx = current_collection_page * COLLECTIONS_PER_PAGE
end_col_idx = min(start_col_idx + COLLECTIONS_PER_PAGE, len(collections_list))
page_collections = collections_list[start_col_idx:end_col_idx]

# Show pagination controls if needed
if total_collection_pages > 1:
    st.markdown(f"**Showing collections {start_col_idx + 1}-{end_col_idx} of {len(collections_list)}** (Page {current_collection_page + 1} of {total_collection_pages})")
    
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 2])
    
    with nav_col1:
        if st.button("⏮️", key="coll_first", disabled=current_collection_page == 0, help="First page"):
            st.session_state.collection_page = 0
            st.rerun()
    
    with nav_col2:
        if st.button("◀️", key="coll_prev", disabled=current_collection_page == 0, help="Previous page"):
            st.session_state.collection_page = max(0, current_collection_page - 1)
            st.rerun()
    
    with nav_col3:
        if st.button("▶️", key="coll_next", disabled=current_collection_page >= total_collection_pages - 1, help="Next page"):
            st.session_state.collection_page = min(total_collection_pages - 1, current_collection_page + 1)
            st.rerun()
    
    with nav_col4:
        if st.button("⏭️", key="coll_last", disabled=current_collection_page >= total_collection_pages - 1, help="Last page"):
            st.session_state.collection_page = total_collection_pages - 1
            st.rerun()
    
    with nav_col5:
        # Page selector
        new_coll_page = st.selectbox(
            "Jump to page:",
            options=list(range(1, total_collection_pages + 1)),
            index=current_collection_page,
            key="coll_page_select",
            help="Select a page to jump to"
        )
        if new_coll_page - 1 != current_collection_page:
            st.session_state.collection_page = new_coll_page - 1
            st.rerun()
    
    st.divider()

for collection in page_collections:
    name = collection['name']
    doc_ids = collection.get('doc_ids', [])
    modified_iso = collection.get('modified_at', 'N/A')
    try: modified_dt = datetime.fromisoformat(modified_iso.replace('Z', '+00:00')); modified_str = modified_dt.strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError): modified_str = "N/A"

    with st.container(border=True):
        # --- Main Collection Row ---
        col1, col2, col3, col4, col5, col6 = st.columns([3.5, 1, 2, 1, 1, 1])
        with col1: st.markdown(f"**{name}**")
        with col2: st.markdown(f"<p style='text-align: center;'>{len(doc_ids)}</p>", unsafe_allow_html=True)
        with col3: st.write(modified_str)
        with col4:
            if st.button("📄 View", key=f"view_{name}", use_container_width=True):
                st.session_state[f"view_visible_{name}"] = not st.session_state.get(f"view_visible_{name}", False)
        with col5:
            if st.button("💬 Dialog", key=f"dialog_{name}", use_container_width=True, disabled=len(doc_ids) == 0):
                st.session_state.dialog_collection_preselect = name
                st.switch_page("pages/12_Document_Dialog.py")
        with col6:
            if st.button("⚙️ Manage", key=f"manage_{name}", use_container_width=True):
                st.session_state[f"manage_visible_{name}"] = not st.session_state.get(f"manage_visible_{name}", False)

        # --- Expander for Viewing Documents ---
        if st.session_state.get(f"view_visible_{name}", False):
            with st.container():
                if not doc_ids:
                    st.info("This collection is empty.")
                else:
                    with st.spinner("Fetching document details..."):
                        results = vector_collection.get(where={"doc_id": {"$in": doc_ids}}, include=["metadatas"])
                        unique_docs = {meta['doc_id']: meta for meta in results.get('metadatas', []) if 'doc_id' in meta}
                        missing_doc_refs = [doc_id for doc_id in doc_ids if doc_id not in unique_docs]

                        if missing_doc_refs:
                            st.warning(
                                f"Detected {len(missing_doc_refs)} stale reference(s) in this collection "
                                "that no longer exist in the Knowledge Base."
                            )
                            c_fix1, c_fix2 = st.columns([1, 2])
                            with c_fix1:
                                if st.button("🩹 Repair References", key=f"repair_refs_{name}", use_container_width=True):
                                    try:
                                        repair_result = collection_mgr.prune_missing_doc_references(name, vector_collection)
                                        removed = repair_result.get("removed_count", 0)
                                        if removed:
                                            st.success(f"✅ Removed {removed} stale reference(s) from '{name}'.")
                                        else:
                                            st.info("No stale references were removed.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to repair collection references: {e}")
                            with c_fix2:
                                st.caption(
                                    "Use this when documents were deleted/reset and collections still reference old doc IDs."
                                )

                        if not unique_docs:
                            st.warning(
                                "Could not retrieve document metadata for this collection. "
                                "Repair references above, or re-ingest documents if needed."
                            )

                        # Tag management controls
                        if unique_docs:
                            with st.expander("🏷️ Tag Management (thematic_tags)", expanded=False):
                                # Collect all existing tags in this collection for reference
                                all_tags_in_collection = set()
                                for doc_id, meta in unique_docs.items():
                                    tags = _normalize_tags(meta.get("thematic_tags"))
                                    all_tags_in_collection.update(tags)

                                if all_tags_in_collection:
                                    st.caption(f"**Existing tags:** {', '.join(sorted(all_tags_in_collection))}")
                                else:
                                    st.caption("**No tags yet** - add tags to organize this collection")

                                st.markdown("---")

                                # Tab-based organization for tag operations
                                tag_tab1, tag_tab2, tag_tab3 = st.tabs(["📄 Tag Selected", "📚 Tag All", "🔄 Rename Tag"])

                                with tag_tab1:
                                    st.markdown("**Add/remove tags from selected documents**")
                                    options = []
                                    for doc_id, meta in unique_docs.items():
                                        label = f"{meta.get('file_name', meta.get('doc_posix_path', doc_id))} ({doc_id})"
                                        options.append((label, doc_id))
                                    options.sort(key=lambda x: x[0].lower())
                                    labels = [opt[0] for opt in options]
                                    label_to_id = {opt[0]: opt[1] for opt in options}
                                    selected_labels = st.multiselect("Select documents", labels, key=f"tag_select_{name}")
                                    selected_ids = [label_to_id[lbl] for lbl in selected_labels]

                                    c1, c2 = st.columns(2)
                                    with c1:
                                        add_tags = st.text_input("Add tags (comma-separated)", key=f"add_tags_{name}")
                                    with c2:
                                        remove_tags = st.text_input("Remove tags (comma-separated)", key=f"remove_tags_{name}")

                                    if st.button("Apply to Selected", key=f"apply_tags_{name}", disabled=not selected_ids):
                                        add_list = [t.strip() for t in add_tags.split(",") if t.strip()]
                                        remove_list = [t.strip() for t in remove_tags.split(",") if t.strip()]
                                        updated = 0
                                        for doc_id in selected_ids:
                                            meta = dict(unique_docs.get(doc_id, {}))
                                            existing = set(_normalize_tags(meta.get("thematic_tags")))
                                            if add_list:
                                                existing.update(add_list)
                                            if remove_list:
                                                existing.difference_update(remove_list)
                                            meta["thematic_tags"] = ", ".join(sorted(existing))
                                            try:
                                                vector_collection.update(ids=[doc_id], metadatas=[meta])
                                                updated += 1
                                            except Exception as e:
                                                logger.warning(f"Failed to update tags for {doc_id}: {e}")
                                        st.success(f"Updated tags for {updated} document(s).")
                                        if updated:
                                            st.rerun()

                                with tag_tab2:
                                    st.markdown("**Apply tags to ALL documents in this collection**")
                                    st.caption(f"This will affect all {len(unique_docs)} documents")

                                    c1, c2 = st.columns(2)
                                    with c1:
                                        add_all_tags = st.text_input("Add tags to all (comma-separated)", key=f"add_all_tags_{name}",
                                                                     placeholder="e.g., proposal-research, 2026")
                                    with c2:
                                        remove_all_tags = st.text_input("Remove tags from all (comma-separated)", key=f"remove_all_tags_{name}")

                                    if st.button("🏷️ Apply to All Documents", key=f"apply_all_tags_{name}", type="primary"):
                                        add_list = [t.strip() for t in add_all_tags.split(",") if t.strip()]
                                        remove_list = [t.strip() for t in remove_all_tags.split(",") if t.strip()]
                                        if add_list or remove_list:
                                            updated = 0
                                            for doc_id, meta in unique_docs.items():
                                                meta = dict(meta)
                                                existing = set(_normalize_tags(meta.get("thematic_tags")))
                                                if add_list:
                                                    existing.update(add_list)
                                                if remove_list:
                                                    existing.difference_update(remove_list)
                                                meta["thematic_tags"] = ", ".join(sorted(existing))
                                                try:
                                                    vector_collection.update(ids=[doc_id], metadatas=[meta])
                                                    updated += 1
                                                except Exception as e:
                                                    logger.warning(f"Failed to update tags for {doc_id}: {e}")
                                            st.success(f"✅ Updated tags for {updated}/{len(unique_docs)} documents!")
                                            if updated:
                                                st.rerun()
                                        else:
                                            st.warning("Please enter tags to add or remove")

                                with tag_tab3:
                                    st.markdown("**Rename a tag across all documents**")
                                    st.caption("Find and replace tag names in this collection")

                                    c1, c2 = st.columns(2)
                                    with c1:
                                        old_tag = st.text_input("Find tag:", key=f"old_tag_{name}",
                                                               placeholder="e.g., old-tag-name")
                                    with c2:
                                        new_tag = st.text_input("Replace with:", key=f"new_tag_{name}",
                                                               placeholder="e.g., new-tag-name")

                                    # Show which docs have this tag
                                    if old_tag and old_tag.strip():
                                        docs_with_tag = [doc_id for doc_id, meta in unique_docs.items()
                                                        if old_tag.strip().lower() in [t.lower() for t in _normalize_tags(meta.get("thematic_tags"))]]
                                        if docs_with_tag:
                                            st.info(f"Found '{old_tag}' in {len(docs_with_tag)} document(s)")
                                        else:
                                            st.warning(f"Tag '{old_tag}' not found in this collection")

                                    if st.button("🔄 Rename Tag", key=f"rename_tag_{name}"):
                                        if old_tag and old_tag.strip() and new_tag and new_tag.strip():
                                            old_tag_clean = old_tag.strip()
                                            new_tag_clean = new_tag.strip()
                                            updated = 0
                                            for doc_id, meta in unique_docs.items():
                                                meta = dict(meta)
                                                existing = _normalize_tags(meta.get("thematic_tags"))
                                                # Case-insensitive find and replace
                                                existing_lower = {t.lower(): t for t in existing}
                                                if old_tag_clean.lower() in existing_lower:
                                                    # Remove old tag, add new tag
                                                    existing_set = set(existing)
                                                    existing_set.discard(existing_lower[old_tag_clean.lower()])
                                                    existing_set.add(new_tag_clean)
                                                    meta["thematic_tags"] = ", ".join(sorted(existing_set))
                                                    try:
                                                        vector_collection.update(ids=[doc_id], metadatas=[meta])
                                                        updated += 1
                                                    except Exception as e:
                                                        logger.warning(f"Failed to rename tag for {doc_id}: {e}")
                                            if updated:
                                                st.success(f"✅ Renamed '{old_tag_clean}' → '{new_tag_clean}' in {updated} document(s)")
                                                st.rerun()
                                            else:
                                                st.warning(f"Tag '{old_tag_clean}' not found in any documents")
                                        else:
                                            st.warning("Please enter both old and new tag names")

                        # Enhanced document display with pagination and sorting
                        display_enhanced_document_list(unique_docs, name, collection_mgr)

        # --- Expander for Management Actions ---
        if st.session_state.get(f"manage_visible_{name}", False):
            with st.container(border=True):
                # Special workflow for default collection
                if name == "default":
                    st.info("💡 **Renaming the 'default' collection**: Create a new collection with your desired name and merge default into it.")

                    st.markdown("**🔄 Rename Default Collection Workflow**")
                    st.markdown("1. Create a new collection with your desired name below")
                    st.markdown("2. All documents from 'default' will be moved to the new collection")
                    st.markdown("3. The 'default' collection will be emptied (but remain available)")

                    new_collection_name = st.text_input("New Collection Name", key=f"new_name_{name}", placeholder="e.g., Deakin")

                    if st.button("✨ Create & Migrate Documents", key=f"create_migrate_{name}", type="primary"):
                        if new_collection_name and new_collection_name.strip():
                            new_name = new_collection_name.strip()
                            if new_name in collection_mgr.get_collection_names():
                                st.error(f"Collection '{new_name}' already exists. Please choose a different name.")
                            else:
                                try:
                                    # Create new collection
                                    if collection_mgr.create_collection(new_name):
                                        # Get all doc IDs from default
                                        default_docs = collection_mgr.collections.get('default', {}).get('doc_ids', [])

                                        if default_docs:
                                            # Add docs to new collection
                                            collection_mgr.add_docs_by_id_to_collection(new_name, default_docs)

                                            # Clear default collection
                                            collection_mgr.collections['default']['doc_ids'] = []
                                            collection_mgr.collections['default']['modified_at'] = datetime.now().isoformat()
                                            collection_mgr._save()

                                            st.success(f"✅ Created '{new_name}' with {len(default_docs)} documents! Default collection cleared.")
                                        else:
                                            st.success(f"✅ Created '{new_name}' collection (default was empty).")

                                        st.rerun()
                                    else:
                                        st.error("Failed to create new collection.")
                                except Exception as e:
                                    st.error(f"Failed to create and migrate: {e}")
                        else:
                            st.error("Please enter a collection name.")

                    st.divider()

                st.markdown("**Combine Collections**")
                merge_options = [c for c in collection_mgr.get_collection_names() if c != name]
                dest_collection = st.selectbox("Merge this collection into:", merge_options, key=f"merge_dest_{name}", index=None, placeholder="Select a destination...")
                if dest_collection:
                    st.warning(f"This will add all documents from **{name}** to **{dest_collection}** and then permanently delete **{name}**.")
                    if st.checkbox("I understand and want to proceed.", key=f"merge_confirm_{name}"):
                        if st.button("Merge and Delete Collection", key=f"merge_btn_{name}"):
                            with st.spinner(f"Merging '{name}' into '{dest_collection}'..."):
                                if collection_mgr.merge_collections(name, dest_collection):
                                    st.success("Merge successful!"); st.rerun()
                                else:
                                    st.error("Merge failed. Please check logs.")
                st.divider()

                st.markdown("**Export Collection**")
                zip_state_key = f"collection_zip_payload_{name}"
                if st.button("📦 Prepare ZIP for Download", key=f"prepare_zip_btn_{name}", use_container_width=True):
                    with st.spinner("Preparing ZIP archive..."):
                        zip_bytes, copied, failed = collection_mgr.create_collection_zip_bytes(name, vector_collection)
                        if zip_bytes:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            safe_name = _safe_collection_archive_name(name)
                            st.session_state[zip_state_key] = {
                                "zip_bytes": zip_bytes,
                                "filename": f"{safe_name}_documents_{timestamp}.zip",
                                "copied_count": len(copied),
                                "failed_count": len(failed),
                            }
                            st.success(f"✅ ZIP prepared with {len(copied)} file(s).")
                            if failed:
                                st.warning(f"{len(failed)} file(s) could not be added to ZIP (missing/unreadable).")
                        else:
                            st.error("No files were available to include in a ZIP for this collection.")

                zip_payload = st.session_state.get(zip_state_key)
                if zip_payload and zip_payload.get("zip_bytes"):
                    st.download_button(
                        label=f"⬇️ Download ZIP ({zip_payload.get('copied_count', 0)} files)",
                        data=zip_payload["zip_bytes"],
                        file_name=zip_payload["filename"],
                        mime="application/zip",
                        key=f"download_zip_btn_{name}",
                        use_container_width=True,
                    )
                    if zip_payload.get("failed_count", 0) > 0:
                        st.caption(f"⚠️ {zip_payload['failed_count']} file(s) were skipped while preparing this archive.")

                with st.expander("Export files directly to a directory", expanded=False):
                    output_dir = st.text_input("Destination Directory", key=f"export_path_{name}", placeholder="e.g., C:\\Users\\YourUser\\Desktop\\Export")
                    if st.button("Export Files", key=f"export_btn_{name}"):
                        if output_dir:
                            with st.spinner("Exporting files..."):
                                copied, failed = collection_mgr.export_collection_files(name, output_dir, vector_collection)
                                st.success(f"✅ Export complete! Copied {len(copied)} files.")
                                if failed: st.warning(f"Could not copy {len(failed)} files: {', '.join(failed)}")
                        else: st.error("Please provide a destination directory.")
                st.divider()

                st.markdown("**Rename Collection**")
                new_name_input = st.text_input("New name", key=f"rename_input_{name}")
                if st.button("Rename", key=f"rename_btn_{name}"):
                    if new_name_input and new_name_input.strip() not in collection_mgr.collections:
                        collection_mgr.rename_collection(name, new_name_input.strip())
                        st.success(f"Renamed '{name}' to '{new_name_input.strip()}'."); st.rerun()
                    else: st.error("New name cannot be empty or already exist.")
                st.divider()

                st.markdown("**Delete Collection**")
                st.warning(f"This action is permanent and cannot be undone.")
                if st.checkbox(f"I understand, permanently delete '{name}'.", key=f"delete_confirm_{name}"):
                    if st.button("DELETE PERMANENTLY", type="primary", key=f"delete_btn_{name}"):
                        collection_mgr.delete_collection(name)
                        st.success(f"Successfully deleted collection '{name}'."); st.rerun()


# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
