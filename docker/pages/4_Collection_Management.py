# ## File: pages/4_Collection_Management.py
# Version: v1.0.1
# Date: 2025-08-16
# Purpose: A UI for managing Working Collections.
#          - REFACTOR (v1.0.1): Updated to use centralized utilities for path handling,
#            logging, and error handling. Removed code duplication.

import streamlit as st
import sys
from pathlib import Path
import os
import shutil
import chromadb
from chromadb.config import Settings as ChromaSettings
from collections import OrderedDict
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import centralized utilities
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.session_state import initialize_app_session_state
from cortex_engine.config import COLLECTION_NAME
from cortex_engine.document_type_manager import get_document_type_manager
from cortex_engine.config_manager import ConfigManager
from cortex_engine.sync_backup_manager import SyncBackupManager

# Set up logging
logger = get_logger(__name__)

@st.cache_resource
def init_chroma_client(db_path):
    if not db_path:
        st.error("Database path not configured. Please configure it in Knowledge Ingest.")
        return None
        
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
    
    
    if not os.path.isdir(chroma_db_path):
        st.error(f"Database path not found: '{chroma_db_path}'. Please configure it in Knowledge Ingest.")
        return None
        
    try:
        db_settings = ChromaSettings(anonymized_telemetry=False)
        return chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")
        return None

st.set_page_config(layout="wide", page_title="Cortex Knowledge & Collection Management")
st.title("📚 4. Knowledge & Collection Management")
st.caption("Manage your knowledge base and curated Working Collections.")

initialize_app_session_state()

config = ConfigManager().get_config()
db_path = config.get("ai_database_path", "")
st.text_input("Knowledge Hub DB Path", value=db_path, disabled=True, help="Set this on the Knowledge Ingest page.")

if not db_path:
    st.warning("Please set a valid AI Database Path on the 'Knowledge Ingest' page first.")
    st.stop()

chroma_client = init_chroma_client(db_path)
if not chroma_client: st.stop()

try:
    vector_collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    st.error(f"Could not connect to collection '{COLLECTION_NAME}'. Please run an ingestion process first. Error: {e}")
    st.stop()

if 'collection_sort_key' not in st.session_state: st.session_state.collection_sort_key = 'modified_at'
if 'collection_sort_order_asc' not in st.session_state: st.session_state.collection_sort_order_asc = False

def set_sort_order(key):
    if st.session_state.collection_sort_key == key: st.session_state.collection_sort_order_asc = not st.session_state.collection_sort_order_asc
    else: st.session_state.collection_sort_key = key; st.session_state.collection_sort_order_asc = True if key == 'name' else False

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
        header_col1, header_col2, header_col3, header_col4, header_col5, header_col6 = st.columns([2.5, 2, 1.5, 1.5, 0.8, 0.5])
        
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
        
        st.markdown("---")
        
        # Display each document
        for doc_id, meta in page_docs:
            doc_col1, doc_col2, doc_col3, doc_col4, doc_col5, doc_col6 = st.columns([2.5, 2, 1.5, 1.5, 0.8, 0.5])
            
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
                except:
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
                        collection_mgr.add_to_collection(selected_collection, [doc_id])
                        st.toast(f"Added '{file_name}' to '{selected_collection}'.")
                        st.rerun()
            
            with doc_col6:
                if st.button("❌", key=f"remove_{doc_id}_from_{collection_name}_enhanced", help=f"Remove '{file_name}' from collection"):
                    collection_mgr.remove_from_collection(collection_name, [doc_id])
                    st.toast(f"Removed '{file_name}' from '{collection_name}'.")
                    st.rerun()
        
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
                        collection_mgr.add_to_collection(bulk_target, doc_ids_to_add)
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

collection_mgr = WorkingCollectionManager()
collections_list = list(collection_mgr.collections.values())

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
        col1, col2, col3, col4, col5 = st.columns([4, 1, 2, 1, 1])
        with col1: st.markdown(f"**{name}**")
        with col2: st.markdown(f"<p style='text-align: center;'>{len(doc_ids)}</p>", unsafe_allow_html=True)
        with col3: st.write(modified_str)
        with col4:
            if st.button("📄 View", key=f"view_{name}", use_container_width=True):
                st.session_state[f"view_visible_{name}"] = not st.session_state.get(f"view_visible_{name}", False)
        with col5:
            if st.button("⚙️ Manage", key=f"manage_{name}", use_container_width=True, disabled=(name=="default")):
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

                        if not unique_docs:
                            st.warning("Could not retrieve details for documents in this collection. They may have been deleted from the Knowledge Base.")

                        # Enhanced document display with pagination and sorting
                        display_enhanced_document_list(unique_docs, name, collection_mgr)

        # --- Expander for Management Actions ---
        if st.session_state.get(f"manage_visible_{name}", False) and name != "default":
            with st.container(border=True):
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

# --- Knowledge Base Backup Management Section ---
st.header("💾 Knowledge Base Backup & Restore")
st.markdown("Create backups and restore your knowledge base for data protection.")

with st.container(border=True):
    backup_tab1, backup_tab2, backup_tab3 = st.tabs(["📦 Create Backup", "📋 Manage Backups", "🔄 Restore"])
    
    with backup_tab1:
        st.subheader("Create New Backup")
        st.markdown("Create a comprehensive backup of your entire knowledge base including documents, entities, and relationships.")
        
        backup_col1, backup_col2 = st.columns([2, 1])
        
        with backup_col1:
            backup_name = st.text_input(
                "Backup Name (optional)",
                placeholder=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Leave empty to auto-generate a name with timestamp"
            )
            
            # Custom backup location
            backup_location = st.text_input(
                "Backup Location (optional)",
                placeholder=r"C:\Users\YourName\Documents\CortexBackups or leave empty for default",
                help="Choose where to store the backup. Leave empty to use default location in database folder. Supports both Windows (C:\\) and Linux (/mnt/) paths."
            )
            
            backup_description = st.text_area(
                "Backup Description (optional)",
                placeholder="Describe what this backup contains or why it was created...",
                help="Optional description for this backup"
            )
        
        with backup_col2:
            include_images = st.checkbox(
                "Include Images",
                value=True,
                help="Include image files in the backup"
            )
            
            compress_backup = st.checkbox(
                "Compress Backup",
                value=True,
                help="Compress the backup file to save space"
            )
        
        st.divider()
        
        if st.button("🚀 Create Backup", type="primary", use_container_width=True):
            with st.spinner("Creating backup... This may take a few minutes for large knowledge bases."):
                try:
                    wsl_db_path = convert_windows_to_wsl_path(db_path)
                    backup_manager = SyncBackupManager(wsl_db_path)
                    
                    # Create backup synchronously
                    backup_metadata = backup_manager.create_backup(
                        backup_name=backup_name.strip() if backup_name.strip() else None,
                        backup_type="full",
                        include_images=include_images,
                        compress=compress_backup,
                        description=backup_description.strip() if backup_description.strip() else None,
                        custom_backup_path=backup_location.strip() if backup_location.strip() else None
                    )
                    
                    # Get user-friendly display path
                    display_path = backup_manager.get_backup_display_path(backup_metadata.backup_path)
                    
                    st.success(f"✅ Backup created successfully!")
                    st.info(f"""
                    **Backup Details:**
                    - ID: `{backup_metadata.backup_id}`
                    - Files: {backup_metadata.file_count:,}
                    - Size: {backup_metadata.total_size / (1024*1024):.1f} MB
                    - Compression: {backup_metadata.compression}
                    - Location: `{display_path}`
                    """)
                    
                    logger.info(f"Backup created via UI: {backup_metadata.backup_id}")
                    
                except Exception as e:
                    st.error(f"❌ Backup failed: {str(e)}")
                    logger.error(f"UI backup creation failed: {e}")
    
    with backup_tab2:
        st.subheader("Manage Existing Backups")
        
        try:
            wsl_db_path = convert_windows_to_wsl_path(db_path)
            backup_manager = SyncBackupManager(wsl_db_path)
            
            # List backups synchronously
            backups = backup_manager.list_backups()
            
            if not backups:
                st.info("No backups found. Create your first backup in the 'Create Backup' tab.")
            else:
                # Sort backups by creation time (newest first)
                backups.sort(key=lambda x: x.creation_time, reverse=True)
                
                st.markdown(f"**Found {len(backups)} backup(s)**")
                
                for i, backup in enumerate(backups):
                    with st.container(border=True):
                        backup_info_col, backup_actions_col = st.columns([3, 1])
                        
                        with backup_info_col:
                            # Parse creation time
                            try:
                                creation_dt = datetime.fromisoformat(backup.creation_time.replace('Z', '+00:00'))
                                formatted_time = creation_dt.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                formatted_time = backup.creation_time
                            
                            st.markdown(f"**🗂️ {backup.backup_id}**")
                            
                            info_col1, info_col2, info_col3 = st.columns(3)
                            with info_col1:
                                st.markdown(f"📅 **Created:** {formatted_time}")
                                st.markdown(f"📁 **Files:** {backup.file_count:,}")
                            with info_col2:
                                st.markdown(f"💾 **Size:** {backup.total_size / (1024*1024):.1f} MB")
                                st.markdown(f"🗜️ **Compression:** {backup.compression}")
                            with info_col3:
                                st.markdown(f"🔐 **Checksum:** `{backup.checksum[:12]}...`")
                                st.markdown(f"📝 **Type:** {backup.backup_type}")
                            
                            if backup.description:
                                st.markdown(f"📋 **Description:** {backup.description}")
                        
                        with backup_actions_col:
                            # Verify backup
                            if st.button("🔍 Verify", key=f"verify_backup_{i}", help="Verify backup integrity"):
                                with st.spinner("Verifying backup integrity..."):
                                    try:
                                        is_valid = backup_manager.verify_backup_integrity(backup.backup_id)
                                        
                                        if is_valid:
                                            st.success("✅ Backup is valid")
                                        else:
                                            st.error("❌ Backup integrity check failed")
                                    except Exception as e:
                                        st.error(f"❌ Verification failed: {str(e)}")
                            
                            # Delete backup
                            if st.button("🗑️ Delete", key=f"delete_backup_{i}", help="Delete this backup"):
                                st.session_state[f"confirm_delete_backup_{backup.backup_id}"] = True
                                st.rerun()
                        
                        # Confirmation for backup deletion
                        if st.session_state.get(f"confirm_delete_backup_{backup.backup_id}", False):
                            st.warning(f"⚠️ Are you sure you want to delete backup `{backup.backup_id}`?")
                            
                            confirm_col1, confirm_col2 = st.columns(2)
                            with confirm_col1:
                                if st.button("✅ Yes, Delete", key=f"confirm_delete_{i}", type="primary"):
                                    try:
                                        success = backup_manager.delete_backup(backup.backup_id)
                                        
                                        if success:
                                            st.success(f"✅ Backup `{backup.backup_id}` deleted successfully")
                                            logger.info(f"Backup deleted via UI: {backup.backup_id}")
                                            del st.session_state[f"confirm_delete_backup_{backup.backup_id}"]
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to delete backup")
                                    except Exception as e:
                                        st.error(f"❌ Delete failed: {str(e)}")
                            
                            with confirm_col2:
                                if st.button("❌ Cancel", key=f"cancel_delete_{i}"):
                                    del st.session_state[f"confirm_delete_backup_{backup.backup_id}"]
                                    st.rerun()
                
                st.divider()
                
                # Cleanup old backups
                st.subheader("🧹 Cleanup Old Backups")
                st.markdown("Automatically delete old backups, keeping only the most recent ones.")
                
                cleanup_col1, cleanup_col2 = st.columns([2, 1])
                
                with cleanup_col1:
                    keep_count = st.number_input(
                        "Number of backups to keep",
                        min_value=1,
                        max_value=50,
                        value=10,
                        help="Keep this many of the most recent backups"
                    )
                
                with cleanup_col2:
                    if st.button("🧹 Cleanup Old Backups", type="secondary"):
                        if len(backups) > keep_count:
                            with st.spinner(f"Cleaning up old backups (keeping {keep_count} most recent)..."):
                                try:
                                    deleted_count = backup_manager.cleanup_old_backups(keep_count=keep_count)
                                    
                                    if deleted_count > 0:
                                        st.success(f"✅ Cleaned up {deleted_count} old backups")
                                        logger.info(f"Cleaned up {deleted_count} backups via UI")
                                        st.rerun()
                                    else:
                                        st.info("No backups needed cleanup")
                                except Exception as e:
                                    st.error(f"❌ Cleanup failed: {str(e)}")
                        else:
                            st.info(f"You only have {len(backups)} backups, no cleanup needed")
        
        except Exception as e:
            st.error(f"❌ Error listing backups: {str(e)}")
    
    with backup_tab3:
        st.subheader("🔄 Restore from Backup")
        st.markdown("Restore your knowledge base from a previous backup.")
        
        try:
            wsl_db_path = convert_windows_to_wsl_path(db_path)
            backup_manager = SyncBackupManager(wsl_db_path)
            
            # List available backups
            backups = backup_manager.list_backups()
            
            if not backups:
                st.warning("No backups available for restore. Create a backup first.")
            else:
                # Sort backups by creation time (newest first)
                backups.sort(key=lambda x: x.creation_time, reverse=True)
                
                backup_options = {}
                for backup in backups:
                    try:
                        creation_dt = datetime.fromisoformat(backup.creation_time.replace('Z', '+00:00'))
                        formatted_time = creation_dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_time = backup.creation_time
                    
                    size_mb = backup.total_size / (1024*1024)
                    label = f"{backup.backup_id} - {formatted_time} ({backup.file_count:,} files, {size_mb:.1f}MB)"
                    backup_options[label] = backup.backup_id
                
                restore_col1, restore_col2 = st.columns([2, 1])
                
                with restore_col1:
                    selected_backup_label = st.selectbox(
                        "Select backup to restore",
                        options=list(backup_options.keys()),
                        help="Choose which backup to restore from"
                    )
                    
                    selected_backup_id = backup_options[selected_backup_label]
                
                with restore_col2:
                    overwrite_existing = st.checkbox(
                        "Overwrite existing data",
                        value=False,
                        help="Replace existing knowledge base data"
                    )
                    
                    verify_checksum = st.checkbox(
                        "Verify backup integrity",
                        value=True,
                        help="Check backup file integrity before restore"
                    )
                
                st.divider()
                
                # Show backup details
                selected_backup = next(b for b in backups if b.backup_id == selected_backup_id)
                
                st.markdown("**📋 Selected Backup Details:**")
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown(f"- **ID:** `{selected_backup.backup_id}`")
                    st.markdown(f"- **Files:** {selected_backup.file_count:,}")
                    st.markdown(f"- **Size:** {selected_backup.total_size / (1024*1024):.1f} MB")
                
                with detail_col2:
                    st.markdown(f"- **Type:** {selected_backup.backup_type}")
                    st.markdown(f"- **Compression:** {selected_backup.compression}")
                    if selected_backup.description:
                        st.markdown(f"- **Description:** {selected_backup.description}")
                
                st.divider()
                
                # Restore warnings
                st.warning("⚠️ **Important:** Restoring will affect your current knowledge base. Make sure you have a recent backup if needed.")
                
                if not overwrite_existing:
                    st.info("💡 **Safe Mode:** Existing files will not be overwritten. This may result in incomplete restore if data conflicts exist.")
                else:
                    st.error("🚨 **Overwrite Mode:** All existing data will be replaced with backup data.")
                
                # Restore button
                if st.button("🔄 Restore Knowledge Base", type="primary", use_container_width=True):
                    with st.spinner("Restoring knowledge base... This may take several minutes."):
                        try:
                            restore_metadata = backup_manager.restore_backup(
                                backup_id=selected_backup_id,
                                overwrite_existing=overwrite_existing,
                                verify_checksum=verify_checksum
                            )
                            
                            if restore_metadata.success:
                                st.success(f"✅ Knowledge base restored successfully!")
                                st.info(f"""
                                **Restore Details:**
                                - Backup ID: `{restore_metadata.backup_id}`
                                - Files Restored: {restore_metadata.files_restored:,}
                                - Restore ID: `{restore_metadata.restore_id}`
                                """)
                                
                                logger.info(f"Knowledge base restored via UI: {restore_metadata.restore_id}")
                                
                                st.success("🔄 **Please restart the application** to ensure all changes take effect.")
                            else:
                                st.error("❌ Restore completed with errors:")
                                for error in restore_metadata.errors:
                                    st.error(f"- {error}")
                        
                        except Exception as e:
                            st.error(f"❌ Restore failed: {str(e)}")
                            logger.error(f"UI restore failed: {e}")
        
        except Exception as e:
            st.error(f"❌ Error loading restore options: {str(e)}")

st.divider()

# --- Bulk Collection Management Section ---
st.header("Bulk Collection Management")
st.markdown("Perform operations on all collections at once.")

with st.container(border=True):
    st.subheader("🗑️ Delete All Collections")
    
    if 'show_confirm_delete_all_collections' not in st.session_state:
        st.session_state.show_confirm_delete_all_collections = False
    
    # Count non-default collections
    non_default_collections = [c for c in collections_list if c.get('name') != 'default']
    collection_count = len(non_default_collections)
    
    if collection_count == 0:
        st.info("No user-created collections to delete (only 'default' collection exists).")
    else:
        if not st.session_state.show_confirm_delete_all_collections:
            # Just show the button initially
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(f"🗑️ Delete All {collection_count} Collections", type="secondary", use_container_width=True):
                    st.session_state.show_confirm_delete_all_collections = True
                    st.rerun()
        
        if st.session_state.show_confirm_delete_all_collections:
            # Show warning and confirmation
            st.markdown("""**⚠️ WARNING**: This will permanently delete ALL user-created collections but keep the 'default' collection.
            
            This will:
            - Delete all named collections (preserving 'default')
            - Keep all documents in the knowledge base
            - Clear collection references but not the actual document data
            
            This action **cannot be undone**.""")
            
            st.error("⚠️ FINAL CONFIRMATION REQUIRED")
            st.markdown(f"You are about to **permanently delete {collection_count} collections**:")
            for collection in non_default_collections:
                st.markdown(f"- **{collection.get('name')}** ({len(collection.get('doc_ids', []))} documents)")
            
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("✅ YES, DELETE ALL COLLECTIONS", type="primary", use_container_width=True):
                    with st.spinner("Deleting all collections..."):
                        deleted_count = 0
                        for collection in non_default_collections:
                            if collection_mgr.delete_collection(collection.get('name')):
                                deleted_count += 1
                        st.success(f"Successfully deleted {deleted_count} collections!")
                        st.session_state.show_confirm_delete_all_collections = False
                    st.rerun()
            with confirm_col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_confirm_delete_all_collections = False
                    st.rerun()

def delete_knowledge_base(db_path: str):
    """Delete the knowledge base with proper error handling and logging."""
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_dir = Path(wsl_db_path) / "knowledge_hub_db"
    graph_file = Path(wsl_db_path) / "knowledge_cortex.gpickle"
    
    try:
        deleted_items = []
        
        # Delete ChromaDB directory
        if chroma_db_dir.exists() and chroma_db_dir.is_dir():
            shutil.rmtree(chroma_db_dir)
            deleted_items.append(f"ChromaDB directory: {chroma_db_dir}")
            logger.info(f"Successfully deleted ChromaDB directory: {chroma_db_dir}")
        
        # Delete knowledge graph file
        if graph_file.exists():
            graph_file.unlink()
            deleted_items.append(f"Knowledge graph: {graph_file}")
            logger.info(f"Successfully deleted knowledge graph: {graph_file}")
        
        if deleted_items:
            st.success(f"✅ Successfully deleted knowledge base components:\\n" + "\\n".join(f"- {item}" for item in deleted_items))
            logger.info("Knowledge base deletion completed successfully")
        else:
            st.warning("⚠️ No knowledge base components found to delete.")
            logger.warning(f"No knowledge base found at: {wsl_db_path}")
            
    except Exception as e:
        error_msg = f"Failed to delete knowledge base: {e}"
        logger.error(f"Knowledge base deletion failed: {e}")
        st.error(f"❌ {error_msg}")
    
    # Reset the confirmation state  
    st.session_state.show_confirm_delete_kb = False

# --- Knowledge Base Management Section ---
st.header("Knowledge Base Management")
st.markdown("Manage the entire knowledge base and database.")

with st.container(border=True):
    # --- Database Deduplication Section ---
    st.subheader("🔧 Database Deduplication")
    st.markdown("Remove duplicate documents from the knowledge base to improve performance and storage efficiency.")
    
    dedup_col1, dedup_col2 = st.columns([2, 1])
    
    with dedup_col1:
        st.markdown("""
        **What does deduplication do?**
        - Identifies documents with identical file hashes or content
        - Keeps the most complete version of each document
        - Removes duplicate entries from ChromaDB
        - Updates collections to remove references to deleted duplicates
        """)
        
        # Initialize deduplication session state
        if 'dedup_analysis_results' not in st.session_state:
            st.session_state.dedup_analysis_results = None
        if 'dedup_analysis_running' not in st.session_state:
            st.session_state.dedup_analysis_running = False
    
    with dedup_col2:
        # Analysis button
        if st.button("🔍 Analyze Duplicates", 
                    key="analyze_duplicates_btn", 
                    type="secondary", 
                    use_container_width=True,
                    disabled=st.session_state.dedup_analysis_running):
            
            st.session_state.dedup_analysis_running = True
            
            with st.spinner("Analyzing knowledge base for duplicates... This may take a few minutes."):
                try:
                    # Perform duplicate analysis (dry run)
                    results = collection_mgr.deduplicate_vector_store(vector_collection, dry_run=True)
                    st.session_state.dedup_analysis_results = results
                    
                    if results.get('status') == 'analysis_complete':
                        st.success(f"✅ Analysis complete!")
                        st.info(f"""
                        **Duplicate Analysis Results:**
                        - Total documents: {results['total_documents']:,}
                        - Duplicates found: {results['duplicates_found']:,}
                        - Duplicate percentage: {results['duplicate_percentage']:.1f}%
                        - Unique files: {results['unique_files']:,}
                        - Duplicate groups: {results['duplicate_groups']:,}
                        """)
                        logger.info(f"Deduplication analysis completed via UI: {results['duplicates_found']} duplicates found")
                    else:
                        st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"UI deduplication analysis failed: {e}")
                finally:
                    st.session_state.dedup_analysis_running = False
                    st.rerun()
    
    # Show analysis results if available
    if st.session_state.dedup_analysis_results:
        results = st.session_state.dedup_analysis_results
        
        if results.get('status') == 'analysis_complete' and results.get('duplicates_found', 0) > 0:
            st.divider()
            
            # Show detailed results
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric("📄 Total Documents", f"{results['total_documents']:,}")
            with result_col2:
                st.metric("🔄 Duplicates Found", f"{results['duplicates_found']:,}")
            with result_col3:
                st.metric("📊 Duplicate %", f"{results['duplicate_percentage']:.1f}%")
            
            st.divider()
            
            # Cleanup options
            st.markdown("**🧹 Cleanup Options**")
            
            # Warning about cleanup
            if results['duplicate_percentage'] > 50:
                st.warning(f"⚠️ High duplicate percentage detected ({results['duplicate_percentage']:.1f}%). This suggests a significant duplication issue that should be resolved.")
            elif results['duplicate_percentage'] > 20:
                st.info(f"💡 Moderate duplication detected ({results['duplicate_percentage']:.1f}%). Cleanup recommended for optimal performance.")
            else:
                st.success(f"✅ Low duplication level ({results['duplicate_percentage']:.1f}%). Cleanup optional but will improve storage efficiency.")
            
            # Cleanup confirmation
            cleanup_col1, cleanup_col2 = st.columns([2, 1])
            
            with cleanup_col1:
                st.markdown(f"""
                **Cleanup will:**
                - Remove {results['duplicates_found']:,} duplicate documents
                - Keep the most complete version of each file
                - Update {len(collection_mgr.get_collection_names())} collections automatically
                - Free up storage space and improve query performance
                """)
            
            with cleanup_col2:
                if st.checkbox("I understand this action cannot be undone", key="dedup_confirm_checkbox"):
                    if st.button("🧹 Remove Duplicates", 
                                key="remove_duplicates_btn", 
                                type="primary", 
                                use_container_width=True):
                        
                        with st.spinner(f"Removing {results['duplicates_found']:,} duplicate documents... This may take several minutes."):
                            try:
                                # Perform actual deduplication
                                cleanup_results = collection_mgr.deduplicate_vector_store(vector_collection, dry_run=False)
                                
                                if cleanup_results.get('status') == 'cleanup_complete':
                                    removed_count = cleanup_results.get('removed_count', 0)
                                    st.success(f"✅ Deduplication complete!")
                                    st.info(f"""
                                    **Cleanup Results:**
                                    - Documents removed: {removed_count:,}
                                    - Storage space freed: ~{removed_count * 0.1:.1f} MB (estimated)
                                    - Collections updated automatically
                                    """)
                                    
                                    # Clear analysis results to force re-analysis
                                    st.session_state.dedup_analysis_results = None
                                    
                                    logger.info(f"Deduplication cleanup completed via UI: {removed_count} documents removed")
                                    
                                    # Show recommendation to restart
                                    st.success("🔄 **Recommendation:** Restart the application to ensure optimal performance with the cleaned database.")
                                    
                                else:
                                    st.error(f"❌ Cleanup failed: {cleanup_results.get('error', 'Unknown error')}")
                                    
                            except Exception as e:
                                st.error(f"❌ Cleanup failed: {str(e)}")
                                logger.error(f"UI deduplication cleanup failed: {e}")
        
        elif results.get('status') == 'analysis_complete' and results.get('duplicates_found', 0) == 0:
            st.success("✅ No duplicates found! Your knowledge base is already optimized.")
            
        elif results.get('status') == 'no_documents':
            st.info("ℹ️ No documents found in the knowledge base.")
    
    st.divider()
    
    st.subheader("🗑️ Delete Entire Knowledge Base")
    
    if 'show_confirm_delete_kb' not in st.session_state:
        st.session_state.show_confirm_delete_kb = False
    
    if not st.session_state.show_confirm_delete_kb:
        # Just show the button initially
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚨 Delete Entire Knowledge Base", type="primary", use_container_width=True):
                st.session_state.show_confirm_delete_kb = True
                st.rerun()
    
    if st.session_state.show_confirm_delete_kb:
        # Show warning and confirmation
        st.markdown("""**⚠️ DANGER ZONE**: This will permanently delete the entire knowledge base including:
        - All indexed documents and embeddings
        - All entity and relationship data (GraphRAG)
        - All vector search data
        - Processing logs and metadata
        
        This action **cannot be undone**. You will need to re-ingest all documents.""")
        
        st.error("⚠️ FINAL CONFIRMATION REQUIRED")
        st.markdown(f"You are about to **permanently delete** the entire knowledge base at: `{db_path}/knowledge_hub_db/`")
        
        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            if st.button("✅ YES, DELETE EVERYTHING", type="primary", use_container_width=True):
                delete_knowledge_base(db_path)
                st.rerun()
        with confirm_col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_confirm_delete_kb = False
                st.rerun()