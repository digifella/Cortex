# ## File: pages/3_Knowledge_Search.py
# Version: 21.0.0 (Utilities Refactor)
# Date: 2025-07-23
# Purpose: A UI for searching the knowledge base, managing collections,
#          and deleting documents from the KB.
#          - REFACTOR (v21.0.0): Updated to use centralized utilities for path handling,
#            logging, and error handling. Removed code duplication.

import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from collections import OrderedDict
import chromadb
from chromadb.config import Settings as ChromaSettings
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import centralized utilities
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.session_state import initialize_app_session_state
from cortex_engine.config import COLLECTION_NAME, INGESTED_FILES_LOG, EMBED_MODEL
from cortex_engine.help_system import help_system

# --- App Config ---
APP_VERSION = "v21.0.0 (Utilities Refactor)"
logger = get_logger(__name__)

# --- Constants ---
# Get dynamic document type options from the document type manager
def get_document_type_options():
    try:
        from cortex_engine.document_type_manager import get_document_type_manager
        doc_type_manager = get_document_type_manager()
        return ["Any"] + doc_type_manager.get_all_document_types()
    except ImportError:
        # Fallback to static list if document type manager is not available
        return [
            "Any", "Project Plan", "Technical Documentation", "Proposal/Quote", "Case Study / Trophy",
            "Final Report", "Draft Report", "Presentation", "Contract/SOW",
            "Meeting Minutes", "Financial Report", "Research Paper", "Email Correspondence", "Other"
        ]

DOC_TYPE_OPTIONS = get_document_type_options()
PROPOSAL_OUTCOME_OPTIONS = ["Any", "Won", "Lost", "Pending", "N/A"]
RESULTS_PAGE_SIZE = 5

# --- Helper Functions ---
# Path handling now centralized in utilities

def heal_and_get_collection_doc_ids(collection_name):
    collection_mgr = WorkingCollectionManager()
    st.session_state.collections = collection_mgr.collections
    collection_obj = st.session_state.collections.get(collection_name, {})
    return set(collection_obj.get("doc_ids", []))

def delete_document_from_kb(doc_id: str, file_path: str, vector_collection, db_path: str):
    try:
        if not vector_collection: st.error("Vector collection not available. Cannot perform deletion."); return False
        nodes_to_delete = vector_collection.get(where={"doc_id": doc_id})
        node_ids_to_delete = nodes_to_delete.get('ids', [])
        if not node_ids_to_delete: st.warning(f"No vector data found for document ID {doc_id}. It may have already been removed.")
        else: vector_collection.delete(ids=node_ids_to_delete); st.toast(f"Deleted {len(node_ids_to_delete)} vector chunks from the knowledge base.")
        chroma_db_path = os.path.join(db_path, "knowledge_hub_db")
        processed_log_path = os.path.join(chroma_db_path, INGESTED_FILES_LOG)
        if os.path.exists(processed_log_path):
            with open(processed_log_path, 'r') as f: log_data = json.load(f)
            key_to_delete = next((key for key in log_data if Path(key).as_posix() == Path(file_path).as_posix()), None)
            if key_to_delete: del log_data[key_to_delete]; st.toast("Removed document from the processed files log.")
            with open(processed_log_path, 'w') as f: json.dump(log_data, f, indent=4)
        collection_mgr = WorkingCollectionManager()
        for coll_name in collection_mgr.get_collection_names(): collection_mgr.remove_from_collection(coll_name, [doc_id])
        st.toast("Removed document from all working collections.")
        st.success(f"Successfully pruned document '{Path(file_path).name}' from the system.")
        return True
    except Exception as e: st.error(f"An error occurred during deletion: {e}"); return False

# --- Core Loading and State Management ---
@st.cache_resource(ttl=3600)
def load_base_index(db_path, model_provider, api_key=None):
    if not db_path or not db_path.strip(): st.warning("Database path is not configured."); return None, None
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
    if not os.path.isdir(chroma_db_path): st.warning(f"🧠 Knowledge base directory not found at '{chroma_db_path}'."); return None, None
    try:
        Settings.llm = Ollama(model="mistral", request_timeout=120.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        db_settings = ChromaSettings(anonymized_telemetry=False)
        db = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=chroma_db_path)
        index = load_index_from_storage(storage_context)
        st.success(f"✅ Knowledge base loaded successfully from '{chroma_db_path}'.")
        return index, chroma_collection
    except Exception as e:
        st.error(f"Backend initialization failed: {e}")
        logger.error(f"Error loading query engine from {chroma_db_path}: {e}", exc_info=True)
        return None, None

def initialize_search_state():
    if 'search_sort_key' not in st.session_state: st.session_state.search_sort_key = 'score'
    if 'search_sort_asc' not in st.session_state: st.session_state.search_sort_asc = False
    if 'search_page' not in st.session_state: st.session_state.search_page = 0
    if 'doc_type_filter' not in st.session_state: st.session_state.doc_type_filter = "Any"
    if 'outcome_filter' not in st.session_state: st.session_state.outcome_filter = "Any"
    if 'filter_operator' not in st.session_state: st.session_state.filter_operator = "AND"
    if 'query_text' not in st.session_state: st.session_state.query_text = ""
    if 'search_results' not in st.session_state: st.session_state.search_results = []

def reset_search_state():
    st.session_state.search_sort_key = 'score'
    st.session_state.search_sort_asc = False
    st.session_state.search_page = 0
    st.session_state.doc_type_filter = "Any"
    st.session_state.outcome_filter = "Any"
    st.session_state.filter_operator = "AND"
    st.session_state.query_text = ""
    st.session_state.search_results = []


# --- UI Rendering ---

def render_sidebar():
    collection_mgr = WorkingCollectionManager()
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.text_input("Database Storage Path", key="db_path_input", disabled=True)
        st.divider()
        st.header("📚 Working Collections")
        st.info("First, select a collection. Then, use the 'Add' buttons on search results to add documents to it.")
        collection_names = collection_mgr.get_collection_names()
        try:
            current_index = collection_names.index(st.session_state.selected_collection)
        except (ValueError, AttributeError):
            current_index = 0
            st.session_state.selected_collection = collection_names[0] if collection_names else "default"
        st.selectbox("Active Collection", options=collection_names, key="selected_collection", index=current_index)
        with st.form("new_collection_form"):
            new_name = st.text_input("Create New Collection")
            if st.form_submit_button("Create"):
                name = new_name.strip()
                if name:
                    if collection_mgr.create_collection(name): st.session_state.collection_to_select = name; st.rerun()
                    else: st.warning("Collection already exists.")
                else: st.warning("Name cannot be empty.")

def render_main_content(base_index, vector_collection):
    st.header("🔎 3. Knowledge Search")
    if not base_index:
        st.markdown("---"); st.info("Backend not loaded. Please check configuration and ensure ingestion has been run."); return

    # --- Search Input UI ---
    st.text_input("Free-Text Search (optional)", key="query_text", placeholder="e.g., project management for AI",
                  help="🔍 **Natural language search**: Ask questions like 'What projects involved Company X?' or use keywords. Leave empty to search by metadata only.")
    st.markdown("**Metadata Filters (optional)**")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: 
        st.selectbox("Document Type", options=DOC_TYPE_OPTIONS, key="doc_type_filter",
                    help="📄 Filter by document type (e.g., 'Proposal/Quote', 'Technical Documentation'). Select 'Any' to include all types.")
    with col2: 
        st.selectbox("Proposal Outcome", options=PROPOSAL_OUTCOME_OPTIONS, key="outcome_filter",
                    help="📊 Filter by proposal result. Useful for finding successful patterns or analyzing losses.")
    num_active_filters = (1 if st.session_state.doc_type_filter != "Any" else 0) + (1 if st.session_state.outcome_filter != "Any" else 0)
    with col3: 
        st.radio("Operator", ["AND", "OR"], key="filter_operator", horizontal=True, disabled=(num_active_filters < 2),
                help="🔗 **AND**: Document must match ALL selected filters. **OR**: Document matches ANY selected filter.")
    st.radio("Search In:", ["Entire Knowledge Base", "Active Collection"], key="search_scope", horizontal=True,
            help="🎯 **Entire Knowledge Base**: Search all documents. **Active Collection**: Search only documents in the currently selected collection.")

    # --- Search Execution Logic ---
    search_button_disabled = not st.session_state.query_text.strip() and num_active_filters == 0
    if st.button("Search Knowledge Base", type="primary", disabled=search_button_disabled):
        query = st.session_state.query_text.strip()

        # --- START: v20.0.0 NATIVE CHROMA FILTER LOGIC ---
        where_clause = {}
        all_conditions = []

        # 1. Build list of metadata conditions from UI
        ui_metadata_conditions = []
        if st.session_state.doc_type_filter != "Any":
            ui_metadata_conditions.append({"document_type": st.session_state.doc_type_filter})
        if st.session_state.outcome_filter != "Any":
            ui_metadata_conditions.append({"proposal_outcome": st.session_state.outcome_filter})

        # 2. Group the UI conditions based on the AND/OR operator
        if len(ui_metadata_conditions) > 1:
            operator_key = f"${st.session_state.filter_operator.lower()}"
            all_conditions.append({operator_key: ui_metadata_conditions})
        elif ui_metadata_conditions:
            all_conditions.extend(ui_metadata_conditions)

        # 3. Build the collection scope condition (always an OR group)
        if st.session_state.search_scope == "Active Collection":
            collection_doc_ids = list(heal_and_get_collection_doc_ids(st.session_state.selected_collection))
            if collection_doc_ids:
                scope_conditions = [{"doc_id": doc_id} for doc_id in collection_doc_ids]
                all_conditions.append({"$or": scope_conditions})
            else:
                st.warning(f"The '{st.session_state.selected_collection}' collection is empty. Searching entire knowledge base instead.")

        # 4. Combine all conditions into a final '$and' group if needed
        if len(all_conditions) > 1:
            where_clause = {"$and": all_conditions}
        elif len(all_conditions) == 1:
            where_clause = all_conditions[0]
        # else: where_clause remains {}

        retriever_kwargs = {"vector_store_kwargs": {"where": where_clause}} if where_clause else {}
        # --- END: v20.0.0 NATIVE CHROMA FILTER LOGIC ---

        with st.spinner("Searching..."):
            retriever = base_index.as_retriever(similarity_top_k=50, **retriever_kwargs)
            search_text = query if query else " "
            response_nodes = retriever.retrieve(search_text)
            unique_results = list(OrderedDict((node.metadata.get('doc_id'), node) for node in response_nodes).values())
            st.session_state.search_results = unique_results
            st.session_state.search_page = 0
            st.session_state.search_sort_key = 'score'
            st.session_state.search_sort_asc = False
            # Reset the save warning for new search results
            st.session_state.results_saved_warning_dismissed = False
            st.rerun()

    # --- Results Viewer ---
    if st.session_state.get("search_results"):
        st.divider()
        st.subheader("Search Results")
        results = st.session_state.search_results

        # --- Sorting and Action Controls ---
        sort_col1, sort_col2, sort_col3, action_col4 = st.columns(4)
        if sort_col1.button("Sort by Relevance (Score)", use_container_width=True): st.session_state.search_sort_key = 'score'; st.session_state.search_sort_asc = False; st.rerun()
        if sort_col2.button("Sort by Filename (A-Z)", use_container_width=True): st.session_state.search_sort_key = 'filename'; st.session_state.search_sort_asc = True; st.rerun()
        if sort_col3.button("Sort by Date (Newest First)", use_container_width=True): st.session_state.search_sort_key = 'date'; st.session_state.search_sort_asc = False; st.rerun()
        with action_col4:
            if st.button("Clear Selection", use_container_width=True):
                st.session_state.search_results = []
                st.rerun()

        # --- Sorting Logic ---
        sort_key = st.session_state.search_sort_key; sort_asc = st.session_state.search_sort_asc
        if sort_key == 'score': results.sort(key=lambda n: n.score or 0, reverse=not sort_asc)
        elif sort_key == 'filename': results.sort(key=lambda n: n.metadata.get('file_name', '').lower(), reverse=not sort_asc)
        elif sort_key == 'date': results.sort(key=lambda n: datetime.fromisoformat(n.metadata.get('last_modified_date', '1970-01-01T00:00:00')) if n.metadata.get('last_modified_date') else datetime.min, reverse=not sort_asc)

        # --- Pagination Logic ---
        total_results = len(results); total_pages = -(-total_results // RESULTS_PAGE_SIZE) or 1
        current_page = st.session_state.search_page
        start_idx = current_page * RESULTS_PAGE_SIZE; end_idx = start_idx + RESULTS_PAGE_SIZE
        paginated_results = results[start_idx:end_idx]
        st.info(f"Showing {len(paginated_results)} of {total_results} results. Page {current_page + 1} of {total_pages}.")

        # --- Bulk Actions ---
        st.markdown("**Bulk Actions**")
        bulk_col1, bulk_col2 = st.columns(2)
        with bulk_col1:
            if st.button("➕ Add All to Collection", type="primary", use_container_width=True, 
                        help=f"Add all {total_results} search results to '{st.session_state.selected_collection}' collection"):
                collection_mgr = WorkingCollectionManager()
                all_doc_ids = [node.metadata.get('doc_id') for node in results if node.metadata.get('doc_id')]
                active_collection_ids = heal_and_get_collection_doc_ids(st.session_state.selected_collection)
                new_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in active_collection_ids]
                
                if new_doc_ids:
                    collection_mgr.add_docs_by_id_to_collection(st.session_state.selected_collection, new_doc_ids)
                    st.success(f"✅ Added {len(new_doc_ids)} documents to '{st.session_state.selected_collection}' collection!")
                    if len(all_doc_ids) - len(new_doc_ids) > 0:
                        st.info(f"ℹ️ {len(all_doc_ids) - len(new_doc_ids)} documents were already in the collection.")
                else:
                    st.info("ℹ️ All search results are already in the selected collection.")
                st.rerun()
        
        with bulk_col2:
            if st.button("➖ Remove All from Collection", use_container_width=True,
                        help=f"Remove all search results that are currently in '{st.session_state.selected_collection}' collection"):
                collection_mgr = WorkingCollectionManager()
                all_doc_ids = [node.metadata.get('doc_id') for node in results if node.metadata.get('doc_id')]
                active_collection_ids = heal_and_get_collection_doc_ids(st.session_state.selected_collection)
                docs_to_remove = [doc_id for doc_id in all_doc_ids if doc_id in active_collection_ids]
                
                if docs_to_remove:
                    collection_mgr.remove_from_collection(st.session_state.selected_collection, docs_to_remove)
                    st.success(f"✅ Removed {len(docs_to_remove)} documents from '{st.session_state.selected_collection}' collection!")
                else:
                    st.info("ℹ️ None of the search results are currently in the selected collection.")
                st.rerun()

        active_collection_ids = heal_and_get_collection_doc_ids(st.session_state.selected_collection)
        collection_mgr = WorkingCollectionManager()
        if not paginated_results:
            st.info("No relevant documents found for your query and filters.")
            if st.button("🔄 Start New Search"):
                reset_search_state()
                st.rerun()

        for node in paginated_results:
            doc_id = node.metadata.get('doc_id')
            file_name = node.metadata.get('file_name', 'Unknown Document')
            is_in_collection = doc_id in active_collection_ids
            col_main, col_actions = st.columns([0.8, 0.2])

            with col_main:
                meta_display = f"**Type:** {node.metadata.get('document_type', 'N/A')} | **Outcome:** {node.metadata.get('proposal_outcome', 'N/A')} | **Score:** {node.score:.2f}"
                with st.expander(f"**{file_name}**"):
                    st.markdown(meta_display)
                    st.text_area("Preview", node.get_content(), height=250, disabled=True, key=f"preview_{doc_id}")
                    st.caption(f"Document ID: {doc_id}")
                    st.divider()
                    if st.checkbox("Show Maintenance Actions", key=f"show_maint_{doc_id}"):
                        st.warning(f"This will permanently delete **{file_name}** from the knowledge base. This action cannot be undone.")
                        if st.button("DELETE PERMANENTLY", key=f"del_btn_{doc_id}", type="primary"):
                            wsl_db_path = convert_windows_to_wsl_path(st.session_state.db_path_input)
                            if delete_document_from_kb(doc_id, node.metadata.get('doc_posix_path', ''), vector_collection, wsl_db_path):
                                st.session_state.search_results = []; st.rerun()
            with col_actions:
                if is_in_collection:
                    if st.button("Remove", key=f"rem_{doc_id}", use_container_width=True):
                        collection_mgr.remove_from_collection(st.session_state.selected_collection, [doc_id]); st.rerun()
                else:
                    if st.button("Add to Collection", key=f"add_{doc_id}", type="secondary", use_container_width=True):
                        collection_mgr.add_docs_by_id_to_collection(st.session_state.selected_collection, [doc_id]); st.rerun()

        # --- Pagination Controls ---
        st.divider()
        page_col1, page_col2, page_col3 = st.columns([1, 6, 1])
        if current_page > 0:
            if page_col1.button("⬅️ Previous Page", use_container_width=True): st.session_state.search_page -= 1; st.rerun()
        if end_idx < total_results:
            if page_col3.button("Next Page ➡️", use_container_width=True): st.session_state.search_page += 1; st.rerun()
        page_col2.write("")

    if st.button("Clear Filters & Reset Search"):
        reset_search_state()
        st.rerun()

# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="Cortex Knowledge Search")
st.title("🧠 Project Cortex Suite")
st.caption(f"App Version: {APP_VERSION}")

# Add help system
help_system.show_help_menu()

# Show help modal if requested
if st.session_state.get("show_help_modal", False):
    help_topic = st.session_state.get("help_topic", "overview")
    help_system.show_help_modal(help_topic)

# Show contextual help for this page
help_system.show_contextual_help("search")

if "collection_to_select" in st.session_state:
    st.session_state.selected_collection = st.session_state.collection_to_select
    del st.session_state.collection_to_select

initialize_app_session_state()
initialize_search_state()

# Check for unsaved search results and warn user
def check_unsaved_search_results():
    """Check if there are unsaved search results and show warning."""
    if (st.session_state.get("search_results") and 
        len(st.session_state.search_results) > 0 and 
        not st.session_state.get("results_saved_warning_dismissed", False)):
        
        st.warning("⚠️ **You have unsaved search results!** Consider saving them to a collection before navigating away.")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if st.button("💾 Save to Default Collection", use_container_width=True):
                # Save to default collection
                collection_mgr = WorkingCollectionManager()
                doc_ids = [node.metadata.get('doc_id') for node in st.session_state.search_results if node.metadata.get('doc_id')]
                collection_mgr.add_docs_by_id_to_collection("default", doc_ids)
                st.success(f"✅ Saved {len(doc_ids)} documents to 'default' collection!")
                st.session_state.results_saved_warning_dismissed = True
                st.rerun()
        
        with col2:
            # Quick save to named collection
            new_collection_name = st.text_input("Or save to new collection:", key="quick_save_collection_name", placeholder="Enter collection name...")
            if new_collection_name and st.button("💾 Create & Save", use_container_width=True):
                collection_mgr = WorkingCollectionManager()
                if collection_mgr.create_collection(new_collection_name):
                    doc_ids = [node.metadata.get('doc_id') for node in st.session_state.search_results if node.metadata.get('doc_id')]
                    collection_mgr.add_docs_by_id_to_collection(new_collection_name, doc_ids)
                    st.success(f"✅ Created '{new_collection_name}' and saved {len(doc_ids)} documents!")
                    st.session_state.results_saved_warning_dismissed = True
                    st.rerun()
                else:
                    st.error(f"Collection '{new_collection_name}' already exists!")
        
        with col3:
            if st.button("❌ Dismiss", use_container_width=True):
                st.session_state.results_saved_warning_dismissed = True
                st.rerun()
        
        st.divider()

# Show the warning if applicable
check_unsaved_search_results()

base_index, vector_collection = load_base_index(st.session_state.db_path_input, "Local")

render_sidebar()
render_main_content(base_index, vector_collection)