# ## File: pages/3_Knowledge_Search.py
# Version: v5.4.0
# Date: 2026-01-19
# Purpose: Advanced knowledge search interface with vector + graph search capabilities.
#          - FEATURE (v5.5.0): Added optional tag input when saving to collections. Tags apply
#            to all saved documents for easier categorization in proposal workflows.
#          - FEATURE (v5.4.0): Streamlined "Save to Collection" UI with dropdown showing existing
#            collections + "Create New Collection" option. Works for bulk and individual adds.
#          - FEATURE (v5.3.0): Background preload of reranker model when page opens. Model loads
#            while user types query, ready by first search. Added result count slider (5-50).
#          - FEATURE (v5.2.0): Added UI toggle for neural reranking in sidebar. Dynamic timeout
#            (300s when reranker enabled for first model load, 30s otherwise). No env vars needed.
#          - FEATURE (v5.1.0): Added optional Qwen3-VL neural reranking for precision boost.
#            Two-stage retrieval: fast recall (embedding) + precision (reranker).
#          - FEATURE (v4.11.0): Added embedding model validation before queries with
#            user-friendly warnings and solutions for embedding model mismatches
#          - GRAPHRAG INTEGRATION (v4.2.1): Re-enabled GraphRAG search modes with radio button
#            selection. Added Traditional Vector Search, GraphRAG Enhanced, and Hybrid Search
#            options with entity relationship analysis and graph context feedback.
#          - BUGFIX (v22.4.3): Fixed UnboundLocalError where 'config' variable was referenced before
#            initialization. Moved config loading to top of main() function.
#          - DOCKER FIX (v22.4.2): Resolved Docker-development database conflicts. Detects when
#            Docker is accessing development databases with schema mismatches and provides solutions.
#          - DOCKER FIX (v22.4.1): Added Docker environment compatibility for collection manager
#            errors. Gracefully handles missing database schema and provides fallback options.
#          - SEARCH ENHANCEMENT (v22.4.0): Implemented multi-strategy search approach to handle
#            complex queries like 'strategy and transformation'. Tries vector search, multi-term
#            search, and text fallback for comprehensive results.
#          - FEATURE RESTORATION (v22.3.0): Restored boolean logic for metadata filters and
#            collection management from pre-refactor version. Implements safe post-search
#            filtering to avoid ChromaDB where clause issues.
#          - UX ENHANCEMENT (v22.2.2): Added progress indicator with st.status to show search
#            activity and prevent user confusion during search operations.
#          - COMPATIBILITY FIX (v22.2.1): Removed LlamaIndex imports to resolve numpy.iterable
#            compatibility issues in Docker environment. Now uses pure ChromaDB approach.
#          - MAJOR FIX (v22.2.0): Bypassed LlamaIndex query engine with direct ChromaDB search
#            to resolve persistent "Expected where to have exactly one operator, got {}" errors.
#          - CRITICAL FIX (v22.0.0): Replaced custom embedding initialization with backend system
#            integration to resolve 'SentenceTransformerWrapper' missing methods error.

import streamlit as st
import os
import pandas as pd
from pathlib import Path
import re
import threading
from typing import Optional

# Import only ChromaDB components needed for direct search
import chromadb
from chromadb.config import Settings as ChromaSettings

# Suppress ChromaDB telemetry errors
import warnings
warnings.filterwarnings("ignore", message=".*capture.*takes.*positional argument.*")
warnings.filterwarnings("ignore", message=".*Failed to send telemetry.*")

# LlamaIndex imports moved to functions where actually needed to avoid numpy.iterable issues

# Import project modules
from cortex_engine.config import (
    EMBED_MODEL, COLLECTION_NAME, KB_LLM_MODEL,
    QWEN3_VL_RERANKER_ENABLED, QWEN3_VL_RERANKER_TOP_K, QWEN3_VL_RERANKER_CANDIDATES
)
from cortex_engine.config_manager import ConfigManager
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils import (
    get_logger,
    convert_to_docker_mount_path,
    convert_windows_to_wsl_path,
    resolve_db_root_path,
)
from cortex_engine.utils.default_paths import get_default_ai_database_path
from cortex_engine.embedding_service import embed_query
from cortex_engine.version_config import VERSION_STRING
from cortex_engine.ui_theme import apply_theme, section_header

# Set up logging
logger = get_logger(__name__)

# Page configuration
PAGE_VERSION = VERSION_STRING

st.set_page_config(page_title="Knowledge Search", layout="wide")

# Apply refined editorial theme
apply_theme()


# =============================================================================
# Reranker Background Preload
# =============================================================================
def _preload_reranker_model():
    """Background task to preload the Qwen3-VL reranker model into GPU memory."""
    try:
        from cortex_engine.qwen3_vl_reranker_service import _load_reranker, Qwen3VLRerankerConfig

        logger.info("🔄 Background preload: Loading Qwen3-VL reranker model...")
        config = Qwen3VLRerankerConfig.auto_select()
        _load_reranker(config)
        logger.info("✅ Background preload: Reranker model ready")

        # Update session state to indicate model is ready
        # Note: This won't trigger UI update, but search will find cached model
    except Exception as e:
        logger.warning(f"Background preload failed (will load on first search): {e}")


def start_reranker_preload():
    """Start background preload of reranker model if enabled and not already loading."""
    # Check if reranker is enabled (from session state or config default)
    reranker_enabled = st.session_state.get('reranker_enabled', QWEN3_VL_RERANKER_ENABLED)

    if reranker_enabled and 'reranker_preload_started' not in st.session_state:
        st.session_state.reranker_preload_started = True
        thread = threading.Thread(target=_preload_reranker_model, daemon=True)
        thread.start()
        logger.info("🚀 Started background reranker model preload")


def get_candidate_db_paths(db_path: str):
    """Build ordered list of possible knowledge base locations."""
    candidates = []
    seen = set()

    def _add(label: str, path_value: str):
        if not path_value:
            return
        normalized = path_value.rstrip('/')
        if not normalized:
            normalized = '/'
        if normalized not in seen:
            candidates.append((label, normalized))
            seen.add(normalized)

    raw_value = db_path or ""
    # Always try the configured/container-normalized path first
    _add("configured", convert_to_docker_mount_path(raw_value))

    in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV') == 'true'
    if in_docker:
        # Allow direct Windows/WSL mount access when bind mount is missing
        _add("windows_mount", convert_windows_to_wsl_path(raw_value))

        # Environment-provided Docker path
        ai_env = os.environ.get('AI_DATABASE_PATH')
        if ai_env:
            _add("docker_env", ai_env.strip())

        # Fall back to standard Docker locations if provided path is empty/missing
        for fallback in filter(None, [
            os.environ.get('WINDOWS_AI_DATABASE_PATH'),
            "/data/ai_databases",
            "/app/data/ai_databases",
            "/home/cortex/data/ai_databases"
        ]):
            _add("fallback", fallback)
    else:
        # Native/WSL environment just needs normalization
        _add("normalized", convert_windows_to_wsl_path(raw_value))

    return candidates


def get_document_type_options():
    """Get document type options - Docker-safe version"""
    st.session_state.pop("kb_fallback_path", None)
    try:
        # In Docker environment, use fallback to static options since collections may not be initialized
        collection_mgr = WorkingCollectionManager()
        collections = collection_mgr.collections
        doc_types = set()
        for collection in collections.values():
            # Handle both 'documents' and 'doc_ids' schemas for compatibility
            docs = collection.get('documents', collection.get('doc_ids', []))
            for doc in docs:
                # For doc_ids schema, doc is just an ID string, so skip metadata extraction
                if isinstance(doc, str):
                    continue
                doc_type = doc.get('metadata', {}).get('document_type', 'Unknown')
                if doc_type and doc_type != 'Unknown':
                    doc_types.add(doc_type)
        return sorted(list(doc_types))
    except Exception as e:
        logger.warning(f"Collections not available (Docker environment?): {e}")
        # Return static fallback options for Docker environment
        return ["Project Plan", "Technical Documentation", "Proposal/Quote", "Case Study / Trophy",
                "Final Report", "Draft Report", "Presentation", "Contract/SOW",
                "Meeting Minutes", "Financial Report", "Research Paper", "Email Correspondence", "Other"]


def get_available_thematic_tags(db_path: str) -> list:
    """Collect unique thematic tags from the ChromaDB collection (best-effort, paged)."""
    try:
        resolved_root = resolve_db_root_path(db_path) if not os.path.exists('/.dockerenv') else Path(db_path)
        if not resolved_root:
            return []
        chroma_db_path = os.path.join(str(resolved_root), "knowledge_hub_db")
        if not os.path.isdir(chroma_db_path):
            return []
        settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=chroma_db_path, settings=settings)
        collection = client.get_collection(COLLECTION_NAME)
        sample = collection.get(include=["metadatas"], limit=2000)
        metadatas = sample.get("metadatas", [])
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        tags = set()
        for meta in metadatas:
            raw = meta.get("thematic_tags")
            if not raw:
                continue
            if isinstance(raw, str):
                parts = [p.strip() for p in raw.split(",") if p.strip()]
            elif isinstance(raw, list):
                parts = [str(p).strip() for p in raw if str(p).strip()]
            else:
                parts = []
            for t in parts:
                tags.add(t)
        return sorted(tags)
    except Exception as e:
        logger.warning(f"Could not gather thematic tags: {e}")
        return []


def find_existing_kb_root(exclude: Optional[str] = None) -> Optional[str]:
    """Look for a populated knowledge base at common fallback paths."""
    normalized_exclude = convert_to_docker_mount_path(exclude) if exclude else ""
    potential_paths = []
    default_path = get_default_ai_database_path()
    env_path = os.getenv("AI_DATABASE_PATH")
    home_path = str(Path.home() / "ai_databases")
    project_path = str(Path(__file__).parent.parent / "data" / "ai_databases")

    for raw in {default_path, env_path, home_path, project_path}:
        if not raw:
            continue
        normalized = convert_to_docker_mount_path(raw)
        if not normalized or normalized == normalized_exclude:
            continue
        chroma_candidate = Path(normalized) / "knowledge_hub_db"
        if chroma_candidate.is_dir():
            return normalized
    return None


def validate_database(db_path, silent=False):
    """Validate database exists and return ChromaDB collection info"""
    import os
    if not db_path or not db_path.strip():
        if not silent:
            st.warning("Database path is not configured.")
        return None

    path_candidates = get_candidate_db_paths(db_path)
    selected_label = None
    selected_base_path = None
    chroma_db_path = None
    attempted = []

    for label, base_path in path_candidates:
        candidate = os.path.join(base_path, "knowledge_hub_db")
        if os.path.isdir(candidate):
            selected_label = label
            selected_base_path = base_path
            chroma_db_path = candidate
            break
        attempted.append((label, candidate))

    if not chroma_db_path:
        if not silent:
            # Check for stale artifacts in the first attempted path
            if attempted:
                first_label, first_path = attempted[0]
                first_base = Path(first_path).parent
                stale_artifacts = []
                for candidate in ["batch_state.json", "staging_ingestion.json", "ingestion_progress.json"]:
                    candidate_path = first_base / candidate
                    if candidate_path.exists():
                        stale_artifacts.append(str(candidate_path))
                progress_dir = first_base / "ingestion_progress"
                if progress_dir.exists():
                    stale_artifacts.append(str(progress_dir))

                # Check for fallback KB location
                safe_db_path = str(first_base)
                fallback = find_existing_kb_root(safe_db_path)
                if fallback:
                    st.session_state["kb_fallback_path"] = fallback
                    st.warning(f"🧠 Knowledge base not found, but we detected data at '{fallback}'. Update your database path to use the populated knowledge base.")
                else:
                    st.session_state.pop("kb_fallback_path", None)
                    if stale_artifacts:
                        artifacts = "\n".join(f"- {path}" for path in stale_artifacts)
                        st.warning(f"🧹 Knowledge base directory not found, and stuck ingestion files were detected:\n{artifacts}\n\nRun **Maintenance → Clean Start** to clear failed ingestion state, then re-ingest.")
                    else:
                        attempt_lines = "\n".join(f"- {lbl}: {path}" for lbl, path in attempted if path)
                        st.warning(
                            "🧠 Knowledge base directory not found in any configured location.\n"
                            f"Tried:\n{attempt_lines}"
                        )
            else:
                st.warning("🧠 Knowledge base directory not found. Please verify your database path.")
        return None
    
    logger.debug(f"Knowledge base resolved to {chroma_db_path} via {selected_label or 'configured'} lookup")
    
    if not silent and selected_label and selected_label != "configured":
        st.info(
            f"Detected knowledge base at `{selected_base_path}` via `{selected_label}` fallback. "
            "Update the sidebar path to this location for full Docker compatibility."
        )
        
    try:
        # Clear any cached ChromaDB connections
        import gc
        gc.collect()
        
        # Simple ChromaDB validation without LlamaIndex
        db_settings = ChromaSettings(
            anonymized_telemetry=False
        )
        client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        
        # Try to get existing collection or create if it doesn't exist
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception as get_e:
            if "does not exist" in str(get_e):
                if not silent:
                    st.warning(f"⚠️ Knowledge base collection '{COLLECTION_NAME}' does not exist. Please run Knowledge Ingest to populate the database.")
                return None
            else:
                raise get_e
        
        # Log collection info
        collection_count = collection.count()
        logger.info(f"Collection '{COLLECTION_NAME}' has {collection_count} documents")
        
        if collection_count > 0:
            if not silent:
                st.success(f"✅ Knowledge base validated: {collection_count} documents available for direct search.")
            return {"path": chroma_db_path, "count": collection_count, "collection": collection}
        else:
            if not silent:
                st.warning("⚠️ Database collection exists but no documents found. Please run Knowledge Ingest to add documents.")
            return None
            
    except Exception as e:
        # Check if this is a collections schema error (development database accessed from Docker)
        if "collections.config_json_str" in str(e):
            if not silent:
                st.warning("🐳 **Development database detected in Docker environment**")
                st.info("📝 **This database was created outside Docker and has schema differences.**\n\n💡 **Options:**\n- Use a separate Docker database path (recommended)\n- Or continue with limited collection features")
            logger.warning(f"Development database schema conflict in Docker: {e}")
            
            # Try to continue with ChromaDB-only access (skip collections)
            try:
                db_settings = ChromaSettings(
                    anonymized_telemetry=False
                )
                client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
                collection = client.get_collection(COLLECTION_NAME)
                collection_count = collection.count()
                
                if collection_count > 0:
                    if not silent:
                        st.success(f"✅ ChromaDB access successful: {collection_count} documents available (collections disabled)")
                    return {"path": chroma_db_path, "count": collection_count, "collection": collection, "collections_disabled": True}
                    
            except Exception as chroma_e:
                if not silent:
                    st.error(f"❌ ChromaDB access also failed: {chroma_e}")
                logger.error(f"ChromaDB access failed: {chroma_e}")
                return None
        else:
            # Handle Chroma tenant/database errors gracefully
            if "tenant" in str(e) or "default_tenant" in str(e):
                msg = (
                    "❌ Database validation failed: Chroma tenant/database mismatch.\n\n"
                    "This can happen after a Chroma upgrade or copying DBs between environments.\n"
                    "Try: deleting the 'knowledge_hub_db' folder under your AI database path and re-ingesting."
                )
                if not silent:
                    st.error(msg)
                logger.error(f"Chroma tenant error at {chroma_db_path}: {e}")
                return None
            if not silent:
                st.error(f"❌ Database validation failed: {e}")
            logger.error(f"Error validating database at {chroma_db_path}: {e}")
            return None


def initialize_search_state():
    """Initialize search-related session state"""
    if 'search_sort_key' not in st.session_state:
        st.session_state.search_sort_key = 'score'
    if 'search_sort_order' not in st.session_state:
        st.session_state.search_sort_order = 'desc'
    if 'last_search_query' not in st.session_state:
        st.session_state.last_search_query = ""
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = []
    # Boolean logic state
    if 'doc_type_filter' not in st.session_state:
        st.session_state.doc_type_filter = "Any"
    if 'outcome_filter' not in st.session_state:
        st.session_state.outcome_filter = "Any"
    if 'thematic_tag_filter' not in st.session_state:
        st.session_state.thematic_tag_filter = []
    if 'filter_operator' not in st.session_state:
        st.session_state.filter_operator = "AND"
    if 'search_scope' not in st.session_state:
        st.session_state.search_scope = "Entire Knowledge Base"
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = "default"
    # Reranker toggle - defaults to config value
    if 'reranker_enabled' not in st.session_state:
        st.session_state.reranker_enabled = QWEN3_VL_RERANKER_ENABLED
    if 'reranker_top_k' not in st.session_state:
        st.session_state.reranker_top_k = QWEN3_VL_RERANKER_TOP_K


def render_sidebar():
    """Render the sidebar with database configuration and filters"""
    st.sidebar.header("⚙️ Configuration")
    
    # Database path configuration - Docker-safe
    from cortex_engine.session_state import initialize_app_session_state
    initialize_app_session_state()
    
    # Handle Docker environment gracefully
    current_path = ""
    try:
        from cortex_engine.config_manager import ConfigManager
        config_manager = ConfigManager()
        current_config = config_manager.get_config()
        current_path = current_config.get("ai_database_path", "")
        resolved_current = resolve_db_root_path(current_path)
        if resolved_current:
            current_path = str(resolved_current)
    except Exception as e:
        logger.warning(f"Config manager not available: {e}")
        # Use proper default path detection
        from cortex_engine.utils.default_paths import get_default_ai_database_path
        current_path = get_default_ai_database_path()
    
    st.sidebar.subheader("📁 Database Storage Path")
    
    # Editable path input
    new_path = st.sidebar.text_input(
        "Database Path:", 
        value=current_path,
        help="Enter the path to your AI databases directory",
        key="db_path_input_widget"
    )
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Update Path", help="Save the new database path"):
            if new_path.strip():
                # Convert and validate path
                normalized_root = resolve_db_root_path(new_path.strip())
                wsl_path = convert_windows_to_wsl_path(str(normalized_root)) if normalized_root else convert_windows_to_wsl_path(new_path.strip())
                
                if os.path.exists(wsl_path):
                    final_path = str(normalized_root) if normalized_root else wsl_path
                    # Update session state
                    st.session_state.db_path_input = final_path
                    
                    # Save to persistent config file - Docker-safe
                    try:
                        config_manager.update_config({"ai_database_path": final_path})
                        st.sidebar.success("✅ Database path updated and saved!")
                    except Exception as config_e:
                        logger.warning(f"Could not save config (Docker environment?): {config_e}")
                        st.sidebar.info("🐳 Running in Docker mode - path updated for this session")
                        
                    # Clear cache to force reload
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Path does not exist: {wsl_path}")
            else:
                st.sidebar.error("❌ Please enter a valid path")
    
    with col2:
        if st.button("🔄 Reset", help="Reset to default path"):
            # Use cross-platform default path detection - Docker-safe
            try:
                from cortex_engine.utils.default_paths import get_default_ai_database_path
                default_path = get_default_ai_database_path()
            except Exception:
                # Use proper default path detection
                from cortex_engine.utils.default_paths import get_default_database_path
                default_path = get_default_database_path()
            
            resolved_default = resolve_db_root_path(default_path)
            st.session_state.db_path_input = str(resolved_default) if resolved_default else default_path
            try:
                config_manager.update_config({"ai_database_path": default_path})
                st.sidebar.success("✅ Path reset to default!")
            except Exception as e:
                logger.warning(f"Could not save config (Docker environment?): {e}")
                st.sidebar.info("🐳 Running in Docker mode - using default Docker path")
            st.cache_resource.clear()
            st.rerun()
    
    # Path validation status
    if current_path:
        wsl_path = convert_windows_to_wsl_path(current_path)
        exists = os.path.exists(wsl_path)
        
        if exists:
            try:
                db_dir = os.path.join(wsl_path, 'knowledge_hub_db')
                db_exists = os.path.exists(db_dir)
                if db_exists:
                    # Count documents
                    try:
                        db_settings = ChromaSettings(
                            anonymized_telemetry=False
                        )
                        client = chromadb.PersistentClient(path=db_dir, settings=db_settings)
                        collections = client.list_collections()
                        if collections:
                            total_docs = sum(collection.count() for collection in collections)
                            st.sidebar.success(f"✅ Database found: {total_docs} documents")
                        else:
                            st.sidebar.info("📂 Database directory exists but no collections found")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Could not read database: {e}")
                else:
                    st.sidebar.warning("⚠️ Database directory not found")
            except Exception as e:
                st.sidebar.error(f"❌ Path validation error: {e}")
        else:
            st.sidebar.error(f"❌ Path does not exist")

    fallback_detected = st.session_state.get("kb_fallback_path")
    if fallback_detected and fallback_detected != current_path:
        st.sidebar.warning(f"Detected populated knowledge base at `{fallback_detected}`.")
        if st.sidebar.button("Use Detected Knowledge Base", key="apply_kb_fallback"):
            st.session_state.db_path_input = fallback_detected
            try:
                config_manager.update_config({"ai_database_path": fallback_detected})
                st.sidebar.success("✅ Updated to detected knowledge base path.")
            except Exception as e:
                logger.warning(f"Failed to save fallback path: {e}")
                st.sidebar.info("Path updated for this session only.")
            st.session_state.pop("kb_fallback_path", None)
            st.cache_resource.clear()
            st.rerun()
    
    st.sidebar.divider()
    
    # Search filters
    st.sidebar.subheader("🔍 Search Filters")
    
    # Document type filter
    doc_types = get_document_type_options()
    doc_type_options = ["Any", "Project Plan", "Technical Documentation", "Proposal/Quote", 
                       "Case Study / Trophy", "Final Report", "Draft Report", "Presentation", 
                       "Contract/SOW", "Meeting Minutes", "Financial Report", "Research Paper", 
                       "Email Correspondence", "Other"] + ([dt for dt in doc_types if dt not in [
                       "Project Plan", "Technical Documentation", "Proposal/Quote", "Case Study / Trophy",
                       "Final Report", "Draft Report", "Presentation", "Contract/SOW",
                       "Meeting Minutes", "Financial Report", "Research Paper", "Email Correspondence", "Other"]])
    
    st.sidebar.selectbox(
        "📄 Document Type:",
        options=doc_type_options,
        key="doc_type_filter",
        help="Filter by document type. Select 'Any' to include all types."
    )
    
    # Proposal outcome filter
    st.sidebar.selectbox(
        "📊 Proposal Outcome:",
        options=["Any", "Won", "Lost", "Pending", "N/A"],
        key="outcome_filter",
        help="Filter by proposal result. Useful for finding successful patterns or analyzing losses."
    )

    # Thematic tags filter
    try:
        tag_options = get_available_thematic_tags(current_path)
    except Exception:
        tag_options = []
    st.sidebar.multiselect(
        "🏷️ Thematic Tags:",
        options=tag_options,
        key="thematic_tag_filter",
        help="Filter by one or more metadata tags."
    )
    
    # Boolean operator (only show if multiple filters are active)
    num_active_filters = sum([
        1 if st.session_state.doc_type_filter != "Any" else 0,
        1 if st.session_state.outcome_filter != "Any" else 0,
        1 if st.session_state.thematic_tag_filter else 0,
    ])
    
    if num_active_filters >= 2:
        st.sidebar.radio(
            "🔗 Filter Operator:",
            ["AND", "OR"],
            key="filter_operator",
            help="**AND**: Document must match ALL selected filters. **OR**: Document matches ANY selected filter."
        )
    else:
        # Ensure operator exists even if not shown
        if 'filter_operator' not in st.session_state:
            st.session_state.filter_operator = "AND"
    
    # Search scope
    st.sidebar.radio(
        "🎯 Search Scope:",
        ["Entire Knowledge Base", "Active Collection"],
        key="search_scope",
        help="**Entire Knowledge Base**: Search all documents. **Active Collection**: Search only documents in the currently selected collection."
    )
    
    # Collection management (only show if Active Collection is selected)
    if st.session_state.search_scope == "Active Collection":
        st.sidebar.subheader("📚 Working Collections")
        
        # Get available collections - Docker-safe
        try:
            collection_mgr = WorkingCollectionManager()
            collection_names = collection_mgr.get_collection_names()
            
            if collection_names:
                st.sidebar.selectbox(
                    "Active Collection:",
                    options=collection_names,
                    key="selected_collection",
                    help="Select which collection to search within."
                )
            else:
                st.sidebar.info("No collections found. Create one in Collection Management.")
                st.session_state.selected_collection = "default"
        except Exception as e:
            logger.warning(f"Collections not available: {e}")
            st.sidebar.info("Collections not available in this environment. Searching entire knowledge base.")
            st.session_state.search_scope = "Entire Knowledge Base"
            st.session_state.selected_collection = "default"
    
    db_path_value = st.session_state.get('db_path_input', current_path)
    resolved_db_value = resolve_db_root_path(db_path_value)
    normalized_db_value = str(resolved_db_value) if resolved_db_value else db_path_value

    # Embedding & Reranker Info Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Search Engine")

    # Show current embedding info and model selector
    try:
        from cortex_engine.embedding_service import get_embedding_info, is_multimodal_enabled
        from cortex_engine.config import QWEN3_VL_MODEL_SIZE, QWEN3_VL_ENABLED

        embed_info = get_embedding_info()
        is_multimodal = is_multimodal_enabled()

        # Detect database dimensions for compatibility check
        db_dimensions = None
        try:
            chroma_path = os.path.join(convert_windows_to_wsl_path(normalized_db_value), "knowledge_hub_db")
            if os.path.exists(chroma_path):
                import chromadb
                from chromadb.config import Settings as ChromaSettings
                db_settings = ChromaSettings(anonymized_telemetry=False)
                client = chromadb.PersistentClient(path=chroma_path, settings=db_settings)
                collection = client.get_collection(COLLECTION_NAME)
                if collection.count() > 0:
                    sample = collection.peek(limit=1)
                    if sample.get('embeddings') and len(sample['embeddings']) > 0:
                        db_dimensions = len(sample['embeddings'][0])
        except Exception:
            pass

        if is_multimodal and QWEN3_VL_ENABLED:
            st.sidebar.success("🔮 **Qwen3-VL Active**")

            # Model size selector for Qwen3-VL
            current_size = QWEN3_VL_MODEL_SIZE.upper() if QWEN3_VL_MODEL_SIZE != "auto" else "AUTO"

            # Determine compatible sizes based on database
            size_options = ["2B", "8B"]
            size_labels = {
                "2B": "2B (2048D, 5GB VRAM)",
                "8B": "8B (4096D, 16GB VRAM)"
            }

            # Show database compatibility warning
            if db_dimensions:
                if db_dimensions == 2048:
                    st.sidebar.info(f"📊 DB: {db_dimensions}D → Use **2B**")
                    recommended = "2B"
                elif db_dimensions == 4096:
                    st.sidebar.info(f"📊 DB: {db_dimensions}D → Use **8B**")
                    recommended = "8B"
                else:
                    st.sidebar.warning(f"📊 DB: {db_dimensions}D (non-Qwen)")
                    recommended = None
            else:
                recommended = None

            # Model size selector
            current_idx = size_options.index(current_size) if current_size in size_options else 0
            selected_size = st.sidebar.selectbox(
                "Model Size:",
                options=size_options,
                index=current_idx,
                format_func=lambda x: size_labels.get(x, x),
                key="search_qwen_model_size",
                help="Select model size matching your database dimensions"
            )

            # Show mismatch warning
            if db_dimensions and recommended and selected_size != recommended:
                st.sidebar.error(f"⚠️ Mismatch! DB needs {recommended}")

            # Apply button if changed
            if selected_size != current_size and current_size != "AUTO":
                if st.sidebar.button("🔄 Apply Model Change", type="primary", use_container_width=True, key="apply_search_model"):
                    os.environ["QWEN3_VL_MODEL_SIZE"] = selected_size
                    try:
                        from cortex_engine.qwen3_vl_embedding_service import reset_service
                        from cortex_engine.config import invalidate_embedding_cache
                        reset_service()
                        invalidate_embedding_cache()
                        st.sidebar.success(f"✅ Switched to {selected_size}")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"❌ Failed: {e}")
            elif current_size == "AUTO":
                # First time - allow setting
                if st.sidebar.button(f"🔄 Set to {selected_size}", type="primary", use_container_width=True, key="set_search_model"):
                    os.environ["QWEN3_VL_MODEL_SIZE"] = selected_size
                    try:
                        from cortex_engine.qwen3_vl_embedding_service import reset_service
                        from cortex_engine.config import invalidate_embedding_cache
                        reset_service()
                        invalidate_embedding_cache()
                        st.sidebar.success(f"✅ Set to {selected_size}")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"❌ Failed: {e}")

            st.sidebar.caption(f"Current: {embed_info.get('embedding_dimension', 'Unknown')}D")
        else:
            model_name = embed_info.get('model_name', EMBED_MODEL)
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            st.sidebar.info(f"📊 **{short_name}**")

        # Reranker toggle control
        st.sidebar.markdown("---")
        reranker_toggle = st.sidebar.checkbox(
            "🎯 Neural Reranking",
            value=st.session_state.reranker_enabled,
            key="reranker_toggle",
            help="Enable Qwen3-VL neural reranking for improved search precision. "
                 "First search takes 2-4 minutes to load the model."
        )
        st.session_state.reranker_enabled = reranker_toggle

        if reranker_toggle:
            # Add slider for controlling number of results
            reranker_top_k = st.sidebar.slider(
                "Results to return:",
                min_value=5,
                max_value=50,
                value=st.session_state.reranker_top_k,
                step=5,
                key="reranker_top_k_slider",
                help="Number of top results after neural reranking"
            )
            st.session_state.reranker_top_k = reranker_top_k
            st.sidebar.caption(f"✅ Active (top-{reranker_top_k} from {QWEN3_VL_RERANKER_CANDIDATES} candidates)")

            # Check if model is already loaded
            try:
                from cortex_engine.qwen3_vl_reranker_service import _reranker_model
                if _reranker_model is not None:
                    st.sidebar.caption("🟢 Model ready")
                elif st.session_state.get('reranker_preload_started'):
                    st.sidebar.caption("🔄 Model loading in background...")
                else:
                    st.sidebar.caption("⚠️ First search loads model (~3 min)")
            except ImportError:
                st.sidebar.caption("⚠️ First search loads model (~3 min)")
        else:
            st.sidebar.caption("📊 Standard embedding search")

    except Exception as e:
        logger.debug(f"Could not get embedding info: {e}")
        st.sidebar.caption(f"📊 Model: {EMBED_MODEL.split('/')[-1]}")

    return {
        'db_path': normalized_db_value,
        'doc_type_filter': st.session_state.doc_type_filter,
        'outcome_filter': st.session_state.outcome_filter,
        'thematic_tag_filter': st.session_state.thematic_tag_filter,
        'filter_operator': st.session_state.filter_operator,
        'search_scope': st.session_state.search_scope,
        'selected_collection': st.session_state.selected_collection,
        'reranker_enabled': st.session_state.reranker_enabled,
        'reranker_top_k': st.session_state.get('reranker_top_k', QWEN3_VL_RERANKER_TOP_K)
    }


def add_search_debug(message: str):
    """Add a message to the search debug log in session state."""
    if 'search_debug_log' not in st.session_state:
        st.session_state.search_debug_log = []
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.search_debug_log.append(f"[{timestamp}] {message}")
    logger.info(message)  # Also log to standard logger


def clear_search_debug():
    """Clear the search debug log."""
    st.session_state.search_debug_log = []


def perform_search(base_index, query, filters, search_mode="Traditional Vector Search"):
    """Perform search - supports Traditional, GraphRAG Enhanced, and Hybrid modes"""
    clear_search_debug()  # Start fresh for each search
    add_search_debug(f"Starting search: mode='{search_mode}', query='{query[:50]}...'")

    try:
        if not query.strip():
            st.warning("⚠️ Please enter a search query")
            return []
        
        # Get database path from config - Docker-safe
        db_path = ""
        try:
            from cortex_engine.config_manager import ConfigManager
            config_manager = ConfigManager()
            current_config = config_manager.get_config()
            db_path = current_config.get("ai_database_path", "")
            resolved_db = resolve_db_root_path(db_path)
            if resolved_db:
                db_path = str(resolved_db)
        except Exception as e:
            logger.warning(f"Config not available: {e}")
            # Try session state as fallback
            from cortex_engine.utils.default_paths import get_default_ai_database_path
            db_path = st.session_state.get('db_path_input', get_default_ai_database_path())
            resolved_db = resolve_db_root_path(db_path)
            if resolved_db:
                db_path = str(resolved_db)
        
        if not db_path:
            st.error("❌ Database path not configured")
            return []
        
        add_search_debug(f"Database path resolved: {db_path}")

        # Choose search strategy based on mode
        if search_mode == "GraphRAG Enhanced":
            add_search_debug("Executing GraphRAG Enhanced search...")
            results = graphrag_enhanced_search(db_path, query, filters)
            add_search_debug(f"GraphRAG search returned {len(results)} results")
            return results
        elif search_mode == "Hybrid Search":
            add_search_debug("Executing Hybrid (Vector + GraphRAG) search...")
            results = hybrid_search(db_path, query, filters)
            add_search_debug(f"Hybrid search returned {len(results)} results")
            return results
        else:
            # Traditional Vector Search (default)
            add_search_debug("Executing Traditional Vector search...")
            results = direct_chromadb_search(db_path, query, filters)
            add_search_debug(f"Traditional search returned {len(results)} results")
            return results
            
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        logger.error(f"Search error: {e}", exc_info=True)
        return []


def graphrag_enhanced_search(db_path, query, filters, top_k=20):
    """GraphRAG enhanced search using knowledge graph context.

    Strategy: Use direct_chromadb_search for vector retrieval (reliable),
    then enhance results with GraphRAG entity/relationship context.
    """
    import concurrent.futures

    def graphrag_search_with_timeout():
        try:
            add_search_debug("Starting GraphRAG Enhanced search...")

            # Step 1: Get vector results using direct_chromadb_search (reliable)
            add_search_debug("Step 1: Getting vector results via direct_chromadb_search...")
            vector_results = direct_chromadb_search(db_path, query, filters, top_k)
            add_search_debug(f"Vector search returned {len(vector_results)} results")

            if not vector_results:
                add_search_debug("No vector results found")
                return []

            # Step 2: Initialize GraphRAG for context enhancement
            add_search_debug("Step 2: Initializing GraphRAG for context enhancement...")
            try:
                from cortex_engine.graphrag_integration import get_graphrag_integration
                graphrag = get_graphrag_integration(db_path)

                # Get graph statistics
                health = graphrag.health_check()
                add_search_debug(f"GraphRAG health: {health.get('graph_nodes', 0)} nodes, {health.get('graph_edges', 0)} edges")

            except Exception as e:
                add_search_debug(f"GraphRAG init failed: {e} - returning vector results only")
                for result in vector_results:
                    result['graph_context'] = 'GraphRAG unavailable'
                return vector_results

            # Step 3: Enhance each result with graph context
            add_search_debug("Step 3: Enhancing results with graph context...")
            enhanced_results = []

            for result in vector_results:
                doc_id = result.get('doc_id', result.get('file_name', ''))

                # Try to get graph context for this document
                try:
                    # Get related entities and relationships
                    related_docs = graphrag.find_related_documents(doc_id, max_results=3)

                    if related_docs:
                        related_names = [d.get('file_name', d.get('doc_id', ''))[:30] for d in related_docs[:3]]
                        result['graph_context'] = f"Related: {', '.join(related_names)}"
                        result['graph_related_count'] = len(related_docs)
                    else:
                        result['graph_context'] = 'No graph relationships found'
                        result['graph_related_count'] = 0

                except Exception as e:
                    result['graph_context'] = 'Graph lookup failed'
                    result['graph_related_count'] = 0

                result['search_source'] = 'GraphRAG Enhanced'
                enhanced_results.append(result)

            add_search_debug(f"Enhanced {len(enhanced_results)} results with graph context")
            return enhanced_results

        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            add_search_debug(f"ERROR: {e} - falling back to traditional search")
            return direct_chromadb_search(db_path, query, filters, top_k)

    # Run GraphRAG search with timeout protection
    # Longer timeout when reranker enabled (first model load takes 3-4 minutes)
    use_reranker = st.session_state.get('reranker_enabled', QWEN3_VL_RERANKER_ENABLED)
    graphrag_timeout = 300 if use_reranker else 45
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(graphrag_search_with_timeout)
            try:
                results = future.result(timeout=graphrag_timeout)

                # Debug logging AFTER thread completes (session_state safe here)
                add_search_debug(f"GraphRAG thread returned {len(results) if results else 0} results")
                if results and len(results) > 0:
                    first = results[0]
                    add_search_debug(f"First result keys: {list(first.keys()) if isinstance(first, dict) else 'not a dict'}")
                    if isinstance(first, dict):
                        add_search_debug(f"First result file_name: {first.get('file_name', 'MISSING')}")
                        add_search_debug(f"First result file_path: {first.get('file_path', 'MISSING')}")
                        text_preview = first.get('text', '')[:100] if first.get('text') else 'NO TEXT'
                        add_search_debug(f"First result text preview: {text_preview}...")

                return results
            except concurrent.futures.TimeoutError:
                logger.error(f"GraphRAG search timed out after {graphrag_timeout} seconds")
                if use_reranker:
                    st.error("⏱️ GraphRAG search timed out. The reranker model may still be loading - try again.")
                else:
                    st.error("⏱️ GraphRAG search timed out. Falling back to traditional search.")
                return direct_chromadb_search(db_path, query, filters, top_k)
    except Exception as e:
        logger.error(f"GraphRAG search wrapper failed: {e}")
        return direct_chromadb_search(db_path, query, filters, top_k)


def hybrid_search(db_path, query, filters, top_k=20):
    """Hybrid search combining vector and GraphRAG results"""
    import concurrent.futures
    
    def hybrid_search_with_timeout():
        try:
            from cortex_engine.graphrag_integration import get_graphrag_integration
            
            # Get both search results (do not halve top_k)
            # Start with vector results so Hybrid never returns fewer than Traditional
            vector_results = direct_chromadb_search(db_path, query, filters, top_k)
            
            try:
                graphrag = get_graphrag_integration(db_path)
                wsl_db_path = convert_windows_to_wsl_path(db_path)
                chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
                
                # Create vector index interface
                vector_index = ChromaVectorIndex(chroma_db_path)
                
                # Get GraphRAG results
                graphrag_raw = graphrag.enhanced_search(
                    query=query,
                    vector_index=vector_index,
                    use_graph_context=True,
                    max_hops=2
                )
                
                # Format GraphRAG results
                graphrag_results = []
                for i, result in enumerate(graphrag_raw[:top_k]):
                    if hasattr(result, 'text'):
                        content = result.text
                        metadata = getattr(result, 'metadata', {})
                    else:
                        content = str(result)
                        metadata = {}
                    
                    formatted_result = {
                        'rank': i + 1,
                        'score': metadata.get('score', 0.75),
                        'text': content,
                        'file_path': metadata.get('file_path', 'Unknown'),
                        'file_name': metadata.get('file_name', 'Unknown'),
                        'document_type': metadata.get('document_type', 'Unknown'),
                        'proposal_outcome': metadata.get('proposal_outcome', 'N/A'),
                        'thematic_tags': metadata.get('thematic_tags', metadata.get('tags', [])),
                        'chunk_id': metadata.get('chunk_id', f'graphrag_chunk_{i}'),
                        'doc_id': metadata.get('doc_id', f'graphrag_doc_{i}'),
                        'search_source': 'GraphRAG Enhanced'
                    }
                    
                    if apply_post_search_filters(formatted_result, filters):
                        graphrag_results.append(formatted_result)
                
            except Exception as e:
                logger.warning(f"GraphRAG portion of hybrid search failed: {e}")
                graphrag_results = []
            
            # Mark vector results
            for result in vector_results:
                result['search_source'] = 'Vector Search'
            
            # Combine: start with vector (ensures at least vector count), then add unique GraphRAG
            combined_by_id = {r['doc_id']: r for r in vector_results}
            for r in graphrag_results:
                if r['doc_id'] not in combined_by_id:
                    combined_by_id[r['doc_id']] = r

            # Sort by score, cap to top_k, update ranks
            final_results = list(combined_by_id.values())
            final_results.sort(key=lambda x: x['score'], reverse=True)
            final_results = final_results[:top_k]
            
            # Update ranks
            for i, result in enumerate(final_results):
                result['rank'] = i + 1
            
            logger.info(f"Hybrid search returned {len(final_results)} combined results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            st.warning(f"⚠️ Hybrid search failed: {e}. Using traditional search.")
            return direct_chromadb_search(db_path, query, filters, top_k)
    
    # Run hybrid search with timeout protection
    # Longer timeout when reranker enabled (first model load takes 3-4 minutes)
    use_reranker = st.session_state.get('reranker_enabled', QWEN3_VL_RERANKER_ENABLED)
    hybrid_timeout = 300 if use_reranker else 60
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(hybrid_search_with_timeout)
            try:
                results = future.result(timeout=hybrid_timeout)
                return results
            except concurrent.futures.TimeoutError:
                logger.error(f"Hybrid search timed out after {hybrid_timeout} seconds")
                if use_reranker:
                    st.error("⏱️ Hybrid search timed out. The reranker model may still be loading - try again.")
                else:
                    st.error("⏱️ Hybrid search timed out. Falling back to traditional search.")
                return direct_chromadb_search(db_path, query, filters, top_k)
    except Exception as e:
        logger.error(f"Hybrid search wrapper failed: {e}")
        return direct_chromadb_search(db_path, query, filters, top_k)


def apply_post_search_filters(result, filters):
    """Apply post-search filtering logic"""
    if not filters or not isinstance(filters, dict):
        return True
    
    # Document type filter
    doc_type_filter = filters.get('doc_type_filter')
    outcome_filter = filters.get('outcome_filter')
    filter_operator = filters.get('filter_operator', 'AND')
    tag_filter = filters.get('thematic_tag_filter', [])
    
    # Check individual filter conditions
    doc_type_match = (doc_type_filter == "Any" or 
                     result.get('document_type') == doc_type_filter)
    
    outcome_match = (outcome_filter == "Any" or 
                    result.get('proposal_outcome') == outcome_filter)

    # Tags: require all selected tags by default
    tag_match = True
    if tag_filter:
        raw_tags = result.get('thematic_tags') or result.get('tags') or result.get('metadata', {}).get('thematic_tags')
        parsed = []
        if isinstance(raw_tags, str):
            parsed = [p.strip() for p in raw_tags.split(",") if p.strip()]
        elif isinstance(raw_tags, list):
            parsed = [str(p).strip() for p in raw_tags if str(p).strip()]
        tag_match = set(tag_filter).issubset(set(parsed))
    
    # Apply boolean logic
    if filter_operator == "AND":
        return doc_type_match and outcome_match and tag_match
    else:  # OR operator
        return (
            doc_type_match
            or outcome_match
            or tag_match
            or (doc_type_filter == "Any" and outcome_filter == "Any" and not tag_filter)
        )


# Simple ChromaDB Vector Index interface for GraphRAG compatibility
class ChromaVectorIndex:
    """Simple interface to make ChromaDB compatible with GraphRAG enhanced search"""
    def __init__(self, chroma_db_path):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from cortex_engine.config import COLLECTION_NAME

        db_settings = ChromaSettings(
            anonymized_telemetry=False
        )
        self.client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        self.collection = self.client.get_collection(COLLECTION_NAME)

    def as_retriever(self, similarity_top_k=10):
        """Create a retriever interface for GraphRAG"""
        return ChromaRetriever(self.collection, similarity_top_k)


class ChromaRetriever:
    """Simple retriever interface for GraphRAG compatibility"""
    def __init__(self, collection, top_k=10):
        self.collection = collection
        self.top_k = top_k

    def retrieve(self, query):
        """Retrieve documents for GraphRAG with text-based fallback"""
        import sys
        retrieved_results = []

        def debug_print(msg):
            print(f"[ChromaRetriever] {msg}", file=sys.stderr, flush=True)

        # Strategy 1: Try embedding-based vector search
        try:
            debug_print(f"Attempting embedding-based search for '{query[:30]}...'")
            query_embedding = embed_query(query)

            if query_embedding and len(query_embedding) > 0:
                debug_print(f"Embedding has {len(query_embedding)} dimensions")

                # Check collection's expected dimensions
                try:
                    peek = self.collection.peek(limit=1)
                    if peek.get('embeddings') and len(peek['embeddings']) > 0:
                        stored_dim = len(peek['embeddings'][0])
                        debug_print(f"Collection has {stored_dim}-dim embeddings stored")
                        if stored_dim != len(query_embedding):
                            debug_print(f"DIMENSION MISMATCH! Query={len(query_embedding)}, Stored={stored_dim}")
                except Exception as peek_e:
                    debug_print(f"Could not peek collection: {peek_e}")

                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.top_k
                )

                if results and results.get('documents') and results['documents'][0]:
                    documents = results['documents'][0]
                    metadatas = results['metadatas'][0] if results['metadatas'] else []
                    distances = results['distances'][0] if results['distances'] else []

                    debug_print(f"Vector search SUCCESS: {len(documents)} results")

                    for doc, metadata, distance in zip(documents, metadatas, distances):
                        result = type('RetrievalResult', (), {
                            'text': doc,
                            'metadata': metadata,
                            'score': 1.0 - distance
                        })()
                        retrieved_results.append(result)
                    return retrieved_results
                else:
                    debug_print("Vector search returned no documents")
            else:
                debug_print("embed_query returned empty embedding")

        except Exception as e:
            debug_print(f"Vector search FAILED: {e}")

        # Strategy 2: Text-based fallback (like direct_chromadb_search)
        debug_print("Falling back to text-based search...")
        try:
            all_results = self.collection.get(limit=min(2000, self.top_k * 20))
            query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
            debug_print(f"Loaded {len(all_results.get('documents', []))} docs, searching for terms: {query_terms}")

            documents = all_results.get('documents', [])
            metadatas = all_results.get('metadatas', [])

            matching_docs = []
            for doc, metadata in zip(documents, metadatas):
                doc_lower = doc.lower()
                matches = sum(1 for term in query_terms if term in doc_lower)
                if matches > 0:
                    score = matches / len(query_terms) if query_terms else 0.5
                    matching_docs.append((doc, metadata, score, matches))

            # Sort by matches and score
            matching_docs.sort(key=lambda x: (x[3], x[2]), reverse=True)

            for doc, metadata, score, _ in matching_docs[:self.top_k]:
                result = type('RetrievalResult', (), {
                    'text': doc,
                    'metadata': metadata,
                    'score': score
                })()
                retrieved_results.append(result)

            debug_print(f"Text-based fallback returned {len(retrieved_results)} results")

        except Exception as e:
            debug_print(f"Text-based fallback FAILED: {e}")

        return retrieved_results


def direct_chromadb_search(db_path, query, filters, top_k=20, use_reranker=None):
    """
    Perform direct ChromaDB search bypassing LlamaIndex entirely.

    Supports optional Qwen3-VL neural reranking for improved precision.
    This resolves the ChromaDB where clause issues.

    Args:
        db_path: Path to the database
        query: Search query text
        filters: Filter options dict
        top_k: Number of results to return
        use_reranker: Enable neural reranking (default from config)

    Returns:
        List of search result dicts
    """
    import time
    import concurrent.futures

    # Determine reranking behavior: UI toggle > explicit param > config default
    if use_reranker is None:
        # Check session state (UI toggle) first, fall back to config
        use_reranker = st.session_state.get('reranker_enabled', QWEN3_VL_RERANKER_ENABLED)

    # Get reranker settings from session state (UI slider) or config
    reranker_top_k = st.session_state.get('reranker_top_k', QWEN3_VL_RERANKER_TOP_K)

    # Get more candidates if reranking is enabled
    candidate_count = QWEN3_VL_RERANKER_CANDIDATES if use_reranker else top_k
    final_top_k = reranker_top_k if use_reranker else top_k

    def search_with_timeout():
        try:
            logger.info(f"Starting ChromaDB search for query: '{query}'")
            wsl_db_path = convert_windows_to_wsl_path(db_path)
            chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
            
            if not os.path.isdir(chroma_db_path):
                st.error(f"Database not found: {chroma_db_path}")
                return []
            
            # Direct ChromaDB client
            logger.info("Connecting to ChromaDB client...")
            db_settings = ChromaSettings(
                anonymized_telemetry=False
            )
            client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
            collection = client.get_collection(COLLECTION_NAME)
            logger.info(f"Connected to collection '{COLLECTION_NAME}'")

            # Validate embedding model compatibility before querying
            try:
                from cortex_engine.utils.embedding_validator import (
                    validate_embedding_compatibility,
                    EmbeddingModelMismatchError
                )
                from cortex_engine.collection_manager import WorkingCollectionManager
                from cortex_engine.config import EMBED_MODEL

                collection_mgr = WorkingCollectionManager()
                collection_metadata = collection_mgr.get_embedding_model_metadata("default")

                validation_result = validate_embedding_compatibility(
                    collection_metadata,
                    current_model=EMBED_MODEL,
                    strict=False  # Don't raise exception, just warn
                )

                if not validation_result["compatible"]:
                    for error in validation_result["errors"]:
                        logger.error(error)
                        st.warning(error)
                    st.warning("⚠️ **Embedding model mismatch detected!** Search results may be unreliable.")
                    st.info("💡 **Solution:** Set `CORTEX_EMBED_MODEL` environment variable or delete database and re-ingest.")
                elif validation_result["warnings"]:
                    for warning in validation_result["warnings"]:
                        logger.warning(warning)

            except Exception as validation_error:
                logger.warning(f"Could not validate embedding model (non-critical): {validation_error}")

            # Try simple get first to test collection access
            logger.info("Testing collection access...")
            try:
                peek = collection.peek(limit=1)
                logger.info(f"Collection peek successful, has embeddings: {bool(peek.get('embeddings'))}")
            except Exception as peek_e:
                logger.warning(f"Collection peek failed: {peek_e}")
            
            # Multi-strategy search approach for better results
            logger.info("Executing multi-strategy ChromaDB search...")
            
            # Strategy 1: Try vector search with original query (explicit embeddings)
            results = None
            search_strategy = "vector"
            
            try:
                q_emb = embed_query(query)
                results = collection.query(
                    query_embeddings=[q_emb],
                    n_results=candidate_count
                )
                logger.info("ChromaDB vector query completed successfully (embeddings)")
                
                # Check if we got good results
                if results and results.get('documents') and results['documents'][0]:
                    search_strategy = "vector"
                else:
                    results = None  # Try other strategies
                    
            except Exception as vector_e:
                logger.warning(f"Vector search failed (embedding mismatch?): {vector_e}")
                # Skip vector search if embedding dimensions don't match
                results = None
            
            # Strategy 2: Skip vector-based searches and go directly to text-based search
            # (Since we have an embedding dimension mismatch)
            
            # Strategy 3: Text-based search (our main fallback)
            if not results or not results.get('documents') or not results['documents'][0]:
                logger.info("Attempting fallback text-based search...")
                
                try:
                    # Get a larger sample of documents
                    all_results = collection.get(limit=min(2000, top_k * 20))
                    
                    matching_docs = []
                    query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
                    
                    documents = all_results.get('documents', [])
                    metadatas = all_results.get('metadatas', [])
                    
                    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                        doc_lower = doc.lower()
                        
                        # Calculate match score based on term presence
                        matches = 0
                        for term in query_terms:
                            if term in doc_lower:
                                matches += 1
                        
                        # Include documents that match at least one term
                        if matches > 0:
                            # Score based on percentage of terms matched and term frequency
                            base_score = matches / len(query_terms)
                            
                            # Boost score for documents with multiple term matches
                            if matches > 1:
                                base_score *= 1.5
                            
                            matching_docs.append({
                                'doc': doc,
                                'meta': metadata,
                                'score': base_score,
                                'matches': matches
                            })
                    
                    # Sort by score and matches
                    if matching_docs:
                        matching_docs.sort(key=lambda x: (x['matches'], x['score']), reverse=True)
                        
                        results = {
                            'documents': [[item['doc'] for item in matching_docs[:top_k]]],
                            'metadatas': [[item['meta'] for item in matching_docs[:top_k]]],
                            'distances': [[1.0 - item['score'] for item in matching_docs[:top_k]]]
                        }
                        
                        search_strategy = "text-fallback"
                        logger.info(f"Text fallback search found {len(matching_docs)} matches")
                    else:
                        results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
                        logger.info("No text matches found in fallback search")
                        
                except Exception as fallback_e:
                    logger.error(f"Fallback search also failed: {fallback_e}")
                    return []
            
            # Log the strategy used
            if results and results.get('documents') and results['documents'][0]:
                logger.info(f"Search completed using '{search_strategy}' strategy with {len(results['documents'][0])} results")
            else:
                logger.warning("All search strategies failed to find results")
            
            # Apply collection scope filtering if needed - SAFE post-search approach
            collection_doc_ids = None
            if filters and filters.get('search_scope') == "Active Collection":
                try:
                    collection_mgr = WorkingCollectionManager()
                    selected_collection = filters.get('selected_collection', 'default')
                    collection_obj = collection_mgr.collections.get(selected_collection, {})
                    collection_doc_ids = set(collection_obj.get("doc_ids", []))
                    logger.info(f"Collection '{selected_collection}' has {len(collection_doc_ids)} documents")
                except Exception as e:
                    logger.warning(f"Could not load collection scope (Docker environment?): {e}")
                    # In Docker/fresh environment, collections may not exist - skip collection filtering
                    collection_doc_ids = None
                    filters['search_scope'] = "Entire Knowledge Base"  # Force full search
            
            # Format results
            formatted_results = []
            if results and results.get('documents'):
                documents = results['documents'][0] if results['documents'] else []
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                distances = results['distances'][0] if results['distances'] else []
                
                logger.info(f"Processing {len(documents)} raw results...")
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    result = {
                        'rank': i + 1,
                        'score': max(0, 1.0 - distance),  # Convert distance to similarity score
                        'text': doc,
                        'file_path': metadata.get('file_path', 'Unknown'),
                        'file_name': metadata.get('file_name', 'Unknown'),
                        'document_type': metadata.get('document_type', 'Unknown'),
                        'proposal_outcome': metadata.get('proposal_outcome', 'N/A'),
                        'thematic_tags': metadata.get('thematic_tags', metadata.get('tags', [])),
                        'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                        'doc_id': metadata.get('doc_id', f'doc_{i}')
                    }
                    
                    # Collection scope filter
                    if (collection_doc_ids is not None and result['doc_id'] not in collection_doc_ids):
                        continue
                    
                    if apply_post_search_filters(result, filters):
                        formatted_results.append(result)
            else:
                logger.warning("No documents returned from ChromaDB query")
            
            logger.info(f"Direct ChromaDB search returned {len(formatted_results)} formatted results")

            # Apply neural reranking for precision boost
            if use_reranker and formatted_results:
                try:
                    from cortex_engine.graph_query import rerank_search_results

                    logger.info(f"Applying Qwen3-VL neural reranking (top_k={final_top_k})")
                    reranked_results = rerank_search_results(
                        query=query,
                        results=formatted_results,
                        top_k=final_top_k,
                        text_key="text"
                    )

                    # Update ranks after reranking
                    for i, result in enumerate(reranked_results):
                        result['rank'] = i + 1

                    logger.info(f"Neural reranking complete: {len(reranked_results)} results")
                    return reranked_results

                except ImportError as e:
                    logger.warning(f"Qwen3-VL reranker not available: {e}")
                except Exception as e:
                    logger.error(f"Reranking failed, returning original results: {e}")

            return formatted_results

        except Exception as e:
            logger.error(f"Direct ChromaDB search failed: {e}", exc_info=True)
            return []
    
    try:
        # Run search with timeout to prevent hanging
        # Longer timeout when reranker enabled (first model load takes 3-4 minutes)
        search_timeout = 300 if use_reranker else 30
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(search_with_timeout)
            try:
                results = future.result(timeout=search_timeout)
                return results
            except concurrent.futures.TimeoutError:
                logger.error(f"Search timed out after {search_timeout} seconds")
                if use_reranker:
                    st.error("Search timed out. The reranker model may still be loading - try again in a moment.")
                else:
                    st.error("Search timed out. The database may be large or there may be connectivity issues.")
                return []
                
    except Exception as e:
        logger.error(f"Search executor failed: {e}")
        st.error(f"Search failed: {e}")
        return []


def show_graphrag_entity_feedback(results, filters):
    """Display entity-based feedback for GraphRAG searches"""
    try:
        # Get database path for GraphRAG integration
        db_path = ""
        try:
            from cortex_engine.config_manager import ConfigManager
            config_manager = ConfigManager()
            current_config = config_manager.get_config()
            db_path = current_config.get("ai_database_path", "")
        except Exception:
            from cortex_engine.utils.default_paths import get_default_ai_database_path
            db_path = st.session_state.get('db_path_input', get_default_ai_database_path())
        
        if not db_path:
            return
            
        from cortex_engine.graphrag_integration import get_graphrag_integration
        graphrag = get_graphrag_integration(db_path)
        
        # Get graph statistics
        stats = graphrag.get_graph_statistics()
        
        # Show entity context in an expandable section
        with st.expander("🧠 GraphRAG Entity Analysis", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Knowledge Graph", f"{stats.get('total_nodes', 0)} entities")
                st.metric("🔗 Relationships", f"{stats.get('total_edges', 0)} connections")
            
            with col2:
                entity_counts = stats.get('entity_counts', {})
                if entity_counts:
                    st.write("**Entity Types Found:**")
                    for entity_type, count in entity_counts.items():
                        if count > 0:
                            st.write(f"• {entity_type.title()}: {count}")
            
            with col3:
                # Show search source breakdown
                search_sources = {}
                for result in results:
                    source = result.get('search_source', 'Traditional')
                    search_sources[source] = search_sources.get(source, 0) + 1
                
                if search_sources:
                    st.write("**Search Sources:**")
                    for source, count in search_sources.items():
                        st.write(f"• {source}: {count}")
            
            # Additional context for GraphRAG enhanced results
            if any(result.get('graph_context') for result in results):
                st.success("✅ Results enhanced with entity relationships and graph context")
            else:
                st.info("ℹ️ Results from vector search with graph validation")
                
    except Exception as e:
        logger.warning(f"Could not show GraphRAG entity feedback: {e}")


def render_search_results(results, filters):
    """Render search results with collection management actions and entity feedback"""
    if not results:
        st.info("🔍 **No results found.**\n\n💡 **Try:**\n- Different search terms (e.g., 'strategy transformation' instead of 'strategy and transformation')\n- Checking sidebar filters - they might be too restrictive\n- Single terms first (e.g., just 'strategy')\n- Verifying your database path contains the right documents")
        return
    
    # Show GraphRAG entity feedback if available
    search_mode = st.session_state.get('search_mode_selection', 'Traditional Vector Search')
    if search_mode in ["GraphRAG Enhanced", "Hybrid Search"] and results:
        show_graphrag_entity_feedback(results, filters)
    
    # Show active filters
    active_filters = []
    if filters.get('doc_type_filter', 'Any') != 'Any':
        active_filters.append(f"Type: {filters['doc_type_filter']}")
    if filters.get('outcome_filter', 'Any') != 'Any':
        active_filters.append(f"Outcome: {filters['outcome_filter']}")
    if filters.get('thematic_tag_filter'):
        active_filters.append("Tags: " + ", ".join(filters['thematic_tag_filter']))
    if filters.get('search_scope', 'Entire Knowledge Base') == 'Active Collection':
        active_filters.append(f"Collection: {filters.get('selected_collection', 'default')}")
    
    if active_filters:
        filter_text = " | ".join(active_filters)
        if len(active_filters) > 1:
            filter_text += f" ({filters.get('filter_operator', 'AND')} logic)"
        st.info(f"🔍 Active filters: {filter_text}")

    # Count unique documents
    unique_docs = len(set(r.get('file_name', 'Unknown') for r in results))
    if unique_docs < len(results):
        st.success(f"✅ Found {len(results)} results ({unique_docs} unique documents)")
    else:
        st.success(f"✅ Found {len(results)} results")
    
    # Bulk collection actions
    if len(results) > 1:
        st.subheader("📋 Save to Collection")

        col1, col2 = st.columns([3, 1])

        with col1:
            try:
                collection_mgr = WorkingCollectionManager()
                collection_names = collection_mgr.get_collection_names()

                # Add "+ Create New Collection" as first option
                collection_options = ["+ Create New Collection"] + collection_names

                selected_option = st.selectbox(
                    "Save all results to:",
                    collection_options,
                    key="bulk_collection_target",
                    help="Select an existing collection or create a new one"
                )

                # Show new collection name input if creating new
                if selected_option == "+ Create New Collection":
                    new_name = st.text_input(
                        "New collection name:",
                        placeholder="e.g., Strategy Workshop Research",
                        key="new_collection_name_bulk"
                    )
                else:
                    new_name = None

                # Optional: Tag these results
                tag_input = st.text_input(
                    "🏷️ Tag these results (optional):",
                    placeholder="e.g., strategy, workshop, 2026",
                    key="bulk_save_tags",
                    help="Comma-separated tags to add to all saved documents"
                )

            except Exception as e:
                st.error(f"Collections not available: {e}")
                selected_option = None
                new_name = None
                tag_input = ""

        with col2:
            st.write("")  # Spacing
            st.write("")  # Align with selectbox
            if st.button("💾 Save Results", type="primary", use_container_width=True):
                if selected_option:
                    try:
                        collection_mgr = WorkingCollectionManager()
                        doc_ids = [result['doc_id'] for result in results]

                        # Determine target collection name
                        if selected_option == "+ Create New Collection":
                            if new_name and new_name.strip():
                                target_collection = new_name.strip()
                                if not collection_mgr.create_collection(target_collection):
                                    st.error(f"Collection '{target_collection}' already exists!")
                                    target_collection = None
                            else:
                                st.warning("Please enter a name for the new collection")
                                target_collection = None
                        else:
                            target_collection = selected_option

                        if target_collection:
                            # Add docs to collection
                            collection_mgr.add_docs_by_id_to_collection(target_collection, doc_ids)

                            # Apply tags if provided
                            tags_applied = 0
                            if tag_input and tag_input.strip():
                                new_tags = [t.strip() for t in tag_input.split(",") if t.strip()]
                                if new_tags:
                                    try:
                                        # Get ChromaDB collection to update metadata
                                        from cortex_engine.config import COLLECTION_NAME
                                        wsl_db_path = convert_windows_to_wsl_path(
                                            st.session_state.get('db_path_input', '')
                                        )
                                        chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
                                        client = chromadb.PersistentClient(
                                            path=chroma_db_path,
                                            settings=ChromaSettings(anonymized_telemetry=False)
                                        )
                                        vector_collection = client.get_collection(COLLECTION_NAME)

                                        # Update each document's tags
                                        for doc_id in doc_ids:
                                            try:
                                                result = vector_collection.get(ids=[doc_id], include=["metadatas"])
                                                if result and result['metadatas']:
                                                    meta = dict(result['metadatas'][0])
                                                    existing_tags = meta.get('thematic_tags', '')
                                                    if isinstance(existing_tags, str):
                                                        existing_set = set(t.strip() for t in existing_tags.split(',') if t.strip())
                                                    else:
                                                        existing_set = set(existing_tags) if existing_tags else set()
                                                    existing_set.update(new_tags)
                                                    meta['thematic_tags'] = ', '.join(sorted(existing_set))
                                                    vector_collection.update(ids=[doc_id], metadatas=[meta])
                                                    tags_applied += 1
                                            except Exception as tag_e:
                                                logger.warning(f"Failed to tag {doc_id}: {tag_e}")
                                    except Exception as e:
                                        logger.warning(f"Could not apply tags: {e}")

                            # Success message
                            if selected_option == "+ Create New Collection":
                                msg = f"✅ Created '{target_collection}' with {len(doc_ids)} documents"
                            else:
                                msg = f"✅ Added {len(doc_ids)} documents to '{target_collection}'"
                            if tags_applied > 0:
                                msg += f" (tagged {tags_applied})"
                            st.success(msg)
                            if selected_option == "+ Create New Collection":
                                st.balloons()

                    except Exception as e:
                        st.error(f"❌ Failed to save: {e}")

        # Clear results button
        if st.button("🆑 Clear Results", help="Clear current search results"):
            st.session_state.last_search_results = []
            st.session_state.last_search_query = ""
            st.rerun()

        st.divider()
    
    # Individual results display
    st.subheader("📊 Search Results")

    # Deduplicate by file_name - keep highest scoring chunk per document
    unique_results = {}
    for result in results:
        file_name = result.get('file_name', 'Unknown')
        if file_name not in unique_results or result['score'] > unique_results[file_name]['score']:
            unique_results[file_name] = result

    deduplicated_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)

    # Show deduplication info
    if len(results) > len(deduplicated_results):
        st.info(f"📄 Showing {len(deduplicated_results)} unique documents (from {len(results)} chunks)")

    for i, result in enumerate(deduplicated_results[:10]):  # Show top 10 unique documents
        with st.expander(f"**{result['rank']}.** {result['file_name']} (Score: {result['score']:.3f})"):
            # Action buttons for individual results
            action_col1, action_col2, action_col3 = st.columns([1, 1, 4])
            
            with action_col1:
                if st.button("➕ Add", key=f"add_{i}", help="Add this document to a collection"):
                    st.session_state[f'show_add_{i}'] = True
                    st.rerun()
            
            # Individual add actions - Docker-safe
            if st.session_state.get(f'show_add_{i}', False):
                try:
                    collection_mgr = WorkingCollectionManager()
                    collection_names = collection_mgr.get_collection_names()

                    # Add "+ Create New Collection" as first option
                    collection_options = ["+ Create New Collection"] + collection_names

                    target_option = st.selectbox(
                        "Add to collection:",
                        collection_options,
                        key=f"target_add_{i}"
                    )

                    # Show new collection name input if creating new
                    if target_option == "+ Create New Collection":
                        new_coll_name = st.text_input(
                            "New collection name:",
                            key=f"new_coll_name_{i}",
                            placeholder="e.g., Strategy Documents"
                        )
                    else:
                        new_coll_name = None

                    col_x, col_y = st.columns(2)
                    with col_x:
                        if st.button("Add", key=f"confirm_add_{i}", type="primary"):
                            try:
                                if target_option == "+ Create New Collection":
                                    if new_coll_name and new_coll_name.strip():
                                        if collection_mgr.create_collection(new_coll_name.strip()):
                                            collection_mgr.add_docs_by_id_to_collection(new_coll_name.strip(), [result['doc_id']])
                                            st.success(f"✅ Created '{new_coll_name}' and added document!")
                                            st.session_state[f'show_add_{i}'] = False
                                            st.rerun()
                                        else:
                                            st.error(f"Collection '{new_coll_name}' already exists!")
                                    else:
                                        st.warning("Please enter a collection name")
                                else:
                                    collection_mgr.add_docs_by_id_to_collection(target_option, [result['doc_id']])
                                    st.success(f"✅ Added to '{target_option}'!")
                                    st.session_state[f'show_add_{i}'] = False
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to add: {e}")

                    with col_y:
                        if st.button("Cancel", key=f"cancel_add_{i}"):
                            st.session_state[f'show_add_{i}'] = False
                            st.rerun()
                except Exception as e:
                    st.warning(f"Collections not available: {e}")
                    if st.button("Close", key=f"close_add_error_{i}"):
                        st.session_state[f'show_add_{i}'] = False
                        st.rerun()
            
            st.write("**Content:**")
            st.write(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
            
            st.write("**Metadata:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📄 **Type:** {result['document_type']}")
                st.write(f"📁 **File:** {result['file_name']}")
                # Show search source if available
                if result.get('search_source'):
                    st.write(f"🔍 **Source:** {result['search_source']}")
            with col2:
                st.write(f"🎯 **Score:** {result['score']:.4f}")
                if result.get('proposal_outcome', 'N/A') != 'N/A':
                    st.write(f"📊 **Outcome:** {result['proposal_outcome']}")
                st.write(f"🔗 **ID:** {result['doc_id']}")
            
            # Show graph context if available
            if result.get('graph_context'):
                st.info(f"🧠 **GraphRAG Context:** {result['graph_context']}")


def main():
    """Main page function"""
    st.title("🔍 3. Knowledge Search")
    st.caption(f"Version: {PAGE_VERSION}")
    
    # Initialize session state first
    initialize_search_state()

    # Start background preload of reranker model (if enabled)
    start_reranker_preload()

    # Render sidebar and get config early
    config = render_sidebar()
    
    # Docker environment detection with better logic
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV') == 'true'
    
    # Check if we're in a development environment (has access to existing database)
    is_dev_with_existing_db = False
    if config and config.get('db_path'):
        db_path_check = config['db_path']
        if db_path_check and os.path.exists(convert_windows_to_wsl_path(db_path_check)):
            try:
                # Try to validate the existing database (silently to avoid duplicate messages)
                db_info_check = validate_database(db_path_check, silent=True)
                if db_info_check and db_info_check.get('count', 0) > 0:
                    is_dev_with_existing_db = True
                    logger.info(f"Development database detected with {db_info_check['count']} documents")
            except Exception:
                pass
    
    # Only show Docker mode message if we're actually in a fresh Docker environment
    if is_docker and not is_dev_with_existing_db:
        st.info("🐳 **Running in Docker mode** - Some collection features may be limited until first setup is complete.")
    elif is_docker and is_dev_with_existing_db:
        st.success("🐳 **Docker mode with existing database detected** - Full functionality available!")
    
    # Validate database
    db_path = config['db_path']
    normalized_db_root = resolve_db_root_path(db_path)
    if normalized_db_root:
        db_path = str(normalized_db_root)
    if not db_path:
        st.error("❌ Database path not configured. Please set it in the sidebar.")
        return
    
    db_info = validate_database(db_path)
    
    if not db_info:
        is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV') == 'true'
        current_db_path = config.get('db_path', '') if config else ''
        
        if is_docker:
            # More specific Docker error handling
            if current_db_path and ('F:\\' in current_db_path or 'C:\\' in current_db_path):
                st.error("❌ **Docker + Windows Database Path Conflict**")
                st.warning("🚫 **Root Cause:** Docker cannot properly access Windows file paths like `F:\\ai_databases`")
                st.info("📝 **Solution:** Use Docker-native paths:\n1. Set path to `/app/data/ai_databases` in sidebar\n2. Go to **Knowledge Ingest** to populate it\n3. Return here to search")
                
                with st.expander("🚑 Emergency Fix"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🚀 Fix Database Path Now"):
                            st.session_state.db_path_input = "/app/data/ai_databases"
                            st.success("Path updated to Docker-compatible location!")
                            st.info("Please refresh the page to apply changes.")
                    with col2:
                        if st.button("🔄 Refresh Page"):
                            st.rerun()
            else:
                st.error("❌ **Docker Setup Required**")
                st.info("📝 **To get started:**\n1. Use the sidebar to set your database path (try `/app/data/ai_databases`)\n2. Go to **Knowledge Ingest** to add documents\n3. Return here to search")
        else:
            st.error("❌ Failed to validate knowledge base. Please check your database path and try again.")
        return
    
    # Search interface
    st.subheader("🔍 Search Knowledge Base")
    
    # GraphRAG search mode selection
    st.markdown("### 🧠 Search Strategy")
    search_mode = st.radio(
        "Choose your search approach:",
        ["Traditional Vector Search", "GraphRAG Enhanced", "Hybrid Search"],
        index=0,
        help="""
        **Traditional Vector Search**: Direct semantic similarity search through document embeddings
        **GraphRAG Enhanced**: Uses knowledge graph relationships and entity context for enhanced results  
        **Hybrid Search**: Combines vector search with graph-based relationship analysis
        """,
        key="search_mode_selection"
    )
    
    # Show GraphRAG status info
    if search_mode in ["GraphRAG Enhanced", "Hybrid Search"]:
        try:
            from cortex_engine.graphrag_integration import get_graphrag_integration
            graphrag = get_graphrag_integration(db_path)
            health = graphrag.health_check()
            
            if health['graph_nodes'] > 0:
                st.success(f"✅ GraphRAG Ready: {health['graph_nodes']} entities, {health['graph_edges']} relationships")
            else:
                st.warning("⚠️ Knowledge graph is empty. GraphRAG will fall back to traditional search.")
                
        except Exception as e:
            st.warning(f"⚠️ GraphRAG initialization issue: {e}")
            logger.warning(f"GraphRAG health check failed: {e}")
    
    st.divider()
    
    # Docker-specific guidance with better detection
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV') == 'true'
    
    if is_docker and not db_info:
        # Check if we're trying to use a development database from Docker
        current_db_path = config.get('db_path', '') if config else ''
        if current_db_path and ('F:\\' in current_db_path or 'C:\\' in current_db_path):
            st.warning("🐳 **Windows Database Path Detected in Docker**")
            st.error("🚫 **Issue:** You're trying to access a Windows database path from inside Docker.")
            st.info("📝 **Solutions:**\n\n**Option 1 (Recommended):** Use separate Docker database\n- Set path to `/app/data/ai_databases`\n- Go to Knowledge Ingest to add documents\n\n**Option 2:** Mount Windows database correctly\n- Ensure Windows path is properly mounted as Docker volume\n- Use mounted path like `/mnt/database` instead of `F:\\ai_databases`")
            
            with st.expander("🔧 Quick Fix: Use Docker Database Path"):
                if st.button("🚀 Set Docker Database Path", type="primary"):
                    st.session_state.db_path_input = "/app/data/ai_databases"
                    st.success("Database path updated! Please refresh the page.")
                    st.rerun()
            return
        else:
            st.warning("🐳 **Docker First-Time Setup Required**")
            st.info("📝 **Next Steps:**\n1. Set database path in sidebar (try `/app/data/ai_databases`)\n2. Go to **Knowledge Ingest** page to add documents\n3. Return here to search your documents")
            return
    
    # Add helpful examples
    with st.expander("💡 Search Examples & Tips", expanded=False):
        st.markdown("""
        **Multi-term searches:**
        - `strategy and transformation` - finds documents about both topics
        - `artificial intelligence machine learning` - AI and ML content
        - `project management agile` - project management with agile methods
        
        **Single term searches:**
        - `pedagogy` - educational approaches
        - `blockchain` - blockchain technology
        - `sustainability` - environmental topics
        
        **Use the sidebar for advanced filtering:**
        - Filter by document type (Technical Documentation, Proposals, etc.)
        - Filter by proposal outcome (Won/Lost/Pending)
        - Search within specific collections
        - Combine filters with AND/OR logic
        """)
    
    # Search input with better examples
    query = st.text_input(
        "Enter your search query:",
        value=st.session_state.get('last_search_query', ''),
        placeholder="e.g., strategy and transformation, artificial intelligence, project management...",
        key="search_query_input",
        help="💡 **Multi-term searches supported!** Try phrases like 'strategy and transformation' or 'machine learning algorithms'. Use sidebar filters for advanced filtering."
    )
    
    # Show filter summary
    filter_summary = []
    if st.session_state.doc_type_filter != 'Any':
        filter_summary.append(f"📄 {st.session_state.doc_type_filter}")
    if st.session_state.outcome_filter != 'Any':
        filter_summary.append(f"📊 {st.session_state.outcome_filter}")
    if st.session_state.search_scope == 'Active Collection':
        filter_summary.append(f"📚 {st.session_state.selected_collection}")
    
    if filter_summary:
        filter_text = " | ".join(filter_summary)
        if len(filter_summary) > 1 and (st.session_state.doc_type_filter != 'Any' and st.session_state.outcome_filter != 'Any'):
            filter_text += f" ({st.session_state.filter_operator} logic)"
        st.info(f"🔍 Active filters: {filter_text}")
    
    # Search button with helpful hints
    search_disabled = not query.strip()
    if search_disabled:
        st.info("💡 **Search Tips:** Try queries like 'artificial intelligence', 'project management', 'strategy and transformation', or use the sidebar filters for specific document types.")
    
    if st.button("🔍 Search Knowledge Base", type="primary", disabled=search_disabled):
        if query.strip():
            # Update last query
            st.session_state.last_search_query = query
            
            # Show progress indicator
            with st.status("🔍 Searching knowledge base...", expanded=True) as status:
                st.write(f"🎯 Query: '{query}'")
                
                # Show search strategy info
                terms = query.split()
                if len(terms) > 1:
                    st.write(f"🧠 Multi-strategy search: trying vector search, individual terms, and text matching")
                    st.write(f"🔎 Search terms: {', '.join(terms)}")
                else:
                    st.write(f"🔍 Single-term vector search")
                
                # Show active filters in progress
                filter_info = []
                if config.get('doc_type_filter', 'Any') != 'Any':
                    filter_info.append(f"Type: {config['doc_type_filter']}")
                if config.get('outcome_filter', 'Any') != 'Any':
                    filter_info.append(f"Outcome: {config['outcome_filter']}")
                if config.get('search_scope', 'Entire Knowledge Base') == 'Active Collection':
                    filter_info.append(f"Collection: {config.get('selected_collection', 'default')}")
                
                if filter_info:
                    filter_text = " | ".join(filter_info)
                    if len(filter_info) > 1:
                        filter_text += f" ({config.get('filter_operator', 'AND')} logic)"
                    st.write(f"🔍 Filters: {filter_text}")
                
                # Show dynamic document count if available
                try:
                    safe_db_path = convert_to_docker_mount_path(config.get('db_path_input') or config.get('ai_database_path') or st.session_state.get('db_path_input', ''))
                    chroma_db_path = os.path.join(safe_db_path, "knowledge_hub_db")
                    db_settings = ChromaSettings(anonymized_telemetry=False)
                    client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
                    try:
                        col = client.get_collection(COLLECTION_NAME)
                        total_docs = col.count()
                    except Exception:
                        # Fallback: sum across collections
                        total_docs = 0
                        for c in client.list_collections():
                            try:
                                total_docs += c.count()
                            except Exception:
                                pass
                    st.write(f"📊 Analyzing {total_docs} documents...")
                except Exception:
                    st.write("📊 Analyzing documents...")
                
                # Perform search with selected mode
                search_mode = st.session_state.get('search_mode_selection', 'Traditional Vector Search')
                results = perform_search(None, query, config, search_mode)
                st.session_state.last_search_results = results
                
                # Update status when complete
                if results and len(results) > 0:
                    # Show which search strategy worked
                    if len(terms) > 1:
                        status.update(label=f"✅ Found {len(results)} results using multi-strategy search", state="complete")
                    else:
                        status.update(label=f"✅ Found {len(results)} results", state="complete")
                else:
                    status.update(label="⚠️ No results found - try the examples above or check sidebar filters", state="complete")
            
            # Display results
            render_search_results(results, config)

            # Show search debug log
            if st.session_state.get('search_debug_log'):
                with st.expander("🔍 Search Debug Log", expanded=False):
                    st.text('\n'.join(st.session_state.search_debug_log))
        # Query validation handled above
        pass
    
    # Display last results if available
    elif st.session_state.get('last_search_results'):
        st.info(f"Showing previous search results for: '{st.session_state.get('last_search_query', '')}'")
        render_search_results(st.session_state.last_search_results, config)

    # Consistent version footer
    try:
        from cortex_engine.ui_components import render_version_footer
        render_version_footer()
    except Exception:
        pass


if __name__ == "__main__":
    main()
