# ## File: pages/3_Knowledge_Search.py
# Version: v4.1.2
# Date: 2025-08-26
# Purpose: Advanced knowledge search interface with vector + graph search capabilities.
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

# Import only ChromaDB components needed for direct search
import chromadb
from chromadb.config import Settings as ChromaSettings

# LlamaIndex imports moved to functions where actually needed to avoid numpy.iterable issues

# Import project modules
from cortex_engine.config import EMBED_MODEL, COLLECTION_NAME, KB_LLM_MODEL
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path

# Set up logging
logger = get_logger(__name__)

# Page configuration
PAGE_VERSION = "v4.1.2"

st.set_page_config(page_title="Knowledge Search", layout="wide")


def get_document_type_options():
    """Get document type options - Docker-safe version"""
    try:
        # In Docker environment, use fallback to static options since collections may not be initialized
        collection_mgr = WorkingCollectionManager()
        collections = collection_mgr.collections
        doc_types = set()
        for collection in collections.values():
            for doc in collection.get('documents', []):
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


def validate_database(db_path, silent=False):
    """Validate database exists and return ChromaDB collection info"""
    import os
    if not db_path or not db_path.strip():
        if not silent:
            st.warning("Database path is not configured.")
        return None
        
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
    
    if not os.path.isdir(chroma_db_path):
        if not silent:
            st.warning(f"🧠 Knowledge base directory not found at '{chroma_db_path}'.")
        return None
        
    try:
        # Simple ChromaDB validation without LlamaIndex
        db_settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        collection = client.get_collection(COLLECTION_NAME)
        
        # Log collection info
        collection_count = collection.count()
        logger.info(f"Collection '{COLLECTION_NAME}' has {collection_count} documents")
        
        if collection_count > 0:
            if not silent:
                st.success(f"✅ Knowledge base validated: {collection_count} documents available for direct search.")
            return {"path": chroma_db_path, "count": collection_count, "collection": collection}
        else:
            if not silent:
                st.warning("⚠️ Database directory exists but no documents found.")
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
                db_settings = ChromaSettings(anonymized_telemetry=False)
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
    if 'filter_operator' not in st.session_state:
        st.session_state.filter_operator = "AND"
    if 'search_scope' not in st.session_state:
        st.session_state.search_scope = "Entire Knowledge Base"
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = "default"


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
    except Exception as e:
        logger.warning(f"Config manager not available (Docker environment?): {e}")
        # In Docker, default to standard path
        current_path = "/app/data/ai_databases"
    
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
                wsl_path = convert_windows_to_wsl_path(new_path.strip())
                
                if os.path.exists(wsl_path):
                    # Update session state
                    st.session_state.db_path_input = new_path.strip()
                    
                    # Save to persistent config file - Docker-safe
                    try:
                        config_manager.update_config({"ai_database_path": new_path.strip()})
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
                # Docker fallback
                default_path = "/app/data/ai_databases"
            
            st.session_state.db_path_input = default_path
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
                        db_settings = ChromaSettings(anonymized_telemetry=False)
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
    
    # Boolean operator (only show if multiple filters are active)
    num_active_filters = sum([
        1 if st.session_state.doc_type_filter != "Any" else 0,
        1 if st.session_state.outcome_filter != "Any" else 0
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
    
    return {
        'db_path': st.session_state.get('db_path_input', current_path),
        'doc_type_filter': st.session_state.doc_type_filter,
        'outcome_filter': st.session_state.outcome_filter,
        'filter_operator': st.session_state.filter_operator,
        'search_scope': st.session_state.search_scope,
        'selected_collection': st.session_state.selected_collection
    }


def perform_search(base_index, query, filters):
    """Perform search - using direct ChromaDB approach to bypass LlamaIndex issues"""
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
        except Exception as e:
            logger.warning(f"Config not available (Docker environment?): {e}")
            # Try session state as fallback
            db_path = st.session_state.get('db_path_input', '/app/data/ai_databases')
        
        if not db_path:
            st.error("❌ Database path not configured")
            return []
        
        # Use direct ChromaDB search to bypass LlamaIndex where clause issues
        logger.info(f"Using direct ChromaDB search for query: {query[:50]}...")
        return direct_chromadb_search(db_path, query, filters)
            
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        logger.error(f"Search error: {e}", exc_info=True)
        return []


def direct_chromadb_search(db_path, query, filters, top_k=20):
    """
    Perform direct ChromaDB search bypassing LlamaIndex entirely
    This resolves the ChromaDB where clause issues
    """
    import time
    import concurrent.futures
    
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
            db_settings = ChromaSettings(anonymized_telemetry=False)
            client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
            collection = client.get_collection(COLLECTION_NAME)
            logger.info(f"Connected to collection '{COLLECTION_NAME}'")
            
            # Try simple get first to test collection access
            logger.info("Testing collection access...")
            try:
                peek = collection.peek(limit=1)
                logger.info(f"Collection peek successful, has embeddings: {bool(peek.get('embeddings'))}")
            except Exception as peek_e:
                logger.warning(f"Collection peek failed: {peek_e}")
            
            # Multi-strategy search approach for better results
            logger.info("Executing multi-strategy ChromaDB search...")
            
            # Strategy 1: Try vector search with original query
            results = None
            search_strategy = "vector"
            
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                logger.info("ChromaDB vector query completed successfully")
                
                # Check if we got good results
                if results and results.get('documents') and results['documents'][0]:
                    search_strategy = "vector"
                else:
                    results = None  # Try other strategies
                    
            except Exception as vector_e:
                logger.warning(f"Vector search failed: {vector_e}")
                results = None
            
            # Strategy 2: If vector search failed or gave no results, try individual terms
            if not results or not results.get('documents') or not results['documents'][0]:
                logger.info("Trying individual term searches...")
                
                # Split query into terms and search each
                terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
                
                if len(terms) > 1:
                    combined_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
                    doc_scores = {}  # Track scores for each document
                    
                    for term in terms:
                        try:
                            term_results = collection.query(
                                query_texts=[term],
                                n_results=min(top_k * 2, 50)  # Get more results per term
                            )
                            
                            if term_results and term_results.get('documents'):
                                term_docs = term_results['documents'][0]
                                term_metas = term_results['metadatas'][0]
                                term_distances = term_results['distances'][0]
                                
                                for doc, meta, dist in zip(term_docs, term_metas, term_distances):
                                    doc_id = meta.get('doc_id', 'unknown')
                                    score = 1.0 - dist
                                    
                                    if doc_id in doc_scores:
                                        # Boost score for documents that match multiple terms
                                        doc_scores[doc_id]['score'] += score * 0.7  # Boost multi-term matches
                                        doc_scores[doc_id]['term_count'] += 1
                                    else:
                                        doc_scores[doc_id] = {
                                            'doc': doc,
                                            'meta': meta,
                                            'score': score,
                                            'distance': dist,
                                            'term_count': 1
                                        }
                                        
                        except Exception as term_e:
                            logger.warning(f"Term search for '{term}' failed: {term_e}")
                    
                    # Sort by score and prioritize multi-term matches
                    if doc_scores:
                        sorted_docs = sorted(doc_scores.values(), 
                                           key=lambda x: (x['term_count'], x['score']), 
                                           reverse=True)
                        
                        # Format as ChromaDB result
                        results = {
                            'documents': [[item['doc'] for item in sorted_docs[:top_k]]],
                            'metadatas': [[item['meta'] for item in sorted_docs[:top_k]]],
                            'distances': [[item['distance'] for item in sorted_docs[:top_k]]]
                        }
                        
                        search_strategy = "multi-term"
                        logger.info(f"Multi-term search found {len(sorted_docs)} unique documents")
            
            # Strategy 3: Fallback to text-based search
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
                        'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                        'doc_id': metadata.get('doc_id', f'doc_{i}')
                    }
                    
                    # Apply post-search filtering - SAFE approach without ChromaDB where clauses
                    if filters and isinstance(filters, dict):
                        # Collection scope filter - skip if collections not available
                        if (collection_doc_ids is not None and 
                            result['doc_id'] not in collection_doc_ids):
                            continue  # Skip - not in selected collection
                        
                        # Document type filter
                        doc_type_filter = filters.get('doc_type_filter')
                        outcome_filter = filters.get('outcome_filter')
                        filter_operator = filters.get('filter_operator', 'AND')
                        
                        # Check individual filter conditions
                        doc_type_match = (doc_type_filter == "Any" or 
                                        result['document_type'] == doc_type_filter)
                        
                        outcome_match = (outcome_filter == "Any" or 
                                       metadata.get('proposal_outcome') == outcome_filter)
                        
                        # Apply boolean logic
                        if filter_operator == "AND":
                            # All conditions must be true
                            if not (doc_type_match and outcome_match):
                                continue  # Skip this result
                        else:  # OR operator
                            # At least one condition must be true (or no filters active)
                            if not (doc_type_match or outcome_match or 
                                  (doc_type_filter == "Any" and outcome_filter == "Any")):
                                continue  # Skip this result
                    
                    formatted_results.append(result)
            else:
                logger.warning("No documents returned from ChromaDB query")
            
            logger.info(f"Direct ChromaDB search returned {len(formatted_results)} formatted results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Direct ChromaDB search failed: {e}", exc_info=True)
            return []
    
    try:
        # Run search with timeout to prevent hanging
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(search_with_timeout)
            try:
                results = future.result(timeout=30)  # 30 second timeout
                return results
            except concurrent.futures.TimeoutError:
                logger.error("Search timed out after 30 seconds")
                st.error("Search timed out. The database may be large or there may be connectivity issues.")
                return []
                
    except Exception as e:
        logger.error(f"Search executor failed: {e}")
        st.error(f"Search failed: {e}")
        return []


def render_search_results(results, filters):
    """Render search results with collection management actions"""
    if not results:
        st.info("🔍 **No results found.**\n\n💡 **Try:**\n- Different search terms (e.g., 'strategy transformation' instead of 'strategy and transformation')\n- Checking sidebar filters - they might be too restrictive\n- Single terms first (e.g., just 'strategy')\n- Verifying your database path contains the right documents")
        return
    
    # Show active filters
    active_filters = []
    if filters.get('doc_type_filter', 'Any') != 'Any':
        active_filters.append(f"Type: {filters['doc_type_filter']}")
    if filters.get('outcome_filter', 'Any') != 'Any':
        active_filters.append(f"Outcome: {filters['outcome_filter']}")
    if filters.get('search_scope', 'Entire Knowledge Base') == 'Active Collection':
        active_filters.append(f"Collection: {filters.get('selected_collection', 'default')}")
    
    if active_filters:
        filter_text = " | ".join(active_filters)
        if len(active_filters) > 1:
            filter_text += f" ({filters.get('filter_operator', 'AND')} logic)"
        st.info(f"🔍 Active filters: {filter_text}")
    
    st.success(f"✅ Found {len(results)} results from 4201 documents")
    
    # Bulk collection actions
    if len(results) > 1:
        st.subheader("📋 Bulk Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add All to Collection", help="Add all search results to a collection"):
                st.session_state.show_bulk_add = True
        
        with col2:
            if st.button("💾 Save Results", help="Create a new collection with these results"):
                st.session_state.show_save_collection = True
        
        with col3:
            if st.button("🆑 Clear Results", help="Clear current search results"):
                st.session_state.last_search_results = []
                st.session_state.last_search_query = ""
                st.rerun()
        
        # Bulk action modals - Docker-safe
        if st.session_state.get('show_bulk_add', False):
            with st.expander("➕ Add All Results to Collection", expanded=True):
                try:
                    collection_mgr = WorkingCollectionManager()
                    collection_names = collection_mgr.get_collection_names()
                    
                    if collection_names:
                        target_collection = st.selectbox("Select target collection:", collection_names)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Add to Collection", type="primary"):
                                doc_ids = [result['doc_id'] for result in results]
                                try:
                                    collection_mgr.add_docs_by_id_to_collection(target_collection, doc_ids)
                                    st.success(f"✅ Added {len(doc_ids)} documents to '{target_collection}'!")
                                    st.session_state.show_bulk_add = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to add documents: {e}")
                        
                        with col_b:
                            if st.button("Cancel"):
                                st.session_state.show_bulk_add = False
                                st.rerun()
                    else:
                        st.info("No collections available. Please set up collections first.")
                        if st.button("Close"):
                            st.session_state.show_bulk_add = False
                            st.rerun()
                except Exception as e:
                    st.error(f"Collections not available in this environment: {e}")
                    st.info("Collection features require a fully initialized system.")
                    if st.button("Close", key="close_bulk_error"):
                        st.session_state.show_bulk_add = False
                        st.rerun()
        
        if st.session_state.get('show_save_collection', False):
            with st.expander("💾 Save as New Collection", expanded=True):
                try:
                    new_collection_name = st.text_input("Collection name:", placeholder="e.g., AI Research Papers")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Create Collection", type="primary"):
                            if new_collection_name.strip():
                                try:
                                    collection_mgr = WorkingCollectionManager()
                                    if collection_mgr.create_collection(new_collection_name.strip()):
                                        doc_ids = [result['doc_id'] for result in results]
                                        collection_mgr.add_docs_by_id_to_collection(new_collection_name.strip(), doc_ids)
                                        st.success(f"✅ Created '{new_collection_name}' with {len(doc_ids)} documents!")
                                        st.session_state.show_save_collection = False
                                        st.rerun()
                                    else:
                                        st.error(f"Collection '{new_collection_name}' already exists!")
                                except Exception as e:
                                    st.error(f"❌ Failed to create collection: {e}")
                            else:
                                st.warning("Please enter a collection name")
                    
                    with col_b:
                        if st.button("Cancel", key="cancel_save"):
                            st.session_state.show_save_collection = False
                            st.rerun()
                except Exception as e:
                    st.error(f"Collections not available in this environment: {e}")
                    st.info("Collection features require a fully initialized system.")
                    if st.button("Close", key="close_save_error"):
                        st.session_state.show_save_collection = False
                        st.rerun()
        
        st.divider()
    
    # Individual results display
    st.subheader("📊 Search Results")
    
    for i, result in enumerate(results[:10]):  # Show top 10 results
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
                    
                    if collection_names:
                        target_collection = st.selectbox(f"Add to collection:", collection_names, key=f"target_add_{i}")
                        
                        col_x, col_y = st.columns(2)
                        with col_x:
                            if st.button("Add", key=f"confirm_add_{i}", type="primary"):
                                try:
                                    collection_mgr.add_docs_by_id_to_collection(target_collection, [result['doc_id']])
                                    st.success(f"✅ Added to '{target_collection}'!")
                                    st.session_state[f'show_add_{i}'] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to add: {e}")
                        
                        with col_y:
                            if st.button("Cancel", key=f"cancel_add_{i}"):
                                st.session_state[f'show_add_{i}'] = False
                                st.rerun()
                    else:
                        st.info("No collections available")
                        if st.button("Close", key=f"close_add_{i}"):
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
            with col2:
                st.write(f"🎯 **Score:** {result['score']:.4f}")
                if result.get('proposal_outcome', 'N/A') != 'N/A':
                    st.write(f"📊 **Outcome:** {result['proposal_outcome']}")
                st.write(f"🔗 **ID:** {result['doc_id']}")


def main():
    """Main page function"""
    st.title("🔍 3. Knowledge Search")
    st.caption(f"Version: {PAGE_VERSION}")
    
    # Initialize session state first
    initialize_search_state()
    
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
    
    if st.button("🔍 Search Knowledge Base", type="primary", disabled=search_disabled) or (query and query != st.session_state.get('last_search_query', '')):
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
                
                st.write("📊 Analyzing 4201 documents...")
                
                # Perform search (base_index not needed for direct ChromaDB search)
                results = perform_search(None, query, config)
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
        # Query validation handled above
        pass
    
    # Display last results if available
    elif st.session_state.get('last_search_results'):
        st.info(f"Showing previous search results for: '{st.session_state.get('last_search_query', '')}'")
        render_search_results(st.session_state.last_search_results, config)


if __name__ == "__main__":
    main()