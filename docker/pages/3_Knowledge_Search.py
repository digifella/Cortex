# ## File: pages/3_Knowledge_Search.py
# Version: 22.2.2 (Progress Indicator Enhancement)
# Date: 2025-08-26
# Purpose: Advanced knowledge search interface with vector + graph search capabilities.
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
PAGE_VERSION = "22.2.2"

st.set_page_config(page_title="Knowledge Search", layout="wide")


def get_document_type_options():
    try:
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
        logger.error(f"Error getting document types: {e}")
        return []


def validate_database(db_path):
    """Validate database exists and return ChromaDB collection info"""
    import os
    if not db_path or not db_path.strip():
        st.warning("Database path is not configured.")
        return None
        
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
    
    if not os.path.isdir(chroma_db_path):
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
            st.success(f"✅ Knowledge base validated: {collection_count} documents available for direct search.")
            return {"path": chroma_db_path, "count": collection_count, "collection": collection}
        else:
            st.warning("⚠️ Database directory exists but no documents found.")
            return None
            
    except Exception as e:
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


def render_sidebar():
    """Render the sidebar with database configuration and filters"""
    st.sidebar.header("⚙️ Configuration")
    
    # Database path configuration
    from cortex_engine.session_state import initialize_app_session_state
    initialize_app_session_state()
    
    from cortex_engine.config_manager import ConfigManager
    config_manager = ConfigManager()
    current_config = config_manager.get_config()
    current_path = current_config.get("ai_database_path", "")
    
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
                    
                    # Save to persistent config file
                    try:
                        config_manager.update_config({"ai_database_path": new_path.strip()})
                        st.sidebar.success("✅ Database path updated and saved!")
                    except Exception as config_e:
                        st.sidebar.warning(f"⚠️ Could not save to config file: {config_e}")
                        
                    # Clear cache to force reload
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Path does not exist: {wsl_path}")
            else:
                st.sidebar.error("❌ Please enter a valid path")
    
    with col2:
        if st.button("🔄 Reset", help="Reset to default path"):
            # Use cross-platform default path detection
            from cortex_engine.utils.default_paths import get_default_ai_database_path
            default_path = get_default_ai_database_path()
            st.session_state.db_path_input = default_path
            try:
                config_manager.update_config({"ai_database_path": default_path})
                st.sidebar.success("✅ Path reset to default!")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not save reset: {e}")
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
    if doc_types:
        selected_doc_type = st.sidebar.selectbox(
            "Document Type:",
            options=["Any"] + doc_types,
            key="doc_type_filter"
        )
    else:
        st.sidebar.info("No document types found")
        selected_doc_type = "Any"
    
    return {
        'db_path': st.session_state.get('db_path_input', current_path),
        'doc_type_filter': selected_doc_type
    }


def perform_search(base_index, query, filters):
    """Perform search - using direct ChromaDB approach to bypass LlamaIndex issues"""
    try:
        if not query.strip():
            st.warning("⚠️ Please enter a search query")
            return []
        
        # Get database path from config
        from cortex_engine.config_manager import ConfigManager
        config_manager = ConfigManager()
        current_config = config_manager.get_config()
        db_path = current_config.get("ai_database_path", "")
        
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
            
            # Direct query - try both vector search and text search approaches
            logger.info("Executing ChromaDB query...")
            
            # First try vector search
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                logger.info("ChromaDB vector query completed successfully")
            except Exception as vector_e:
                logger.warning(f"Vector search failed: {vector_e}")
                # Fallback to get all and filter (last resort)
                logger.info("Attempting fallback text-based search...")
                try:
                    # Get a subset of documents and filter by text content
                    all_results = collection.get(limit=min(1000, top_k * 10))  # Get larger sample
                    
                    # Simple text matching as fallback
                    matching_docs = []
                    query_lower = query.lower()
                    
                    documents = all_results.get('documents', [])
                    metadatas = all_results.get('metadatas', [])
                    
                    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                        if query_lower in doc.lower():
                            matching_docs.append({
                                'documents': [doc],
                                'metadatas': [metadata], 
                                'distances': [0.5]  # Arbitrary similarity score
                            })
                    
                    # Format as ChromaDB query result
                    if matching_docs:
                        results = {
                            'documents': [[item['documents'][0] for item in matching_docs[:top_k]]],
                            'metadatas': [[item['metadatas'][0] for item in matching_docs[:top_k]]],
                            'distances': [[item['distances'][0] for item in matching_docs[:top_k]]]
                        }
                        logger.info(f"Fallback text search found {len(matching_docs)} matches")
                    else:
                        results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
                        logger.info("No text matches found in fallback search")
                        
                except Exception as fallback_e:
                    logger.error(f"Fallback search also failed: {fallback_e}")
                    return []
            
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
                        'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                        'doc_id': metadata.get('doc_id', f'doc_{i}')
                    }
                    
                    # Apply post-search filtering if needed
                    if filters and isinstance(filters, dict):
                        doc_type_filter = filters.get('doc_type_filter')
                        if (doc_type_filter and doc_type_filter != "Any" and 
                            result['document_type'] != doc_type_filter):
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
    """Render search results"""
    if not results:
        st.info("No results found. Try different search terms or check your database configuration.")
        return
    
    st.success(f"✅ Found {len(results)} results")
    
    # Results display
    for i, result in enumerate(results[:10]):  # Show top 10 results
        with st.expander(f"**{result['rank']}.** {result['file_name']} (Score: {result['score']:.3f})"):
            st.write("**Content:**")
            st.write(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
            
            st.write("**Metadata:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📄 **Type:** {result['document_type']}")
                st.write(f"📁 **File:** {result['file_name']}")
            with col2:
                st.write(f"🎯 **Score:** {result['score']:.4f}")
                st.write(f"🔗 **Path:** {result['file_path']}")


def main():
    """Main page function"""
    st.title("🔍 3. Knowledge Search")
    st.caption(f"Version: {PAGE_VERSION}")
    
    # Initialize session state
    initialize_search_state()
    
    # Render sidebar and get config
    config = render_sidebar()
    
    # Validate database
    db_path = config['db_path']
    if not db_path:
        st.error("❌ Database path not configured. Please set it in the sidebar.")
        return
    
    db_info = validate_database(db_path)
    
    if not db_info:
        st.error("❌ Failed to validate knowledge base. Please check your database path and try again.")
        return
    
    # Search interface
    st.subheader("🔍 Search Knowledge Base")
    
    # Search input
    query = st.text_input(
        "Enter your search query:",
        value=st.session_state.get('last_search_query', ''),
        placeholder="e.g., machine learning algorithms, project management techniques...",
        key="search_query_input"
    )
    
    # Search button
    if st.button("🔍 Search Knowledge Base", type="primary") or (query and query != st.session_state.get('last_search_query', '')):
        if query.strip():
            # Update last query
            st.session_state.last_search_query = query
            
            # Show progress indicator
            with st.status("🔍 Searching knowledge base...", expanded=True) as status:
                st.write(f"🎯 Query: '{query}'")
                st.write("📊 Analyzing documents...")
                
                # Perform search (base_index not needed for direct ChromaDB search)
                results = perform_search(None, query, config)
                st.session_state.last_search_results = results
                
                # Update status when complete
                if results and len(results) > 0:
                    status.update(label=f"✅ Found {len(results)} results", state="complete")
                else:
                    status.update(label="⚠️ No results found", state="complete")
            
            # Display results
            render_search_results(results, config)
        else:
            st.warning("⚠️ Please enter a search query")
    
    # Display last results if available
    elif st.session_state.get('last_search_results'):
        st.info(f"Showing previous search results for: '{st.session_state.get('last_search_query', '')}'")
        render_search_results(st.session_state.last_search_results, config)


if __name__ == "__main__":
    main()