# ## File: pages/3_Knowledge_Search.py
# Version: 22.0.0 (Backend Integration Fix)
# Date: 2025-08-26
# Purpose: Advanced knowledge search interface with vector + graph search capabilities.
#          - CRITICAL FIX (v22.0.0): Replaced custom embedding initialization with backend system
#            integration to resolve 'SentenceTransformerWrapper' missing methods error.
#          - CRITICAL FIX (v21.3.0): Fixed emergency embedding model to use correct 768 dimensions
#            matching BAAI/bge-base-en-v1.5 standard, resolving ChromaDB dimension mismatch.
#          - CRITICAL BUGFIX (v21.2.3): Fixed path not being retained by saving database path
#            changes to persistent config file. Session state was being overridden on page loads.

import streamlit as st
import os
import pandas as pd
from pathlib import Path
import re

# Import backend components
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

# Import project modules
from cortex_engine.config import EMBED_MODEL, COLLECTION_NAME, KB_LLM_MODEL
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path

# Set up logging
logger = get_logger(__name__)

# Page configuration
PAGE_VERSION = "22.0.0"

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


@st.cache_resource(ttl=3600)
def load_base_index(db_path, model_provider, api_key=None):
    """Load the base index with backend embedding system integration"""
    import os
    if not db_path or not db_path.strip():
        st.warning("Database path is not configured.")
        return None, None
        
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
    
    if not os.path.isdir(chroma_db_path):
        st.warning(f"🧠 Knowledge base directory not found at '{chroma_db_path}'.")
        return None, None
        
    try:
        # Check if Ollama is available
        from cortex_engine.utils.ollama_utils import check_ollama_service, get_ollama_status_message
        
        is_running, error_msg = check_ollama_service()
        if not is_running:
            st.warning(f"⚠️ {get_ollama_status_message(is_running, error_msg)}")
            st.info("📋 **Limited functionality:** Vector search available, but AI-powered query enhancement is disabled.")
            Settings.llm = None
        else:
            Settings.llm = Ollama(model="mistral", request_timeout=120.0)
        
        # Use backend embedding system for full compatibility
        import torch
        logger.info(f"Initializing embedding model {EMBED_MODEL} with PyTorch {torch.__version__}")
        
        # Initialize backend system which properly configures Settings.embed_model
        from cortex_engine.query_cortex import setup_models
        setup_models()
        
        # Verify backend system worked
        if Settings.embed_model:
            test_embedding = Settings.embed_model.get_text_embedding("test")
            logger.info(f"✅ Backend embedding model successful: {type(Settings.embed_model).__name__}, dimension: {len(test_embedding)}")
            
            # Verify critical method exists
            if hasattr(Settings.embed_model, 'get_agg_embedding_from_queries'):
                logger.info("✅ get_agg_embedding_from_queries method available")
            else:
                logger.error("❌ Missing get_agg_embedding_from_queries method")
        else:
            st.error("❌ Backend embedding model not available")
            return None, None
        
        # Set up ChromaDB
        db_settings = ChromaSettings(anonymized_telemetry=False)
        db = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        
        # Log collection info
        try:
            collection_count = chroma_collection.count()
            logger.info(f"Collection '{COLLECTION_NAME}' has {collection_count} documents")
            
            if collection_count > 0:
                sample = chroma_collection.peek(limit=1)
                if sample.get('embeddings') and len(sample['embeddings']) > 0:
                    expected_dim = len(sample['embeddings'][0])
                    logger.info(f"Collection expects embeddings with dimension: {expected_dim}")
        except Exception as e:
            logger.warning(f"Could not inspect collection: {e}")
            
        # Create vector store and index
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=chroma_db_path)
        index = load_index_from_storage(storage_context)
        
        if Settings.llm:
            st.success(f"✅ Knowledge base loaded successfully from '{chroma_db_path}' with full AI capabilities.")
        else:
            st.success(f"✅ Knowledge base loaded from '{chroma_db_path}' (basic search mode).")
        
        return index, chroma_collection
        
    except Exception as e:
        st.error(f"❌ Backend initialization failed: {e}")
        logger.error(f"Error loading query engine from {chroma_db_path}: {e}", exc_info=True)
        return None, None


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
            default_path = "/mnt/f/ai_databases"
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
    """Perform search with the configured index"""
    try:
        if not base_index:
            st.error("❌ Knowledge base not loaded")
            return []
        
        if not query.strip():
            st.warning("⚠️ Please enter a search query")
            return []
        
        # Create query engine
        query_engine = base_index.as_query_engine(
            similarity_top_k=20,
            response_mode="no_text"  # We only want the source nodes
        )
        
        # Perform query
        with st.spinner("🔍 Searching knowledge base..."):
            response = query_engine.query(query)
            
            # Extract results
            results = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    try:
                        # Extract metadata
                        metadata = node.metadata if hasattr(node, 'metadata') else {}
                        
                        result = {
                            'rank': i + 1,
                            'score': getattr(node, 'score', 0.0),
                            'text': node.text if hasattr(node, 'text') else str(node),
                            'file_path': metadata.get('file_path', 'Unknown'),
                            'file_name': metadata.get('file_name', 'Unknown'),
                            'document_type': metadata.get('document_type', 'Unknown'),
                            'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                            'doc_id': getattr(node, 'node_id', f'node_{i}')
                        }
                        results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Error processing search result {i}: {e}")
                        continue
            
            logger.info(f"Search completed: {len(results)} results for query: {query[:50]}...")
            return results
            
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        logger.error(f"Search error: {e}", exc_info=True)
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
    
    # Load base index
    db_path = config['db_path']
    if not db_path:
        st.error("❌ Database path not configured. Please set it in the sidebar.")
        return
    
    base_index, vector_collection = load_base_index(db_path, "Local")
    
    if not base_index:
        st.error("❌ Failed to load knowledge base. Please check your database path and try again.")
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
            
            # Perform search
            results = perform_search(base_index, query, config)
            st.session_state.last_search_results = results
            
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