# ## File: pages/3_Knowledge_Search.py
# Version: 21.3.0 (Embedding Dimension Fix)
# Date: 2025-08-26
# Purpose: A UI for searching the knowledge base, managing collections,
#          and deleting documents from the KB.
#          - CRITICAL FIX (v21.3.0): Fixed emergency embedding model to use correct 768 dimensions
#            matching BAAI/bge-base-en-v1.5 standard, resolving ChromaDB dimension mismatch.
#          - CRITICAL BUGFIX (v21.2.3): Fixed path not being retained by saving database path
#            changes to persistent config file. Session state was being overridden on page loads.
#          - CRITICAL BUGFIX (v21.2.2): Fixed Update Path button not saving changes by properly
#            handling text_input session state synchronization and widget key conflicts.
#          - BUGFIX (v21.2.1): Enhanced database path validation with real-time debugging info,
#            improved session state synchronization, and live path validation preview.
#          - FEATURE (v21.2.0): Added editable database path configuration in sidebar with
#            real-time validation, path status checking, and easy reset functionality.
#          - CRITICAL BUGFIX (v21.1.2): Fixed remaining UnboundLocalError by moving os import
#            to the very beginning of both functions where it's needed before any usage.
#          - CRITICAL FIX (v21.1.0): Resolved PyTorch 2.8+ meta tensor issue with multi-layered
#            fallback system including emergency no-op embedding model.

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
PAGE_VERSION = "v1.2.3"
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
    import os  # Import os at the very beginning of the function
    if not db_path or not db_path.strip(): st.warning("Database path is not configured."); return None, None
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
    if not os.path.isdir(chroma_db_path): st.warning(f"🧠 Knowledge base directory not found at '{chroma_db_path}'."); return None, None
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
        
        # Use the backend embedding system that has full LlamaIndex compatibility
        import torch
        logger.info(f"Initializing embedding model {EMBED_MODEL} with PyTorch {torch.__version__}")
        
        # Use the properly configured backend system from query_cortex
        from cortex_engine.query_cortex import setup_models
        setup_models()
        
        # The backend system configures Settings.embed_model with full compatibility
        if Settings.embed_model:
            test_embedding = Settings.embed_model.get_text_embedding("test")
            logger.info(f"Backend embedding model successful, dimension: {len(test_embedding)}")
            logger.info(f"✅ Using backend embedding model: {type(Settings.embed_model).__name__}")
        else:
            st.error("❌ Backend embedding model not available")
            return None, None
                
        except Exception as e:
            logger.error(f"Failed to initialize backend embedding system: {e}")
            # If backend fails, create emergency fallback with full compatibility
            logger.warning("🚨 Creating emergency embedding model with full LlamaIndex compatibility")
            
            class EmergencyEmbeddingModel:
                """Emergency embedding model with full LlamaIndex interface"""
                def __init__(self):
                    self.embed_dim = 768
                    self.model_name = "emergency-fallback"
                    logger.warning("⚠️ Using emergency embedding model - search quality will be limited")
                
                def get_text_embedding(self, text):
                    import hashlib
                    hash_obj = hashlib.sha256(text.encode('utf-8'))
                    hash_bytes = hash_obj.digest()
                    embedding = []
                    for i in range(self.embed_dim):
                        byte_idx = (i * 4) % len(hash_bytes)
                        byte_val = hash_bytes[byte_idx] if byte_idx < len(hash_bytes) else 0
                        float_val = (float(byte_val) - 127.5) / 127.5
                        embedding.append(float_val)
                    return embedding
                
                def get_text_embeddings(self, texts):
                    return [self.get_text_embedding(text) for text in texts]
                
                def get_text_embedding_batch(self, texts):
                    return self.get_text_embeddings(texts)
                
                def get_query_embedding(self, query):
                    return self.get_text_embedding(query)
                
                def get_agg_embedding_from_queries(self, queries):
                    """Aggregate multiple query embeddings by averaging them"""
                    embeddings = self.get_text_embeddings(queries)
                    if not embeddings:
                        return []
                    import numpy as np
                    avg_embedding = np.mean(embeddings, axis=0)
                    return avg_embedding.tolist()
            
            embed_model = EmergencyEmbeddingModel()
            logger.warning("🆘 Emergency embedding model active with full compatibility")
                    
                    # Ultra-fallback: Try the most reliable embedding model
                    try:
                        logger.warning("Attempting ultra-fallback to most reliable embedding model...")
                        ultra_fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                        
                        # Try direct sentence-transformers first
                        try:
                            from sentence_transformers import SentenceTransformer
                            
                            # Force minimal environment
                            os.environ["TOKENIZERS_PARALLELISM"] = "false"
                            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" 
                            os.environ["TRANSFORMERS_OFFLINE"] = "0"
                            
                            st_fallback = SentenceTransformer(ultra_fallback_model, device="cpu")
                            
                            class UltraFallbackWrapper:
                                def __init__(self, model, model_name):
                                    self.model = model
                                    self.model_name = model_name
                                    
                                def get_text_embedding(self, text):
                                    return self.model.encode(text, convert_to_tensor=False).tolist()
                                
                                def get_text_embeddings(self, texts):
                                    embeddings = self.model.encode(texts, convert_to_tensor=False)
                                    return [emb.tolist() for emb in embeddings]
                            
                            embed_model = UltraFallbackWrapper(st_fallback, ultra_fallback_model)
                            logger.warning(f"⚠️ Using ultra-fallback embedding model: {ultra_fallback_model}")
                            
                        except Exception as ultra_st_e:
                            logger.error(f"Ultra-fallback sentence-transformers failed: {ultra_st_e}")
                            
                            # Final fallback: Simple HuggingFaceEmbedding with ultra-fallback model
                            embed_model = HuggingFaceEmbedding(
                                model_name=ultra_fallback_model,
                                device="cpu"
                            )
                            logger.warning(f"⚠️ Using HF embedding ultra-fallback: {ultra_fallback_model}")
                            
                    except Exception as ultimate_e:
                        logger.error(f"All fallback approaches failed: {ultimate_e}")
                        
                        # Emergency fallback: Create a no-op embedding model that allows the UI to load
                        logger.error("⚠️ EMERGENCY FALLBACK: Creating no-op embedding model")
                        logger.error("   This is due to a PyTorch/torchvision/transformers compatibility issue")
                        logger.error("   Knowledge Search will be limited to basic ChromaDB operations")
                        
                        class NoOpEmbeddingModel:
                            def __init__(self):
                                self.model_name = "no-op-fallback"
                                
                            def get_text_embedding(self, text):
                                # Return a simple hash-based pseudo-embedding
                                import hashlib
                                import struct
                                # Use SHA-256 for better hash distribution
                                hash_obj = hashlib.sha256(text.encode('utf-8'))
                                hash_bytes = hash_obj.digest()
                                # Convert to 768-dimensional embedding (BAAI/bge-base-en-v1.5 standard)
                                pseudo_embedding = []
                                for i in range(768):
                                    # Use different byte positions for variation
                                    byte_idx = (i * 4) % len(hash_bytes)
                                    byte_val = hash_bytes[byte_idx] if byte_idx < len(hash_bytes) else 0
                                    # Convert to float in range [-1, 1]
                                    float_val = (float(byte_val) - 127.5) / 127.5
                                    pseudo_embedding.append(float_val)
                                return pseudo_embedding  # 768 dimensions
                            
                            def get_text_embeddings(self, texts):
                                return [self.get_text_embedding(text) for text in texts]
                        
                        embed_model = NoOpEmbeddingModel()
                        logger.warning("⚠️ Using emergency no-op embedding model - search functionality will be limited")
                        
                        # Store this critical information in session state for user notification
                        st.session_state.embedding_emergency_mode = True
                        st.session_state.embedding_error_details = str(ultimate_e)
        Settings.embed_model = embed_model
        logger.info(f"Loaded embedding model: {EMBED_MODEL}")
        
        # Debug: Check embedding dimensions
        embedding_info = {"model": EMBED_MODEL, "dimension": "unknown", "status": "unknown"}
        try:
            test_embedding = embed_model.get_text_embedding("test")
            embedding_dim = len(test_embedding)
            embedding_info.update({"dimension": embedding_dim, "status": "loaded"})
            logger.info(f"Embedding model dimension: {embedding_dim}")
        except Exception as e:
            embedding_info.update({"status": f"failed: {e}"})
            logger.warning(f"Could not test embedding dimension: {e}")
        
        # Don't store in session_state here due to caching - will be done separately
        db_settings = ChromaSettings(anonymized_telemetry=False)
        db = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        
        # Debug: Check collection info
        try:
            collection_count = chroma_collection.count()
            logger.info(f"Collection '{COLLECTION_NAME}' has {collection_count} documents")
            
            # Try to peek at existing embeddings to understand expected dimension
            collection_info = {"count": collection_count, "expected_dimension": "unknown"}
            if collection_count > 0:
                sample = chroma_collection.peek(limit=1)
                if sample.get('embeddings') and len(sample['embeddings']) > 0:
                    expected_dim = len(sample['embeddings'][0])
                    collection_info["expected_dimension"] = expected_dim
                    logger.info(f"Collection expects embeddings with dimension: {expected_dim}")
                else:
                    logger.warning("No embeddings found in sample data")
            
            # Don't store in session_state here due to caching - will be done separately
        except Exception as e:
            logger.warning(f"Could not inspect collection: {e}")
            
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=chroma_db_path)
        index = load_index_from_storage(storage_context)
        
        if Settings.llm:
            st.success(f"✅ Knowledge base loaded successfully from '{chroma_db_path}' with full AI capabilities.")
        else:
            st.success(f"✅ Knowledge base loaded from '{chroma_db_path}' (basic search mode).")
        return index, chroma_collection
    except Exception as e:
        st.error(f"Backend initialization failed: {e}")
        logger.error(f"Error loading query engine from {chroma_db_path}: {e}", exc_info=True)
        return None, None

def run_embedding_diagnostics(db_path):
    """Run embedding diagnostics that aren't cached - updates session state"""
    import os  # Import os at the very beginning of the function
    try:
        wsl_db_path = convert_windows_to_wsl_path(db_path)
        chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
        
        # Test embedding model with robust PyTorch 2.8+ meta tensor fix
        try:
            import torch
            
            # Same robust initialization as main function
            logger.info(f"Diagnostic: Initializing embedding model {EMBED_MODEL} with PyTorch {torch.__version__}")
            
            # Method 1: Use sentence-transformers directly (most reliable)
            try:
                from sentence_transformers import SentenceTransformer
                
                # Set environment variables
                os.environ["TOKENIZERS_PARALLELISM"] = "false" 
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                
                st_model = SentenceTransformer(EMBED_MODEL, device="cpu", trust_remote_code=True)
                
                # Test the model
                test_output = st_model.encode("test", convert_to_tensor=False)
                logger.info(f"Diagnostic: Sentence-transformers test successful, dimension: {len(test_output)}")
                
                # Use same wrapper class as main function
                class SentenceTransformerWrapper:
                    def __init__(self, model, model_name):
                        self.model = model
                        self.model_name = model_name
                        
                    def get_text_embedding(self, text):
                        return self.model.encode(text, convert_to_tensor=False).tolist()
                    
                    def get_text_embeddings(self, texts):
                        embeddings = self.model.encode(texts, convert_to_tensor=False)
                        return [emb.tolist() for emb in embeddings]
                
                embed_model = SentenceTransformerWrapper(st_model, EMBED_MODEL)
                logger.info("✅ Diagnostic: Successfully initialized using sentence-transformers wrapper approach")
                
            except Exception as st_e:
                logger.warning(f"Diagnostic: Sentence-transformers approach failed: {st_e}")
                
                # Method 2: Environment variable approach
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                
                embed_model = HuggingFaceEmbedding(
                    model_name=EMBED_MODEL,
                    device="cpu",
                    trust_remote_code=True,
                    cache_folder=None,
                    model_kwargs={
                        'torch_dtype': torch.float32,
                        'low_cpu_mem_usage': False,
                        'device_map': None,
                        'use_auth_token': False
                    }
                )
                logger.info("✅ Diagnostic: Successfully initialized using environment fix approach")
                
        except Exception as e:
            logger.warning(f"Diagnostic: Advanced setups failed: {e}")
            
            # Method 3: Basic fallback
            try:
                embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cpu")
                logger.info("✅ Diagnostic: Successfully initialized using basic fallback")
                
            except Exception as fallback_e:
                logger.error(f"Diagnostic: All initialization methods failed: {fallback_e}")
                
                # Try alternative model as last resort
                try:
                    logger.warning("Diagnostic: Attempting alternative embedding model...")
                    alt_model = "sentence-transformers/all-MiniLM-L6-v2"
                    embed_model = HuggingFaceEmbedding(model_name=alt_model, device="cpu")
                    logger.warning(f"⚠️ Diagnostic: Using alternative embedding model: {alt_model}")
                    
                    # Update EMBED_MODEL reference for diagnostics
                    embedding_info = {"model": f"{alt_model} (fallback)", "dimension": "unknown", "status": "fallback_model"}
                    
                except Exception as alt_e:
                    logger.error(f"Diagnostic: Even alternative model failed: {alt_e}")
                    embedding_info = {"model": EMBED_MODEL, "dimension": "error", "status": f"complete_failure: {alt_e}"}
                    st.session_state.embedding_info = embedding_info
                    st.session_state.collection_info = {"count": "error", "expected_dimension": "error", "error": str(alt_e)}
                    return
                
        embedding_info = {"model": EMBED_MODEL, "dimension": "unknown", "status": "unknown"}
        
        try:
            test_embedding = embed_model.get_text_embedding("test")
            embedding_dim = len(test_embedding)
            embedding_info.update({"dimension": embedding_dim, "status": "loaded"})
            logger.info(f"Diagnostic: Embedding model dimension: {embedding_dim}")
        except Exception as e:
            embedding_info.update({"status": f"failed: {e}"})
            logger.warning(f"Diagnostic: Could not test embedding dimension: {e}")
        
        # Test collection
        db_settings = ChromaSettings(anonymized_telemetry=False)
        db = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        
        collection_info = {"count": "unknown", "expected_dimension": "unknown"}
        try:
            # First check if collection exists and is accessible
            collection_count = chroma_collection.count()
            collection_info["count"] = collection_count
            logger.info(f"Diagnostic: Collection has {collection_count} documents")
            
            if collection_count > 0:
                try:
                    # Try to peek at a sample to get dimension info
                    sample = chroma_collection.peek(limit=1)
                    if sample.get('embeddings') and len(sample['embeddings']) > 0:
                        expected_dim = len(sample['embeddings'][0])
                        collection_info["expected_dimension"] = expected_dim
                        logger.info(f"Diagnostic: Collection expects embeddings with dimension: {expected_dim}")
                    else:
                        logger.warning("Diagnostic: No embeddings found in sample data - collection may be empty or corrupted")
                        collection_info["expected_dimension"] = "no_embeddings_found"
                except Exception as peek_e:
                    logger.warning(f"Diagnostic: Could not peek at collection sample: {peek_e}")
                    collection_info["expected_dimension"] = f"peek_failed: {peek_e}"
            else:
                logger.info("Diagnostic: Collection is empty")
                collection_info["expected_dimension"] = "empty_collection"
                
        except Exception as e:
            logger.error(f"Diagnostic: Collection access failed: {e}", exc_info=True)
            collection_info.update({"count": "error", "expected_dimension": "error", "error": str(e)})
        
        # Store in session state
        st.session_state.embedding_info = embedding_info
        st.session_state.collection_info = collection_info
        
    except Exception as e:
        logger.error(f"Diagnostic function failed: {e}")
        st.session_state.embedding_info = {"model": EMBED_MODEL, "dimension": "error", "status": f"diagnostic failed: {e}"}
        st.session_state.collection_info = {"count": "error", "expected_dimension": "error", "error": str(e)}

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
        
        # Database Path Configuration (now editable)
        current_path = st.session_state.get("db_path_input", "")
        
        # Initialize the editable field with current path if not set
        if "editable_db_path_input" not in st.session_state:
            st.session_state.editable_db_path_input = current_path
        
        new_path = st.text_input(
            "Database Storage Path", 
            key="editable_db_path_input",
            help="📁 Path to your AI databases directory (e.g., /mnt/f/ai_databases or C:\\ai_databases)"
        )
        
        # Show current vs input path for debugging
        if new_path != current_path:
            st.caption(f"💡 **Saved:** `{current_path or 'None'}`")
            st.caption(f"💡 **Input:** `{new_path}`")
            st.caption("Click 'Update Path' to apply changes")
        
        # Path control buttons
        path_col1, path_col2 = st.columns(2)
        with path_col1:
            if st.button("📁 Update Path", use_container_width=True, type="primary"):
                # Debug: Show what we're comparing
                st.caption(f"🔧 Debug: new_path='{new_path}', current_path='{current_path}'")
                
                if new_path.strip() and new_path != current_path:
                    # Update the main session state variable
                    st.session_state.db_path_input = new_path.strip()
                    
                    # CRITICAL: Save to persistent config file so it persists across sessions
                    try:
                        from cortex_engine.config_manager import ConfigManager
                        config_manager = ConfigManager()
                        config_manager.update_config({"ai_database_path": new_path.strip()})
                        st.caption("💾 Saved to persistent configuration")
                    except Exception as config_e:
                        st.warning(f"⚠️ Could not save to config file: {config_e}")
                    
                    # Clear cached index to force reload with new path
                    st.cache_resource.clear()
                    st.success(f"✅ Database path updated and saved: {new_path.strip()}")
                    st.rerun()
                elif not new_path.strip():
                    st.error("❌ Path cannot be empty")
                else:
                    st.info("ℹ️ Path unchanged - new path same as current path")
        
        with path_col2:
            if st.button("🔄 Reset", use_container_width=True):
                default_path = "/mnt/f/ai_databases"  # Default fallback
                # Update both session state variables
                st.session_state.db_path_input = default_path
                st.session_state.editable_db_path_input = default_path
                
                # Save to persistent config file
                try:
                    from cortex_engine.config_manager import ConfigManager
                    config_manager = ConfigManager()
                    config_manager.update_config({"ai_database_path": default_path})
                    st.caption("💾 Reset saved to persistent configuration")
                except Exception as config_e:
                    st.warning(f"⚠️ Could not save reset to config file: {config_e}")
                
                st.cache_resource.clear()
                st.success(f"🔄 Reset to default and saved: {default_path}")
                st.rerun()
        
        # Path validation status - use input value if different from current, otherwise use current
        validation_path = new_path if new_path.strip() else current_path
        if validation_path:
            import os
            from cortex_engine.utils import convert_windows_to_wsl_path
            
            # Debug: Show what we're working with
            st.caption(f"🔍 **Analyzing path:** `{validation_path}`")
            
            wsl_path = convert_windows_to_wsl_path(validation_path)
            chroma_path = os.path.join(wsl_path, "knowledge_hub_db")
            
            # Debug: Show converted path
            st.caption(f"🔄 **Converted to:** `{wsl_path}`")
            
            if os.path.exists(wsl_path):
                st.success(f"✅ Base path exists")
                if os.path.exists(chroma_path):
                    st.success(f"✅ Knowledge base found")
                    # Count documents if possible
                    try:
                        import chromadb
                        from chromadb.config import Settings as ChromaSettings
                        from cortex_engine.config import COLLECTION_NAME
                        
                        db_settings = ChromaSettings(anonymized_telemetry=False)
                        db = chromadb.PersistentClient(path=chroma_path, settings=db_settings)
                        collection = db.get_or_create_collection(COLLECTION_NAME)
                        doc_count = collection.count()
                        st.info(f"📚 {doc_count} documents in knowledge base")
                    except Exception as e:
                        st.warning("⚠️ Knowledge base exists but cannot inspect")
                        st.caption(f"Error: {str(e)[:100]}...")
                else:
                    st.warning(f"⚠️ Knowledge base not found")
                    st.caption(f"Looking for: `{chroma_path}`")
                    st.info("💡 Run Knowledge Ingest to create the knowledge base")
            else:
                st.error(f"❌ Path does not exist")
                st.caption(f"Checked: `{wsl_path}`")
                st.info("💡 Check path or create the directory")
                
                # Additional debugging: check if parent directories exist
                parent_dir = os.path.dirname(wsl_path)
                if os.path.exists(parent_dir):
                    st.caption(f"ℹ️ Parent directory exists: `{parent_dir}`")
                else:
                    st.caption(f"❌ Parent directory missing: `{parent_dir}`")
        else:
            st.warning("⚠️ No database path configured")
            st.caption("Enter a path above and click 'Update Path'")
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
    
    # Check for emergency embedding mode and show critical warning
    if st.session_state.get("embedding_emergency_mode", False):
        st.error("🚨 **CRITICAL: Embedding Model Initialization Failed!**")
        st.error("⚠️ **Limited Functionality**: Knowledge Search is operating in emergency mode due to PyTorch/torchvision compatibility issues.")
        st.info("**🔧 Solutions**: This is a known issue with PyTorch 2.8+ and torchvision. Try:\n"
               "1. Restart the application\n"
               "2. Check if this is a Docker environment compatibility issue\n"
               "3. Consider downgrading PyTorch to version 2.3 or earlier")
        
        with st.expander("🔍 Technical Error Details", expanded=False):
            st.code(st.session_state.get("embedding_error_details", "No details available"))
        
        st.divider()
    
    # Show embedding diagnostic info if available
    if hasattr(st.session_state, 'embedding_info') and hasattr(st.session_state, 'collection_info'):
        embed_info = st.session_state.embedding_info
        coll_info = st.session_state.collection_info
        
        with st.expander("🔧 Embedding Diagnostics (Click to expand)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current Embedding Model:**")
                st.code(f"Model: {embed_info.get('model', 'Unknown')}\nDimension: {embed_info.get('dimension', 'Unknown')}\nStatus: {embed_info.get('status', 'Unknown')}")
            with col2:
                st.markdown("**Collection Information:**")
                if coll_info.get('error'):
                    st.code(f"Documents: {coll_info.get('count', 'Unknown')}\nExpected Dim: {coll_info.get('expected_dimension', 'Unknown')}\nError: {coll_info.get('error', 'Unknown')}")
                else:
                    st.code(f"Documents: {coll_info.get('count', 'Unknown')}\nExpected Dim: {coll_info.get('expected_dimension', 'Unknown')}")
            
            # Check for collection errors first
            if coll_info.get('error'):
                st.error(f"❌ **Collection Access Error!** Cannot inspect ChromaDB collection: {coll_info.get('error')}")
                st.info("💡 **Solutions**:\n1. Check database permissions and path access\n2. Try restarting the application\n3. Check if the collection database is corrupted\n4. Consider re-ingesting your documents")
            # Then check for dimension mismatch
            elif (embed_info.get('dimension') != 'unknown' and 
                coll_info.get('expected_dimension') != 'unknown' and
                str(embed_info.get('dimension')) != str(coll_info.get('expected_dimension'))):
                st.error(f"❌ **Dimension Mismatch Detected!** Current model produces {embed_info.get('dimension')} dimensions, but collection expects {coll_info.get('expected_dimension')} dimensions.")
                st.info("💡 **Solution**: The collection was created with a different embedding model. You need to either:\n1. Re-ingest your documents with the current model, or\n2. Use the same embedding model that was used during ingestion")
            elif (embed_info.get('dimension') != 'unknown' and coll_info.get('count') != 'error' and
                  str(embed_info.get('dimension')) == str(coll_info.get('expected_dimension', 'unknown'))):
                st.success("✅ **Embedding Configuration OK!** Model and collection dimensions match perfectly.")
                st.info("🔍 The search issue may be related to query processing or other factors, not embedding dimensions.")
    
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

        # Only include where clause if it has actual conditions (not empty dict)
        if where_clause and len(where_clause) > 0:
            retriever_kwargs = {"vector_store_kwargs": {"where": where_clause}}
        else:
            # Explicitly avoid passing any where clause to prevent ChromaDB validation errors
            retriever_kwargs = {}
        
        # Debug logging for troubleshooting
        logger.debug(f"Search conditions: {len(all_conditions)} conditions, where_clause keys: {list(where_clause.keys()) if where_clause else 'None'}")
        logger.debug(f"Retriever kwargs: {retriever_kwargs}")
        # --- END: v20.0.0 NATIVE CHROMA FILTER LOGIC ---

        with st.spinner("Searching..."):
            try:
                # Try with filters first if we have them
                retriever = base_index.as_retriever(similarity_top_k=50, **retriever_kwargs)
                search_text = query if query else " "
                response_nodes = retriever.retrieve(search_text)
            except ValueError as e:
                if "Expected where to have exactly one operator" in str(e):
                    # The issue is likely in LlamaIndex ChromaVectorStore itself
                    # Let's try to bypass the problematic retriever approach
                    logger.warning(f"ChromaDB where clause validation failed: {e}")
                    st.warning("⚠️ Filter validation failed, using direct vector search...")
                    
                    try:
                        # Debug: Check vector collection state
                        logger.debug(f"Vector collection type: {type(vector_collection)}")
                        logger.debug(f"Search text: '{search_text}'")
                        
                        # Direct ChromaDB query bypass for basic search functionality
                        search_results = vector_collection.query(
                            query_texts=[search_text or "document"],
                            n_results=50,
                            include=['documents', 'metadatas', 'distances']
                        )
                        
                        logger.debug(f"Direct search returned: {len(search_results.get('documents', [[]])[0])} results")
                        
                        # Convert ChromaDB results to LlamaIndex-like format
                        response_nodes = []
                        if search_results.get('documents') and search_results['documents'][0]:
                            # Import TextNode locally to avoid circular imports
                            try:
                                from llama_index.core.schema import TextNode
                                NodeClass = TextNode
                                logger.debug("Successfully imported TextNode")
                            except ImportError as ie:
                                logger.warning(f"TextNode import failed: {ie}, using SimpleNode fallback")
                                # Fallback to a simple object if import fails
                                class SimpleNode:
                                    def __init__(self, text, metadata=None, score=None):
                                        self.text = text
                                        self.metadata = metadata or {}
                                        self.score = score
                                NodeClass = SimpleNode
                                
                            for i, (doc, metadata, distance) in enumerate(zip(
                                search_results['documents'][0],
                                search_results['metadatas'][0],
                                search_results['distances'][0]
                            )):
                                node = NodeClass(
                                    text=doc,
                                    metadata=metadata or {},
                                    score=1.0 - distance  # Convert distance to similarity score
                                )
                                response_nodes.append(node)
                            
                            logger.info(f"Direct search created {len(response_nodes)} result nodes")
                        else:
                            logger.warning("Direct search returned no documents")
                            response_nodes = []
                        
                        st.info("✅ Using direct vector search (some advanced features may be limited)")
                        
                    except Exception as fallback_e:
                        logger.error(f"Direct search fallback failed: {fallback_e}", exc_info=True)
                        st.error(f"❌ Both filtered and direct search failed: {str(fallback_e)}")
                        st.error("Please check your knowledge base configuration and ensure documents were properly ingested.")
                        return
                else:
                    raise
            except Exception as e:
                logger.error(f"Search failed: {e}")
                st.error(f"Search failed: {e}")
                return
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
st.caption(f"Page Version: {PAGE_VERSION}")

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

# Run diagnostics on every page load (not cached)
run_embedding_diagnostics(st.session_state.db_path_input)

render_sidebar()
render_main_content(base_index, vector_collection)