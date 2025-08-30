# ## File: pages/13_Maintenance.py
# Version: v4.4.0
# Date: 2025-08-31
# Purpose: Consolidated maintenance and administrative functions for the Cortex Suite.
#          Combines database maintenance, system terminal, and other administrative functions
#          from various pages into a single, organized maintenance interface.
#          - BUGFIX (v1.0.1): Fixed import error by using ConfigManager instead of load_config
#          - ENHANCEMENT (v4.4.0): Added Clean Start function for complete system reset
#            Specifically addresses ChromaDB schema conflicts and provides fresh installation state

import streamlit as st
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import time

# Configure page
st.set_page_config(
    page_title="Maintenance", 
    page_icon="🔧",
    layout="wide"
)

# Page configuration
PAGE_VERSION = "v4.4.0"

# Import Cortex modules
try:
    from cortex_engine.config import INGESTED_FILES_LOG
    from cortex_engine.config_manager import ConfigManager
    from cortex_engine.utils import get_logger, convert_windows_to_wsl_path, ensure_directory
    from cortex_engine.utils.command_executor import display_command_executor_widget, SafeCommandExecutor
    from cortex_engine.ingestion_recovery import IngestionRecoveryManager
    from cortex_engine.collection_manager import WorkingCollectionManager
    from cortex_engine.setup_manager import SetupManager
    from cortex_engine.backup_manager import BackupManager
    from cortex_engine.sync_backup_manager import SyncBackupManager
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from cortex_engine.config import COLLECTION_NAME
except ImportError as e:
    st.error(f"Failed to import required Cortex modules: {e}")
    st.stop()

# Set up logging
logger = get_logger(__name__)

# Initialize session state
if 'show_confirm_clear_log' not in st.session_state:
    st.session_state.show_confirm_clear_log = False
if 'show_confirm_delete_kb' not in st.session_state:
    st.session_state.show_confirm_delete_kb = False
if 'show_confirm_clean_start' not in st.session_state:
    st.session_state.show_confirm_clean_start = False
if 'maintenance_config' not in st.session_state:
    st.session_state.maintenance_config = None

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

def perform_clean_start(db_path: str):
    """Perform complete system reset - removes all data, collections, logs, and configurations for fresh start."""
    
    # Handle Docker environment path resolution properly
    if os.path.exists('/.dockerenv'):
        # In Docker container - check multiple possible mount points
        possible_docker_paths = ["/data", "/app/data", "/home/cortex/data", db_path]
        
        docker_db_path = None
        for path in possible_docker_paths:
            if os.path.exists(path):
                docker_db_path = path
                break
        
        if not docker_db_path:
            # Fallback to /data if nothing exists (will be created)
            docker_db_path = "/data"
            
        st.info(f"🐳 **Docker Mode:** Using container path `{docker_db_path}` (checked: {possible_docker_paths})")
        final_db_path = docker_db_path
    else:
        # Non-Docker environment, use WSL path conversion
        final_db_path = convert_windows_to_wsl_path(db_path)
        st.info(f"💻 **Host Mode:** Converted `{db_path}` to `{final_db_path}`")
    
    try:
        deleted_items = []
        
        # Debug information
        st.info(f"🔍 **Debug Info:**\n- Original DB Path: `{db_path}`\n- Final Path: `{final_db_path}`\n- Running in Docker: `{os.path.exists('/.dockerenv')}`")
        
        with st.spinner("🧹 Performing Clean Start - Complete System Reset..."):
            # FIRST: Clear ingested files log from database directory BEFORE deleting the directory
            chroma_db_dir = Path(final_db_path) / "knowledge_hub_db"
            st.write(f"🔍 Looking for ChromaDB directory at: `{chroma_db_dir}`")
            st.write(f"🔍 Directory exists: `{chroma_db_dir.exists()}`")
            
            if chroma_db_dir.exists():
                # List all files in the directory for debugging
                try:
                    files_in_dir = list(chroma_db_dir.iterdir())
                    st.write(f"🔍 Files in ChromaDB directory: `{[f.name for f in files_in_dir if f.is_file()]}`")
                except Exception as e:
                    st.write(f"🔍 Could not list directory contents: {e}")
                
                ingested_files_log = chroma_db_dir / "ingested_files.log"
                st.write(f"🔍 Looking for ingested files log at: `{ingested_files_log}`")
                st.write(f"🔍 Ingested files log exists: `{ingested_files_log.exists()}`")
                
                if ingested_files_log.exists():
                    try:
                        ingested_files_log.unlink()
                        deleted_items.append(f"Ingested files log: {ingested_files_log}")
                        logger.info(f"Clean Start: Cleared ingested files log: {ingested_files_log}")
                        st.write(f"✅ Deleted ingested files log")
                    except Exception as e:
                        st.write(f"❌ Failed to delete ingested files log: {e}")
                        logger.error(f"Clean Start: Failed to delete ingested files log: {e}")
                else:
                    st.write(f"ℹ️ No ingested files log found to delete")
            else:
                st.write(f"ℹ️ ChromaDB directory does not exist - nothing to clean")
            
            # 1. Delete ChromaDB directory (vector database) - AFTER clearing the log
            if chroma_db_dir.exists() and chroma_db_dir.is_dir():
                shutil.rmtree(chroma_db_dir)
                deleted_items.append(f"ChromaDB vector database: {chroma_db_dir}")
                logger.info(f"Clean Start: Deleted ChromaDB directory: {chroma_db_dir}")
            
            # 2. Delete knowledge graph file
            graph_file = Path(final_db_path) / "knowledge_cortex.gpickle"
            if graph_file.exists():
                graph_file.unlink()
                deleted_items.append(f"Knowledge graph: {graph_file}")
                logger.info(f"Clean Start: Deleted knowledge graph: {graph_file}")
            
            # 3. Clear working collections
            project_root = Path(__file__).parent.parent
            collections_file = project_root / "working_collections.json"
            if collections_file.exists():
                collections_file.unlink()
                deleted_items.append(f"Working collections: {collections_file}")
                logger.info(f"Clean Start: Cleared working collections: {collections_file}")
            
            # 4. Clear ALL ingestion logs and metadata (comprehensive)
            logs_dir = project_root / "logs"
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    log_file.unlink()
                    deleted_items.append(f"Log file: {log_file}")
                logger.info(f"Clean Start: Cleared logs directory: {logs_dir}")
            
            # 5. Clear ALL staging and batch ingestion files (comprehensive)
            staging_patterns = [
                "staging_ingestion.json",
                "staging_*.json", 
                "staging_test.json",
                "batch_progress.json",
                "batch_state.json",
                "ingestion_progress.json",
                "failed_ingestion.json"
            ]
            
            for pattern in staging_patterns:
                for staging_file in project_root.glob(pattern):
                    if staging_file.is_file():
                        staging_file.unlink()
                        deleted_items.append(f"Staging/batch file: {staging_file}")
                        logger.info(f"Clean Start: Cleared staging file: {staging_file}")
            
            # 6. Clear session state and cached configuration files
            config_files_to_clear = [
                project_root / "cortex_config.json",
                project_root / "boilerplate.json"
            ]
            
            for config_file in config_files_to_clear:
                if config_file.exists():
                    # For cortex_config.json, only reset database paths, keep other settings
                    if config_file.name == "cortex_config.json":
                        try:
                            with open(config_file, 'r') as f:
                                config_data = json.load(f)
                            # Reset database-specific paths but keep other settings
                            config_data.pop('ai_database_path', None)
                            config_data.pop('knowledge_source_path', None) 
                            with open(config_file, 'w') as f:
                                json.dump(config_data, f, indent=2)
                            deleted_items.append(f"Reset database paths in: {config_file}")
                        except Exception as e:
                            logger.warning(f"Could not reset config file {config_file}: {e}")
                    else:
                        config_file.unlink()
                        deleted_items.append(f"Configuration file: {config_file}")
            
            # 7. Clear ingestion recovery metadata  
            recovery_metadata_dir = Path(final_db_path) / "recovery_metadata"
            if recovery_metadata_dir.exists():
                shutil.rmtree(recovery_metadata_dir)
                deleted_items.append(f"Recovery metadata: {recovery_metadata_dir}")
                logger.info(f"Clean Start: Cleared recovery metadata: {recovery_metadata_dir}")
            
            # 8. Clear Streamlit session state cache files
            streamlit_cache_patterns = [".streamlit/cache", "__pycache__", "*.pyc"]
            for pattern in streamlit_cache_patterns:
                for cache_item in project_root.rglob(pattern):
                    if cache_item.is_file():
                        cache_item.unlink()
                        deleted_items.append(f"Cache file: {cache_item}")
                    elif cache_item.is_dir() and cache_item.name in ['.streamlit', '__pycache__']:
                        shutil.rmtree(cache_item)
                        deleted_items.append(f"Cache directory: {cache_item}")
            
            # 9. Clear any remaining temporary and state files
            temp_patterns = ["*.tmp", "*.temp", "temp_*", "*.pid", "*.lock", "*_state.json"]
            for pattern in temp_patterns:
                for temp_file in project_root.glob(pattern):
                    if temp_file.is_file():
                        temp_file.unlink()
                        deleted_items.append(f"Temporary file: {temp_file}")
        
        # Success message
        st.success("✅ **Clean Start completed successfully!**")
        st.markdown("### 🗑️ Cleaned Items:")
        for item in deleted_items:
            st.write(f"- {item}")
        
        st.info("""
        ### 🚀 Next Steps:
        1. **Navigate to Knowledge Ingest page** to set database path
        2. **Run document ingestion** to rebuild your knowledge base  
        3. **Start fresh** - all schema conflicts should be resolved
        
        Your system now has a completely clean slate and should work without any ChromaDB schema errors.
        """)
        
        logger.info(f"Clean Start completed successfully - {len(deleted_items)} items cleaned")
        
    except Exception as e:
        error_msg = f"Clean Start failed: {e}"
        logger.error(f"Clean Start error: {e}")
        st.error(f"❌ {error_msg}")
        st.error("""
        **If Clean Start fails:**
        1. Try manually deleting the database directory
        2. Restart the application 
        3. Check logs for detailed error information
        """)

@st.cache_resource
def init_chroma_client(db_path):
    """Initialize ChromaDB client for maintenance operations"""
    if not db_path:
        return None
        
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
    
    if not os.path.isdir(chroma_db_path):
        return None
        
    try:
        db_settings = ChromaSettings(anonymized_telemetry=False)
        return chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return None

def load_maintenance_config():
    """Load configuration for maintenance operations"""
    if st.session_state.maintenance_config is None:
        try:
            config_manager = ConfigManager()
            config = config_manager.get_config()
            
            # Map ConfigManager keys to expected keys
            maintenance_config = {
                'db_path': config.get('ai_database_path', '/tmp/cortex_db'),
                'knowledge_source_path': config.get('knowledge_source_path', ''),
            }
            
            st.session_state.maintenance_config = maintenance_config
            return maintenance_config
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            return None
    return st.session_state.maintenance_config

def clear_ingestion_log_file():
    """Clear the ingestion log file to allow re-ingestion of all files"""
    try:
        config = load_maintenance_config()
        if not config:
            return
        
        db_path = config.get('db_path', '/tmp/cortex_db')
        wsl_db_path = convert_windows_to_wsl_path(db_path)
        log_path = Path(wsl_db_path) / "knowledge_hub_db" / INGESTED_FILES_LOG
        
        if log_path.exists():
            os.remove(log_path)
            st.success(f"✅ Ingestion log cleared successfully: {log_path}")
            logger.info(f"Ingestion log cleared: {log_path}")
        else:
            st.warning(f"Log file not found: {log_path}")
    except Exception as e:
        st.error(f"❌ Failed to clear ingestion log: {e}")
        logger.error(f"Failed to clear ingestion log: {e}")

def delete_knowledge_base(db_path):
    """Permanently delete the entire knowledge base"""
    try:
        wsl_db_path = convert_windows_to_wsl_path(db_path)
        kb_dir = Path(wsl_db_path) / "knowledge_hub_db"
        
        if kb_dir.exists():
            shutil.rmtree(kb_dir)
            st.success(f"✅ Knowledge base deleted successfully: {kb_dir}")
            logger.info(f"Knowledge base deleted: {kb_dir}")
        else:
            st.warning(f"Knowledge base directory not found: {kb_dir}")
    except Exception as e:
        st.error(f"❌ Failed to delete knowledge base: {e}")
        logger.error(f"Failed to delete knowledge base: {e}")

def display_header():
    """Display page header with navigation and information"""
    st.title("🔧 Maintenance & Administration")
    st.caption(f"Version: {PAGE_VERSION} • Consolidated System Maintenance Interface")
    
    st.markdown("""
    **⚠️ Important:** This page contains powerful system maintenance functions that can modify or delete data.  
    Please use these functions with caution and ensure you have appropriate backups before proceeding.
    """)
    
    st.divider()

def display_database_maintenance():
    """Display database maintenance and recovery functions"""
    st.header("🗄️ Database Maintenance")
    
    config = load_maintenance_config()
    if not config:
        st.error("Cannot load configuration for database operations")
        return
    
    db_path = config.get('db_path', '/tmp/cortex_db')
    
    with st.expander("🚀 Clean Start - Complete System Reset", expanded=True):
        st.markdown("### 🧹 Clean Start Function")
        st.warning("""
        **NEW:** Complete system reset function that addresses database schema issues, collection conflicts, and provides a fresh start.
        This function is specifically designed to resolve ChromaDB schema errors like 'collections.config_json_str' column missing.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Clean Start will:**
            - ✅ Delete entire knowledge base directory (ChromaDB)
            - ✅ Delete knowledge graph file (.gpickle)  
            - ✅ Clear ALL ingestion logs and progress files
            - ✅ Remove ingested files log from database directory
            - ✅ Clear ALL staging and batch ingestion files (including failed ingests)
            - ✅ Reset working collections (working_collections.json)
            - ✅ Clear ingestion recovery metadata
            - ✅ Remove Streamlit cache and session state files
            - ✅ Clear temporary files, lock files, and state files
            - ✅ Reset database configuration paths
            - ✅ Fix ChromaDB schema conflicts and version issues
            - ✅ Provide completely fresh installation state
            """)
            
            st.info("**Use Clean Start when:**")
            st.markdown("""
            - Getting 'collections.config_json_str' schema errors
            - Collection Management shows connection errors  
            - Docker vs non-Docker database conflicts
            - ChromaDB version compatibility issues
            - System appears corrupted or inconsistent
            - **Failed batch ingests** showing up in Knowledge Ingest page
            - Half-finished ingestion operations need clearing
            - Want completely fresh system without any residual files
            """)
        
        with col2:
            if st.button("🚀 Clean Start Reset", use_container_width=True, type="primary"):
                st.session_state.show_confirm_clean_start = True
                
            if st.session_state.get("show_confirm_clean_start"):
                st.error("⚠️ **COMPLETE SYSTEM RESET**")
                st.warning("This will delete ALL data and provide a completely fresh start. All knowledge base content, collections, and configurations will be lost.")
                
                c1, c2 = st.columns(2)
                if c1.button("✅ YES, CLEAN START", use_container_width=True):
                    perform_clean_start(db_path)
                    st.session_state.show_confirm_clean_start = False
                    st.rerun()
                if c2.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_confirm_clean_start = False
                    st.rerun()

    with st.expander("⚙️ Basic Database Operations", expanded=False):
        st.subheader("Clear Ingestion Log")
        st.info("This action allows all files to be scanned and re-ingested. Useful for rebuilding the knowledge base from scratch.")
        
        if st.button("Clear Ingestion Log..."):
            st.session_state.show_confirm_clear_log = True
            
        if st.session_state.get("show_confirm_clear_log"):
            st.warning(f"This will delete the **{INGESTED_FILES_LOG}** file. The system will then see all source files as new on the next scan. Are you sure?")
            c1, c2 = st.columns(2)
            if c1.button("YES, Clear the Log", use_container_width=True, type="primary"):
                clear_ingestion_log_file()
                st.session_state.show_confirm_clear_log = False
                st.rerun()
            if c2.button("Cancel", use_container_width=True):
                st.session_state.show_confirm_clear_log = False
                st.rerun()
        
        st.divider()
        
        st.subheader("Delete Entire Knowledge Base")
        st.error("⚠️ **DANGER:** This is the most destructive action.")
        
        if st.button("Permanently Delete Knowledge Base...", type="primary"):
            st.session_state.show_confirm_delete_kb = True
            
        if st.session_state.get("show_confirm_delete_kb"):
            st.warning(f"This will permanently delete the entire **knowledge_hub_db** directory. This action cannot be undone.")
            c1, c2 = st.columns(2)
            if c1.button("YES, DELETE EVERYTHING", use_container_width=True):
                delete_knowledge_base(db_path)
                st.session_state.show_confirm_delete_kb = False
                st.rerun()
            if c2.button("Cancel Deletion", use_container_width=True):
                st.session_state.show_confirm_delete_kb = False
                st.rerun()

    # Add database deduplication section
    with st.expander("🔧 Database Deduplication & Optimization", expanded=False):
        st.subheader("🔧 Database Deduplication")
        st.markdown("Remove duplicate documents from the knowledge base to improve performance and storage efficiency.")
        
        # Initialize ChromaDB client for deduplication
        chroma_client = init_chroma_client(db_path)
        if not chroma_client:
            st.warning("ChromaDB not accessible. Cannot perform deduplication operations.")
        else:
            try:
                vector_collection = chroma_client.get_collection(name=COLLECTION_NAME)
                collection_mgr = WorkingCollectionManager()
                
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
                                    logger.info(f"Deduplication analysis completed via Maintenance UI: {results['duplicates_found']} duplicates found")
                                else:
                                    st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                                    
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                                logger.error(f"Maintenance UI deduplication analysis failed: {e}")
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
                                                
                                                logger.info(f"Deduplication cleanup completed via Maintenance UI: {removed_count} documents removed")
                                                
                                                # Show recommendation to restart
                                                st.success("🔄 **Recommendation:** Restart the application to ensure optimal performance with the cleaned database.")
                                                
                                            else:
                                                st.error(f"❌ Cleanup failed: {cleanup_results.get('error', 'Unknown error')}")
                                                
                                        except Exception as e:
                                            st.error(f"❌ Cleanup failed: {str(e)}")
                                            logger.error(f"Maintenance UI deduplication cleanup failed: {e}")
                    
                    elif results.get('status') == 'analysis_complete' and results.get('duplicates_found', 0) == 0:
                        st.success("✅ No duplicates found! Your knowledge base is already optimized.")
                        
                    elif results.get('status') == 'no_documents':
                        st.info("ℹ️ No documents found in the knowledge base.")
                        
            except Exception as e:
                st.error(f"Could not access vector collection: {e}")

    with st.expander("🔧 Advanced Database Recovery & Repair", expanded=False):
        st.markdown("""
        **Recover from failed ingestions** or **repair inconsistencies** in your knowledge base.
        Use this when ingestion processes are interrupted or documents seem to be missing.
        """)
        
        try:
            recovery_manager = IngestionRecoveryManager(db_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 Analyze Current State")
                if st.button("🔍 Analyze Ingestion State", use_container_width=True, key="analyze_recovery_state"):
                    with st.spinner("🔄 Analyzing knowledge base state..."):
                        analysis = recovery_manager.analyze_ingestion_state()
                        st.session_state.recovery_analysis = analysis
                        st.rerun()
                
                if "recovery_analysis" in st.session_state:
                    analysis = st.session_state.recovery_analysis
                    
                    # Show key statistics
                    if "statistics" in analysis:
                        stats = analysis["statistics"]
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("📁 Ingested Files", stats.get("ingested_files_count", 0))
                        with stat_col2:
                            st.metric("📄 KB Documents", stats.get("chromadb_docs_count", 0))
                        with stat_col3:
                            st.metric("🚨 Orphaned", stats.get("orphaned_count", 0))
                    
                    # Show issues found
                    if analysis.get("issues_found"):
                        st.warning("**Issues Found:**")
                        for issue in analysis["issues_found"]:
                            st.write(f"• {issue}")
                    else:
                        st.success("✅ No issues detected")
            
            with col2:
                st.markdown("#### 🛠️ Recovery Actions")
                
                # Quick recovery collection creation
                if st.button("🚀 Quick Recovery: Create Collection from Recent Files", use_container_width=True, key="quick_recovery_recent"):
                    collection_name = st.text_input("Collection name:", value="recovered_files", key="quick_recovery_name")
                    
                    if collection_name:
                        with st.spinner(f"🔄 Creating recovery collection '{collection_name}'..."):
                            result = recovery_manager.create_recovery_collection_from_recent(collection_name, hours_back=24)
                            
                            if result["status"] == "success":
                                st.success(f"✅ Created '{collection_name}' with {result['documents_added']} documents!")
                            else:
                                st.error(f"❌ Recovery failed: {result.get('error', 'Unknown error')}")
                
                # Orphaned document recovery
                if st.session_state.get("recovery_analysis", {}).get("statistics", {}).get("orphaned_count", 0) > 0:
                    st.markdown("**Orphaned Documents Detected**")
                    orphaned_count = st.session_state.recovery_analysis["statistics"]["orphaned_count"]
                    collection_name = st.text_input(f"Recover {orphaned_count} orphaned documents to collection:", 
                                                   value="recovered_orphaned", key="orphaned_recovery_name")
                    
                    if st.button(f"🔄 Recover {orphaned_count} Documents", use_container_width=True, key="recover_orphaned_docs"):
                        if collection_name:
                            with st.spinner("🔄 Recovering orphaned documents..."):
                                result = recovery_manager.recover_orphaned_documents(collection_name)
                                
                                if result["status"] == "success":
                                    st.success(f"✅ Recovered {result['recovered_count']} documents to '{collection_name}'!")
                                else:
                                    st.error(f"❌ Recovery failed: {result.get('error', 'Unknown error')}")
                
                # Collection repair
                if st.button("🔧 Auto-Repair Collections", use_container_width=True, key="auto_repair_collections"):
                    with st.spinner("🔄 Repairing collection inconsistencies..."):
                        result = recovery_manager.auto_repair_collections()
                        
                        if result["status"] == "success":
                            if result["collections_cleaned"] > 0:
                                st.success(f"✅ Repaired {result['collections_cleaned']} collections, removed {result['invalid_refs_removed']} invalid references")
                            else:
                                st.info("✅ No repairs needed - all collections are consistent")
                        else:
                            st.error(f"❌ Repair failed: {result.get('error', 'Unknown error')}")
            
            # Show recommendations if available
            if st.session_state.get("recovery_analysis", {}).get("recommendations"):
                st.markdown("---")
                st.markdown("#### 💡 Recommended Actions")
                recommendations = st.session_state.recovery_analysis["recommendations"]
                for rec in recommendations:
                    st.info(f"💡 {rec}")
                    
        except Exception as e:
            st.error(f"Failed to initialize recovery manager: {e}")

def display_system_terminal():
    """Display system terminal and command execution interface"""
    st.header("💻 System Terminal")
    
    st.markdown("""
    This secure terminal interface allows you to execute system commands safely within the Cortex Suite environment.
    Only whitelisted commands are permitted to ensure system security.
    """)
    
    # Quick action buttons for common tasks
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📦 Check Models", use_container_width=True):
            st.session_state.quick_command = "ollama list"
            st.rerun()
    
    with col2:
        if st.button("🔍 System Status", use_container_width=True):
            st.session_state.quick_command = "docker ps"
            st.rerun()
    
    with col3:
        if st.button("📊 Disk Usage", use_container_width=True):
            st.session_state.quick_command = "df -h"
            st.rerun()
    
    with col4:
        if st.button("🔄 Clear Terminal", use_container_width=True):
            st.session_state.quick_command = None
            if 'terminal_output' in st.session_state:
                del st.session_state.terminal_output
            st.rerun()
    
    # Display command executor widget
    try:
        display_command_executor_widget()
    except Exception as e:
        st.error(f"Failed to load command executor: {e}")

def display_setup_maintenance():
    """Display setup and installation maintenance functions"""
    st.header("⚙️ Setup & Installation")
    
    with st.expander("🔄 Reset System Setup", expanded=False):
        st.markdown("""
        Reset the system setup state if installation gets stuck or needs to be rerun.
        This will clear setup progress and allow you to run the setup wizard again.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset Setup State", use_container_width=True):
                try:
                    setup_manager = SetupManager()
                    setup_manager.reset_setup()
                    
                    # Clear related session state
                    keys_to_clear = [k for k in st.session_state.keys() if 'setup' in k.lower() or 'installation' in k.lower()]
                    for key in keys_to_clear:
                        del st.session_state[key]
                    
                    st.success("✅ Setup state reset successfully!")
                    logger.info("Setup state reset via maintenance page")
                    
                except Exception as e:
                    st.error(f"❌ Failed to reset setup: {e}")
        
        with col2:
            st.info("After resetting, you can navigate to the Setup Wizard to reconfigure the system.")

def display_backup_management():
    """Display backup management functions"""
    st.header("💾 Backup Management")
    
    config = load_maintenance_config()
    if not config:
        st.error("Cannot load configuration for backup operations")
        return
    
    try:
        db_path = config.get('db_path', '/tmp/cortex_db')
        # Convert to proper WSL path format for backup manager
        wsl_db_path = convert_windows_to_wsl_path(db_path)
        
        # Ensure the backups directory exists using centralized utility
        backups_dir = ensure_directory(Path(wsl_db_path) / "backups")
        
        backup_manager = BackupManager(wsl_db_path)
        
        with st.expander("📦 Create New Backup", expanded=False):
            backup_name = st.text_input("Backup name (optional):", placeholder="my_backup_2025_08_27")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Create Backup", use_container_width=True):
                    with st.spinner("Creating backup..."):
                        try:
                            backup_id = backup_manager.create_backup(backup_name if backup_name else None)
                            st.success(f"✅ Backup created successfully: {backup_id}")
                        except Exception as e:
                            st.error(f"❌ Backup failed: {e}")
        
        with st.expander("📋 Manage Existing Backups", expanded=False):
            try:
                backups = backup_manager.list_backups()
                
                if not backups:
                    st.info("No backups found.")
                else:
                    st.write(f"Found {len(backups)} backup(s):")
                    
                    for i, backup in enumerate(backups):
                        with st.container(border=True):
                            backup_col1, backup_col2, backup_col3 = st.columns([3, 1, 1])
                            
                            with backup_col1:
                                st.markdown(f"**{backup.backup_id}**")
                                st.caption(f"Created: {backup.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                                st.caption(f"Size: {backup.size_mb:.1f} MB • Files: {backup.file_count}")
                            
                            with backup_col2:
                                if st.button("🔄 Restore", key=f"restore_backup_{i}", help="Restore this backup"):
                                    st.warning("⚠️ Restoring will overwrite current data!")
                                    if st.button("✅ Confirm Restore", key=f"confirm_restore_{i}"):
                                        with st.spinner("Restoring backup..."):
                                            try:
                                                success = backup_manager.restore_backup(backup.backup_id)
                                                if success:
                                                    st.success("✅ Backup restored successfully!")
                                                else:
                                                    st.error("❌ Restore failed")
                                            except Exception as e:
                                                st.error(f"❌ Restore failed: {e}")
                            
                            with backup_col3:
                                if st.button("🗑️ Delete", key=f"delete_backup_{i}", help="Delete this backup"):
                                    st.session_state[f"confirm_delete_backup_{backup.backup_id}"] = True
                                    st.rerun()
                        
                        # Confirmation for backup deletion
                        if st.session_state.get(f"confirm_delete_backup_{backup.backup_id}", False):
                            st.warning(f"⚠️ Are you sure you want to delete backup `{backup.backup_id}`?")
                            
                            confirm_col1, confirm_col2 = st.columns(2)
                            with confirm_col1:
                                if st.button("✅ Yes, Delete", key=f"confirm_delete_final_{i}", type="primary"):
                                    try:
                                        success = backup_manager.delete_backup(backup.backup_id)
                                        if success:
                                            st.success(f"✅ Backup `{backup.backup_id}` deleted successfully")
                                            del st.session_state[f"confirm_delete_backup_{backup.backup_id}"]
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to delete backup")
                                    except Exception as e:
                                        st.error(f"❌ Delete failed: {e}")
                            
                            with confirm_col2:
                                if st.button("❌ Cancel", key=f"cancel_delete_{i}"):
                                    del st.session_state[f"confirm_delete_backup_{backup.backup_id}"]
                                    st.rerun()
                                    
            except Exception as e:
                st.error(f"Failed to list backups: {e}")
                
    except Exception as e:
        st.error(f"Failed to initialize backup manager: {e}")

def display_changelog_viewer():
    """Display the project changelog viewer"""
    st.markdown("## 📋 Project Changelog")
    st.markdown("View the complete development history and version changes for the Cortex Suite.")
    
    # Get project root path
    project_root = Path(__file__).parent.parent
    changelog_path = project_root / "CHANGELOG.md"
    
    if not changelog_path.exists():
        st.error("❌ CHANGELOG.md not found in project root")
        return
    
    try:
        # Read the changelog
        with open(changelog_path, 'r', encoding='utf-8') as f:
            changelog_content = f.read()
        
        # Display changelog info
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info(f"📁 **File Location:** `{changelog_path.relative_to(project_root)}`")
        
        with col2:
            # Get file stats
            stat = changelog_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            st.info(f"🕒 **Last Updated:** {last_modified}")
        
        with col3:
            # File size
            size_kb = stat.st_size / 1024
            st.info(f"📏 **Size:** {size_kb:.1f} KB")
        
        st.divider()
        
        # Display options
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            show_full = st.checkbox("📖 Show Full Changelog", 
                                  value=st.session_state.get("changelog_show_full", False),
                                  key="changelog_show_full_checkbox")
        
        with col2:
            if st.button("🔄 Refresh", key="changelog_refresh"):
                st.rerun()
        
        with col3:
            # Download button
            st.download_button(
                label="💾 Download Changelog",
                data=changelog_content,
                file_name="CHANGELOG.md",
                mime="text/markdown",
                key="changelog_download"
            )
        
        st.divider()
        
        # Parse and display changelog sections
        if show_full:
            # Show entire changelog
            st.markdown("### 📚 Complete Changelog")
            st.markdown(changelog_content)
        else:
            # Show recent versions (first few entries)
            lines = changelog_content.split('\n')
            
            # Find recent version entries (lines starting with ##)
            version_lines = []
            current_section = []
            version_count = 0
            
            for line in lines:
                if line.startswith('## v') and version_count < 5:  # Show last 5 versions
                    if current_section:
                        version_lines.append('\n'.join(current_section))
                        current_section = []
                        version_count += 1
                    current_section = [line]
                elif line.startswith('## ') and not line.startswith('## ['):
                    # Stop at non-version headers
                    break
                elif current_section:
                    current_section.append(line)
            
            # Add the last section
            if current_section and version_count < 5:
                version_lines.append('\n'.join(current_section))
            
            if version_lines:
                st.markdown("### 🆕 Recent Updates (Last 5 Versions)")
                for section in version_lines:
                    st.markdown(section)
                    st.divider()
                
                st.info("💡 **Tip:** Check 'Show Full Changelog' above to see complete version history.")
            else:
                st.warning("⚠️ Could not parse changelog sections. Showing raw content:")
                st.text(changelog_content[:2000] + "..." if len(changelog_content) > 2000 else changelog_content)
    
    except Exception as e:
        st.error(f"❌ Failed to read changelog: {e}")
        logger.error(f"Changelog viewer error: {e}")

def main():
    """Main function to orchestrate the maintenance interface"""
    display_header()
    
    # Create tabs for different maintenance categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗄️ Database", 
        "💻 Terminal", 
        "⚙️ Setup", 
        "💾 Backups",
        "📋 Changelog",
        "ℹ️ Info"
    ])
    
    with tab1:
        display_database_maintenance()
    
    with tab2:
        display_system_terminal()
    
    with tab3:
        display_setup_maintenance()
    
    with tab4:
        display_backup_management()
    
    with tab5:
        display_changelog_viewer()
    
    with tab6:
        st.markdown("""
        ## 📋 Maintenance Information
        
        This maintenance interface consolidates system administration functions from across the Cortex Suite:
        
        **Database Functions:**
        - Clear ingestion logs for re-processing files
        - Delete and rebuild knowledge base
        - Analyze and repair database inconsistencies
        - Recover orphaned documents and failed ingestions
        
        **System Functions:**
        - Execute safe system commands
        - Check model availability and system status  
        - Monitor disk usage and resource consumption
        
        **Setup Functions:**
        - Reset installation state if setup gets stuck
        - Reconfigure system components
        
        **Backup Functions:**
        - Create and restore knowledge base backups
        - Manage backup lifecycle and storage
        
        **⚠️ Important Notes:**
        - Always backup your data before performing destructive operations
        - Some functions require system administrator privileges
        - Monitor system resources during intensive operations
        - Check logs for detailed error information if operations fail
        """)
        
        st.markdown(f"**Page Version:** {PAGE_VERSION} | **Date:** 2025-08-27")

if __name__ == "__main__":
    main()