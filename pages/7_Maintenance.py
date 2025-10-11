# ## File: pages/7_Maintenance.py
# Version: v4.10.1
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
PAGE_VERSION = "v4.10.1"

# Import Cortex modules
try:
    from cortex_engine.config import INGESTED_FILES_LOG
    from cortex_engine.config_manager import ConfigManager
    from cortex_engine.utils import get_logger, convert_windows_to_wsl_path, ensure_directory
    from cortex_engine.utils.command_executor import display_command_executor_widget, SafeCommandExecutor
    from cortex_engine.utils.performance_monitor import get_performance_monitor, get_all_stats, get_session_summary
    from cortex_engine.utils.gpu_monitor import get_gpu_memory_info, get_device_recommendations
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

def delete_ingested_document_database(db_path: str):
    """Delete the ingested document database with proper error handling and logging."""
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_dir = Path(wsl_db_path) / "knowledge_hub_db"
    graph_file = Path(wsl_db_path) / "knowledge_cortex.gpickle"
    collections_file = Path(wsl_db_path) / "working_collections.json"
    batch_state_file = Path(wsl_db_path) / "batch_state.json"
    staging_file = Path(wsl_db_path) / "staging_ingestion.json"
    
    try:
        deleted_items = []
        errors = []
        
        # Delete ChromaDB directory
        if chroma_db_dir.exists() and chroma_db_dir.is_dir():
            try:
                shutil.rmtree(chroma_db_dir)
                deleted_items.append(f"ChromaDB directory: {chroma_db_dir}")
                logger.info(f"Successfully deleted ChromaDB directory: {chroma_db_dir}")
            except Exception as e:
                error_msg = f"Failed to delete ChromaDB directory: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Delete knowledge graph file
        if graph_file.exists():
            try:
                graph_file.unlink()
                deleted_items.append(f"Knowledge graph: {graph_file}")
                logger.info(f"Successfully deleted knowledge graph: {graph_file}")
            except Exception as e:
                error_msg = f"Failed to delete knowledge graph file: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Delete collections file (only stored in KB database path now)
        if collections_file.exists():
            try:
                collections_file.unlink()
                deleted_items.append(f"Collections file: {collections_file}")
                logger.info(f"Successfully deleted collections file: {collections_file}")
            except Exception as e:
                error_msg = f"Failed to delete collections file: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Delete batch state file
        if batch_state_file.exists():
            try:
                batch_state_file.unlink()
                deleted_items.append(f"Batch state file: {batch_state_file}")
                logger.info(f"Successfully deleted batch state file: {batch_state_file}")
            except Exception as e:
                error_msg = f"Failed to delete batch state file: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Delete staging ingestion file
        if staging_file.exists():
            try:
                staging_file.unlink()
                deleted_items.append(f"Staging file: {staging_file}")
                logger.info(f"Successfully deleted staging file: {staging_file}")
            except Exception as e:
                error_msg = f"Failed to delete staging file: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Report results
        if deleted_items:
            st.success(f"✅ Successfully deleted ingested document database components:\\n" + "\\n".join(f"- {item}" for item in deleted_items))
            logger.info("Ingested document database deletion completed successfully")
        
        if errors:
            st.error(f"❌ Some items could not be deleted:\\n" + "\\n".join(f"- {error}" for error in errors))
            
        if not deleted_items and not errors:
            st.warning("⚠️ No ingested document database components found to delete.")
            logger.warning(f"No ingested document database found at: {wsl_db_path}")
            
    except Exception as e:
        error_msg = f"Failed to delete ingested document database: {e}"
        logger.error(f"Ingested document database deletion failed: {e}")
        st.error(f"❌ {error_msg}")
    
    # Reset the confirmation state  
    st.session_state.show_confirm_delete_kb = False

def perform_clean_start(db_path: str):
    """Perform complete system reset - removes all data, collections, logs, and configurations for fresh start."""
    
    # Initialize comprehensive debug log
    debug_log_lines = []
    debug_log_lines.append("=" * 80)
    debug_log_lines.append("CORTEX SUITE CLEAN START DEBUG LOG")
    debug_log_lines.append("=" * 80)
    debug_log_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    debug_log_lines.append(f"Original DB Path: {db_path}")
    debug_log_lines.append(f"Running in Docker: {os.path.exists('/.dockerenv')}")
    
    # Handle Docker environment path resolution properly
    if os.path.exists('/.dockerenv'):
        # In Docker container - use the configured path directly
        # Docker volumes handle the path mapping, so we use the path as configured by user
        final_db_path = db_path
        msg = f"🐳 **Docker Mode:** Using configured database path `{final_db_path}` (Docker handles volume mapping)"
        st.info(msg)
        debug_log_lines.append("ENVIRONMENT: Docker Container")
        debug_log_lines.append(f"Final DB Path: {final_db_path}")
        debug_log_lines.append("Path Conversion: None (Docker handles volume mapping)")
    else:
        # Non-Docker environment, use WSL path conversion for Windows paths
        final_db_path = convert_windows_to_wsl_path(db_path)
        msg = f"💻 **Host Mode:** Converted `{db_path}` to `{final_db_path}`"
        st.info(msg)
        debug_log_lines.append("ENVIRONMENT: Host/WSL")
        debug_log_lines.append(f"Final DB Path: {final_db_path}")
        debug_log_lines.append("Path Conversion: Applied WSL conversion")
    
    debug_log_lines.append("")
    debug_log_lines.append("OPERATIONS LOG:")
    debug_log_lines.append("-" * 40)

    try:
        deleted_items = []
        
        # Show current knowledge base location before deletion
        chroma_db_dir = Path(final_db_path) / "knowledge_hub_db"
        debug_log_lines.append(f"Target ChromaDB Directory: {chroma_db_dir}")
        debug_log_lines.append(f"Directory Exists: {chroma_db_dir.exists()}")
        
        if chroma_db_dir.exists():
            st.warning(f"📁 **Current Knowledge Base Found:** `{chroma_db_dir}`\n\nThis directory and all contents will be permanently deleted.")
            debug_log_lines.append("STATUS: Knowledge base directory found - will be deleted")
        else:
            st.info(f"📁 **Knowledge Base Location:** `{chroma_db_dir}`\n\nDirectory does not exist - will clean up other logs and configurations only.")
            debug_log_lines.append("STATUS: No knowledge base directory found - cleanup only")
        
        # Show debug information with clear path mapping and current file status
        with st.container(border=True):
            st.markdown("#### 🔍 Pre‑Operation Debug Information")
            st.write(f"Windows path: `{db_path}`")
            st.write(f"Resolved path: `{final_db_path}`")
            # Current state check
            st.markdown("**Current State Check**")
            items = [
                ("ChromaDB directory", Path(final_db_path) / "knowledge_hub_db"),
                ("Collections file", Path(final_db_path) / "working_collections.json"),
                ("Staging file", Path(final_db_path) / "staging_ingestion.json"),
                ("Batch state", Path(final_db_path) / "batch_state.json"),
                ("Knowledge graph", Path(final_db_path) / "knowledge_cortex.gpickle"),
            ]
            for label, p in items:
                exists = p.exists()
                icon = "✅" if exists else "⚪"
                st.write(f"{icon} {label}: `{p}` (exists={exists})")
            # Append to debug buffer as well
            debug_display = "\n".join(debug_log_lines)
            st.text_area("Debug Log", value=debug_display, height=140, help="Copy this info if you need to report issues")
        
        st.divider()
        st.header("🧹 Clean Start Operations")
        st.info("**The following operations will be performed step by step. You can read each step as it completes.**")
        
        # STEP 1: Check ChromaDB directory
        st.subheader("Step 1: Checking ChromaDB Directory")
        chroma_db_dir = Path(final_db_path) / "knowledge_hub_db"
        
        st.success(f"🔍 Looking for ChromaDB directory at: `{chroma_db_dir}`")
        debug_log_lines.append(f"OPERATION: Checking ChromaDB directory at {chroma_db_dir}")
        
        st.success(f"🔍 Directory exists: `{chroma_db_dir.exists()}`")
        debug_log_lines.append(f"RESULT: Directory exists = {chroma_db_dir.exists()}")
        
        # Show current step results (non-expander)
        with st.container(border=True):
            st.markdown("#### 📋 Step 1 Results")
            step1_results = f"""Path being checked: {chroma_db_dir}
Directory exists: {chroma_db_dir.exists()}
"""
            st.text_area("Step 1 Debug Information", value=step1_results, height=100)
        
        if chroma_db_dir.exists():
            st.subheader("Step 2: Analyzing Directory Contents")
            
            # List all files in the directory for debugging
            try:
                files_in_dir = list(chroma_db_dir.iterdir())
                file_names = [f.name for f in files_in_dir if f.is_file()]
                dir_names = [f.name for f in files_in_dir if f.is_dir()]
                st.success(f"🔍 Files in ChromaDB directory: `{file_names}`")
                st.success(f"🔍 Subdirectories: `{dir_names}`")
                debug_log_lines.append(f"FILES FOUND: {file_names}")
                debug_log_lines.append(f"SUBDIRECTORIES FOUND: {dir_names}")
                
                # Show directory analysis results (non-expander)
                with st.container(border=True):
                    st.markdown("#### 📋 Step 2 Results - Directory Analysis")
                    step2_results = f"""Directory contents:
Files found: {file_names}
Subdirectories found: {dir_names}
Total items: {len(files_in_dir)}
"""
                    st.text_area("Step 2 Debug Information", value=step2_results, height=120)
                    
            except Exception as e:
                st.error(f"❌ Could not list directory contents: {e}")
                debug_log_lines.append(f"ERROR: Could not list directory contents: {e}")
            
            st.subheader("Step 3: Checking Ingested Files Log")
            ingested_files_log = chroma_db_dir / "ingested_files.log"
            st.success(f"🔍 Looking for ingested files log at: `{ingested_files_log}`")
            st.success(f"🔍 Ingested files log exists: `{ingested_files_log.exists()}`")
            debug_log_lines.append(f"INGESTED_LOG_PATH: {ingested_files_log}")
            debug_log_lines.append(f"INGESTED_LOG_EXISTS: {ingested_files_log.exists()}")
            
            # Show ingested log check results (non-expander)
            with st.container(border=True):
                st.markdown("#### 📋 Step 3 Results - Ingested Log Check")
                step3_results = f"""Ingested files log path: {ingested_files_log}
Log file exists: {ingested_files_log.exists()}
"""
                st.text_area("Step 3 Debug Information", value=step3_results, height=100)
            
            if ingested_files_log.exists():
                st.subheader("Step 4: Deleting Ingested Files Log")
                try:
                    ingested_files_log.unlink()
                    deleted_items.append(f"Ingested files log: {ingested_files_log}")
                    logger.info(f"Clean Start: Cleared ingested files log: {ingested_files_log}")
                    st.success(f"✅ Successfully deleted ingested files log")
                    debug_log_lines.append(f"SUCCESS: Deleted ingested files log")
                except Exception as e:
                    st.error(f"❌ Failed to delete ingested files log: {e}")
                    logger.error(f"Clean Start: Failed to delete ingested files log: {e}")
                    debug_log_lines.append(f"ERROR: Failed to delete ingested files log: {e}")
            else:
                st.info(f"ℹ️ No ingested files log found to delete - skipping this step")
                debug_log_lines.append("INFO: No ingested files log found")
        else:
            st.warning(f"ℹ️ ChromaDB directory does not exist - skipping file-specific operations")
            debug_log_lines.append("INFO: ChromaDB directory does not exist")
        
        # CRITICAL STEP: Delete ChromaDB directory (vector database)
        st.subheader("⚠️ CRITICAL STEP: Delete ChromaDB Directory")
        debug_log_lines.append("")
        debug_log_lines.append("OPERATION: Attempting to delete ChromaDB directory")
        
        # Re-check directory existence for deletion step
        chroma_exists_for_deletion = chroma_db_dir.exists() and chroma_db_dir.is_dir()
        st.success(f"🔍 Directory exists for deletion: `{chroma_exists_for_deletion}`")
        
        if chroma_exists_for_deletion:
            st.warning(f"⚠️ **ABOUT TO DELETE:** `{chroma_db_dir}`")
            try:
                shutil.rmtree(chroma_db_dir)
                deleted_items.append(f"ChromaDB vector database: {chroma_db_dir}")
                logger.info(f"Clean Start: Deleted ChromaDB directory: {chroma_db_dir}")
                st.success(f"✅ **SUCCESS: Deleted ChromaDB directory** `{chroma_db_dir}`")
                debug_log_lines.append(f"SUCCESS: Deleted ChromaDB directory {chroma_db_dir}")
            except Exception as e:
                st.error(f"❌ **FAILED to delete ChromaDB directory:** {e}")
                debug_log_lines.append(f"ERROR: Failed to delete ChromaDB directory: {e}")
        else:
            st.info("ℹ️ ChromaDB directory does not exist - nothing to delete")
            debug_log_lines.append("SKIP: ChromaDB directory does not exist or is not a directory")
        
        # Show critical step results (non-expander)
        with st.container(border=True):
            st.markdown("#### 📋 Critical Step Results - ChromaDB Directory Deletion")
            critical_results = f"""Target directory: {chroma_db_dir}
Directory existed before deletion: {chroma_exists_for_deletion}
Directory exists after deletion attempt: {chroma_db_dir.exists() if hasattr(chroma_db_dir, 'exists') else 'Unknown'}
Deletion successful: {'Yes' if chroma_exists_for_deletion and not chroma_db_dir.exists() else 'No' if chroma_exists_for_deletion else 'N/A - Directory did not exist'}
"""
            st.text_area("Critical Step Debug Information", value=critical_results, height=150)
            
            # 2. Delete knowledge graph file
            graph_file = Path(final_db_path) / "knowledge_cortex.gpickle"
            if graph_file.exists():
                graph_file.unlink()
                deleted_items.append(f"Knowledge graph: {graph_file}")
                logger.info(f"Clean Start: Deleted knowledge graph: {graph_file}")
            
            # 3. Clear working collections (check both locations)
            project_root = Path(__file__).parent.parent
            old_collections_file = project_root / "working_collections.json"
            new_collections_file = Path(final_db_path) / "working_collections.json"
            
            # Delete from project root if it exists
            if old_collections_file.exists():
                old_collections_file.unlink()
                deleted_items.append(f"Working collections (old location): {old_collections_file}")
                logger.info(f"Clean Start: Cleared working collections from project root: {old_collections_file}")
            
            # Delete from KB database path if it exists
            if new_collections_file.exists():
                new_collections_file.unlink()
                deleted_items.append(f"Working collections: {new_collections_file}")
                logger.info(f"Clean Start: Cleared working collections from KB database: {new_collections_file}")
            
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
            
            # Clear from project root (legacy location)
            for pattern in staging_patterns:
                for staging_file in project_root.glob(pattern):
                    if staging_file.is_file():
                        staging_file.unlink()
                        deleted_items.append(f"Staging/batch file (project): {staging_file}")
                        logger.info(f"Clean Start: Cleared staging file from project: {staging_file}")
            
            # Clear from database path (current location)
            db_path_obj = Path(final_db_path)
            for pattern in staging_patterns:
                for staging_file in db_path_obj.glob(pattern):
                    if staging_file.is_file():
                        staging_file.unlink()
                        deleted_items.append(f"Staging/batch file (database): {staging_file}")
                        logger.info(f"Clean Start: Cleared staging file from database: {staging_file}")
            
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
        
        # Add final debug log entries
        debug_log_lines.append("")
        debug_log_lines.append("FINAL RESULTS:")
        debug_log_lines.append("-" * 40)
        debug_log_lines.append(f"Total items cleaned: {len(deleted_items)}")
        debug_log_lines.append("Deleted items:")
        for item in deleted_items:
            debug_log_lines.append(f"  - {item}")
        debug_log_lines.append("")
        debug_log_lines.append("OPERATION COMPLETED SUCCESSFULLY")
        debug_log_lines.append("=" * 80)
        
        # Create comprehensive final debug log
        final_debug_log = "\n".join(debug_log_lines)
        
        # Success message
        st.success("✅ **Clean Start completed successfully!**")
        st.markdown("### 🗑️ Cleaned Items:")
        for item in deleted_items:
            st.write(f"- {item}")

        # Final verification
        with st.container(border=True):
            st.markdown("#### ✅ Final Verification")
            items = [
                ("ChromaDB directory", Path(final_db_path) / "knowledge_hub_db"),
                ("Collections file", Path(final_db_path) / "working_collections.json"),
                ("Staging file", Path(final_db_path) / "staging_ingestion.json"),
                ("Batch state", Path(final_db_path) / "batch_state.json"),
                ("Knowledge graph", Path(final_db_path) / "knowledge_cortex.gpickle"),
            ]
            for label, p in items:
                exists = p.exists()
                icon = "❌" if exists else "✅"
                st.write(f"{icon} {label}: `{p}` (exists={exists})")
        
        # Show final comprehensive debug log on screen
        st.subheader("📋 Complete Debug Log")
        st.info("**This shows everything that happened during the Clean Start operation. You can copy this information if needed.**")
        
        with st.container(border=True):
            st.markdown("#### 📋 Complete Debug Log - All Operations")
            st.text_area("Complete Debug Log", value=final_debug_log, height=400, help="Copy this entire log if you need to share it for troubleshooting")
        
        st.success("🎉 **CLEAN START IS COMPLETE - PLEASE READ THE DEBUG INFORMATION ABOVE**")
        st.success("🔍 **IMPORTANT:** Check the debug logs above to see exactly what was deleted and any issues that occurred.")
        
        # Add a prominent message that stops auto-progression
        st.warning("⏸️ **PAUSED FOR REVIEW** - Study the debug information above, then refresh the page to continue using the system.")
        
        st.info("""
        ### 🚀 Next Steps:
        1. **Review the debug logs above** to understand what was cleaned
        2. **Navigate to Knowledge Ingest page** to set database path
        3. **Run document ingestion** to rebuild your knowledge base  
        4. **Start fresh** - all schema conflicts should be resolved
        
        Your system now has a completely clean slate and should work without any ChromaDB schema errors.
        """)
        
        # STOP EXECUTION HERE so user can read everything
        st.stop()
        
        logger.info(f"Clean Start completed successfully - {len(deleted_items)} items cleaned")
        
    except Exception as e:
        # Add error to debug log
        debug_log_lines.append("")
        debug_log_lines.append("CRITICAL ERROR:")
        debug_log_lines.append("-" * 40)
        debug_log_lines.append(f"Error: {str(e)}")
        debug_log_lines.append(f"Error Type: {type(e).__name__}")
        debug_log_lines.append("OPERATION FAILED")
        debug_log_lines.append("=" * 80)
        
        # Create debug log even for failed operations
        error_debug_log = "\n".join(debug_log_lines)
        
        error_msg = f"Clean Start failed: {e}"
        logger.error(f"Clean Start error: {e}")
        st.error(f"❌ {error_msg}")
        
        # Show error debug log on screen (avoid nested expanders)
        st.subheader("📋 Error Debug Log")
        st.error("**An error occurred during Clean Start. This shows what was attempted before the error:**")
        with st.container(border=True):
            st.markdown("#### 📋 Error Debug Log - All Operations Before Error")
            st.text_area("Error Debug Log", value=error_debug_log, height=400, help="Copy this error log for troubleshooting")
        
        st.error("🚨 **CLEAN START FAILED - PLEASE READ THE ERROR DEBUG INFORMATION ABOVE**")
        st.error("🔍 **IMPORTANT:** The error debug log above shows exactly what was attempted before the failure.")
        
        # Add a prominent message that stops auto-progression
        st.warning("⏸️ **PAUSED FOR ERROR REVIEW** - Study the error information above for troubleshooting.")
        
        st.error("""
        ### 🛠️ Troubleshooting Steps:
        1. **Review the error debug log above** to understand what failed
        2. Try manually deleting the database directory if permissions issue
        3. Restart the application if needed
        4. **Copy the error debug log above** for detailed troubleshooting information
        5. Check Docker container has proper permissions to the mounted directories
        """)
        
        # STOP EXECUTION HERE so user can read error information
        st.stop()

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
            
            # Map ConfigManager keys to expected keys - use proper default path detection
            from cortex_engine.utils.default_paths import get_default_ai_database_path
            default_db_path = get_default_ai_database_path()
            maintenance_config = {
                'db_path': config.get('ai_database_path', default_db_path),
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
        
        from cortex_engine.utils.default_paths import get_default_ai_database_path
        db_path = config.get('db_path', get_default_ai_database_path())
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

def delete_ingested_document_database_simple(db_path):
    """Permanently delete the entire ingested document database (simple version)"""
    try:
        wsl_db_path = convert_windows_to_wsl_path(db_path)
        kb_dir = Path(wsl_db_path) / "knowledge_hub_db"
        collections_file = Path(wsl_db_path) / "working_collections.json"
        graph_file = Path(wsl_db_path) / "knowledge_cortex.gpickle"
        batch_state_file = Path(wsl_db_path) / "batch_state.json"
        staging_file = Path(wsl_db_path) / "staging_ingestion.json"
        
        deleted_items = []
        errors = []
        
        if kb_dir.exists():
            try:
                shutil.rmtree(kb_dir)
                deleted_items.append("ChromaDB database")
                logger.info(f"ChromaDB database deleted: {kb_dir}")
            except Exception as e:
                error_msg = f"Failed to delete ChromaDB database: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Delete collections file (only stored in KB database path now)
        if collections_file.exists():
            try:
                collections_file.unlink()
                deleted_items.append("Collections")
                logger.info(f"Collections file deleted: {collections_file}")
            except Exception as e:
                error_msg = f"Failed to delete collections file: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        if graph_file.exists():
            try:
                graph_file.unlink()
                deleted_items.append("Knowledge graph")
                logger.info(f"Knowledge graph deleted: {graph_file}")
            except Exception as e:
                error_msg = f"Failed to delete knowledge graph: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Delete batch state file
        if batch_state_file.exists():
            try:
                batch_state_file.unlink()
                deleted_items.append("Batch state")
                logger.info(f"Batch state file deleted: {batch_state_file}")
            except Exception as e:
                error_msg = f"Failed to delete batch state file: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Delete staging file
        if staging_file.exists():
            try:
                staging_file.unlink()
                deleted_items.append("Staging file")
                logger.info(f"Staging file deleted: {staging_file}")
            except Exception as e:
                error_msg = f"Failed to delete staging file: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        if deleted_items:
            st.success(f"✅ Ingested document database deleted successfully: {', '.join(deleted_items)}")
        
        if errors:
            st.error(f"❌ Some items could not be deleted: {', '.join(errors)}")
            
        if not deleted_items and not errors:
            st.warning("No ingested document database components found to delete.")
    except Exception as e:
        st.error(f"❌ Failed to delete ingested document database: {e}")
        logger.error(f"Failed to delete ingested document database: {e}")

def display_header():
    """Display page header with navigation and information"""
    st.title("🔧 7. Maintenance & Administration")
    st.caption(f"Version: {PAGE_VERSION} • Consolidated System Maintenance Interface")
    
    # Quick access to System Terminal
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💻 Open System Terminal", use_container_width=True, help="Access the secure system terminal for command execution"):
            st.switch_page("pages/_System_Terminal.py")
    
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
    
    # Use proper default path detection
    from cortex_engine.utils.default_paths import get_default_ai_database_path
    default_db_path = get_default_ai_database_path()
    db_path = config.get('ai_database_path', config.get('db_path', default_db_path))

    # Database Path Configuration block
    with st.container(border=True):
        st.subheader("📁 Database Path Configuration")
        docker_mode = os.path.exists('/.dockerenv')
        st.caption(f"Environment: {'🐳 Docker' if docker_mode else '💻 Host'}")

        # Current value and normalized preview
        current_input = st.text_input(
            "AI Database Path",
            value=db_path,
            help="Enter the folder that contains your knowledge base (e.g., C:/ai_databases)."
        )
        # Keep the most recent user input in session state for downstream actions
        st.session_state["maintenance_current_db_input"] = current_input
        try:
            preview = convert_windows_to_wsl_path(current_input)
            st.code(f"Resolved path: {preview}")
        except Exception:
            pass

        # Quick scan for likely locations
        def scan_candidates():
            cands = []
            drives = list("CDEFGHIJKLMNOPQRSTUVWXYZ")
            for d in drives:
                base = f"{d}:/ai_databases"
                wsl = convert_windows_to_wsl_path(base)
                kb = os.path.join(wsl, "knowledge_hub_db")
                if os.path.isdir(wsl) or os.path.isdir(kb):
                    cands.append(base)
            # Add Docker defaults
            if docker_mode:
                for p in ["/app/data/ai_databases", "/data", "/workspace/data/ai_databases"]:
                    if os.path.isdir(p) or os.path.isdir(os.path.join(p, "knowledge_hub_db")):
                        cands.append(p)
            return sorted(set(cands))

        cols = st.columns([1,1,1])
        with cols[0]:
            if st.button("🔎 Scan Common Locations", use_container_width=True, key="scan_db_locations"):
                st.session_state.discovered_db_paths = scan_candidates()
                st.rerun()
        with cols[1]:
            if st.button("💾 Save Path", use_container_width=True, key="save_db_path"):
                try:
                    ConfigManager().update_config({"ai_database_path": current_input})
                    if 'maintenance_config' in st.session_state and st.session_state.maintenance_config:
                        st.session_state.maintenance_config['db_path'] = current_input
                    st.success("✅ Database path saved")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save path: {e}")

        # Show discovered candidates, if any
        if st.session_state.get('discovered_db_paths'):
            choice = st.selectbox(
                "Discovered database locations",
                st.session_state.discovered_db_paths,
                help="Select a discovered location to populate the field above."
            )
            if st.button("Use Selected Location", key="use_selected_db_loc"):
                try:
                    ConfigManager().update_config({"ai_database_path": choice})
                    if 'maintenance_config' in st.session_state and st.session_state.maintenance_config:
                        st.session_state.maintenance_config['db_path'] = choice
                    st.success(f"✅ Path set to {choice}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to set path: {e}")
    
    st.markdown("## 🚀 Clean Start - Complete System Reset")
    st.markdown("### 🧹 Clean Start Function")
    st.warning("""
    **Complete system reset function** that addresses database schema issues, collection conflicts, and provides a fresh start.
    This function is specifically designed to resolve ChromaDB schema errors like 'collections.config_json_str' column missing.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Clean Start will:**
        - ✅ Delete entire ingested document database (ChromaDB)
        - ✅ Delete knowledge graph file (.gpickle)  
        - ✅ Delete collections file (working_collections.json)
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
        st.info("**⚠️ Most Dangerous Operations**\n\nComplete system reset functions are now located in the **Advanced Database Recovery & Repair** section below for safety.")

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
        
        st.subheader("Delete Ingested Document Database")
        st.error("⚠️ **DANGER:** This is the most destructive action.")
        st.caption("This will delete the processed/ingested documents database, NOT your source Knowledge Base files.")
        
        if st.button("Permanently Delete Ingested Document Database...", type="primary"):
            st.session_state.show_confirm_delete_kb = True
            
        if st.session_state.get("show_confirm_delete_kb"):
            st.warning(f"This will permanently delete the **ingested document database** (ChromaDB + Collections + Knowledge Graph). Your source documents will NOT be affected. This action cannot be undone.")
            c1, c2 = st.columns(2)
            if c1.button("YES, DELETE INGESTED DATABASE", use_container_width=True):
                delete_ingested_document_database(db_path)
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
        
        # Clean Start Reset - Moved here for safety
        st.markdown("---")
        st.markdown("### ⚠️ **DANGER ZONE - Complete System Reset**")
        st.error("**This section contains destructive operations that cannot be undone!**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **🚀 Clean Start Reset**
            
            Complete system reset function that addresses database schema issues, collection conflicts, and provides a fresh start.
            This function is specifically designed to resolve ChromaDB schema errors like 'collections.config_json_str' column missing.
            
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
            
            **Use Clean Start when:**
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
            st.warning("⚠️ **COMPLETE SYSTEM RESET**\n\nThis will delete ALL data and provide a completely fresh start. All knowledge base content, collections, and configurations will be lost.")
            
            if st.button("🚀 Clean Start Reset", use_container_width=True, type="secondary", help="⚠️ DANGER: This will delete everything!"):
                st.session_state.show_confirm_clean_start = True
                
            if st.session_state.get("show_confirm_clean_start"):
                st.error("⚠️ **FINAL WARNING - COMPLETE SYSTEM RESET**")
                st.warning("This will delete ALL data and provide a completely fresh start. All knowledge base content, collections, and configurations will be lost.")
                
                c1, c2 = st.columns(2)
                if c1.button("✅ YES, CLEAN START", use_container_width=True, type="primary"):
                    # Use freshest input from the configuration block if available
                    fresh_path = st.session_state.get('maintenance_current_db_input', db_path)
                    perform_clean_start(fresh_path)
                    st.session_state.show_confirm_clean_start = False
                    st.rerun()
                if c2.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_confirm_clean_start = False
                    st.rerun()
        
        st.markdown("---")
        
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
        from cortex_engine.utils.default_paths import get_default_ai_database_path
        db_path = config.get('db_path', get_default_ai_database_path())
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

        # Export to external location (Host/Filesystem) — non-Docker parity
        with st.expander("📤 Export Backup To External Location", expanded=False):
            st.caption("Copies your entire knowledge base (Chroma, graph, collections, logs) to a folder you choose.")

            def _resolve_destination(p: str) -> Path:
                # Convert Windows or POSIX input to a usable local path (handles WSL too)
                p = (p or '').strip()
                if not p:
                    return Path('')
                resolved = convert_windows_to_wsl_path(p)
                return Path(resolved)

            dest_root_input = st.text_input(
                "Destination folder (e.g., C:/CortexBackups or /home/user/CortexBackups)",
                placeholder="C:/CortexBackups or /home/you/CortexBackups",
                key="kb_external_backup_dest_host"
            )

            backup_name_hint = datetime.now().strftime("cortex_kb_backup_%Y%m%d_%H%M%S")
            dest_backup_name = st.text_input("Backup folder name", value=backup_name_hint, key="kb_external_backup_name_host")

            if st.button("📤 Export Now", use_container_width=True, key="btn_export_external_backup_host"):
                try:
                    dest_root = _resolve_destination(dest_root_input)
                    if not dest_root or str(dest_root) == ".":
                        st.error("Please provide a valid destination path.")
                        st.stop()

                    # Ensure destination base exists
                    dest_root.mkdir(parents=True, exist_ok=True)

                    # Source paths
                    src_base = Path(convert_windows_to_wsl_path(db_path))
                    chroma_src = src_base / "knowledge_hub_db"
                    files_to_copy = [
                        (src_base / "working_collections.json", "working_collections.json"),
                        (src_base / "knowledge_cortex.gpickle", "knowledge_cortex.gpickle"),
                        (src_base / "batch_state.json", "batch_state.json"),
                        (src_base / "staging_ingestion.json", "staging_ingestion.json"),
                    ]

                    # Prevent exporting into the same KB folder
                    dest_dir = dest_root / dest_backup_name
                    if str(dest_dir.resolve()) == str(src_base.resolve()) or str(dest_root.resolve()) == str(src_base.resolve()):
                        st.error("Destination cannot be the same as the knowledge base path.")
                        st.stop()

                    # Create destination
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    # Copy ChromaDB directory if exists
                    if chroma_src.exists():
                        st.info("Copying ChromaDB directory (this may take a while)...")
                        shutil.copytree(chroma_src, dest_dir / "knowledge_hub_db", dirs_exist_ok=True)

                    # Copy other files
                    copied_files = 0
                    for src_path, name in files_to_copy:
                        if src_path.exists():
                            shutil.copy2(src_path, dest_dir / name)
                            copied_files += 1

                    st.success(f"✅ Export complete to: {dest_dir}")
                    st.caption(f"Copied extra files: {copied_files}")
                except PermissionError as pe:
                    st.error(f"Permission denied writing to destination: {pe}")
                except FileNotFoundError as fe:
                    st.error(f"Destination not found: {fe}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
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

def display_performance_dashboard():
    """Display performance monitoring dashboard with metrics and analytics"""
    st.markdown("## 📊 Performance Monitoring Dashboard")
    st.markdown("Real-time performance metrics for critical operations (v4.9.0+)")

    # Get performance monitor
    monitor = get_performance_monitor()

    # Create columns for high-level metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        session_summary = get_session_summary()
        st.metric(
            "Total Operations",
            session_summary.get("total_operations", 0),
            help="Total number of monitored operations this session"
        )

    with col2:
        cache_stats = monitor.get_cache_stats()
        hit_rate = cache_stats.get("hit_rate", 0.0) * 100
        st.metric(
            "Cache Hit Rate",
            f"{hit_rate:.1f}%",
            help="Percentage of queries served from cache"
        )

    with col3:
        gpu_info = get_gpu_memory_info()
        device_name = gpu_info.device_name
        if len(device_name) > 20:
            device_name = device_name[:17] + "..."
        st.metric(
            "Device",
            device_name,
            help=f"Full name: {gpu_info.device_name}"
        )

    with col4:
        session_duration = session_summary.get("session_duration_formatted", "N/A")
        st.metric(
            "Session Duration",
            session_duration,
            help="Performance monitoring session duration"
        )

    st.markdown("---")

    # Overview Charts
    all_stats = get_all_stats()
    if all_stats:
        st.markdown("### 📊 Performance Overview")

        # Create two columns for overview charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Operation distribution pie chart
            import pandas as pd
            import plotly.express as px

            op_data = []
            for op_type, stats in all_stats.items():
                op_data.append({
                    'Operation': op_type.replace('_', ' ').title(),
                    'Count': stats.total_operations,
                    'Time (s)': stats.total_duration
                })

            if op_data:
                df_ops = pd.DataFrame(op_data)

                fig_pie = px.pie(
                    df_ops,
                    values='Count',
                    names='Operation',
                    title='Operation Distribution',
                    hole=0.4  # Donut chart
                )
                fig_pie.update_layout(height=350)
                st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            # Total time by operation (bar chart)
            if op_data:
                fig_bar = px.bar(
                    df_ops,
                    x='Operation',
                    y='Time (s)',
                    title='Total Time by Operation Type',
                    color='Time (s)',
                    color_continuous_scale='Blues'
                )
                fig_bar.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

    # GPU/Device Information
    with st.expander("🖥️ Device & GPU Information", expanded=True):
        gpu_info = get_gpu_memory_info()
        device_recs = get_device_recommendations()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Device Information")
            st.markdown(f"**Device:** {gpu_info.device_name}")
            st.markdown(f"**Type:** {'CUDA GPU' if gpu_info.is_cuda else ('Apple Silicon (MPS)' if gpu_info.is_mps else 'CPU')}")

            if gpu_info.is_cuda:
                st.markdown(f"**Total Memory:** {gpu_info.total_memory_gb:.2f} GB")
                st.markdown(f"**Free Memory:** {gpu_info.free_memory_gb:.2f} GB")
                st.markdown(f"**Utilization:** {gpu_info.utilization_percent:.1f}%")

                # Progress bar for memory utilization
                st.progress(min(gpu_info.utilization_percent / 100.0, 1.0))

        with col_b:
            st.markdown("### Batch Size Recommendations")
            batch_recs = device_recs.get("batch_recommendations", {})
            perf_tier = device_recs.get("performance_tier", "Unknown")

            st.markdown(f"**Recommended Batch Size:** {batch_recs.get('recommended', 32)}")
            st.markdown(f"**Conservative:** {batch_recs.get('conservative', 32)}")
            st.markdown(f"**Aggressive:** {batch_recs.get('aggressive', 32)}")
            st.markdown(f"**Performance Tier:** {perf_tier}")

    # Operation Statistics
    with st.expander("⏱️ Operation Performance Statistics", expanded=True):
        all_stats = get_all_stats()

        if not all_stats:
            st.info("No performance data collected yet. Run some operations (ingestion, search, etc.) to see metrics.")
        else:
            # Create tabs for different operation types
            op_types = list(all_stats.keys())
            if op_types:
                tabs = st.tabs([f"{'📸' if op == 'image_processing' else '🔢' if op == 'embedding_batch' else '🔍' if op == 'query' else '⚡'} {op.replace('_', ' ').title()}" for op in op_types])

                for tab, op_type in zip(tabs, op_types):
                    with tab:
                        stats = all_stats[op_type]

                        # Metrics row
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Total Ops", stats.total_operations)
                        with m2:
                            st.metric("Successful", stats.successful_operations)
                        with m3:
                            st.metric("Failed", stats.failed_operations)
                        with m4:
                            success_rate = (stats.successful_operations / stats.total_operations * 100) if stats.total_operations > 0 else 0
                            st.metric("Success Rate", f"{success_rate:.1f}%")

                        # Timing statistics
                        st.markdown("### ⏱️ Timing Statistics")

                        t1, t2, t3 = st.columns(3)
                        with t1:
                            st.markdown(f"**Average Duration:** {stats.avg_duration:.3f}s")
                            st.markdown(f"**Min Duration:** {stats.min_duration:.3f}s")
                            st.markdown(f"**Max Duration:** {stats.max_duration:.3f}s")

                        with t2:
                            st.markdown(f"**Median (P50):** {stats.p50_duration:.3f}s")
                            st.markdown(f"**95th Percentile:** {stats.p95_duration:.3f}s")
                            st.markdown(f"**99th Percentile:** {stats.p99_duration:.3f}s")

                        with t3:
                            st.markdown(f"**Total Time:** {stats.total_duration:.2f}s")
                            st.markdown(f"**First Seen:** {stats.first_seen.split('T')[1][:8]}")
                            st.markdown(f"**Last Seen:** {stats.last_seen.split('T')[1][:8]}")

                        # Performance Charts
                        recent = monitor.get_recent_metrics(op_type, limit=50)
                        if recent and len(recent) >= 2:
                            import pandas as pd
                            import plotly.express as px
                            import plotly.graph_objects as go

                            st.markdown("### 📈 Performance Charts")

                            # Create DataFrame for plotting
                            df = pd.DataFrame([
                                {
                                    'timestamp': metric.timestamp,
                                    'duration': metric.duration,
                                    'success': metric.success,
                                    'operation': op_type
                                }
                                for metric in recent
                            ])

                            # Chart 1: Duration over time (line chart)
                            fig1 = px.line(
                                df,
                                x='timestamp',
                                y='duration',
                                title=f'{op_type.replace("_", " ").title()} - Duration Over Time',
                                labels={'duration': 'Duration (seconds)', 'timestamp': 'Time'},
                                markers=True
                            )
                            fig1.add_hline(y=stats.avg_duration, line_dash="dash", line_color="red",
                                          annotation_text=f"Avg: {stats.avg_duration:.3f}s")
                            fig1.update_layout(height=400)
                            st.plotly_chart(fig1, use_container_width=True)

                            # Chart 2: Duration distribution (histogram)
                            fig2 = px.histogram(
                                df,
                                x='duration',
                                title=f'{op_type.replace("_", " ").title()} - Duration Distribution',
                                labels={'duration': 'Duration (seconds)', 'count': 'Frequency'},
                                nbins=20
                            )
                            fig2.add_vline(x=stats.p50_duration, line_dash="dash", line_color="green",
                                          annotation_text=f"P50: {stats.p50_duration:.3f}s")
                            fig2.add_vline(x=stats.p95_duration, line_dash="dash", line_color="orange",
                                          annotation_text=f"P95: {stats.p95_duration:.3f}s")
                            fig2.update_layout(height=350)
                            st.plotly_chart(fig2, use_container_width=True)

                        # Recent operations (text summary)
                        if recent and len(recent) >= 1:
                            st.markdown("### 📋 Recent Operations (Last 5)")
                            for i, metric in enumerate(list(recent)[-5:], 1):
                                status = "✅" if metric.success else "❌"
                                metadata_str = ", ".join(f"{k}={v}" for k, v in metric.metadata.items())
                                st.markdown(f"{i}. {status} **{metric.duration:.3f}s** - {metadata_str}")

    # Cache Statistics
    with st.expander("💾 Query Cache Statistics", expanded=False):
        cache_stats = monitor.get_cache_stats()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Queries", cache_stats.get("total_queries", 0))
        with c2:
            st.metric("Cache Hits", cache_stats.get("cache_hits", 0))
        with c3:
            st.metric("Cache Misses", cache_stats.get("cache_misses", 0))
        with c4:
            hit_rate = cache_stats.get("hit_rate", 0.0) * 100
            st.metric("Hit Rate", f"{hit_rate:.1f}%")

        if cache_stats.get("total_queries", 0) > 0:
            st.markdown(f"**Cache Efficiency:** {cache_stats.get('cache_hits', 0)} instant responses out of {cache_stats.get('total_queries', 0)} queries")

            # Calculate time saved
            query_stats = all_stats.get("query")
            if query_stats and cache_stats.get("cache_hits", 0) > 0:
                avg_search_time = query_stats.avg_duration
                time_saved = cache_stats.get("cache_hits", 0) * avg_search_time
                st.success(f"⚡ Estimated time saved by caching: **{time_saved:.2f} seconds**")

    # Actions
    with st.expander("⚙️ Performance Monitoring Actions", expanded=False):
        col_x, col_y = st.columns(2)

        with col_x:
            if st.button("💾 Save Metrics to File", help="Export current metrics to JSON file"):
                try:
                    file_path = monitor.save_to_file()
                    st.success(f"✅ Metrics saved to: `{file_path}`")
                except Exception as e:
                    st.error(f"❌ Failed to save metrics: {e}")

            if st.button("🔄 Refresh Display", help="Reload performance data"):
                st.rerun()

        with col_y:
            if st.button("🧹 Clear Metrics", help="Reset all performance metrics (new session)", type="primary"):
                monitor.clear()
                st.success("✅ Performance metrics cleared - new session started")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.info("💡 **Tip:** Performance metrics are collected automatically during ingestion, search, and other operations. The data resets when you clear metrics or restart the application.")


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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗄️ Database",
        "💻 Terminal",
        "⚙️ Setup",
        "💾 Backups",
        "📊 Performance",
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
        display_performance_dashboard()

    with tab6:
        display_changelog_viewer()

    with tab7:
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

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
