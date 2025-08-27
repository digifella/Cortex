# ## File: pages/13_Maintenance.py
# Version: v1.0.0
# Date: 2025-08-27
# Purpose: Consolidated maintenance and administrative functions for the Cortex Suite.
#          Combines database maintenance, system terminal, and other administrative functions
#          from various pages into a single, organized maintenance interface.

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
PAGE_VERSION = "v1.0.0"

# Import Cortex modules
try:
    from cortex_engine.config import load_config, INGESTED_FILES_LOG
    from cortex_engine.utils import get_logger, convert_windows_to_wsl_path
    from cortex_engine.utils.command_executor import display_command_executor_widget, SafeCommandExecutor
    from cortex_engine.ingestion_recovery import IngestionRecoveryManager
    from cortex_engine.collection_manager import WorkingCollectionManager
    from cortex_engine.setup_manager import SetupManager
    from cortex_engine.backup_manager import BackupManager
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
if 'maintenance_config' not in st.session_state:
    st.session_state.maintenance_config = None

def load_maintenance_config():
    """Load configuration for maintenance operations"""
    if st.session_state.maintenance_config is None:
        try:
            config = load_config()
            st.session_state.maintenance_config = config
            return config
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
        backup_manager = BackupManager(db_path)
        
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

def main():
    """Main function to orchestrate the maintenance interface"""
    display_header()
    
    # Create tabs for different maintenance categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗄️ Database", 
        "💻 Terminal", 
        "⚙️ Setup", 
        "💾 Backups",
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