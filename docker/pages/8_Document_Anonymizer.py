# ## File: pages/8_Document_Anonymizer.py
# Version: 1.0.0
# Date: 2025-07-30
# Purpose: GUI for document anonymization functionality.
#          Processes documents to replace identifying information with generic placeholders.

import streamlit as st
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# --- Project Setup ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from cortex_engine.anonymizer import DocumentAnonymizer, AnonymizationMapping
from cortex_engine.utils import get_logger
from cortex_engine.utils.path_utils import (
    normalize_path, ensure_directory, validate_path_exists,
    process_multiple_drag_drop_paths, get_file_size_display
)
from cortex_engine.config_manager import ConfigManager
from cortex_engine.help_system import help_system

# Set up logging
logger = get_logger(__name__)

st.set_page_config(layout="wide", page_title="Document Anonymizer")

# Add global CSS for left-aligned folder buttons
st.markdown("""
<style>
/* Left-align folder directory navigation buttons */
.stButton > button:has-text("📁") {
    text-align: left !important;
    justify-content: flex-start !important;
}
/* Alternative selector for buttons containing folder emoji */
button[data-testid="baseButton-secondary"]:contains("📁") {
    text-align: left !important;
    justify-content: flex-start !important;
}
/* Fallback: target all secondary buttons in directory navigation columns */
div[data-testid="column"] .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def is_supported_file(file_path: Path) -> bool:
    """Check if file format is supported for anonymization."""
    supported_extensions = {'.txt', '.pdf', '.docx'}
    return file_path.suffix.lower() in supported_extensions

def process_drag_drop_paths(raw_paths: str) -> List[Path]:
    """Process drag-and-drop paths from various platforms using enhanced path utilities."""
    paths = process_multiple_drag_drop_paths(raw_paths)
    
    # Show warnings for any invalid paths
    if raw_paths.strip():
        input_lines = [line.strip() for line in raw_paths.strip().split('\n') if line.strip()]
        if len(paths) < len(input_lines):
            invalid_count = len(input_lines) - len(paths)
            st.warning(f"{invalid_count} path(s) were invalid or not found")
    
    return paths

def get_file_info(file_path: Path) -> Dict:
    """Get basic file information for display."""
    try:
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': get_file_size_display(file_path),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
            'type': file_path.suffix.upper()[1:] if file_path.suffix else 'Unknown',
            'supported': is_supported_file(file_path)
        }
    except Exception as e:
        return {
            'name': file_path.name,
            'size': 'Unknown',
            'modified': 'Unknown',
            'type': 'Unknown',
            'supported': False,
            'error': str(e)
        }

# --- Main Interface ---

st.title("📝 Document Anonymizer")
st.caption("Replace identifying information in documents with generic placeholders")

help_system.show_help_menu()

# Introduction
st.markdown("""
The Document Anonymizer processes your documents to replace identifying information such as:
- **People names** → Person A, Person B, etc.
- **Company names** → Company 1, Company 2, etc.
- **Project names** → Project 1, Project 2, etc.
- **Email addresses** → [EMAIL]
- **Phone numbers** → [PHONE]
- **URLs** → [URL]
- **Headers/footers** → [HEADER/FOOTER REMOVED]

**Supported formats:** TXT, PDF, DOCX
""")

st.divider()

# Configuration Section
with st.expander("⚙️ Configuration", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        shared_mapping = st.checkbox(
            "Use shared entity mapping across files",
            value=True,
            help="When enabled, the same person/company will get the same anonymous name across all files"
        )
        
        preserve_structure = st.checkbox(
            "Preserve document structure",
            value=True,
            help="Maintain original formatting and structure in output files"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Entity detection confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower values catch more entities but may include false positives"
        )

# File Selection Section
st.subheader("📁 Select Files to Anonymize")

# Tab for different input methods
tab1, tab2, tab3 = st.tabs(["📋 Drag & Drop", "📂 Browse Files", "📁 Browse Directory"])

selected_files = []

with tab1:
    st.markdown("**Drag and drop files or folders here:**")
    
    # Text area for drag-drop (works on all platforms)
    drag_drop_input = st.text_area(
        "Paste file paths here (one per line):",
        height=100,
        placeholder="Drag files here or paste paths like:\n/path/to/document.pdf\n/Users/name/Documents/file.docx\nC:\\Documents\\file.txt",
        key="drag_drop_paths"
    )
    
    if drag_drop_input:
        dropped_files = process_drag_drop_paths(drag_drop_input)
        if dropped_files:
            st.success(f"Found {len(dropped_files)} file(s)")
            selected_files.extend(dropped_files)

with tab2:
    st.markdown("**Browse and select individual files:**")
    
    # Initialize session state for file browser
    if "anonymizer_directory_path" not in st.session_state:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        st.session_state.anonymizer_directory_path = config.get('last_directory', str(Path.home()))
    
    if "anonymizer_file_selections" not in st.session_state:
        st.session_state.anonymizer_file_selections = {}
    
    def reset_file_browser():
        st.session_state.anonymizer_file_selections = {}
    
    # Directory path input
    new_path = st.text_input(
        "Browse Directory:",
        value=st.session_state.anonymizer_directory_path,
        help="📁 Navigate to the directory containing files to anonymize"
    )
    
    # Update session state if path changed manually
    if new_path != st.session_state.anonymizer_directory_path:
        st.session_state.anonymizer_directory_path = new_path
        reset_file_browser()
    
    # Directory navigation
    current_path = normalize_path(st.session_state.anonymizer_directory_path)
    if current_path and current_path.exists() and current_path.is_dir():
        try:
            # Navigation buttons
            c1, c2, c3 = st.columns(3)
            
            # Get current directory contents
            files = []
            subdirs = []
            for item in current_path.iterdir():
                if item.is_file() and is_supported_file(item):
                    files.append(item)
                elif item.is_dir():
                    subdirs.append(item.name)
            
            files = sorted(files, key=lambda x: x.name.lower())
            subdirs = sorted(subdirs, key=str.lower)
            
            if c1.button("Select All Files", use_container_width=True):
                for file_path in files:
                    st.session_state.anonymizer_file_selections[str(file_path)] = True
                st.rerun()
            
            if c2.button("Deselect All Files", use_container_width=True):
                for file_path in files:
                    st.session_state.anonymizer_file_selections[str(file_path)] = False
                st.rerun()
            
            # Go up one level button
            if current_path != current_path.parent:
                if c3.button("⬆️ Go Up One Level", use_container_width=True):
                    st.session_state.anonymizer_directory_path = str(current_path.parent)
                    reset_file_browser()
                    st.rerun()
            
            st.markdown("---")
            
            # Show subdirectories for navigation
            if subdirs:
                st.markdown("**📁 Directories:**")
                cols = st.columns(3)
                for i, dirname in enumerate(subdirs):
                    col = cols[i % 3]
                    with col:
                        full_path = current_path / dirname
                        if st.button(f"📁 {dirname}", key=f"nav_dir_{dirname}", help=f"Navigate to {dirname}", use_container_width=True):
                            st.session_state.anonymizer_directory_path = str(full_path)
                            reset_file_browser()
                            st.rerun()
            
            # Show files for selection
            if files:
                st.markdown("**📄 Files:**")
                for file_path in files:
                    info = get_file_info(file_path)
                    is_selected = st.session_state.anonymizer_file_selections.get(str(file_path), False)
                    
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        new_selection = st.checkbox(
                            f"📄 {info['name']} ({info['size']}, {info['type']})",
                            value=is_selected,
                            key=f"file_cb_{file_path}"
                        )
                        st.session_state.anonymizer_file_selections[str(file_path)] = new_selection
                        
                        if new_selection and file_path not in selected_files:
                            selected_files.append(file_path)
                        elif not new_selection and file_path in selected_files:
                            selected_files.remove(file_path)
                    
                    with col2:
                        if info['supported']:
                            st.success("✓")
                        else:
                            st.error("❌")
            else:
                st.info("No supported files found in this directory")
                
        except Exception as e:
            st.error(f"Error reading directory: {e}")
    else:
        st.error("Invalid or non-existent directory")

with tab3:
    st.markdown("**Browse directory and select files for processing:**")
    
    # Directory browser for batch processing
    if "anonymizer_batch_directory" not in st.session_state:
        st.session_state.anonymizer_batch_directory = str(Path.home())
    if "anonymizer_batch_files" not in st.session_state:
        st.session_state.anonymizer_batch_files = []
    if "anonymizer_file_selections" not in st.session_state:
        st.session_state.anonymizer_file_selections = {}
    if "anonymizer_current_page" not in st.session_state:
        st.session_state.anonymizer_current_page = 0
    
    # Directory input and scan controls
    col1, col2 = st.columns([3, 1])
    with col1:
        batch_dir = st.text_input(
            "Directory to scan:",
            value=st.session_state.anonymizer_batch_directory,
            key="anonymizer_batch_directory",
            help="Directory containing files to anonymize"
        )
    
    with col2:
        include_subdirs = st.checkbox(
            "Include subdirectories",
            value=True,
            help="Scan subdirectories recursively"
        )
    
    if st.button("🔍 Scan Directory", type="secondary", use_container_width=True):
        try:
            batch_path = normalize_path(batch_dir)
            if batch_path and batch_path.exists() and batch_path.is_dir():
                # Find supported files
                supported_files = []
                if include_subdirs:
                    for file_path in batch_path.rglob('*'):
                        if file_path.is_file() and is_supported_file(file_path):
                            supported_files.append(file_path)
                else:
                    for file_path in batch_path.iterdir():
                        if file_path.is_file() and is_supported_file(file_path):
                            supported_files.append(file_path)
                
                if supported_files:
                    # Store files and reset state
                    st.session_state.anonymizer_batch_files = supported_files
                    st.session_state.anonymizer_file_selections = {str(f): False for f in supported_files}
                    st.session_state.anonymizer_current_page = 0
                    st.success(f"Found {len(supported_files)} supported files")
                    st.rerun()
                else:
                    st.warning("No supported files found in directory" + (" and subdirectories" if include_subdirs else ""))
            else:
                st.error("Invalid or non-existent directory path")
        except Exception as e:
            st.error(f"Error scanning directory: {e}")
    
    # File browser with pagination (similar to Knowledge Ingest)
    if st.session_state.anonymizer_batch_files:
        FILES_PER_PAGE = 10  # Same as Knowledge Ingest
        total_files = len(st.session_state.anonymizer_batch_files)
        total_pages = (total_files + FILES_PER_PAGE - 1) // FILES_PER_PAGE
        current_page = st.session_state.anonymizer_current_page
        
        # Sorting controls
        st.markdown("---")
        st.markdown("### 📁 File Browser")
        
        def sort_by_name():
            st.session_state.anonymizer_batch_files.sort(key=lambda f: f.name.lower())
            st.session_state.anonymizer_current_page = 0
        
        def sort_by_date():
            st.session_state.anonymizer_batch_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            st.session_state.anonymizer_current_page = 0
        
        sc1, sc2 = st.columns(2)
        sc1.button("🔤 Sort by Name (A-Z)", on_click=sort_by_name, use_container_width=True)
        sc2.button("📅 Sort by Date (Newest First)", on_click=sort_by_date, use_container_width=True)
        
        # Pagination info and selection summary
        start_idx = current_page * FILES_PER_PAGE
        end_idx = min(start_idx + FILES_PER_PAGE, total_files)
        paginated_files = st.session_state.anonymizer_batch_files[start_idx:end_idx]
        
        num_selected = sum(1 for selected in st.session_state.anonymizer_file_selections.values() if selected)
        st.info(f"Found **{total_files}** files. Currently selecting **{num_selected}** for processing. Displaying page {current_page + 1} of {total_pages}.")
        
        # Bulk selection controls
        c1, c2, c3, c4 = st.columns(4)
        
        def select_page():
            for f in paginated_files:
                st.session_state.anonymizer_file_selections[str(f)] = True
        
        def deselect_page():
            for f in paginated_files:
                st.session_state.anonymizer_file_selections[str(f)] = False
        
        def select_all():
            for f in st.session_state.anonymizer_batch_files:
                st.session_state.anonymizer_file_selections[str(f)] = True
        
        def deselect_all():
            for f in st.session_state.anonymizer_batch_files:
                st.session_state.anonymizer_file_selections[str(f)] = False
        
        c1.button("✅ Select All on Page", on_click=select_page, use_container_width=True)
        c2.button("❌ Deselect All on Page", on_click=deselect_page, use_container_width=True)
        c3.button("✅ Select All (All Pages)", on_click=select_all, use_container_width=True)
        c4.button("❌ Deselect All (All Pages)", on_click=deselect_all, use_container_width=True)
        
        st.markdown("---")
        
        # File list with checkboxes
        def update_selection(file_path, new_value):
            st.session_state.anonymizer_file_selections[str(file_path)] = new_value
        
        for fp in paginated_files:
            from datetime import datetime
            mod_time = datetime.fromtimestamp(fp.stat().st_mtime)
            info = get_file_info(fp)
            
            # Create label similar to Knowledge Ingest
            label = f"**{fp.name}** ({info['size']}) - `{mod_time.strftime('%Y-%m-%d %H:%M:%S')}`"
            
            is_selected = st.session_state.anonymizer_file_selections.get(str(fp), False)
            new_selection = st.checkbox(label, value=is_selected, key=f"cb_{fp}")
            
            if new_selection != is_selected:
                update_selection(fp, new_selection)
                st.rerun()
        
        st.divider()
        
        # Navigation controls (same as Knowledge Ingest)
        nav_cols = st.columns([1, 1, 5])
        
        if current_page > 0:
            if nav_cols[0].button("⬅️ Previous", use_container_width=True):
                st.session_state.anonymizer_current_page = current_page - 1
                st.rerun()
        
        if end_idx < total_files:
            if nav_cols[1].button("Next ➡️", use_container_width=True):
                st.session_state.anonymizer_current_page = current_page + 1
                st.rerun()
        
        nav_cols[2].write(f"Page {current_page + 1} of {total_pages}")
        
        st.divider()
        
        # Process selected files button
        if num_selected > 0:
            # Show processing section
            st.markdown("""
            <div style="background-color: #e8f4fd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
                <h3 style="margin: 0; color: #1f77b4;">🚀 Ready to Process</h3>
                <p style="margin: 0.5rem 0 0 0;">Click the button below to start anonymizing your selected files</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(f"🔒 Process {num_selected} Selected Files", type="primary", use_container_width=True):
                    # Add selected files to processing queue
                    selected_files.clear()
                    for file_path_str, is_selected in st.session_state.anonymizer_file_selections.items():
                        if is_selected:
                            file_path = Path(file_path_str)
                            if file_path.exists():
                                selected_files.append(file_path)
                    
                    if selected_files:
                        # Store files and trigger processing
                        st.session_state.files_to_process = selected_files.copy()
                        st.session_state.processing_status = "starting"
                        st.session_state.auto_start_processing = True
                        
                        st.success(f"🚀 Starting anonymization of {len(selected_files)} files...")
                        st.rerun()
                    else:
                        st.error("No valid files to process")
        else:
            st.info("Select files above to begin anonymization")

# Handle files from session state (when processing was initiated from batch)
if st.session_state.get("files_to_process") and not selected_files:
    selected_files = st.session_state.files_to_process.copy()
    # Clear the session state files after loading
    if st.session_state.get("processing_status") == "completed":
        st.session_state.files_to_process = []

# Display Selected Files
if selected_files:
    st.subheader("📋 Selected Files")
    
    # Remove duplicates while preserving order
    unique_files = []
    seen = set()
    for f in selected_files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    
    selected_files = unique_files
    
    for i, file_path in enumerate(selected_files):
        info = get_file_info(file_path)
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.text(f"📄 {info['name']} ({info['size']}, {info['type']})")
        
        with col2:
            if not info['supported']:
                st.error("Unsupported")
            else:
                st.success("✓ Ready")
        
        with col3:
            if st.button("Remove", key=f"remove_{i}"):
                selected_files.pop(i)
                st.rerun()

# Output Directory Selection
st.subheader("📁 Output Configuration")

col1, col2 = st.columns([2, 1])

with col1:
    output_directory = st.text_input(
        "Output directory (leave empty to use same directory as input files):",
        placeholder="/path/to/output or C:\\Output",
        help="Directory where anonymized files will be saved"
    )

with col2:
    if st.button("Create Directory"):
        if output_directory:
            try:
                output_path = normalize_path(output_directory)
                if output_path:
                    ensure_directory(output_path)
                    st.success("Directory created/verified")
                else:
                    st.error("Invalid path")
            except Exception as e:
                st.error(f"Error creating directory: {e}")

# Anonymization Execution
st.divider()

if selected_files:
    st.subheader("🚀 Run Anonymization")
    
    # Summary
    supported_count = sum(1 for f in selected_files if is_supported_file(f))
    st.info(f"Ready to anonymize {supported_count} supported files")
    
    # Check processing status and show feedback
    processing_status = st.session_state.get("processing_status", None)
    if processing_status == "starting":
        st.info("⏳ **Processing initiated...** Preparing anonymization engine...")
    
    # Check if auto-processing was triggered from batch selection
    auto_start = st.session_state.get("auto_start_processing", False)
    if auto_start:
        st.session_state.auto_start_processing = False  # Reset flag
        st.session_state.processing_status = "running"  # Update status
        process_files = True
        st.info("🔄 **Processing started!** Anonymizing your selected files...")
    else:
        process_files = st.button("🔒 Start Anonymization", type="primary", use_container_width=True)
    
    if process_files:
        if supported_count == 0:
            st.error("No supported files selected")
        else:
            # Create anonymizer
            anonymizer = DocumentAnonymizer()
            
            # Prepare output directory
            output_dir = None
            if output_directory:
                output_dir = normalize_path(output_directory)
                if output_dir:
                    ensure_directory(output_dir)
                else:
                    st.error("Invalid output directory")
                    st.stop()
            
            # Filter to supported files only
            files_to_process = [f for f in selected_files if is_supported_file(f)]
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            try:
                # Run anonymization
                status_text.text("Starting anonymization...")
                
                results = anonymizer.anonymize_batch(
                    files_to_process,
                    output_dir,
                    shared_mapping=shared_mapping,
                    confidence_threshold=confidence_threshold
                )
                
                # Display results
                progress_bar.progress(1.0)
                status_text.text("Anonymization complete!")
                
                # Reset processing status
                st.session_state.processing_status = "completed"
                
                with results_container:
                    st.success(f"Successfully processed {len(results)} files")
                    
                    # Results table
                    st.markdown("### 📊 Anonymization Results")
                    
                    success_count = 0
                    error_count = 0
                    
                    for input_file, output_file in results.items():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.text(f"📄 {Path(input_file).name}")
                        
                        with col2:
                            if output_file.startswith("ERROR:"):
                                st.error(output_file)
                                error_count += 1
                            else:
                                st.success(f"✓ → {Path(output_file).name}")
                                success_count += 1
                        
                        with col3:
                            if not output_file.startswith("ERROR:"):
                                if st.button("View", key=f"view_{input_file}"):
                                    # Show preview of anonymized content
                                    try:
                                        with open(output_file, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        st.text_area(
                                            f"Preview: {Path(output_file).name}",
                                            content[:1000] + "..." if len(content) > 1000 else content,
                                            height=200
                                        )
                                    except Exception as e:
                                        st.error(f"Error reading file: {e}")
                    
                    # Summary
                    st.markdown(f"**Summary:** {success_count} successful, {error_count} errors")
                    
                    # Mapping report link
                    if shared_mapping and output_dir:
                        mapping_file = output_dir / "anonymization_mapping.txt"
                        if mapping_file.exists():
                            st.info(f"📋 Anonymization mapping saved to: {mapping_file}")
                            
                            if st.button("Show Mapping Preview"):
                                try:
                                    with open(mapping_file, 'r', encoding='utf-8') as f:
                                        mapping_content = f.read()
                                    st.text_area(
                                        "Anonymization Mapping",
                                        mapping_content,
                                        height=300
                                    )
                                except Exception as e:
                                    st.error(f"Error reading mapping file: {e}")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("Anonymization failed!")
                
                # Reset processing status on error
                st.session_state.processing_status = "failed"
                
                st.error(f"Anonymization failed: {e}")
                logger.error(f"Anonymization failed: {e}")

else:
    st.info("👆 Select files above to begin anonymization")

# Help Section
st.divider()

with st.expander("❓ Help & Tips"):
    st.markdown("""
    ### Supported File Formats
    - **TXT**: Plain text files
    - **PDF**: Adobe PDF documents
    - **DOCX**: Microsoft Word documents
    
    ### How It Works
    1. **Entity Detection**: Uses AI to identify people, companies, projects, and other sensitive information
    2. **Smart Replacement**: Replaces entities with consistent anonymous names (Person A, Company 1, etc.)
    3. **Structure Preservation**: Maintains document formatting and readability
    4. **Mapping Report**: Creates a reference file showing original → anonymous mappings
    
    ### Tips for Best Results
    - **Test First**: Try with a small sample before processing large batches
    - **Review Output**: Always review anonymized documents before use
    - **Keep Mappings Secure**: The mapping file can reverse anonymization - store it securely
    - **Shared Mapping**: Use shared mapping for related documents to maintain consistency
    
    ### Platform Compatibility
    - **Mac**: Supports drag-drop from Finder
    - **Windows**: Supports drag-drop from File Explorer
    - **Linux**: Supports standard file paths
    - **Docker**: Fully supported in containerized environments
    
    ### Troubleshooting
    - **File Not Found**: Ensure file paths are accessible from the current environment
    - **Unsupported Format**: Convert files to TXT, PDF, or DOCX
    - **Permission Errors**: Check file and directory permissions
    - **Memory Issues**: Process large files individually rather than in batches
    """)

# Show help modal if requested
if st.session_state.get("show_help_modal", False):
    help_topic = st.session_state.get("help_topic", "anonymizer")
    help_system.show_help_modal(help_topic)