# ## File: pages/8_Document_Anonymizer.py
# Version: v4.0.4
# Date: 2025-08-28
# Purpose: Simplified document anonymization interface with drag-and-drop processing.
#          Processes documents to replace identifying information with generic placeholders.
#          Auto-processes files on upload with download-only results interface.

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
            value=st.session_state.get("shared_mapping", True),
            key="shared_mapping_checkbox",
            help="When enabled, the same person/company will get the same anonymous name across all files"
        )
        
        preserve_structure = st.checkbox(
            "Preserve document structure",
            value=st.session_state.get("preserve_structure", True),
            key="preserve_structure_checkbox",
            help="Maintain original formatting and structure in output files"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Entity detection confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.get("confidence_threshold", 0.3),
            step=0.1,
            key="confidence_threshold_slider",
            help="Lower values catch more entities but may include false positives"
        )

# Store configuration values immediately
st.session_state.shared_mapping = shared_mapping
st.session_state.confidence_threshold = confidence_threshold
st.session_state.preserve_structure = preserve_structure

# File Upload Section
st.subheader("📁 Drop Files to Anonymize")

# Simple file uploader for drag-and-drop functionality
uploaded_files = st.file_uploader(
    "Drag and drop files here or click to browse:",
    type=['txt', 'pdf', 'docx'],
    accept_multiple_files=True,
    help="Supported formats: TXT, PDF, DOCX",
    key="anonymizer_uploader"
)

# Process files immediately when uploaded
if uploaded_files:
    st.success(f"📤 Processing {len(uploaded_files)} file(s)...")
    
    # Save uploaded files to temporary directory for processing
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    files_to_process = []
    
    for uploaded_file in uploaded_files:
        # Save uploaded file to temporary location
        temp_file_path = temp_dir / uploaded_file.name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        files_to_process.append(temp_file_path)
    
    # Store temp directory in session state for cleanup later
    st.session_state.temp_anonymizer_dir = temp_dir
    
    # Process immediately
    st.session_state.files_to_process = files_to_process
    st.session_state.auto_process = True

# Output Directory Selection
st.subheader("📁 Output Configuration")

col1, col2 = st.columns([2, 1])

with col1:
    output_directory = st.text_input(
        "Output directory (leave empty to use Downloads folder):",
        value=st.session_state.get("output_directory", ""),
        placeholder="/path/to/output or C:\\Output",
        key="output_directory_input",
        help="Directory where anonymized files will be saved. Leave empty for Downloads folder."
    )

with col2:
    if st.button("Create Directory", key="create_output_dir"):
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

# Store output directory immediately  
st.session_state.output_directory = output_directory

# Automatic Processing
if st.session_state.get("files_to_process") and st.session_state.get("auto_process"):
    files_to_process = st.session_state.files_to_process
    st.session_state.auto_process = False  # Reset flag
    
    st.divider()
    st.subheader("🔄 Processing Files...")
    
    # Filter to supported files only
    supported_files = [f for f in files_to_process if is_supported_file(f)]
    
    if len(supported_files) == 0:
        st.error("No supported files found")
        st.stop()
    
    # Get configuration values
    output_directory = st.session_state.get("output_directory", "")
    shared_mapping = st.session_state.get("shared_mapping", True)
    confidence_threshold = st.session_state.get("confidence_threshold", 0.3)
    
    # Prepare output directory
    output_dir = None
    if output_directory:
        output_dir = normalize_path(output_directory)
        if output_dir:
            ensure_directory(output_dir)
        else:
            st.error("Invalid output directory")
            st.stop()
    
    # Create anonymizer
    anonymizer = DocumentAnonymizer()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    try:
        # Run anonymization
        status_text.text("Starting anonymization...")
        
        logger.info(f"Anonymizing {len(supported_files)} files")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Files: {[str(f) for f in supported_files]}")
        
        results = anonymizer.anonymize_batch(
            supported_files,
            output_dir,
            shared_mapping=shared_mapping,
            confidence_threshold=confidence_threshold
        )
        
        logger.info(f"Anonymization results: {results}")
        
        # Display results
        progress_bar.progress(1.0)
        status_text.text("Anonymization complete!")
        
        # Reset processing status
        st.session_state.processing_status = "completed"
        
        # Handle output files from temporary directory
        final_results = {}
        if hasattr(st.session_state, 'temp_anonymizer_dir'):
            import shutil
            from pathlib import Path
            
            # If no output directory specified, move files to Downloads or current directory
            if not output_dir:
                # Try to use Downloads folder, fallback to current directory
                try:
                    downloads_path = Path.home() / "Downloads"
                    if downloads_path.exists():
                        final_output_dir = downloads_path
                    else:
                        final_output_dir = Path.cwd()
                except Exception:
                    final_output_dir = Path.cwd()
                    
                ensure_directory(final_output_dir)
                
                # Move output files to permanent location
                for input_file, output_file in results.items():
                    if not output_file.startswith("ERROR:"):
                        temp_output_path = Path(output_file)
                        if temp_output_path.parent == st.session_state.temp_anonymizer_dir:
                            # Move file to permanent location
                            permanent_path = final_output_dir / temp_output_path.name
                            shutil.move(str(temp_output_path), str(permanent_path))
                            final_results[input_file] = str(permanent_path)
                        else:
                            final_results[input_file] = output_file
                    else:
                        final_results[input_file] = output_file
            else:
                final_results = results
            
            # Clean up temporary input files
            try:
                for item in st.session_state.temp_anonymizer_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                st.session_state.temp_anonymizer_dir.rmdir()
                del st.session_state.temp_anonymizer_dir
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")
        else:
            final_results = results
        
        # Store final results for download access
        st.session_state.anonymizer_results = final_results
        
        with results_container:
            st.success(f"✅ Successfully processed {len(final_results)} files")
            
            # Show where files were saved
            if not output_dir:
                downloads_path = Path.home() / "Downloads"
                if downloads_path.exists():
                    st.info(f"📁 **Files saved to:** {downloads_path}")
                else:
                    st.info(f"📁 **Files saved to:** {Path.cwd()}")
            else:
                st.info(f"📁 **Files saved to:** {output_dir}")
            
            st.markdown("### 📥 Download Anonymized Files")
            
            success_count = 0
            error_count = 0
            
            for input_file, output_file in final_results.items():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if output_file.startswith("ERROR:"):
                        st.error(f"❌ {Path(input_file).name}: {output_file}")
                        error_count += 1
                    else:
                        st.success(f"✅ {Path(input_file).name} → {Path(output_file).name}")
                        success_count += 1
                
                with col2:
                    if not output_file.startswith("ERROR:"):
                        # Download button only
                        try:
                            output_path = Path(output_file)
                            if output_path.exists():
                                with open(output_file, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                st.download_button(
                                    label="💾 Download",
                                    data=file_content,
                                    file_name=output_path.name,
                                    mime="text/plain",
                                    key=f"download_{Path(input_file).stem}"
                                )
                            else:
                                st.error("File not found")
                        except Exception as e:
                            st.error(f"Error: {str(e)[:30]}...")
            
            st.markdown(f"**Summary:** {success_count} successful, {error_count} errors")
            
            # Show mapping file if it exists
            if shared_mapping and not output_dir:
                downloads_path = Path.home() / "Downloads"
                if downloads_path.exists():
                    mapping_file = downloads_path / "anonymization_mapping.txt"
                else:
                    mapping_file = Path.cwd() / "anonymization_mapping.txt"
                
                if mapping_file.exists():
                    st.info(f"📋 Anonymization mapping saved to: {mapping_file}")
                    
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Anonymization failed!")
        
        # Clean up temporary files on error
        if hasattr(st.session_state, 'temp_anonymizer_dir'):
            try:
                import shutil
                shutil.rmtree(st.session_state.temp_anonymizer_dir, ignore_errors=True)
                del st.session_state.temp_anonymizer_dir
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")
        
        st.error(f"Anonymization failed: {e}")
        logger.error(f"Anonymization failed: {e}")

# Clear previous results when new files are uploaded
if not st.session_state.get("files_to_process") and not uploaded_files:
    if "anonymizer_results" in st.session_state:
        del st.session_state.anonymizer_results

# Help Section  
st.divider()

with st.expander("❓ Help & Tips"):
    st.markdown("""
    ### How It Works
    1. **Drop Files**: Drag and drop or browse for TXT, PDF, or DOCX files
    2. **Automatic Processing**: Files are processed immediately upon upload  
    3. **Download Results**: Use the download buttons to get your anonymized files
    
    ### What Gets Anonymized
    - **People names** → Person A, Person B, etc.
    - **Company names** → Company 1, Company 2, etc.
    - **Project names** → Project 1, Project 2, etc.
    - **Email addresses** → [EMAIL]
    - **Phone numbers** → [PHONE]
    - **URLs** → [URL]
    
    ### File Output
    - **Default Location**: Downloads folder (or current directory as fallback)
    - **Custom Location**: Specify in Output Configuration above
    - **File Format**: Anonymized content saved as .txt files
    - **Mapping File**: Optional reference showing original → anonymous mappings
    
    ### Tips for Best Results
    - **Test First**: Try with a small sample before processing large batches
    - **Review Output**: Always verify anonymized documents before use  
    - **Keep Mappings Secure**: The mapping file can reverse anonymization
    - **Use Shared Mapping**: Enable for consistent names across multiple files
    """)

# Show help modal if requested
if st.session_state.get("show_help_modal", False):
    help_topic = st.session_state.get("help_topic", "anonymizer")
    help_system.show_help_modal(help_topic)