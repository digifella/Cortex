# ## File: pages/8_Document_Anonymizer.py
# Version: v5.2.0
# Date: 2025-08-30
# Purpose: Streamlined document anonymization interface with clean file browsing.
#          Replaces identifying information with generic placeholders using modern UI patterns.

import streamlit as st
import sys
from pathlib import Path
import os
import tempfile
import time
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from cortex_engine.anonymizer import DocumentAnonymizer, AnonymizationMapping
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path
from cortex_engine.config_manager import ConfigManager

# Set up logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(page_title="Document Anonymizer", layout="wide", page_icon="🎭")

# Page metadata
PAGE_VERSION = "v5.2.0"

def main():
    """Main Document Anonymizer application."""
    
    # Initialize session state
    if 'anonymizer_results' not in st.session_state:
        st.session_state.anonymizer_results = {}
    
    if 'current_anonymization' not in st.session_state:
        st.session_state.current_anonymization = None
    
    # Header
    st.title("🎭 Document Anonymizer")
    st.caption(f"Version: {PAGE_VERSION} • Streamlined privacy protection for your documents")
    
    # Info section
    with st.expander("ℹ️ About Document Anonymizer", expanded=False):
        st.markdown("""
        **Protect sensitive information** by replacing identifying details with generic placeholders.
        
        **✨ Features:**
        - **Smart Entity Detection**: Automatically finds people, companies, and locations
        - **Consistent Replacement**: Same entity always gets the same placeholder (e.g., "John Smith" → "Person A")
        - **Privacy Safe**: Perfect for sharing confidential documents for review
        - **Multiple Formats**: PDF, Word, and text file support
        - **Docker Compatible**: Works seamlessly in containerized environments
        
        **🎯 Perfect For:**
        - Preparing documents for external review
        - Creating demo versions of confidential files
        - Protecting client privacy in case studies
        - Sanitizing documents for training purposes
        
        **🔄 Replacement Examples:**
        - **People**: John Smith → Person A, Jane Doe → Person B
        - **Companies**: Acme Corp → Company 1, BigTech Inc → Company 2  
        - **Projects**: Project Alpha → Project 1, Initiative Beta → Project 2
        - **Contact Info**: emails → [EMAIL], phones → [PHONE], URLs → [URL]
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Document Input")
        
        # File input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Browse Knowledge Base"],
            help="Upload a new file or select from your knowledge base documents"
        )
        
        selected_file = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a document to anonymize:",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, Word documents, Text files"
            )
            
            if uploaded_file:
                # Save uploaded file temporarily with proper permissions
                try:
                    temp_dir = Path(tempfile.gettempdir()) / "cortex_anonymizer"
                    temp_dir.mkdir(exist_ok=True, mode=0o755)
                    
                    # Create temporary file with proper extension
                    file_extension = Path(uploaded_file.name).suffix
                    temp_filename = f"upload_{int(time.time())}_{uploaded_file.name}"
                    selected_file = str(temp_dir / temp_filename)
                    
                    # Write file with proper permissions
                    with open(selected_file, 'wb') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                    
                    # Set file permissions
                    os.chmod(selected_file, 0o644)
                    
                    st.success(f"📄 File uploaded: {uploaded_file.name}")
                    st.info(f"Size: {uploaded_file.size / 1024:.1f} KB")
                    
                except Exception as e:
                    st.error(f"❌ Failed to save uploaded file: {str(e)}")
                    selected_file = None
        
        else:  # Browse Knowledge Base
            # Load configuration to find knowledge base documents
            config_manager = ConfigManager()
            config = config_manager.get_config()
            
            # Look for documents in knowledge base locations
            possible_dirs = []
            if config.get('db_path'):
                base_path = Path(convert_windows_to_wsl_path(config['db_path']))
                possible_dirs.extend([
                    base_path / "documents",
                    base_path / "source_documents",
                    base_path.parent / "documents",
                    base_path.parent / "source_documents"
                ])
            
            # Also check common project directories
            project_dirs = [
                project_root / "documents",
                project_root / "source_documents", 
                project_root / "test_documents"
            ]
            possible_dirs.extend(project_dirs)
            
            knowledge_files = []
            for dir_path in possible_dirs:
                if dir_path.exists():
                    for file_path in dir_path.glob("**/*"):
                        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
                            knowledge_files.append(file_path)
            
            if knowledge_files:
                file_names = [f"{f.name} ({f.parent.name})" for f in knowledge_files]
                selected_idx = st.selectbox(
                    "Select document from knowledge base:",
                    range(len(file_names)),
                    format_func=lambda x: file_names[x],
                    index=None,
                    placeholder="Choose a document..."
                )
                
                if selected_idx is not None:
                    selected_file = str(knowledge_files[selected_idx])
                    file_info = knowledge_files[selected_idx]
                    st.success(f"📄 Selected: {file_info.name}")
                    st.info(f"Location: {file_info.parent}")
            else:
                st.warning("📁 No documents found in knowledge base directories")
                st.info("💡 Try uploading a file instead, or add documents to your knowledge base first")
        
        # Anonymization settings
        if selected_file:
            st.divider()
            st.subheader("⚙️ Anonymization Settings")
            
            confidence_threshold = st.slider(
                "Entity Detection Confidence:",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.1,
                help="Lower values detect more entities (may include false positives)"
            )
            
            st.session_state.confidence_threshold = confidence_threshold
    
    with col2:
        st.header("🎭 Anonymization Process")
        
        if selected_file:
            # Show file preview info
            file_path = Path(selected_file)
            st.markdown(f"**File:** `{file_path.name}`")
            
            # Process button
            if st.button("🚀 Start Anonymization", type="primary", use_container_width=True):
                
                # Processing with progress
                progress_bar = st.progress(0, "🔄 Initializing anonymization...")
                status_container = st.container()
                
                try:
                    with status_container:
                        st.info("📖 Reading document content...")
                    progress_bar.progress(25, "📖 Reading document...")
                    
                    # Initialize anonymizer
                    anonymizer = DocumentAnonymizer()
                    mapping = AnonymizationMapping()
                    
                    with status_container:
                        st.info("🔍 Detecting entities and sensitive information...")
                    progress_bar.progress(50, "🔍 Detecting entities...")
                    
                    # Process the file
                    result_path, result_mapping = anonymizer.anonymize_single_file(
                        input_path=selected_file,
                        output_path=None,  # Let it auto-generate
                        mapping=mapping,
                        confidence_threshold=st.session_state.confidence_threshold
                    )
                    
                    with status_container:
                        st.info("🎭 Applying anonymization...")
                    progress_bar.progress(75, "🎭 Anonymizing content...")
                    
                    # Store results
                    st.session_state.current_anonymization = {
                        'original_file': selected_file,
                        'anonymized_file': result_path,
                        'mapping': result_mapping,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    progress_bar.progress(100, "✅ Anonymization complete!")
                    
                    with status_container:
                        st.success("🎉 **Anonymization completed successfully!**")
                        
                        # Summary stats
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("People", len([k for k, v in result_mapping.mappings.items() if v.startswith("Person")]))
                        with col_b:
                            st.metric("Companies", len([k for k, v in result_mapping.mappings.items() if v.startswith("Company")]))
                        with col_c:
                            st.metric("Projects", len([k for k, v in result_mapping.mappings.items() if v.startswith("Project")]))
                    
                except Exception as e:
                    progress_bar.progress(0, "❌ Anonymization failed")
                    st.error(f"❌ **Anonymization failed:** {str(e)}")
                    logger.error(f"Anonymization error: {e}", exc_info=True)
        
        # Results section
        if st.session_state.current_anonymization:
            st.divider()
            st.subheader("📋 Anonymization Results")
            
            result = st.session_state.current_anonymization
            
            # File info
            col_orig, col_anon = st.columns(2)
            with col_orig:
                st.markdown("**📄 Original File:**")
                st.code(Path(result['original_file']).name)
            
            with col_anon:
                st.markdown("**🎭 Anonymized File:**")
                st.code(Path(result['anonymized_file']).name)
            
            # Download section
            st.markdown("### 💾 Download Results")
            
            try:
                # Read anonymized content
                with open(result['anonymized_file'], 'r', encoding='utf-8') as f:
                    anonymized_content = f.read()
                
                # Download button for anonymized file
                st.download_button(
                    label="📄 Download Anonymized Document",
                    data=anonymized_content,
                    file_name=Path(result['anonymized_file']).name,
                    mime="text/plain",
                    use_container_width=True
                )
                
                # Generate mapping report
                mapping_content = generate_mapping_report(result['mapping'])
                st.download_button(
                    label="🗂️ Download Mapping Reference", 
                    data=mapping_content,
                    file_name=f"anonymization_mapping_{int(time.time())}.txt",
                    mime="text/plain",
                    help="Reference file showing original → anonymized mappings (keep secure!)"
                )
                
                # Preview section
                with st.expander("👁️ Preview Anonymized Content", expanded=False):
                    preview_content = anonymized_content[:2000]
                    if len(anonymized_content) > 2000:
                        preview_content += "\n\n... [Content truncated for preview] ..."
                    st.text_area("Preview:", preview_content, height=300)
                    
                # Mapping preview
                if result['mapping'].mappings:
                    with st.expander("🔍 Entity Mappings", expanded=False):
                        mapping_df = []
                        for original, anonymized in result['mapping'].mappings.items():
                            mapping_df.append({
                                "Original": original,
                                "Anonymized": anonymized,
                                "Type": get_entity_type_from_anonymized(anonymized)
                            })
                        
                        if mapping_df:
                            import pandas as pd
                            df = pd.DataFrame(mapping_df)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                    
            except Exception as e:
                st.error(f"❌ Could not load results: {str(e)}")
        
        elif selected_file:
            st.info("👆 Click **Start Anonymization** to process your document")
        else:
            st.info("👈 Select a document from the left panel to get started")

def generate_mapping_report(mapping: AnonymizationMapping) -> str:
    """Generate a formatted mapping report."""
    report = []
    report.append("ANONYMIZATION MAPPING REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("⚠️  KEEP THIS FILE SECURE AND SEPARATE FROM ANONYMIZED DOCUMENTS")
    report.append("")
    
    if mapping.mappings:
        # Group by type
        people = []
        companies = []  
        projects = []
        other = []
        
        for original, anonymized in mapping.mappings.items():
            if anonymized.startswith("Person"):
                people.append((original, anonymized))
            elif anonymized.startswith("Company"):
                companies.append((original, anonymized))
            elif anonymized.startswith("Project"):
                projects.append((original, anonymized))
            else:
                other.append((original, anonymized))
        
        if people:
            report.append("👥 PEOPLE:")
            for original, anonymized in sorted(people):
                report.append(f"  {original} → {anonymized}")
            report.append("")
        
        if companies:
            report.append("🏢 COMPANIES:")
            for original, anonymized in sorted(companies):
                report.append(f"  {original} → {anonymized}")
            report.append("")
        
        if projects:
            report.append("📋 PROJECTS:")
            for original, anonymized in sorted(projects):
                report.append(f"  {original} → {anonymized}")
            report.append("")
        
        if other:
            report.append("🔧 OTHER:")
            for original, anonymized in sorted(other):
                report.append(f"  {original} → {anonymized}")
    else:
        report.append("No entity mappings found.")
    
    return "\n".join(report)

def get_entity_type_from_anonymized(anonymized: str) -> str:
    """Get entity type from anonymized name."""
    if anonymized.startswith("Person"):
        return "👥 Person"
    elif anonymized.startswith("Company"):
        return "🏢 Company" 
    elif anonymized.startswith("Project"):
        return "📋 Project"
    elif anonymized.startswith("Location"):
        return "📍 Location"
    else:
        return "🔧 Other"

if __name__ == "__main__":
    main()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
