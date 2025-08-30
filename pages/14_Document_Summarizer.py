# ## File: pages/14_Document_Summarizer.py
# Version: v4.4.0
# Date: 2025-08-28
# Purpose: Advanced Document Summarizer with multiple detail levels.
#          Leverages Docling, LLM infrastructure, and intelligent chunking.

import streamlit as st
import sys
from pathlib import Path
import os
import tempfile
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from cortex_engine.document_summarizer import DocumentSummarizer, SummaryResult
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path
from cortex_engine.config_manager import ConfigManager

# Set up logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(page_title="Document Summarizer", layout="wide", page_icon="📄")

# Page metadata
PAGE_VERSION = "v4.4.0"

def main():
    """Main Document Summarizer application."""
    
    # Initialize session state
    if 'summarizer_results' not in st.session_state:
        st.session_state.summarizer_results = {}
    
    if 'current_summary' not in st.session_state:
        st.session_state.current_summary = None
    
    # Header
    st.title("📄 Document Summarizer")
    st.caption(f"Version: {PAGE_VERSION} • Advanced AI-powered document analysis with multiple detail levels")
    
    # Info section
    with st.expander("ℹ️ About Document Summarizer", expanded=False):
        st.markdown("""
        **Transform any document into clear, actionable summaries** using advanced AI analysis.
        
        **✨ Features:**
        - **Multiple Detail Levels**: Choose from Highlights, Summary, or Detailed analysis
        - **Smart Processing**: Handles large documents with intelligent chunking
        - **Advanced Extraction**: Uses Docling for superior PDF and Office document processing
        - **Markdown Output**: Clean, structured summaries ready for use
        - **File Support**: PDF, DOCX, PPTX, XLSX, TXT files
        - **Integration Ready**: Works with anonymized documents from Document Anonymizer
        
        **🎯 Perfect For:**
        - Executive briefings from lengthy reports
        - Key insights from research papers
        - Quick overviews of contracts and proposals
        - Structured analysis of meeting notes
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Document Input")
        
        # File upload method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Browse Anonymized Documents"],
            help="Upload a new file or select from previously anonymized documents"
        )
        
        selected_file = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a document to summarize:",
                type=['pdf', 'docx', 'pptx', 'xlsx', 'txt'],
                help="Supported formats: PDF, Word, PowerPoint, Excel, Text files"
            )
            
            if uploaded_file:
                # Save uploaded file temporarily with proper permissions
                try:
                    temp_dir = Path(tempfile.gettempdir()) / "cortex_summaries"
                    temp_dir.mkdir(exist_ok=True, mode=0o755)
                    
                    # Create temporary file with proper extension
                    file_extension = Path(uploaded_file.name).suffix
                    temp_filename = f"uploaded_{int(time.time())}_{uploaded_file.name}"
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
        
        else:  # Browse Anonymized Documents
            # Load configuration to find anonymized documents directory
            config_manager = ConfigManager()
            config = config_manager.get_config()
            
            # Look for anonymized documents in common locations
            possible_dirs = []
            if config.get('db_path'):
                base_path = Path(convert_windows_to_wsl_path(config['db_path']))
                possible_dirs.extend([
                    base_path / "anonymized_documents",
                    base_path / "anonymized",
                    base_path.parent / "anonymized_documents"
                ])
            
            # Also check project directory
            project_anonymized = project_root / "anonymized_documents"
            if project_anonymized.exists():
                possible_dirs.append(project_anonymized)
            
            anonymized_files = []
            for dir_path in possible_dirs:
                if dir_path.exists():
                    for file_path in dir_path.glob("**/*"):
                        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.pptx', '.xlsx']:
                            anonymized_files.append(file_path)
            
            if anonymized_files:
                file_names = [f"{f.name} ({f.parent.name})" for f in anonymized_files]
                selected_idx = st.selectbox(
                    "Select anonymized document:",
                    range(len(file_names)),
                    format_func=lambda x: file_names[x],
                    index=None,
                    placeholder="Choose a document..."
                )
                
                if selected_idx is not None:
                    selected_file = str(anonymized_files[selected_idx])
                    st.success(f"📄 Selected: {anonymized_files[selected_idx].name}")
            else:
                st.info("📂 No anonymized documents found. Use the Document Anonymizer first or upload a file.")
        
        # Summary level selection
        st.header("🎯 Summary Level")
        
        summary_level = st.radio(
            "Choose detail level:",
            ["highlights", "summary", "detailed"],
            format_func=lambda x: {
                "highlights": "📋 Highlights Only",
                "summary": "📄 Summary", 
                "detailed": "📖 Detailed Analysis"
            }[x],
            help="Highlights: 2-3 key points • Summary: 1-2 paragraphs per section • Detailed: Comprehensive analysis"
        )
        
        # Level descriptions
        level_descriptions = {
            "highlights": "**2-3 key bullet points** with executive summary. Perfect for quick briefings.",
            "summary": "**1-2 paragraphs per major section** with key findings and actionable insights.",
            "detailed": "**Comprehensive analysis** with section-by-section breakdown, evidence, and implementation guidance."
        }
        
        st.info(level_descriptions[summary_level])
        
        # Process button
        if selected_file and st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            process_document(selected_file, summary_level)
    
    with col2:
        st.header("📊 Summary Results")
        
        if st.session_state.current_summary:
            display_summary_results(st.session_state.current_summary)
        else:
            st.info("👈 Select a document and summary level to get started")
            
            # Show example of what summaries look like
            with st.expander("👀 Preview: What summaries look like", expanded=False):
                st.markdown("""
                **📋 Highlights Only Example:**
                ```
                ## 📋 Document Highlights
                This quarterly report shows 15% revenue growth with strong market expansion in Q3.
                
                **Key Takeaways:**
                - Revenue increased 15% YoY to $2.4M
                - Customer acquisition cost decreased 22%
                - Recommended focus on expanding digital channels
                ```
                
                **📄 Summary Example:**
                ```
                ## 📄 Document Summary
                
                ### Purpose & Context
                Quarterly business review covering financial performance, market analysis, and strategic recommendations...
                
                ### Key Findings
                Revenue growth driven by digital transformation initiatives...
                ```
                
                **📖 Detailed Analysis Example:**
                ```
                ## 📖 Detailed Document Analysis
                
                ### Executive Summary
                Comprehensive analysis of Q3 performance...
                
                ### Financial Performance Analysis
                [Detailed breakdown with supporting data]
                
                ### Market Conditions & Opportunities
                [In-depth market analysis]
                ```
                """)

def process_document(file_path: str, summary_level: str):
    """Process document and generate summary."""
    
    # Check if this is a temporary uploaded file
    is_temp_file = "/tmp/" in file_path or "cortex_summaries" in file_path
    
    # Initialize progress tracking with better visibility
    st.markdown("### 🔄 Processing Status")
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        # Add estimated time info
        st.info("⏱️ **Estimated processing time:** Small docs (30-60s), Large docs (2-5 mins)")
        st.info("💡 **Tip:** For very large documents, try 'Highlights' level first - it's faster and more reliable!")
    
    def progress_callback(message: str, percent: float):
        progress_bar.progress(percent / 100)
        status_text.markdown(f"**🔄 {message}**")
        
        # Add helpful details based on progress stage
        if percent < 30:
            detail_text.text("📖 Reading and extracting text from document...")
        elif percent < 50:
            detail_text.text("🧮 Analyzing document structure and content...")
        elif percent < 85:
            detail_text.text("🤖 AI is generating your summary (loading model to GPU if needed)...")
        else:
            detail_text.text("✨ Finalizing summary format and preparing results...")
    
    try:
        # Initialize summarizer
        status_text.markdown("**🚀 Initializing document summarizer...**")
        summarizer = DocumentSummarizer()
        
        # Process document - remove the spinner since we have detailed progress
        result = summarizer.summarize_document(
            file_path=file_path,
            summary_level=summary_level,
            progress_callback=progress_callback
        )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if result.success:
            # Store result in session state
            st.session_state.current_summary = result
            
            # Success message
            st.success(f"✅ Summary generated successfully!")
            st.info(f"📊 Processed {result.word_count:,} words in {result.processing_time:.1f} seconds")
            
            # Log successful processing
            logger.info(f"Document summarized successfully: {result.metadata.get('filename', 'unknown')} ({summary_level} level)")
            
        else:
            st.error(f"❌ Summarization failed: {result.error}")
            logger.error(f"Document summarization failed: {result.error}")
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error in document processing: {e}")
    
    finally:
        # Clean up temporary files
        if is_temp_file and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"🗑️ Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {file_path}: {cleanup_error}")

def display_summary_results(result: SummaryResult):
    """Display the summary results with download options."""
    
    # Metadata display
    metadata = result.metadata
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Document", metadata.get('filename', 'Unknown'))
    with col2:
        st.metric("📊 Words", f"{result.word_count:,}")
    with col3:
        st.metric("⏱️ Processing", f"{result.processing_time:.1f}s")
    
    # Summary level indicator
    level_icons = {
        'highlights': '📋 Highlights Only',
        'summary': '📄 Summary',
        'detailed': '📖 Detailed Analysis'
    }
    
    st.subheader(f"{level_icons.get(metadata.get('summary_level', 'summary'), '📄 Summary')}")
    
    # Display the summary
    st.markdown(result.summary)
    
    # Download options
    st.subheader("💾 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate filename for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(metadata.get('filename', 'document')).stem
        summary_level = metadata.get('summary_level', 'summary')
        
        download_filename = f"{base_name}_{summary_level}_{timestamp}.md"
        
        st.download_button(
            label="📥 Download as Markdown",
            data=result.summary,
            file_name=download_filename,
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        # Copy to clipboard button (using st.code for easy copying)
        if st.button("📋 Show Raw Markdown", use_container_width=True):
            st.code(result.summary, language="markdown")
    
    # Processing details
    with st.expander("📊 Processing Details", expanded=False):
        st.json({
            "Document Information": {
                "Filename": metadata.get('filename'),
                "File Size": f"{metadata.get('file_size_mb', 0):.2f} MB",
                "File Type": metadata.get('file_extension'),
                "Estimated Pages": result.page_count
            },
            "Processing Information": {
                "Summary Level": metadata.get('summary_level'),
                "Processing Time": f"{result.processing_time:.2f} seconds",
                "Word Count": result.word_count,
                "Model Used": "mistral-small3.2"
            }
        })
    
    # Action buttons
    st.subheader("🔄 Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process Another Level", use_container_width=True):
            st.session_state.current_summary = None
            st.rerun()
    
    with col2:
        if st.button("📄 New Document", use_container_width=True):
            st.session_state.current_summary = None
            st.session_state.summarizer_results = {}
            st.rerun()
    
    with col3:
        # Option to save to knowledge base (future feature)
        st.button("💾 Save to Knowledge Base", use_container_width=True, disabled=True, help="Coming soon: Save summaries directly to your knowledge base")

if __name__ == "__main__":
    main()