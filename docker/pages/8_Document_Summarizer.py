# ## File: pages/14_Document_Summarizer.py
# Version: v5.6.0
# Date: 2026-01-26
# Purpose: Advanced Document Summarizer with multiple detail levels.
#          Leverages Docling, LLM infrastructure, and intelligent chunking.
#          NEW: Hardware-aware model selection and Document Q&A feature.

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
from cortex_engine.document_summarizer import (
    DocumentSummarizer, SummaryResult, QAResult, SUMMARIZER_MODELS
)
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path
from cortex_engine.config_manager import ConfigManager
from cortex_engine.version_config import VERSION_STRING

# Set up logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(page_title="Document Summarizer", layout="wide", page_icon="📄")

# Page metadata
PAGE_VERSION = VERSION_STRING

def get_installed_ollama_models() -> set:
    """Get set of actually installed Ollama models."""
    import requests
    installed = set()
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for m in models:
                name = m.get('name', '')
                installed.add(name)
                # Also add base name without tag
                if ':' in name:
                    installed.add(name.split(':')[0])
    except Exception as e:
        logger.debug(f"Could not fetch Ollama models: {e}")
    return installed


def install_ollama_model(model_name: str) -> bool:
    """Install an Ollama model. Returns True if successful."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for large models
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to install model {model_name}: {e}")
        return False


def render_model_selector():
    """Render the model selection sidebar with clear installed/available sections."""
    st.sidebar.header("🤖 Model Selection")

    # Get hardware info
    try:
        from cortex_engine.utils.smart_model_selector import detect_nvidia_gpu
        has_nvidia, gpu_info = detect_nvidia_gpu()
        available_vram = gpu_info.get("memory_total_gb", 0) if has_nvidia else 0
    except Exception:
        has_nvidia = False
        available_vram = 0

    # Get actually installed models from Ollama
    installed_models = get_installed_ollama_models()

    # Categorize models
    installed_ready = []  # Installed and can run
    installed_heavy = []  # Installed but needs more VRAM
    not_installed = []    # Not installed

    recommended_model = None

    for model_name, config in SUMMARIZER_MODELS.items():
        vram_needed = config.get("vram_gb", 0)
        can_run = available_vram >= vram_needed if has_nvidia else vram_needed <= 4.0

        # Check if installed (check both full name and base name)
        base_name = model_name.split(':')[0]
        is_installed = model_name in installed_models or base_name in installed_models

        model_info = {
            "name": model_name,
            "config": config,
            "can_run": can_run,
            "is_installed": is_installed
        }

        if is_installed:
            if can_run:
                installed_ready.append(model_info)
                # Track best recommended model (prefer non-vision, highest tier that fits)
                if not config.get("multimodal") and (
                    recommended_model is None or
                    config.get("vram_gb", 0) > SUMMARIZER_MODELS.get(recommended_model, {}).get("vram_gb", 0)
                ):
                    recommended_model = model_name
            else:
                installed_heavy.append(model_info)
        else:
            if can_run:  # Only show installable models that can actually run
                not_installed.append(model_info)

    # Initialize session state
    if 'summarizer_model' not in st.session_state:
        st.session_state.summarizer_model = recommended_model or (
            installed_ready[0]["name"] if installed_ready else "mistral:latest"
        )

    # ============ INSTALLED & READY MODELS ============
    if installed_ready:
        st.sidebar.markdown("### ✅ Installed Models")

        # Build options for dropdown
        model_options = []
        model_labels = {}

        for m in installed_ready:
            name = m["name"]
            config = m["config"]
            vram = config.get("vram_gb", 0)
            vision = " 👁️" if config.get("multimodal") else ""
            rec = " ⭐" if name == recommended_model else ""

            label = f"{name}{vision} ({vram}GB){rec}"
            model_options.append(name)
            model_labels[name] = label

        # Ensure current selection is valid
        current = st.session_state.summarizer_model
        if current not in model_options:
            current = model_options[0]
            st.session_state.summarizer_model = current

        selected_model = st.sidebar.selectbox(
            "Select model:",
            options=model_options,
            format_func=lambda x: model_labels.get(x, x),
            index=model_options.index(current) if current in model_options else 0,
            key="model_selector_main"
        )

        # Update session state
        if selected_model != st.session_state.summarizer_model:
            st.session_state.summarizer_model = selected_model

        # Show selected model details
        if selected_model in SUMMARIZER_MODELS:
            config = SUMMARIZER_MODELS[selected_model]
            st.sidebar.caption(f"📝 {config.get('description', '')}")
            st.sidebar.caption(f"📊 Context: {config.get('context_window', 8192):,} tokens")

        if recommended_model:
            st.sidebar.info(f"⭐ **Recommended:** {recommended_model}")

    else:
        st.sidebar.warning("⚠️ No compatible models installed!")
        selected_model = None

    # ============ MODELS THAT NEED MORE VRAM ============
    if installed_heavy:
        with st.sidebar.expander(f"⚠️ Installed but need more VRAM ({len(installed_heavy)})", expanded=False):
            for m in installed_heavy:
                config = m["config"]
                st.markdown(f"**{m['name']}** - needs {config.get('vram_gb', 0)}GB (you have {available_vram:.1f}GB)")

    # ============ AVAILABLE TO INSTALL ============
    if not_installed:
        with st.sidebar.expander(f"📥 Available to Install ({len(not_installed)})", expanded=not installed_ready):
            st.caption("These models can run on your hardware:")

            for m in not_installed:
                config = m["config"]
                name = m["name"]
                vram = config.get("vram_gb", 0)
                vision = " 👁️" if config.get("multimodal") else ""
                desc = config.get("description", "")

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{name}**{vision}")
                    st.caption(f"{desc} ({vram}GB)")

                with col2:
                    # Install button with unique key
                    if st.button("Install", key=f"install_{name}", use_container_width=True):
                        with st.spinner(f"Installing {name}... This may take several minutes."):
                            success = install_ollama_model(name)
                            if success:
                                st.success(f"✅ Installed {name}!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to install {name}")

    # ============ HARDWARE INFO ============
    with st.sidebar.expander("💻 Your Hardware", expanded=False):
        if has_nvidia:
            st.markdown(f"**GPU:** {gpu_info.get('device_name', 'Unknown')}")
            st.markdown(f"**VRAM:** {available_vram:.1f}GB")
        else:
            st.markdown("**GPU:** None detected")
            st.markdown("Running on CPU (limited to small models)")

    return selected_model or "mistral:latest"


def main():
    """Main Document Summarizer application."""

    # Initialize session state
    if 'summarizer_results' not in st.session_state:
        st.session_state.summarizer_results = {}

    if 'current_summary' not in st.session_state:
        st.session_state.current_summary = None

    if 'document_content' not in st.session_state:
        st.session_state.document_content = None

    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    # Render model selector in sidebar
    selected_model = render_model_selector()

    # Header
    st.title("📄 Document Summarizer")
    st.caption(f"Version: {PAGE_VERSION} • Advanced AI-powered document analysis with model selection and Q&A")
    
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
        - **🆕 Model Selection**: Choose from available local models based on your hardware
        - **🆕 Document Q&A**: Ask follow-up questions about the document after summarization

        **🎯 Perfect For:**
        - Executive briefings from lengthy reports
        - Key insights from research papers
        - Quick overviews of contracts and proposals
        - Structured analysis of meeting notes
        - Deep-dive Q&A sessions with documents
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
            process_document(selected_file, summary_level, selected_model)
    
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

def process_document(file_path: str, summary_level: str, model_name: str):
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
        st.info(f"🤖 **Using model:** {model_name}")
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
            detail_text.text(f"🤖 AI ({model_name}) is generating your summary...")
        else:
            detail_text.text("✨ Finalizing summary format and preparing results...")

    try:
        # Initialize summarizer with selected model
        status_text.markdown(f"**🚀 Initializing document summarizer with {model_name}...**")
        summarizer = DocumentSummarizer(model_name=model_name)

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
            # Store document content for Q&A
            st.session_state.document_content = result.document_content
            # Clear Q&A history for new document
            st.session_state.qa_history = []

            # Success message
            st.success(f"✅ Summary generated successfully!")
            st.info(f"📊 Processed {result.word_count:,} words in {result.processing_time:.1f} seconds using {model_name}")

            # Log successful processing
            logger.info(f"Document summarized successfully: {result.metadata.get('filename', 'unknown')} ({summary_level} level) with {model_name}")

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
    """Display the summary results with download options and Q&A."""

    # Metadata display
    metadata = result.metadata

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Document", metadata.get('filename', 'Unknown'))
    with col2:
        st.metric("📊 Words", f"{result.word_count:,}")
    with col3:
        st.metric("⏱️ Processing", f"{result.processing_time:.1f}s")
    with col4:
        st.metric("🤖 Model", metadata.get('model_used', 'Unknown').split(':')[0])

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

    # ==================== NEW: Document Q&A Section ====================
    st.divider()
    st.subheader("💬 Ask Questions About This Document")
    st.caption("Ask follow-up questions about the document content. The AI will answer based only on what's in the document.")

    # Q&A input
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What are the main conclusions? Who are the key stakeholders?",
        key="qa_question_input"
    )

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True, disabled=not question)
    with col_clear:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.qa_history = []
            st.rerun()

    # Process question
    if ask_button and question:
        with st.spinner(f"🤖 Thinking with {st.session_state.get('summarizer_model', 'AI')}..."):
            try:
                summarizer = DocumentSummarizer(model_name=st.session_state.get('summarizer_model'))
                qa_result = summarizer.query_document(
                    question=question,
                    document_content=st.session_state.get('document_content'),
                    summary=result.summary
                )

                if qa_result.success:
                    # Add to history
                    st.session_state.qa_history.append({
                        'question': question,
                        'answer': qa_result.answer,
                        'time': qa_result.processing_time
                    })
                    st.rerun()
                else:
                    st.error(f"❌ Could not answer: {qa_result.error}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display Q&A history
    if st.session_state.get('qa_history'):
        st.markdown("---")
        st.markdown("**📜 Q&A History:**")
        for i, qa in enumerate(reversed(st.session_state.qa_history)):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.qa_history) - i}:** {qa['question']}")
                st.markdown(qa['answer'])
                st.caption(f"⏱️ {qa['time']:.1f}s")
                st.markdown("---")

    # ==================== Extended Q&A with Document Dialog ====================
    st.divider()
    st.subheader("💬 Want Multi-Turn Conversations?")

    with st.container(border=True):
        st.markdown("""
        **Document Dialog** enables rich, multi-turn Q&A with source citations across document collections.

        To use Document Dialog with this document:
        1. **Ingest the document** via Knowledge Ingest (adds embeddings, entity extraction, metadata)
        2. **Add to a collection** via Knowledge Search or Collection Management
        3. **Open Document Dialog** for conversational Q&A with citations
        """)

        col_nav1, col_nav2 = st.columns(2)

        with col_nav1:
            if st.button("📥 Go to Knowledge Ingest", use_container_width=True):
                st.switch_page("pages/2_Knowledge_Ingest.py")

        with col_nav2:
            if st.button("💬 Go to Document Dialog", use_container_width=True):
                st.switch_page("pages/12_Document_Dialog.py")

        st.caption("💡 This page is for quick one-off analysis. For persistent knowledge base Q&A, use the ingest workflow above.")

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
                "Model Used": metadata.get('model_used', 'Unknown')
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
            st.session_state.document_content = None
            st.session_state.qa_history = []
            st.rerun()

    with col3:
        # Option to save to knowledge base (future feature)
        st.button("💾 Save to Knowledge Base", use_container_width=True, disabled=True, help="Coming soon: Save summaries directly to your knowledge base")

if __name__ == "__main__":
    main()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
