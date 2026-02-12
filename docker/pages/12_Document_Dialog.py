# ## File: pages/12_Document_Dialog.py
# Version: v5.6.0
# Date: 2026-01-26
# Purpose: Conversational Q&A interface for document collections.
#          Enables multi-turn conversations with ingested documents
#          using RAG retrieval and optional neural reranking.

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from cortex_engine.document_dialog import (
    DocumentDialogEngine, DialogSession, DialogResponse
)
from cortex_engine.document_summarizer import SUMMARIZER_MODELS
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.config_manager import ConfigManager
from cortex_engine.config import QWEN3_VL_RERANKER_ENABLED
from cortex_engine.version_config import VERSION_STRING
from cortex_engine.utils import get_logger

# Set up logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Document Dialog",
    layout="wide",
    page_icon="💬"
)

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
                if ':' in name:
                    installed.add(name.split(':')[0])
    except Exception as e:
        logger.debug(f"Could not fetch Ollama models: {e}")
    return installed


def render_sidebar():
    """Render sidebar with collection selector and settings."""
    st.sidebar.header("💬 Document Dialog")
    st.sidebar.caption(f"Version: {PAGE_VERSION}")

    # Get database path
    config_manager = ConfigManager()
    config = config_manager.get_config()
    db_path = config.get("ai_database_path", "")

    if not db_path:
        st.sidebar.error("Database path not configured. Please set it in Knowledge Search.")
        return None, None, None, True

    # Initialize collection manager
    try:
        collection_mgr = WorkingCollectionManager()
        collection_names = collection_mgr.get_collection_names()
    except Exception as e:
        st.sidebar.error(f"Could not load collections: {e}")
        return None, None, None, True

    if not collection_names:
        st.sidebar.warning("No collections found. Create one in Knowledge Search or Collection Management.")
        return None, None, None, True

    # Collection selector
    st.sidebar.subheader("📚 Collection")

    # Check for preselected collection from navigation
    preselected = st.session_state.get("dialog_collection_preselect")
    if preselected and preselected in collection_names:
        default_idx = collection_names.index(preselected)
        # Clear preselect after use
        st.session_state.pop("dialog_collection_preselect", None)
    else:
        default_idx = 0

    selected_collection = st.sidebar.selectbox(
        "Select collection:",
        options=collection_names,
        index=default_idx,
        key="dialog_collection_select",
        help="Choose a collection to have a conversation with"
    )

    # Show collection info
    doc_ids = collection_mgr.get_doc_ids_by_name(selected_collection)
    st.sidebar.info(f"📄 {len(doc_ids)} documents in collection")

    # Model selector
    st.sidebar.subheader("🤖 Model")

    # Get hardware info
    try:
        from cortex_engine.utils.smart_model_selector import detect_nvidia_gpu
        has_nvidia, gpu_info = detect_nvidia_gpu()
        available_vram = gpu_info.get("memory_total_gb", 0) if has_nvidia else 0
    except Exception:
        has_nvidia = False
        available_vram = 0

    installed_models = get_installed_ollama_models()

    # Build model options from installed models that can run
    model_options = []
    model_labels = {}
    recommended_model = None

    for model_name, config in SUMMARIZER_MODELS.items():
        vram_needed = config.get("vram_gb", 0)
        can_run = available_vram >= vram_needed if has_nvidia else vram_needed <= 4.0

        base_name = model_name.split(':')[0]
        is_installed = model_name in installed_models or base_name in installed_models

        if is_installed and can_run:
            model_options.append(model_name)
            vram = config.get("vram_gb", 0)
            vision = " 👁️" if config.get("multimodal") else ""
            model_labels[model_name] = f"{model_name}{vision} ({vram}GB)"

            # Track best recommended model
            if not config.get("multimodal") and (
                recommended_model is None or
                config.get("vram_gb", 0) > SUMMARIZER_MODELS.get(recommended_model, {}).get("vram_gb", 0)
            ):
                recommended_model = model_name

    if not model_options:
        st.sidebar.warning("No compatible models installed. Install one via Document Summarizer.")
        model_options = ["mistral:latest"]
        model_labels = {"mistral:latest": "mistral:latest (default)"}

    # Initialize model in session state
    if 'dialog_model' not in st.session_state:
        st.session_state.dialog_model = recommended_model or model_options[0]

    # Ensure current selection is valid
    if st.session_state.dialog_model not in model_options:
        st.session_state.dialog_model = model_options[0]

    selected_model = st.sidebar.selectbox(
        "Select model:",
        options=model_options,
        format_func=lambda x: model_labels.get(x, x),
        index=model_options.index(st.session_state.dialog_model),
        key="dialog_model_select"
    )

    if selected_model != st.session_state.dialog_model:
        st.session_state.dialog_model = selected_model

    # Show model description
    if selected_model in SUMMARIZER_MODELS:
        st.sidebar.caption(f"📝 {SUMMARIZER_MODELS[selected_model].get('description', '')}")

    # Reranker toggle
    st.sidebar.subheader("⚡ Settings")

    use_reranker = st.sidebar.checkbox(
        "Enable Neural Reranking",
        value=st.session_state.get("dialog_use_reranker", QWEN3_VL_RERANKER_ENABLED),
        key="dialog_reranker_toggle",
        help="Use Qwen3-VL reranker for improved precision (~95% vs ~85%)"
    )
    st.session_state.dialog_use_reranker = use_reranker

    # Result count
    top_k = st.sidebar.slider(
        "Context chunks:",
        min_value=3,
        max_value=10,
        value=st.session_state.get("dialog_top_k", 5),
        key="dialog_top_k_slider",
        help="Number of document chunks to retrieve per query"
    )
    st.session_state.dialog_top_k = top_k

    return selected_collection, selected_model, db_path, False


def initialize_session_if_needed(
    collection_name: str,
    model_name: str,
    db_path: str,
    engine: DocumentDialogEngine
):
    """Initialize or update dialog session based on current settings."""
    current_session = st.session_state.get("dialog_session")

    # Check if we need a new session
    needs_new_session = (
        current_session is None or
        current_session.collection_name != collection_name or
        current_session.model_name != model_name
    )

    if needs_new_session:
        engine.model_name = model_name
        session = engine.initialize_session(collection_name)
        st.session_state.dialog_session = session
        st.session_state.dialog_messages = []
        logger.info(f"Created new dialog session for collection '{collection_name}'")

    return st.session_state.dialog_session


def render_document_list(engine: DocumentDialogEngine, collection_name: str):
    """Render expandable document list in sidebar."""
    with st.sidebar.expander("📋 Documents in Collection", expanded=False):
        docs = engine.get_collection_documents(collection_name)
        if docs:
            for doc in docs:
                file_name = doc.get("file_name", "Unknown")
                doc_type = doc.get("document_type", "")
                st.markdown(f"• **{file_name}**")
                if doc_type and doc_type != "Unknown":
                    st.caption(f"  {doc_type}")
        else:
            st.info("No documents found in collection")


def render_chat_interface(engine: DocumentDialogEngine, session: DialogSession):
    """Render the main chat interface."""
    # Initialize message display list
    if "dialog_messages" not in st.session_state:
        st.session_state.dialog_messages = []

    # Sync messages from session to display list
    if len(session.messages) > len(st.session_state.dialog_messages):
        st.session_state.dialog_messages = [
            {"role": msg.role, "content": msg.content, "sources": msg.sources}
            for msg in session.messages
        ]

    # Display chat messages
    for message in st.session_state.dialog_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for src in message["sources"]:
                        st.markdown(f"**[{src.get('source_number', '?')}]** {src.get('file_name', 'Unknown')}")
                        if src.get('document_type') and src.get('document_type') != 'Unknown':
                            st.caption(f"Type: {src.get('document_type')}")

    # Empty state message
    if not st.session_state.dialog_messages:
        st.info("👋 Start a conversation by asking a question about the documents in this collection.")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to display
        st.session_state.dialog_messages.append({"role": "user", "content": prompt, "sources": []})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {session.model_name}..."):
                response = engine.query(
                    question=prompt,
                    session=session,
                    top_k=st.session_state.get("dialog_top_k", 5),
                    use_reranker=st.session_state.get("dialog_use_reranker", True)
                )

                if response.success:
                    st.markdown(response.answer)

                    # Show sources
                    if response.sources:
                        with st.expander("📚 Sources", expanded=False):
                            for src in response.sources:
                                st.markdown(f"**[{src.get('source_number', '?')}]** {src.get('file_name', 'Unknown')}")
                                if src.get('document_type') and src.get('document_type') != 'Unknown':
                                    st.caption(f"Type: {src.get('document_type')}")

                    # Add to display list
                    st.session_state.dialog_messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources
                    })

                    # Show processing info
                    st.caption(f"⏱️ {response.processing_time:.1f}s • {response.context_chunks} chunks")

                else:
                    st.error(f"Failed to generate response: {response.error}")


def render_action_buttons(engine: DocumentDialogEngine, session: DialogSession):
    """Render action buttons for conversation management."""
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🗑️ Clear History", use_container_width=True):
            # Clear messages but keep session
            session.messages.clear()
            st.session_state.dialog_messages = []
            st.rerun()

    with col2:
        if st.button("📥 Export Conversation", use_container_width=True):
            if session.messages:
                export_content = engine.export_conversation(session, format="markdown")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dialog_{session.collection_name}_{timestamp}.md"

                st.download_button(
                    label="💾 Download Markdown",
                    data=export_content,
                    file_name=filename,
                    mime="text/markdown",
                    use_container_width=True
                )
            else:
                st.info("No conversation to export yet.")


def main():
    """Main Document Dialog application."""
    # Header
    st.title("💬 Document Dialog")
    st.caption(f"Version: {PAGE_VERSION} • Conversational Q&A with your document collections")

    # Info section
    with st.expander("ℹ️ About Document Dialog", expanded=False):
        st.markdown("""
        **Have conversations with your document collections** using advanced RAG retrieval.

        **✨ Features:**
        - **Multi-turn Conversations**: Ask follow-up questions with context awareness
        - **Source Citations**: See exactly which documents inform each answer
        - **Collection-based**: Query any of your curated document collections
        - **Neural Reranking**: Optional precision boost using Qwen3-VL reranker
        - **Export**: Save conversations to markdown for sharing or reference

        **🎯 Perfect For:**
        - Exploring research across multiple papers
        - Q&A sessions about project documentation
        - Due diligence on proposal collections
        - Quick answers from technical document sets
        """)

    # Render sidebar and get configuration
    collection_name, model_name, db_path, has_error = render_sidebar()

    if has_error:
        st.warning("👈 Please configure settings in the sidebar to start a conversation.")
        return

    # Initialize engine
    try:
        engine = DocumentDialogEngine(model_name=model_name, db_path=db_path)
    except Exception as e:
        st.error(f"Failed to initialize dialog engine: {e}")
        return

    # Show document list in sidebar
    render_document_list(engine, collection_name)

    # Initialize/update session
    session = initialize_session_if_needed(collection_name, model_name, db_path, engine)

    # Check for empty collection
    if not session.doc_ids:
        st.warning(f"📂 Collection '{collection_name}' is empty. Add documents via Knowledge Search or Knowledge Ingest.")
        return

    # Main chat interface
    st.divider()
    render_chat_interface(engine, session)

    # Action buttons at bottom
    st.divider()
    render_action_buttons(engine, session)


if __name__ == "__main__":
    main()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
