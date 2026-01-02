"""
Knowledge Synthesizer - Generate new insights from existing knowledge
Version: v1.0.0
Date: 2026-01-01
Purpose: Synthesize novel ideas by combining concepts from knowledge collections
"""

import streamlit as st
import sys
from pathlib import Path

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Knowledge Synthesizer",
    page_icon="✨",
    layout="wide"
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import theme and components
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.ui_components import (
    llm_provider_selector,
    collection_selector,
    error_display,
    render_version_footer
)
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.knowledge_synthesizer import run_synthesis
from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Apply theme IMMEDIATELY
apply_theme()

# ============================================
# PAGE HEADER
# ============================================

st.title("✨ Knowledge Synthesizer")
st.markdown("""
**Generate novel insights** by combining concepts from your knowledge collections.
The synthesizer helps you discover connections, patterns, and opportunities across your knowledge base.
""")

st.caption("💡 Use the sidebar (←) to navigate between pages")
st.markdown("---")

# ============================================
# COLLECTION SELECTION
# ============================================

section_header("📚", "Select Knowledge Collection", "Choose the collection to synthesize from")

try:
    collection_manager = WorkingCollectionManager()
    selected_collection = collection_selector(
        collection_manager,
        key_prefix="synthesizer",
        required=True
    )

    if not selected_collection:
        st.info("📋 No collections found. Please create a collection in 'Collection Management' first.")
        st.stop()

except Exception as e:
    error_display(
        str(e),
        error_type="Collection Loading Error",
        recovery_suggestion="Check that the database path is configured correctly"
    )
    logger.error(f"Failed to load collections: {e}", exc_info=True)
    st.stop()

# ============================================
# LLM PROVIDER CONFIGURATION
# ============================================

section_header("🤖", "Configure AI Model", "Choose your AI provider for synthesis")

try:
    provider_display, llm_status = llm_provider_selector(
        task_type="research",
        key_prefix="synthesizer"
    )

    # Convert display name to legacy format for compatibility
    if "Gemini" in provider_display:
        provider = "gemini"
    elif "OpenAI" in provider_display:
        provider = "openai"
    else:
        provider = "ollama"

    # Show detailed error if LLM service not available
    if llm_status["status"] == "error":
        error_display(
            llm_status["message"],
            error_type="LLM Service Issue",
            recovery_suggestion="Configure your API key in .env file or ensure Ollama is running"
        )
        provider = None

except Exception as e:
    error_display(
        str(e),
        error_type="LLM Configuration Error",
        recovery_suggestion="Check your LLM provider settings"
    )
    logger.error(f"LLM provider configuration failed: {e}", exc_info=True)
    provider = None

# ============================================
# SEED IDEAS INPUT
# ============================================

section_header("💡", "Provide Seed Ideas", "Guide the synthesis with your initial thoughts")

seed_ideas = st.text_area(
    "Enter your initial ideas, questions, or themes:",
    height=150,
    placeholder="e.g., 'How can we combine our knowledge of AI in healthcare with our project management expertise?' or 'Explore the intersection of our research on renewable energy and smart grid technology.'",
    help="Provide context and direction for the AI to synthesize relevant knowledge connections"
)

# ============================================
# SYNTHESIS EXECUTION
# ============================================

st.divider()

# Disable synthesis if no collection or provider unavailable
synthesis_disabled = not selected_collection or provider is None

if provider is None and selected_collection:
    st.warning("⚠️ **Synthesis disabled**: Please configure a working LLM provider above to continue.")

if st.button("✨ Synthesize New Ideas", type="primary", disabled=synthesis_disabled, use_container_width=True):
    if not seed_ideas:
        st.error("❌ Please provide some seed ideas to start the synthesis process.")
    else:
        try:
            with st.spinner("🔄 Synthesizing knowledge... This may take a few moments."):
                synthesis_output = run_synthesis(
                    collection_name=selected_collection,
                    seed_ideas=seed_ideas,
                    llm_provider=provider
                )
                st.session_state.synthesis_output = synthesis_output
                logger.info(f"Synthesis completed for collection: {selected_collection}")
                st.success("✅ Synthesis complete! View results below.")

        except Exception as e:
            error_display(
                str(e),
                error_type="Synthesis Error",
                recovery_suggestion="Try simplifying your seed ideas or check your collection data"
            )
            logger.error(f"Synthesis failed: {e}", exc_info=True)

# ============================================
# OUTPUT DISPLAY
# ============================================

if "synthesis_output" in st.session_state:
    section_header("📊", "Synthesis Results", "Novel insights and connections")

    st.markdown(st.session_state.synthesis_output)

    # Export options
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            "📥 Download as Markdown",
            data=st.session_state.synthesis_output,
            file_name=f"synthesis_{selected_collection}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            del st.session_state.synthesis_output
            st.rerun()

# ============================================
# FOOTER
# ============================================

render_version_footer()
