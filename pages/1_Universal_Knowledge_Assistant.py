"""
Universal Knowledge Assistant - Streamlined Interface for Knowledge Work
Version: 1.0.0
Date: 2026-01-01

Purpose: Single unified interface replacing AI Assisted Research and Knowledge Synthesizer.
Combines internal RAG + external sources with real-time streaming synthesis.
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path

# Set page config FIRST
st.set_page_config(
    page_title="Universal Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import theme and components
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.ui_components import (
    collection_selector,
    error_display,
    render_version_footer
)
from cortex_engine.universal_assistant import (
    UniversalKnowledgeAssistant,
    SourceConfig,
    DepthLevel,
    IntentType
)
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils.logging_utils import get_logger
from llama_index.core import VectorStoreIndex

logger = get_logger(__name__)

# Apply theme IMMEDIATELY
apply_theme()

# ============================================
# PAGE HEADER
# ============================================

st.title("🧠 Universal Knowledge Assistant")
st.markdown("""
**One interface for all knowledge work** - Research, synthesis, ideation, and exploration.
Ask anything, and I'll intelligently search your knowledge base and academic papers, then synthesize insights in real-time.
""")

st.caption("💡 Use the sidebar (←) to navigate between pages")
st.markdown("---")

# ============================================
# SESSION STATE
# ============================================

if 'assistant_history' not in st.session_state:
    st.session_state.assistant_history = []


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_rag_index(collection_name: str = None):
    """Get RAG index for the specified collection."""
    try:
        collection_manager = WorkingCollectionManager()

        if collection_name:
            index = collection_manager.load_index(collection_name)
        else:
            collections = collection_manager.list_collections()
            if collections:
                index = collection_manager.load_index(collections[0])
            else:
                return None

        return index
    except Exception as e:
        logger.error(f"Error loading RAG index: {e}")
        return None


async def process_query_async(
    user_input: str,
    rag_index,
    sources: SourceConfig,
    depth: DepthLevel,
    collection_name: str = None
):
    """Process user query with streaming updates."""

    assistant = UniversalKnowledgeAssistant(
        rag_index=rag_index,
        collection_name=collection_name
    )

    status_placeholder = st.empty()
    output_placeholder = st.empty()

    full_response = []
    current_status = ""

    try:
        async for chunk in assistant.process_query(
            user_input=user_input,
            sources=sources,
            depth=depth
        ):
            if chunk.metadata and "status" in chunk.metadata:
                current_status = chunk.content
                status_placeholder.info(current_status)
            else:
                full_response.append(chunk.content)
                output_placeholder.markdown("".join(full_response))

        status_placeholder.success("✅ Complete!")

        # Save to history
        st.session_state.assistant_history.append({
            "query": user_input,
            "response": "".join(full_response),
            "depth": depth.value,
            "sources": {
                "internal_rag": sources.internal_rag,
                "internal_graph": sources.internal_graph,
                "external_papers": sources.external_papers
            }
        })

        return "".join(full_response)

    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        logger.error(error_msg)
        error_display(
            error_msg,
            error_type="Processing Error",
            recovery_suggestion="Try simplifying your query or selecting different sources"
        )
        return None


# ============================================
# SIDEBAR CONFIGURATION
# ============================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # Collection selection
    st.subheader("Knowledge Base")
    try:
        collection_manager = WorkingCollectionManager()
        collections = collection_manager.list_collections()

        if collections:
            selected_collection = st.selectbox(
                "Working Collection",
                options=["All Collections"] + collections,
                help="Select a specific collection to search, or 'All Collections' for global search"
            )

            if selected_collection == "All Collections":
                selected_collection = None
                st.info("Searching across all collections")
            else:
                st.success(f"Searching in: **{selected_collection}**")
        else:
            st.warning("No collections found. Ingest documents first.")
            selected_collection = None
    except Exception as e:
        error_display(str(e), "Collection Loading Error")
        selected_collection = None

    st.divider()

    # Knowledge sources
    st.subheader("Knowledge Sources")

    source_internal_rag = st.checkbox(
        "Internal Knowledge Base",
        value=True,
        help="Search your ingested documents via RAG"
    )

    source_internal_graph = st.checkbox(
        "Knowledge Graph",
        value=True,
        help="Search entities and relationships"
    )

    source_external_papers = st.checkbox(
        "Academic Papers",
        value=False,
        help="Search Semantic Scholar for academic research"
    )

    st.divider()

    # Depth level
    st.subheader("Analysis Depth")

    depth_option = st.radio(
        "Select depth level",
        options=["Quick", "Thorough", "Deep"],
        index=1,  # Default to Thorough
        help="Quick (1-2 min) | Thorough (3-5 min) | Deep (10+ min)"
    )

    depth_map = {
        "Quick": DepthLevel.QUICK,
        "Thorough": DepthLevel.THOROUGH,
        "Deep": DepthLevel.DEEP
    }
    selected_depth = depth_map[depth_option]

    # Depth descriptions
    depth_descriptions = {
        "Quick": "⚡ Fast overview - Top 5 results, 2-3 paragraphs",
        "Thorough": "⚖️ Balanced analysis - Top 10 results, 4-6 paragraphs",
        "Deep": "🔬 Comprehensive research - Top 20 results, extensive analysis"
    }
    st.caption(depth_descriptions[depth_option])

    st.divider()

    # Model info
    with st.expander("🤖 Model Information"):
        st.markdown("""
        **Router Model:** llama3.2:3b (intent classification)
        **Power Model:** qwen2.5:72b or llama3.3:70b (synthesis)

        Models are automatically selected based on task requirements and your GPU capabilities.
        """)


# ============================================
# MAIN CONTENT
# ============================================

section_header("🔍", "Knowledge Query", "Enter your question, topic, or goal")

# Query input
user_query = st.text_area(
    "What would you like to explore?",
    placeholder="Examples:\n- How does GraphRAG improve retrieval quality?\n- Synthesize key insights about knowledge management from my documents\n- Generate innovative ideas for improving our RAG pipeline",
    height=120,
    help="Enter your question, topic, or goal. The assistant will automatically determine the best approach.",
    label_visibility="collapsed"
)

# Action buttons
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    generate_button = st.button(
        "🚀 Generate Knowledge",
        type="primary",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "🗑️ Clear History",
        use_container_width=True
    )

if clear_button:
    st.session_state.assistant_history = []
    st.rerun()

# Process query
if generate_button and user_query:
    st.markdown("---")
    section_header("📊", "Results", f"Query: {user_query[:50]}...")

    # Create source config
    sources = SourceConfig(
        internal_rag=source_internal_rag,
        internal_graph=source_internal_graph,
        external_papers=source_external_papers
    )

    # Validate at least one source is selected
    if not any([sources.internal_rag, sources.internal_graph, sources.external_papers]):
        st.error("⚠️ Please select at least one knowledge source!")
    else:
        # Load RAG index
        with st.spinner("Loading knowledge base..."):
            rag_index = get_rag_index(
                selected_collection if selected_collection != "All Collections" else None
            )

            if not rag_index and source_internal_rag:
                st.warning("Could not load RAG index. Internal search disabled.")
                sources.internal_rag = False

        # Process query
        with st.spinner("Processing your query..."):
            try:
                response = asyncio.run(
                    process_query_async(
                        user_input=user_query,
                        rag_index=rag_index,
                        sources=sources,
                        depth=selected_depth,
                        collection_name=selected_collection if selected_collection != "All Collections" else None
                    )
                )

                if response:
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Response (Markdown)",
                        data=response,
                        file_name=f"knowledge_assistant_{user_query[:30].replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=False
                    )

            except Exception as e:
                error_display(
                    str(e),
                    error_type="Query Processing Error",
                    recovery_suggestion="Try simplifying your query or check your connection",
                    show_details=True
                )
                logger.error(f"Error in process_query_async: {e}", exc_info=True)

elif generate_button:
    st.warning("⚠️ Please enter a query first!")


# ============================================
# QUERY HISTORY
# ============================================

if st.session_state.assistant_history:
    st.markdown("---")
    section_header("📚", "Query History", f"{len(st.session_state.assistant_history)} previous queries")

    with st.expander("View Previous Queries", expanded=False):
        for i, item in enumerate(reversed(st.session_state.assistant_history), 1):
            st.markdown(f"**{i}. {item['query']}**")
            st.caption(f"Depth: {item['depth']} | Sources: {', '.join([k for k, v in item['sources'].items() if v])}")

            with st.container():
                preview = item['response'][:500] + "..." if len(item['response']) > 500 else item['response']
                st.markdown(preview)

            st.divider()


# ============================================
# FOOTER
# ============================================

render_version_footer()
