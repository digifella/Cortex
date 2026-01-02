"""
Universal Knowledge Assistant - Streamlined Interface for Knowledge Work
Version: 2.0.0
Date: 2026-01-02

Purpose: Single unified interface replacing AI Assisted Research and Knowledge Synthesizer.
Combines internal RAG + external sources with real-time streaming synthesis.
Now with Mixture of Experts (MoE) for superior analysis quality.
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
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from cortex_engine.config_manager import ConfigManager
        from cortex_engine.config import COLLECTION_NAME, EMBED_MODEL
        from cortex_engine.utils import convert_to_docker_mount_path
        from cortex_engine.embedding_adapters import EmbeddingServiceAdapter
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        import os

        # Get database path from configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        db_path = config.get('ai_database_path', '/mnt/f/ai_databases')

        # Convert to proper path
        safe_db_path = convert_to_docker_mount_path(db_path)
        chroma_db_path = os.path.join(safe_db_path, "knowledge_hub_db")

        # Set up embeddings
        try:
            Settings.embed_model = EmbeddingServiceAdapter(model_name=EMBED_MODEL)
        except Exception:
            Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cuda")

        # Connect to ChromaDB
        db_settings = ChromaSettings(anonymized_telemetry=False)
        db = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=chroma_collection)
        )

        return index

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error loading RAG index: {e}")
        logger.error(f"Full traceback: {error_details}")
        return None


async def process_query_async(
    user_input: str,
    rag_index,
    sources: SourceConfig,
    depth: DepthLevel,
    creativity: float = 1.0,
    manual_model: str = None,
    use_moe: bool = False,
    collection_name: str = None
):
    """Process user query with streaming updates."""

    assistant = UniversalKnowledgeAssistant(
        rag_index=rag_index,
        collection_name=collection_name
    )

    # Map creativity to temperature
    # 0.0 -> 0.2 (very factual)
    # 1.0 -> 0.7 (balanced)
    # 2.0 -> 1.2 (creative)
    # 3.0 -> 2.0 (maximum creativity)
    temperature = 0.2 + (creativity * 0.6)

    status_placeholder = st.empty()
    output_placeholder = st.empty()
    model_display = st.empty()

    full_response = []
    current_status = ""
    selected_model = None

    try:
        async for chunk in assistant.process_query(
            user_input=user_input,
            sources=sources,
            depth=depth,
            temperature=temperature,
            force_model=manual_model,
            use_moe=use_moe
        ):
            # Show selected model on first chunk
            if selected_model is None and chunk.metadata and "model" in chunk.metadata:
                selected_model = chunk.metadata["model"]
                reasoning = chunk.metadata.get("reasoning", "")

                if use_moe and "expert_models" in chunk.metadata:
                    # MoE mode - show all expert models
                    expert_models = chunk.metadata["expert_models"]
                    model_info = f"🎯 **MoE Mode Active**\n\n"
                    model_info += f"**Expert Models:** {', '.join(expert_models)}\n\n"
                    model_info += f"**Meta-Synthesis Model:** {selected_model}\n\n"
                    model_info += f"**Temperature:** {temperature:.2f}"
                else:
                    # Single model mode
                    model_info = f"🤖 **Model:** {selected_model} | **Temperature:** {temperature:.2f}"
                    if reasoning:
                        model_info += f"\n\n**Why this model?** {reasoning}"

                model_display.info(model_info)

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

    # Creativity slider
    st.subheader("🎨 Creativity Level")

    creativity_level = st.slider(
        "Adjust creativity/temperature",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.5,
        help="0 = Factual & Conservative | 1 = Balanced | 2 = Creative | 3 = Highly Experimental"
    )

    creativity_labels = {
        0.0: "📊 Factual (Conservative)",
        0.5: "📈 Mostly Factual",
        1.0: "⚖️ Balanced",
        1.5: "💡 Creative",
        2.0: "🎨 Highly Creative",
        2.5: "🚀 Very Experimental",
        3.0: "🌟 Maximum Creativity"
    }
    st.caption(creativity_labels.get(creativity_level, "⚖️ Balanced"))

    st.divider()

    # MoE (Mixture of Experts) option
    st.subheader("🎯 Quality Enhancement")

    use_moe = st.checkbox(
        "Use Multiple Models (MoE)",
        value=False,
        help="Run 2-3 expert models in parallel and combine their outputs for higher quality (2-3x slower)"
    )

    if use_moe:
        st.info("💡 **MoE Mode:** Will run multiple models and synthesize the best insights from each")

    st.divider()

    # Model selection
    st.subheader("🤖 Model Selection")

    # Auto-select model
    use_auto_model = st.checkbox(
        "Auto-select model",
        value=True,
        help="Let the system choose the best model based on your query"
    )

    if not use_auto_model:
        # Get available models
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                available_models = [line.split()[0] for line in lines if line.strip()]

                # Filter for power models (70B+)
                power_models = [m for m in available_models if any(x in m.lower() for x in ['70b', '72b'])]

                if power_models:
                    manual_model = st.selectbox(
                        "Override model",
                        options=power_models,
                        help="Manually select which model to use for synthesis"
                    )
                else:
                    st.warning("No 70B+ models found")
                    manual_model = None
            else:
                st.error("Could not list Ollama models")
                manual_model = None
        except Exception as e:
            st.error(f"Error listing models: {e}")
            manual_model = None
    else:
        manual_model = None

    st.divider()

    # Model info
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Router Model:** llama3.2:3b or qwen2.5:3b (intent classification)
        **Power Models:** qwen2.5:72b, llama3.3:70b, nemotron:70b (synthesis)

        Models are automatically selected based on task requirements and GPU capabilities.
        You can override auto-selection above.
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
                        creativity=creativity_level,
                        manual_model=manual_model if not use_auto_model else None,
                        use_moe=use_moe,
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
