
import streamlit as st
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.knowledge_synthesizer import run_synthesis

st.set_page_config(page_title="Knowledge Synthesizer", layout="wide")

st.title("🧠 Knowledge Synthesizer")

# --- Collection Selection ---

st.subheader("1. Select Knowledge Collection")
collection_manager = WorkingCollectionManager()
collections = collection_manager.get_collection_names()

if not collections:
    st.warning("No collections found. Please create a collection in 'Collection Management' first.")
    st.stop()

selected_collection = st.selectbox("Choose a collection to synthesize from:", collections)

# --- LLM Provider Selection ---

st.subheader("2. Configure AI Model")

# Use standardized LLM provider selector
from cortex_engine.ui_components import llm_provider_selector

try:
    provider_display, llm_status = llm_provider_selector("research", "synthesis", 
                                                        "Choose your AI provider for knowledge synthesis")
    
    # Convert display name to legacy format for compatibility
    if "Gemini" in provider_display:
        provider = "gemini"
    elif "OpenAI" in provider_display:
        provider = "openai"
    else:
        provider = "ollama"
    
    # Show detailed error if LLM service not available
    if llm_status["status"] == "error":
        st.error(f"⚠️ **LLM Service Issue**: {llm_status['message']}")
        if "GEMINI_API_KEY" in llm_status["message"]:
            st.info("💡 **Setup Required**: Add your Gemini API key to the `.env` file")
        provider = None
        
except Exception as e:
    st.error(f"❌ LLM Provider configuration error: {e}")
    provider = None

# --- Seed Ideas ---

st.subheader("3. Provide Seed Ideas")
seed_ideas = st.text_area(
    "Enter your initial ideas, questions, or themes to guide the synthesis:",
    height=150,
    placeholder="e.g., 'How can we combine our knowledge of AI in healthcare with our project management expertise?' or 'Explore the intersection of our research on renewable energy and smart grid technology.'"
)

# --- Synthesis Button ---

st.divider()
# Disable synthesis if no collection or provider unavailable
synthesis_disabled = not selected_collection or provider is None

if provider is None and selected_collection:
    st.warning("⚠️ **Synthesis disabled**: Please configure a working LLM provider above to continue.")

if st.button("✨ Synthesize New Ideas", type="primary", disabled=synthesis_disabled):
    if not seed_ideas:
        st.error("Please provide some seed ideas to start the synthesis process.")
    else:
        with st.spinner("Synthesizing knowledge... This may take a few moments."):
            synthesis_output = run_synthesis(
                collection_name=selected_collection,
                seed_ideas=seed_ideas,
                llm_provider=provider
            )
            st.session_state.synthesis_output = synthesis_output

# --- Output Display ---

if "synthesis_output" in st.session_state:
    st.subheader("Synthesis Results")
    st.markdown(st.session_state.synthesis_output)
