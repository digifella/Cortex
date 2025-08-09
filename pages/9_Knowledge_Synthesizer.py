
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
llm_choice = st.selectbox(
    "AI Provider:",
    ["🌩️ Gemini (Cloud - More Capable)", "🏠 Local Mistral (Fast & Private)"],
    help="Gemini: More powerful but requires internet. Local: Private and fast.",
    key="synthesis_llm_choice"
)

if "Gemini" in llm_choice:
    provider = "gemini"
else:
    provider = "ollama"

# --- Seed Ideas ---

st.subheader("3. Provide Seed Ideas")
seed_ideas = st.text_area(
    "Enter your initial ideas, questions, or themes to guide the synthesis:",
    height=150,
    placeholder="e.g., 'How can we combine our knowledge of AI in healthcare with our project management expertise?' or 'Explore the intersection of our research on renewable energy and smart grid technology.'"
)

# --- Synthesis Button ---

st.divider()
if st.button("✨ Synthesize New Ideas", type="primary", disabled=not selected_collection):
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
