# AI-Assisted Research Page
# Version: v4.8.0
# Multi-agent research and synthesis engine UI

import streamlit as st
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.synthesise import (
    agent_foundational_query_crafter,
    agent_exploratory_query_crafter,
    step1_fetch_foundational_sources,
    step2_fetch_exploratory_sources,
    agent_thematic_analyser,
    step3_go_deeper,
    step4_run_synthesis,
    step5_run_deep_synthesis,
    build_context_from_sources,
    save_outputs_to_custom_dir
)

st.set_page_config(page_title="Cortex AI Research Assistant", layout="wide")

# Page configuration
PAGE_VERSION = "v4.8.0"

# --- Initialize Session State ---
def initialize_session_state():
    """
    Initializes session state keys ONLY if they are not already present.
    This is critical for preserving state across reruns.
    """
    defaults = {
        "research_step": "start",
        "research_topic": "",
        "research_log": [],
        "foundational_queries": {},
        "foundational_sources": [],
        "curated_foundational_sources": [],
        "exploratory_queries": {},
        "exploratory_sources": [],
        "curated_exploratory_sources": [],
        "research_themes": [],
        "research_final_results": None, # Will hold (note_path, map_path)
        "deep_research_result": None, # Will hold path to the deep research note
        "research_output_path": "",
        "fetch_failures": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_research_state():
    """Resets the entire research workflow state to its defaults."""
    defaults = {
        "research_step": "start",
        "research_topic": "",
        "research_log": [],
        "foundational_queries": {},
        "foundational_sources": [],
        "curated_foundational_sources": [],
        "exploratory_queries": {},
        "exploratory_sources": [],
        "curated_exploratory_sources": [],
        "research_themes": [],
        "research_final_results": None,
        "deep_research_result": None,
        "research_output_path": "",
        "fetch_failures": []
    }
    for key, value in defaults.items():
        st.session_state[key] = value
    st.rerun()

initialize_session_state()


# --- UI Layout & Callbacks ---
st.title("🤖 1. AI Assisted Research")
st.caption(f"Page Version: {PAGE_VERSION}")
st.caption("Using a **Double Diamond** approach to find, analyze, and synthesize external knowledge.")

# FIX: Centralized header management to fix UI step numbering bug
STEP_HEADERS = {
    "start": "Step 1: Define Research Topic",
    "foundational_curation": "Step 2: Curate Foundational Sources",
    "exploratory_queries": "Step 3: Refine Exploratory Queries",
    "exploratory_curation": "Step 4: Curate Exploratory Sources",
    "theme_curation": "Step 5: Refine Themes & Synthesize Initial Report",
    "synthesis_complete": "Step 6: Initial Synthesis Complete",
    "deep_research_complete": "✅ Research Complete: Final Report Generated"
}
current_step = st.session_state.research_step
st.header(STEP_HEADERS.get(current_step, "AI Research"))


log_expander = st.expander("Show Live Log", expanded=True)

def execute_and_log(step_function, *args, **kwargs):
    """
    Correctly wraps a backend function call to inject a logging callback
    and display live updates in the UI.
    """
    log_placeholder = log_expander.empty()
    st.session_state.research_log.append(f"--- Running: {step_function.__name__} ---")

    def ui_log_callback(message):
        st.session_state.research_log.append(message)
        log_placeholder.code("\n".join(st.session_state.research_log), language="log")

    kwargs['status_callback'] = ui_log_callback
    result = step_function(*args, **kwargs)

    with log_expander:
        st.code("\n".join(st.session_state.research_log), language="log")
    return result

# --- Reusable UI Components ---
def display_source_list(sources, selection_key, title):
    st.subheader(title)
    st.markdown(f"Found {len(sources)} sources. Select the most relevant ones to include in the final synthesis.")

    if not sources and not st.session_state.fetch_failures:
        st.info("No sources were found in this step.")
        return

    if st.session_state.fetch_failures:
        with st.expander("⚠️ View Search Failures", expanded=False):
            for failure in st.session_state.fetch_failures:
                st.error(failure)

    if not sources:
        return

    # FIX: Only initialize the selection state if it doesn't exist.
    if selection_key not in st.session_state:
        st.session_state[selection_key] = sources.copy()

    with st.container(border=True):
        col1, col2 = st.columns(2)
        if col1.button(f"Select All ({len(sources)})", use_container_width=True, key=f"sel_all_{selection_key}"):
            st.session_state[selection_key] = sources.copy(); st.rerun()
        if col2.button("Deselect All", use_container_width=True, key=f"desel_all_{selection_key}"):
            st.session_state[selection_key] = []; st.rerun()
    st.divider()

    selected_urls = {src['url'] for src in st.session_state[selection_key]}

    def toggle_source_selection(source_obj):
        current_selection = st.session_state.get(selection_key, [])
        if source_obj['url'] in {s['url'] for s in current_selection}:
            st.session_state[selection_key] = [s for s in current_selection if s['url'] != source_obj['url']]
        else:
            current_selection.append(source_obj)
            st.session_state[selection_key] = current_selection

    for i, source in enumerate(sources):
        is_selected = source['url'] in selected_urls
        cite_count = f" (Citations: {source.get('citations', 0)})" if 'citations' in source else ''
        label = f"**[{source['source_type'].upper()}]** {source['title']}{cite_count}"
        st.checkbox(label, value=is_selected, key=f"{selection_key}_{i}", on_change=toggle_source_selection, args=(source,))


# --- Main Workflow Controller ---
if current_step == "start":
    st.markdown("This tool automates research using a two-phase 'Double Diamond' approach. First, it finds foundational, highly-cited papers. Second, it explores the topic more broadly to build thematic understanding.")
    st.divider()
    
    # LLM Provider Choice for Research
    col1, col2 = st.columns([2, 1])
    with col1:
        research_topic = st.text_input(
            "**Enter your research topic:**",
            placeholder="e.g., Best practices for AI project management",
            key="research_topic_input"
        )
    with col2:
        # Use standardized LLM provider selector with error handling
        from cortex_engine.ui_components import llm_provider_selector
        
        try:
            provider_display, llm_status = llm_provider_selector("research", "research_ai", 
                                                                "Choose between cloud power or local privacy for research")
            
            # Convert display name to legacy format for compatibility with synthesise.py
            if "Gemini" in provider_display:
                provider = "gemini"
            elif "OpenAI" in provider_display:
                provider = "openai"
            else:
                provider = "ollama"
            
            # Store provider choice in session state for legacy compatibility
            st.session_state.research_provider = provider
            
            # Show detailed error if LLM service not available
            if llm_status["status"] == "error":
                st.error(f"⚠️ **LLM Service Issue**: {llm_status['message']}")
                if "GEMINI_API_KEY" in llm_status["message"]:
                    st.info("💡 **Setup Required**: Add your Gemini API key to the `.env` file in your project root:")
                    st.code('GEMINI_API_KEY="your_gemini_api_key_here"')
                    st.info("🔗 **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)")
                provider = None  # Disable research when LLM unavailable
                
        except Exception as e:
            st.error(f"❌ LLM Provider configuration error: {e}")
            provider = None
    # Disable button if no research topic or LLM provider unavailable
    research_disabled = not research_topic or provider is None
    
    if provider is None and research_topic:
        st.warning("⚠️ **Research disabled**: Please configure a working LLM provider above to continue.")
    
    if st.button("💎 Start Diamond 1: Find Foundational Sources", disabled=research_disabled, type="primary"):
        st.session_state.research_topic = research_topic
        st.session_state.research_log = [f"--- Starting New Research on '{st.session_state.research_topic}' ---"]
        with st.spinner("AI is generating queries for foundational papers..."):
            st.session_state.foundational_queries = execute_and_log(agent_foundational_query_crafter, topic=st.session_state.research_topic)
        with st.spinner("AI is fetching highly-cited and review papers... This may take a moment."):
            # Call to the (now restored) robust fetcher function
            fetch_results = execute_and_log(
                step1_fetch_foundational_sources,
                queries=st.session_state.foundational_queries
            )
            st.session_state.foundational_sources = fetch_results.get("sources", [])
            st.session_state.fetch_failures = fetch_results.get("failures", [])
        st.session_state.research_step = "foundational_curation"
        st.rerun()

elif current_step == "foundational_curation":
    display_source_list(st.session_state.foundational_sources, 'curated_foundational_sources', "Foundational Papers (Highly-Cited & Reviews)")

    st.divider()
    num_selected = len(st.session_state.get('curated_foundational_sources', []))
    st.info(f"You have selected **{num_selected}** foundational source(s). These will be locked in and included in the final analysis.")
    if st.button("💎 Start Diamond 2: Explore Thematically", type="primary"):
        st.session_state.fetch_failures = [] # Clear failures before next step
        with st.spinner("AI is generating queries for broader, thematic exploration..."):
            st.session_state.exploratory_queries = execute_and_log(agent_exploratory_query_crafter, topic=st.session_state.research_topic)
        st.session_state.research_step = "exploratory_queries"
        st.rerun()

elif current_step == "exploratory_queries":
    st.markdown("The AI has generated broader queries. Edit, add, or delete them before fetching additional sources.")

    edited_queries = {"scholar_queries": [], "youtube_queries": []}
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 Academic Queries")
        queries = st.session_state.exploratory_queries.get("scholar_queries", [""])
        for i, q in enumerate(queries):
            q_input = st.text_input(f"Academic Query {i+1}", value=q, key=f"sq_{i}", label_visibility="collapsed")
            if q_input: edited_queries["scholar_queries"].append(q_input)
        if st.button("➕ Add Academic Query"): st.session_state.exploratory_queries["scholar_queries"].append(""); st.rerun()
    with col2:
        st.subheader("📺 Video Queries")
        queries = st.session_state.exploratory_queries.get("youtube_queries", [""])
        for i, q in enumerate(queries):
            q_input = st.text_input(f"YouTube Query {i+1}", value=q, key=f"yq_{i}", label_visibility="collapsed")
            if q_input: edited_queries["youtube_queries"].append(q_input)
        if st.button("➕ Add Video Query"): st.session_state.exploratory_queries["youtube_queries"].append(""); st.rerun()

    st.session_state.exploratory_queries = edited_queries
    st.divider()
    if st.button("▶️ Fetch Exploratory Sources", type="primary"):
        with st.spinner("AI is fetching additional papers and videos... This may take a moment."):
            fetch_results = execute_and_log(step2_fetch_exploratory_sources, queries=st.session_state.exploratory_queries)
            st.session_state.exploratory_sources = fetch_results.get("sources", [])
            st.session_state.fetch_failures.extend(fetch_results.get("failures", []))
        st.session_state.research_step = "exploratory_curation"
        st.rerun()

elif current_step == "exploratory_curation":
    display_source_list(st.session_state.exploratory_sources, 'curated_exploratory_sources', "Exploratory Sources (Papers & Videos)")

    st.divider()
    num_selected_foundational = len(st.session_state.get('curated_foundational_sources', []))
    num_selected_exploratory = len(st.session_state.get('curated_exploratory_sources', []))
    st.info(f"You have selected **{num_selected_foundational}** foundational and **{num_selected_exploratory}** exploratory sources.")

    if st.button("▶️ Identify Themes from All Sources", type="primary"):
        if not st.session_state.get('curated_foundational_sources') and not st.session_state.get('curated_exploratory_sources'):
            st.error("You must select at least one source to continue."); st.stop()

        with st.spinner("AI is performing thematic analysis on all curated sources..."):
            all_curated_sources = st.session_state.get('curated_foundational_sources', []) + st.session_state.get('curated_exploratory_sources', [])
            context = build_context_from_sources(st.session_state.research_topic, all_curated_sources)
            st.session_state.research_themes = execute_and_log(agent_thematic_analyser, context=context, existing_themes=None)
        st.session_state.research_step = "theme_curation"
        st.rerun()

elif current_step == "theme_curation":
    st.markdown("Review, re-rank, edit, or delete the themes. Use 'Go Deeper' to find more sources for a specific theme, then generate the initial report.")

    if not st.session_state.research_themes or "Could not determine" in st.session_state.research_themes[0]:
        st.error("Could not identify themes. The content may be too diverse or insufficient.")
    else:
        with st.container(border=True):
            for i, theme in enumerate(st.session_state.research_themes):
                col1, col2, col3, col4, col5 = st.columns([0.7, 0.1, 0.05, 0.05, 0.05])
                new_theme = col1.text_input(f"Theme {i+1}", value=theme, key=f"theme_edit_{i}", label_visibility="collapsed")
                if new_theme != theme: st.session_state.research_themes[i] = new_theme; st.rerun()
                if col2.button("🔎 Go Deeper", key=f"deep_{i}"):
                    with st.spinner(f"Searching for more sources on '{theme}'..."):
                        new_sources = execute_and_log(step3_go_deeper, theme_query=theme)
                        if new_sources:
                            existing_urls = {src['url'] for src in st.session_state.exploratory_sources}
                            unique_new = [s for s in new_sources if s['url'] not in existing_urls]
                            if unique_new:
                                st.session_state.exploratory_sources.extend(unique_new)
                                st.toast(f"✅ Added {len(unique_new)} new source(s)! Returning to source curation.")
                                st.session_state.research_step = "exploratory_curation"; st.rerun()
                if col3.button("⬆️", key=f"up_{i}", disabled=(i==0)): st.session_state.research_themes.insert(i-1, st.session_state.research_themes.pop(i)); st.rerun()
                if col4.button("⬇️", key=f"down_{i}", disabled=(i==len(st.session_state.research_themes)-1)): st.session_state.research_themes.insert(i+1, st.session_state.research_themes.pop(i)); st.rerun()
                if col5.button("🗑️", key=f"del_{i}"): st.session_state.research_themes.pop(i); st.rerun()

        if st.button("➕ Add New Theme"): st.session_state.research_themes.append("New theme - please edit"); st.rerun()

    st.divider()
    if st.button("▶️ Generate Initial Report", type="primary"):
        final_themes = [theme for theme in st.session_state.research_themes if theme.strip()]
        all_final_sources = st.session_state.get('curated_foundational_sources', []) + st.session_state.get('curated_exploratory_sources', [])
        with st.spinner("AI is generating the initial Discovery Note and Mind Map..."):
            st.session_state.research_final_results = execute_and_log(
                step4_run_synthesis,
                sources=all_final_sources,
                themes=final_themes,
                topic=st.session_state.research_topic
            )
        st.session_state.research_step = "synthesis_complete"
        st.rerun()

elif current_step == "synthesis_complete":
    st.success("The initial `discovery_note.md` and `mindmap.png` have been generated. You can now ingest this note or proceed to the final deep research step.")

    if st.session_state.research_final_results and st.session_state.research_final_results[0]:
        note_path, map_path = st.session_state.research_final_results
        try:
            with open(note_path, 'r', encoding='utf-8') as f: discovery_note = f.read()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Discovery Note")
                st.markdown(discovery_note, unsafe_allow_html=True)
                st.download_button("Download Note (.md)", discovery_note, f"{Path(note_path).name}")
            if map_path and Path(map_path).exists():
                 with open(map_path, 'rb') as f: map_image = f.read()
                 with col2:
                    st.subheader("🎨 Mind Map")
                    st.image(map_image)
                    st.download_button("Download Mind Map (.png)", map_image, f"{Path(map_path).name}", "image/png")
        except Exception as e:
            st.error(f"Error displaying results: {e}")

        st.divider()
        st.header("🚀 Final Step: Generate Deep Research Report")
        st.info("This optional final step instructs the AI to take all the information gathered so far (including the themes and mind map) and conduct a final, deeper round of analysis and synthesis to produce a comprehensive research report.")
        if st.button("🧠 Generate Deep Research Report", type="primary", use_container_width=True):
            with st.spinner("The Deep Research Agent is now working. This is an intensive process and may take several minutes..."):
                discovery_note_content = ""
                try:
                    with open(st.session_state.research_final_results[0], 'r', encoding='utf-8') as f:
                        discovery_note_content = f.read()
                except Exception:
                    st.error("Could not read initial discovery note to begin deep research.")
                    st.stop()

                final_report_path = execute_and_log(
                    step5_run_deep_synthesis,
                    topic=st.session_state.research_topic,
                    initial_synthesis_note=discovery_note_content
                )
                st.session_state.deep_research_result = final_report_path
            st.session_state.research_step = "deep_research_complete"
            st.rerun()
    else:
        st.error("Synthesis failed to produce output files.")

elif current_step == "deep_research_complete":
    st.balloons()
    st.success("The final, comprehensive research report has been generated and saved to the `external_research` folder.")

    if st.session_state.deep_research_result and Path(st.session_state.deep_research_result).exists():
        report_path = st.session_state.deep_research_result
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                deep_research_note = f.read()

            st.subheader("📜 Deep Research Report")
            st.markdown(deep_research_note, unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Deep Research Report (.md)",
                deep_research_note,
                file_name=Path(report_path).name
            )
        except Exception as e:
            st.error(f"Error displaying final report: {e}")
    else:
        st.error("Could not find or display the generated deep research report.")

if current_step != "start":
    st.divider()
    if st.button("↩️ Start New Research Topic"):
        reset_research_state()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
