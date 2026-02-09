"""
Knowledge Synthesizer - Generate new insights from existing knowledge
Version: v2.0.0
Date: 2026-02-09
Purpose: 2-stage ideation (default) with optional classic 4-stage mode.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Knowledge Synthesizer",
    page_icon="✨",
    layout="wide"
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.knowledge_synthesizer import (
    discover_themes,
    export_ideation_markdown,
    generate_ideas_by_theme,
    run_four_stage_ideation,
)
from cortex_engine.ui_components import (
    collection_selector,
    error_display,
    llm_provider_selector,
    render_version_footer,
)
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Apply theme IMMEDIATELY
apply_theme()

CREDIBILITY_OPTIONS = [
    (0, "0 - Unclassified+"),
    (1, "1 - Commentary+"),
    (2, "2 - Editorial+"),
    (3, "3 - Pre-Print+"),
    (4, "4 - Institutional+"),
    (5, "5 - Peer-Reviewed only"),
]


def _provider_key(provider_display: str) -> str:
    if "Gemini" in provider_display:
        return "gemini"
    if "OpenAI" in provider_display:
        return "openai"
    return "ollama"


def _render_discovered_themes() -> list[str]:
    themes_payload = st.session_state.get("ideation_discovery_payload", {})
    themes = themes_payload.get("themes", []) if isinstance(themes_payload, dict) else []

    if not themes:
        return []

    st.markdown("### Themes")
    default_selected = [t.get("name", "") for t in themes if isinstance(t, dict) and t.get("name")]

    for idx, theme in enumerate(themes, start=1):
        if not isinstance(theme, dict):
            continue
        name = str(theme.get("name", "")).strip()
        desc = str(theme.get("description", "")).strip()
        if not name:
            continue
        with st.container(border=True):
            st.markdown(f"**{idx}. {name}**")
            if desc:
                st.caption(desc)

    selected = st.multiselect(
        "Select themes for idea generation",
        options=default_selected,
        default=default_selected,
        key="ideation_selected_themes",
    )
    return selected


def _render_two_stage(provider: str, selected_collection: str, min_credibility_tier: int) -> None:
    section_header("1️⃣", "Discover Themes", "Find high-value synthesis themes from your KB")

    seed_topic = st.text_input(
        "Seed topic",
        key="ideation_seed_topic",
        placeholder="e.g., AI in healthcare education",
    )

    discover_col1, discover_col2 = st.columns([1, 1])
    with discover_col1:
        if st.button("Discover Themes", type="primary", use_container_width=True):
            if not seed_topic.strip():
                st.error("Please enter a seed topic.")
            else:
                with st.spinner("Discovering themes..."):
                    payload = discover_themes(
                        collection_name=selected_collection,
                        seed_topic=seed_topic.strip(),
                        llm_provider=provider,
                        min_credibility_tier=min_credibility_tier,
                        top_k_chunks=15,
                    )
                st.session_state["ideation_discovery_payload"] = payload
                if payload.get("status") == "success":
                    count = len(payload.get("themes", []))
                    st.success(f"Discovered {count} themes.")
                else:
                    st.error(payload.get("error", "Theme discovery failed."))

    with discover_col2:
        if st.button("Clear Discovery", use_container_width=True):
            st.session_state.pop("ideation_discovery_payload", None)
            st.session_state.pop("ideation_generate_payload", None)
            st.rerun()

    st.markdown("---")
    section_header("2️⃣", "Generate Ideas", "Generate ideas from selected themes")

    selected_themes = _render_discovered_themes()

    innovation_goals = st.text_area(
        "Innovation goals (optional)",
        key="ideation_innovation_goals",
        placeholder="e.g., focus on practical applications and contrarian opportunities",
        height=110,
    )

    if st.button("Generate Ideas", type="primary", use_container_width=True):
        if not selected_themes:
            st.error("Select at least one theme.")
        else:
            with st.spinner("Generating ideas..."):
                payload = generate_ideas_by_theme(
                    collection_name=selected_collection,
                    themes=selected_themes,
                    innovation_goals=innovation_goals.strip(),
                    llm_provider=provider,
                    min_credibility_tier=min_credibility_tier,
                    chunks_per_theme=3,
                )
            st.session_state["ideation_generate_payload"] = payload
            if payload.get("status") == "success":
                st.success("Idea generation complete.")
            else:
                st.error(payload.get("error", "Idea generation failed."))

    payload = st.session_state.get("ideation_generate_payload")
    if isinstance(payload, dict) and payload.get("status") == "success":
        st.markdown("---")
        section_header("📊", "Results", "Theme-grouped ideas")

        theme_groups = payload.get("theme_groups", [])
        for group in theme_groups:
            theme = group.get("theme", "Theme")
            with st.container(border=True):
                st.markdown(f"### {theme}")
                for idea in group.get("ideas", []):
                    st.markdown(f"**{idea.get('title', 'Idea')}**")
                    desc = idea.get("description", "")
                    if desc:
                        st.write(desc)
                    impact = idea.get("impact", "")
                    if impact:
                        st.caption(f"Impact: {impact}")

        md = export_ideation_markdown(payload, title="Knowledge Synthesizer (2-Stage)")
        st.download_button(
            "📥 Export as Markdown",
            data=md,
            file_name="knowledge_synthesizer_2stage.md",
            mime="text/markdown",
            use_container_width=True,
        )


def _render_four_stage(provider: str, selected_collection: str) -> None:
    section_header("🧭", "Classic 4-Stage", "Optional deeper ideation workflow")

    seed_topic = st.text_input(
        "Seed topic",
        key="classic_seed_topic",
        placeholder="e.g., AI in healthcare education",
    )
    innovation_goals = st.text_area(
        "Innovation goals",
        key="classic_innovation_goals",
        placeholder="Optional constraints, success criteria, or strategic intent",
        height=110,
    )
    preselected_themes = st.text_area(
        "Optional preselected themes (one per line)",
        key="classic_preselected_themes",
        placeholder="Theme A\nTheme B\nTheme C",
        height=100,
    )

    if st.button("Run 4-Stage Workflow", type="primary", use_container_width=True):
        if not seed_topic.strip():
            st.error("Please enter a seed topic.")
        else:
            selected_themes = [t.strip() for t in preselected_themes.splitlines() if t.strip()]
            with st.spinner("Running 4-stage ideation..."):
                payload = run_four_stage_ideation(
                    collection_name=selected_collection,
                    seed_topic=seed_topic.strip(),
                    innovation_goals=innovation_goals.strip(),
                    llm_provider=provider,
                    selected_themes=selected_themes or None,
                )
            st.session_state["classic_ideation_payload"] = payload

    payload = st.session_state.get("classic_ideation_payload")
    if isinstance(payload, dict):
        if payload.get("status") != "success":
            st.error(payload.get("error", "4-stage ideation failed."))
            return

        st.markdown("---")
        section_header("📊", "4-Stage Results", "Discovery, define, develop, deliver")

        st.markdown("### Discovery")
        for theme in payload.get("discovery", {}).get("themes", []):
            st.write(f"- {theme}")

        st.markdown("### Develop")
        for group in payload.get("develop", {}).get("idea_groups", []):
            with st.container(border=True):
                st.markdown(f"**Problem:** {group.get('problem_statement', 'N/A')}")
                for idea in group.get("ideas", []):
                    st.write(f"- {idea.get('title', 'Idea')}: {idea.get('description', '')}")

        st.markdown("### Deliver")
        deliver = payload.get("deliver", {})
        st.write(deliver.get("summary", ""))
        if deliver.get("recommendation"):
            st.caption(deliver.get("recommendation"))

        md = export_ideation_markdown(payload, title="Knowledge Synthesizer (4-Stage)")
        st.download_button(
            "📥 Export as Markdown",
            data=md,
            file_name="knowledge_synthesizer_4stage.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ============================================
# PAGE HEADER
# ============================================

st.title("✨ Knowledge Synthesizer")
st.markdown(
    """
**Generate innovative ideas grounded in your knowledge base.**
Default mode is fast 2-stage ideation (Discover Themes → Generate Ideas), with optional classic 4-stage workflow.
"""
)
st.caption("KB-only mode. Uses collection and credibility filters.")
st.markdown("---")

# ============================================
# COLLECTION + MODEL CONFIG
# ============================================

section_header("📚", "Select Knowledge Collection", "Choose the source collection")

try:
    collection_manager = WorkingCollectionManager()
    selected_collection = collection_selector(collection_manager, key_prefix="synthesizer", required=True)
    if not selected_collection:
        st.info("No collections found. Create one in Collection Management first.")
        st.stop()
except Exception as e:
    error_display(
        str(e),
        error_type="Collection Loading Error",
        recovery_suggestion="Check that the database path is configured correctly",
    )
    logger.error(f"Failed to load collections: {e}", exc_info=True)
    st.stop()

section_header("🤖", "Configure AI Model", "Choose provider and workflow")

provider = None
try:
    provider_display, llm_status = llm_provider_selector(task_type="research", key_prefix="synthesizer")
    provider = _provider_key(provider_display)
    if llm_status.get("status") == "error":
        error_display(
            llm_status.get("message", "LLM service not available"),
            error_type="LLM Service Issue",
            recovery_suggestion="Configure API keys or ensure your local model service is running",
        )
        provider = None
except Exception as e:
    error_display(
        str(e),
        error_type="LLM Configuration Error",
        recovery_suggestion="Check provider setup",
    )
    logger.error(f"LLM provider configuration failed: {e}", exc_info=True)

workflow_mode = st.radio(
    "Workflow mode",
    options=["2-Stage (Default)", "4-Stage (Classic)"],
    horizontal=True,
    key="ideation_workflow_mode",
)

credibility_labels = [label for _, label in CREDIBILITY_OPTIONS]
selected_credibility_label = st.selectbox(
    "Minimum credibility tier",
    options=credibility_labels,
    index=0,
    help="Only documents at or above this credibility tier are used in ideation context.",
)
min_credibility_tier = next(v for v, lbl in CREDIBILITY_OPTIONS if lbl == selected_credibility_label)

st.divider()

if provider is None:
    st.warning("Synthesis disabled until a working LLM provider is available.")
else:
    if workflow_mode == "2-Stage (Default)":
        _render_two_stage(provider, selected_collection, min_credibility_tier)
    else:
        _render_four_stage(provider, selected_collection)

# footer
render_version_footer()
