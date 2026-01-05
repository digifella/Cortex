"""
Proposal Copilot v2.0 - Proof of Concept
Version: 2.0.0 (POC)
Date: 2026-01-02

Purpose: Demonstrate flexible, hint-based MoE assistance for tender documents.
NO rigid [INSTRUCTION] format required - works with ANY document structure.

Key Features:
- Auto-detects sections (headings, questions, blanks)
- User provides hints in natural language
- MoE assistance available anywhere
- Flexible approach selection
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
import docx
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Proposal Copilot v2.0 POC",
    page_icon="🚀",
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import new flexible system
from cortex_engine.proposals import (
    FlexibleTemplateParser,
    FlexibleSection,
    HintBasedAssistant,
    AssistanceRequest,
    AssistanceMode,
    ContentStatus
)
from cortex_engine.adaptive_model_manager import AdaptiveModelManager
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.collection_manager import WorkingCollectionManager

# Apply theme
apply_theme()

# ============================================
# TITLE & INTRO
# ============================================

st.title("🚀 Proposal Copilot v2.0 - Proof of Concept")

st.markdown("""
**NEW Flexible System** - No rigid `[INSTRUCTION]` tags required!

Upload ANY tender document and:
- ✅ Auto-detects sections (headings, questions, tables)
- ✅ Identifies what needs work (blanks, placeholders)
- ✅ You provide hints in **natural language**
- ✅ MoE assistance available for complex sections
- ✅ Works with ANY document structure
""")

st.info("💡 **Key Difference:** You don't need to manually insert instruction tags. Just upload your tender, and the system will understand it.")

st.markdown("---")

# ============================================
# SESSION STATE
# ============================================

if 'parsed_sections' not in st.session_state:
    st.session_state.parsed_sections = []

if 'section_content' not in st.session_state:
    st.session_state.section_content = {}

if 'uploaded_doc' not in st.session_state:
    st.session_state.uploaded_doc = None

# ============================================
# STEP 1: UPLOAD DOCUMENT
# ============================================

section_header("📄", "Step 1: Upload Tender Document", "Any .docx format - no special tags required")

uploaded_file = st.file_uploader(
    "Upload your tender document (.docx)",
    type=['docx'],
    help="The system will automatically detect sections, questions, and areas needing work"
)

if uploaded_file:
    if st.button("🔍 Parse Document (Auto-Detect Sections)", type="primary"):
        with st.spinner("Analyzing document structure..."):
            # Load document
            doc = docx.Document(BytesIO(uploaded_file.read()))

            # Parse with flexible parser
            parser = FlexibleTemplateParser()
            sections = parser.parse_document(doc)

            st.session_state.parsed_sections = sections
            st.session_state.uploaded_doc = uploaded_file.name

            st.success(f"✅ Detected {len(sections)} sections")
            st.info(f"📊 Sections needing work: {sum(1 for s in sections if s.needs_work)}")

# ============================================
# STEP 2: REVIEW DETECTED SECTIONS
# ============================================

if st.session_state.parsed_sections:
    st.markdown("---")
    section_header("📋", "Step 2: Review Auto-Detected Sections", f"{len(st.session_state.parsed_sections)} sections found")

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        show_filter = st.selectbox(
            "Show:",
            options=["All Sections", "Needs Work Only", "Complete Only"],
            index=1
        )

    with col2:
        complexity_filter = st.selectbox(
            "Complexity:",
            options=["All", "Simple", "Moderate", "Complex"],
            index=0
        )

    # Filter sections
    filtered_sections = st.session_state.parsed_sections

    if show_filter == "Needs Work Only":
        filtered_sections = [s for s in filtered_sections if s.needs_work]
    elif show_filter == "Complete Only":
        filtered_sections = [s for s in filtered_sections if not s.needs_work]

    if complexity_filter != "All":
        filtered_sections = [s for s in filtered_sections if s.complexity.lower() == complexity_filter.lower()]

    st.info(f"Showing {len(filtered_sections)} sections")

    # Display sections in expandable cards
    for i, section in enumerate(filtered_sections):
        # Status indicator
        if section.needs_work:
            status_emoji = "⚠️"
            status_color = "orange"
        else:
            status_emoji = "✅"
            status_color = "green"

        # Complexity indicator
        complexity_emoji = {
            "simple": "🟢",
            "moderate": "🟡",
            "complex": "🔴"
        }.get(section.complexity, "⚪")

        with st.expander(
            f"{status_emoji} {section.heading} | {complexity_emoji} {section.complexity.title()} | {section.section_type.value}",
            expanded=i < 3  # Auto-expand first 3
        ):
            # Section info
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Section ID:** `{section.section_id}`")
                st.markdown(f"**Type:** {section.section_type.value}")
                if section.numbering:
                    st.markdown(f"**Numbering:** {section.numbering}")
                if section.parent_heading:
                    st.markdown(f"**Parent:** {section.parent_heading}")

            with col2:
                st.markdown(f"**Status:** {section.status.value}")
                st.markdown(f"**Complexity:** {section.complexity}")
                st.markdown(f"**Needs Work:** {'Yes' if section.needs_work else 'No'}")

            # Current content
            st.markdown("### Current Content")
            if section.content:
                st.text_area(
                    "Content",
                    value=section.content,
                    height=100,
                    key=f"content_display_{section.section_id}",
                    label_visibility="collapsed"
                )
            else:
                st.warning("⚠️ No content - section is empty")

            # AI suggestion
            if section.suggested_approach:
                st.info(f"💡 **Suggested Approach:** {section.suggested_approach}")

            # User can provide hint and request assistance
            if section.needs_work:
                st.markdown("---")
                st.markdown("### 🤖 AI Assistance")

                # User hint
                user_hint = st.text_area(
                    "How should AI help with this section?",
                    placeholder="Example: Answer this question using our technical capabilities from the knowledge base...",
                    height=80,
                    key=f"hint_{section.section_id}"
                )

                # Options
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    mode = st.selectbox(
                        "Assistance Mode:",
                        options=[
                            AssistanceMode.GENERATE_NEW.value,
                            AssistanceMode.ANSWER_QUESTION.value,
                            AssistanceMode.REFINE_EXISTING.value,
                            AssistanceMode.BRAINSTORM.value,
                            AssistanceMode.EXPAND_BRIEF.value,
                            AssistanceMode.REWRITE_PROFESSIONAL.value,
                            AssistanceMode.ADD_EVIDENCE.value
                        ],
                        key=f"mode_{section.section_id}"
                    )

                with col_b:
                    use_moe = st.checkbox(
                        "Use MoE (Multiple Experts)",
                        value=section.complexity == "complex",
                        help="Use 2-3 expert models and synthesize their outputs",
                        key=f"moe_{section.section_id}"
                    )

                with col_c:
                    creativity = st.slider(
                        "Creativity:",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.7,
                        step=0.1,
                        key=f"creativity_{section.section_id}"
                    )

                # Generate button
                if st.button(
                    f"🚀 Generate Content" + (" with MoE" if use_moe else ""),
                    type="primary",
                    key=f"generate_{section.section_id}"
                ):
                    if not user_hint:
                        st.warning("Please provide a hint about how AI should help!")
                    else:
                        # This would be the actual generation
                        st.info("🔄 **DEMO MODE:** In production, this would:")
                        st.markdown(f"""
                        1. Analyze your hint: "{user_hint}"
                        2. Select optimal model(s): {use_moe and '2-3 expert models' or '1 model'}
                        3. Retrieve relevant KB content
                        4. Generate using mode: {mode}
                        5. {use_moe and 'Synthesize expert outputs' or 'Return result'}
                        """)

                        st.success("✅ Generation complete! (demo)")

                        # Placeholder result
                        st.markdown("### Generated Content (Demo)")
                        st.info(f"""
This is a demo placeholder. In production, you would see the actual AI-generated content here based on:
- Your hint: {user_hint}
- Mode: {mode}
- MoE: {use_moe}
- Creativity: {creativity}
                        """)

# ============================================
# STEP 3: BATCH GENERATION (Future)
# ============================================

if st.session_state.parsed_sections:
    st.markdown("---")
    section_header("⚡", "Step 3: Batch Generation (Coming Soon)", "Generate all sections at once")

    needs_work_count = sum(1 for s in st.session_state.parsed_sections if s.needs_work)

    if needs_work_count > 0:
        st.info(f"📊 {needs_work_count} sections need work and could be generated in parallel")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Generate All Sections (Demo)", disabled=True):
                st.info("This will generate all sections in parallel using optimal models")

        with col2:
            if st.button("⚙️ Configure Batch Settings", disabled=True):
                st.info("This will let you customize generation for each section")
    else:
        st.success("✅ All sections appear complete!")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
### 🎯 Next Steps for Full Implementation

This POC demonstrates the **flexible parsing and hint-based assistance**. Full implementation will include:

1. ✅ Flexible template parsing (DONE - this POC)
2. ✅ Auto-section detection (DONE - this POC)
3. ✅ Hint-based assistance design (DONE - this POC)
4. ⏳ Actual MoE integration (connect to AdaptiveModelManager)
5. ⏳ Knowledge base context building
6. ⏳ Parallel batch generation
7. ⏳ Quality validation & metrics
8. ⏳ Document assembly with generated content

**Test this POC** with your real tender documents to validate the flexible parsing works for your use cases!
""")

st.info("💡 **Feedback:** Does the auto-detection work well with your tender documents? Any sections it misses?")
