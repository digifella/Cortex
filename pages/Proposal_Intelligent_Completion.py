"""
Proposal Intelligent Completion
Version: 2.0.0
Date: 2026-01-20

Purpose: Interactive two-tier intelligent proposal completion workflow.
- Tier 1: Auto-complete simple fields from entity profile
- Tier 2: Interactive human-in-the-loop responses for substantive questions

Key Features:
- Strict field extraction to focus on real substantive questions
- Questions grouped by type for organized review
- Per-question actions: Skip / Auto-fill / Generate / Manual
- Evidence panel with knowledge collection search
"""

import streamlit as st
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.document_chunker import DocumentChunker
from cortex_engine.llm_interface import LLMInterface
from cortex_engine.config_manager import ConfigManager
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils import convert_windows_to_wsl_path

# New intelligent completion components
from cortex_engine.field_classifier import (
    FieldClassifier, FieldTier, QuestionType, ClassifiedField
)
from cortex_engine.evidence_retriever import EvidenceRetriever, Evidence
from cortex_engine.response_generator import ResponseGenerator, DraftResponse

st.set_page_config(
    page_title="Intelligent Completion - Cortex Suite",
    page_icon="🧠",
    layout="wide"
)

# Load config
config = ConfigManager().get_config()
db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))

# Initialize managers
workspace_manager = WorkspaceManager(Path(db_path) / "workspaces")
entity_manager = EntityProfileManager(Path(db_path))
collection_manager = WorkingCollectionManager()
field_classifier = FieldClassifier()
llm = LLMInterface(model="qwen2.5:72b-instruct-q4_K_M")
chunker = DocumentChunker(target_chunk_size=4000, max_chunk_size=6000)

# Question type icons and colors
QTYPE_ICONS = {
    QuestionType.CAPABILITY: "💪",
    QuestionType.METHODOLOGY: "📋",
    QuestionType.VALUE_PROPOSITION: "💎",
    QuestionType.COMPLIANCE: "✅",
    QuestionType.INNOVATION: "💡",
    QuestionType.RISK: "⚠️",
    QuestionType.PERSONNEL: "👥",
    QuestionType.PRICING: "💰",
    QuestionType.GENERAL: "📝",
}

QTYPE_DISPLAY = {
    QuestionType.CAPABILITY: "Capability & Experience",
    QuestionType.METHODOLOGY: "Methodology & Approach",
    QuestionType.VALUE_PROPOSITION: "Value Proposition",
    QuestionType.COMPLIANCE: "Compliance & Standards",
    QuestionType.INNOVATION: "Innovation",
    QuestionType.RISK: "Risk Management",
    QuestionType.PERSONNEL: "Personnel & Team",
    QuestionType.PRICING: "Pricing",
    QuestionType.GENERAL: "General Questions",
}

# Professional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2D5F4F;
        margin-bottom: 0.5rem;
    }

    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    .tier-auto {
        background: #E8F5E9;
        color: #2E7D32;
    }

    .tier-intelligent {
        background: #E3F2FD;
        color: #1565C0;
    }

    .confidence-high { color: #2E7D32; }
    .confidence-medium { color: #F57C00; }
    .confidence-low { color: #C62828; }

    .evidence-card {
        background: #FAFAFA;
        border-left: 3px solid #2D5F4F;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .question-type-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1565C0;
        padding: 0.75rem 0;
        border-bottom: 2px solid #E3F2FD;
        margin-bottom: 1rem;
    }

    .question-type-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .qtype-capability { background: #E8EAF6; color: #3949AB; }
    .qtype-methodology { background: #FFF3E0; color: #EF6C00; }
    .qtype-value { background: #E8F5E9; color: #388E3C; }
    .qtype-compliance { background: #FFEBEE; color: #C62828; }
    .qtype-innovation { background: #F3E5F5; color: #7B1FA2; }
    .qtype-risk { background: #FFF8E1; color: #FF8F00; }
    .qtype-personnel { background: #E0F7FA; color: #00838F; }
    .qtype-pricing { background: #FCE4EC; color: #C2185B; }
    .qtype-general { background: #ECEFF1; color: #546E7A; }

    .question-status {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .status-pending { background: #ECEFF1; color: #546E7A; }
    .status-skipped { background: #FFEBEE; color: #C62828; }
    .status-completed { background: #E8F5E9; color: #2E7D32; }
    .status-manual { background: #E3F2FD; color: #1565C0; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🧠 Intelligent Proposal Completion</div>', unsafe_allow_html=True)
st.caption("Interactive human-in-the-loop workflow: review each substantive question with evidence support")

# ============================
# SESSION STATE INITIALIZATION
# ============================

if 'ic_classified_fields' not in st.session_state:
    st.session_state.ic_classified_fields = None
    st.session_state.ic_auto_complete_fields = []
    st.session_state.ic_intelligent_fields = []
    st.session_state.ic_questions_by_type = {}
    st.session_state.ic_question_status = {}  # {field_text: {'status': 'pending'|'skipped'|'completed'|'manual', 'response': str}}
    st.session_state.ic_current_question_idx = 0
    st.session_state.ic_evidence_cache = {}  # Cache evidence lookups

# ============================
# SIDEBAR: Configuration
# ============================

with st.sidebar:
    st.header("Configuration")

    # Workspace selection
    workspaces = workspace_manager.list_workspaces()
    if not workspaces:
        st.warning("No workspaces found. Create one in Proposal Workspace.")
        st.stop()

    workspace_options = {
        f"{ws.metadata.workspace_name}": ws
        for ws in workspaces
    }

    selected_workspace_name = st.selectbox(
        "Workspace",
        options=list(workspace_options.keys()),
        key="ic_workspace_select"
    )

    workspace = workspace_options[selected_workspace_name]

    # Check entity is bound
    if not workspace.metadata.entity_id:
        st.error("No entity bound to workspace")
        st.stop()

    # Show entity info
    entity = entity_manager.get_entity_profile(workspace.metadata.entity_id)
    if entity:
        st.success(f"Entity: {getattr(entity, 'company_name', workspace.metadata.entity_id)}")

    st.divider()

    # Knowledge collection selection
    st.subheader("Knowledge Collection")
    st.caption("Select collection for evidence search")

    collections = collection_manager.get_collection_names()
    if not collections:
        st.warning("No collections available. Create one in Collection Management.")
        selected_collection = None
    else:
        selected_collection = st.selectbox(
            "Evidence Source",
            options=collections,
            key="ic_collection_select",
            help="Substantive responses will search this collection for evidence"
        )

        if selected_collection:
            doc_count = len(collection_manager.get_doc_ids_by_name(selected_collection))
            st.caption(f"{doc_count} documents in collection")

    st.divider()

    # Processing options
    st.subheader("Options")
    use_reranker = st.checkbox(
        "Use Neural Reranker",
        value=True,
        help="Use Qwen3-VL reranker for better evidence precision (slower)"
    )

    max_evidence = st.slider(
        "Max Evidence per Question",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of evidence passages to retrieve"
    )

    use_strict_filter = st.checkbox(
        "Strict Question Filtering",
        value=True,
        help="Only extract questions matching substantive patterns (reduces noise)"
    )

# ============================
# MAIN CONTENT
# ============================

# Find document
documents_dir = workspace.workspace_path / "documents"
doc_path = None

if (documents_dir / workspace.metadata.original_filename).exists():
    doc_path = documents_dir / workspace.metadata.original_filename
elif (documents_dir / "tender_original.txt").exists():
    doc_path = documents_dir / "tender_original.txt"
else:
    txt_files = list(documents_dir.glob("*.txt"))
    if txt_files:
        doc_path = txt_files[0]

if not doc_path:
    st.error("No document file found in workspace")
    st.stop()

# Load document
try:
    with open(doc_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
except Exception as e:
    st.error(f"Error reading document: {str(e)}")
    st.stop()

# ============================
# STEP 1: Field Extraction & Classification
# ============================

st.subheader("Step 1: Extract & Classify Fields")

col1, col2 = st.columns([3, 1])

with col1:
    st.info("""
    Scan document for fillable fields. Questions grouped by type for organized review.
    - **Auto-Complete**: Simple fields filled from entity profile
    - **Substantive Questions**: Review interactively with evidence support
    """)

with col2:
    if st.button("Extract Fields", type="primary", use_container_width=True):
        with st.spinner("Extracting and classifying fields..."):
            # Extract fields using document chunker to find completable sections
            chunks = chunker.create_chunks(document_text)
            completable_chunks = chunker.filter_completable_chunks(chunks)

            # For each chunk, extract field-like content and classify
            classified_fields = []

            for chunk in completable_chunks:
                # Patterns that indicate fillable fields
                field_patterns = [
                    # Label: [blank] patterns
                    r'^([A-Za-z][A-Za-z\s\(\)\']+):\s*$',
                    # Question patterns
                    r'^([A-Z][^?]+\?)\s*$',
                    # "Please provide" patterns (with substance)
                    r'((?:Please\s+)?(?:provide|describe|detail|outline|explain)\s+[^.]{20,}\.)',
                    # "How will you" patterns
                    r'(How\s+(?:will|would|do|can)\s+you\s+[^?]+\?)',
                    # Field with placeholder
                    r'^([A-Za-z][A-Za-z\s\']+):\s*[\[\<\_]',
                ]

                lines = chunk.content.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or len(line) < 5:
                        continue

                    for pattern in field_patterns:
                        match = re.match(pattern, line, re.IGNORECASE)
                        if match:
                            field_text = match.group(1).strip()
                            if len(field_text) > 5:
                                classified = field_classifier.classify(
                                    field_text,
                                    context=chunk.title,
                                    strict_filter=use_strict_filter
                                )
                                # Avoid duplicates
                                if not any(cf.field_text == classified.field_text for cf in classified_fields):
                                    classified_fields.append(classified)
                            break

            # Separate by tier
            auto_fields = [f for f in classified_fields if f.tier == FieldTier.AUTO_COMPLETE]
            intel_fields = [f for f in classified_fields if f.tier == FieldTier.INTELLIGENT]

            # Group intelligent fields by question type
            questions_by_type = defaultdict(list)
            for field in intel_fields:
                qtype = field.question_type or QuestionType.GENERAL
                questions_by_type[qtype].append(field)

            # Initialize status for each question
            question_status = {}
            for field in intel_fields:
                question_status[field.field_text] = {
                    'status': 'pending',
                    'response': '',
                    'evidence': []
                }

            st.session_state.ic_classified_fields = classified_fields
            st.session_state.ic_auto_complete_fields = auto_fields
            st.session_state.ic_intelligent_fields = intel_fields
            st.session_state.ic_questions_by_type = dict(questions_by_type)
            st.session_state.ic_question_status = question_status
            st.session_state.ic_evidence_cache = {}

            st.rerun()

# ============================
# Show Classification Results
# ============================

if st.session_state.ic_classified_fields:
    auto_fields = st.session_state.ic_auto_complete_fields
    intel_fields = st.session_state.ic_intelligent_fields
    questions_by_type = st.session_state.ic_questions_by_type
    question_status = st.session_state.ic_question_status

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Fields", len(st.session_state.ic_classified_fields))
    col2.metric("Auto-Complete", len(auto_fields))
    col3.metric("Substantive", len(intel_fields))
    col4.metric("Question Types", len(questions_by_type))

    st.divider()

    # ============================
    # STEP 2: Auto-Complete Fields
    # ============================

    st.subheader("Step 2: Auto-Complete Fields")

    if auto_fields:
        known_fields = [f for f in auto_fields if f.auto_complete_mapping]
        unknown_fields = [f for f in auto_fields if not f.auto_complete_mapping]

        if known_fields:
            st.success(f"{len(known_fields)} fields can be auto-completed from entity profile")

            with st.expander("Review Auto-Completed Fields", expanded=False):
                for field in known_fields:
                    col1, col2, col3 = st.columns([2, 3, 1])

                    with col1:
                        st.markdown(f"**{field.field_text}**")

                    with col2:
                        profile_field = field.auto_complete_mapping
                        value = "Not set"

                        if entity and profile_field:
                            value = getattr(entity, profile_field, None)
                            if value is None and hasattr(entity, 'data'):
                                value = entity.data.get(profile_field)
                            if value is None:
                                value = f"[Not set: {profile_field}]"

                        st.code(str(value) if value else "Not set", language=None)

                    with col3:
                        st.caption(f"-> {profile_field}")

        if unknown_fields:
            st.warning(f"{len(unknown_fields)} fields need manual entry (not recognized as substantive)")
            with st.expander("Fields Needing Manual Entry", expanded=False):
                for field in unknown_fields:
                    st.text(f"- {field.field_text}")
    else:
        st.info("No auto-complete fields found")

    st.divider()

    # ============================
    # STEP 3: Interactive Substantive Questions
    # ============================

    st.subheader("Step 3: Substantive Questions (by Type)")

    if not intel_fields:
        st.info("No substantive questions found")
    elif not selected_collection:
        st.warning("Select a knowledge collection in the sidebar to enable evidence search")
    else:
        # Progress summary
        total_questions = len(intel_fields)
        completed_count = sum(1 for s in question_status.values() if s['status'] in ['completed', 'skipped', 'manual'])
        pending_count = total_questions - completed_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Pending", pending_count)
        col2.metric("Completed", sum(1 for s in question_status.values() if s['status'] == 'completed'))
        col3.metric("Skipped/Manual", sum(1 for s in question_status.values() if s['status'] in ['skipped', 'manual']))

        st.divider()

        # Initialize evidence retriever and response generator
        evidence_retriever = EvidenceRetriever(db_path)
        response_generator = ResponseGenerator(llm, entity_manager)

        # Display questions grouped by type
        for qtype in QuestionType:
            if qtype not in questions_by_type or not questions_by_type[qtype]:
                continue

            type_questions = questions_by_type[qtype]
            icon = QTYPE_ICONS.get(qtype, "📝")
            display_name = QTYPE_DISPLAY.get(qtype, qtype.value.title())

            # Count status for this type
            type_pending = sum(1 for q in type_questions if question_status.get(q.field_text, {}).get('status') == 'pending')
            type_done = len(type_questions) - type_pending

            st.markdown(f"""
            <div class="question-type-header">
                {icon} {display_name} ({type_done}/{len(type_questions)} done)
            </div>
            """, unsafe_allow_html=True)

            for q_idx, field in enumerate(type_questions):
                field_key = field.field_text
                status_data = question_status.get(field_key, {'status': 'pending', 'response': '', 'evidence': []})
                current_status = status_data['status']

                # Status badge
                status_badge_class = f"status-{current_status}"
                status_label = current_status.upper()

                with st.container(border=True):
                    # Header row with question and status
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"**Q{q_idx + 1}:** {field.field_text}")
                        if field.word_limit:
                            st.caption(f"Word limit: {field.word_limit}")

                    with col2:
                        st.markdown(f'<span class="question-status {status_badge_class}">{status_label}</span>', unsafe_allow_html=True)

                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        if st.button("Skip", key=f"skip_{qtype.value}_{q_idx}", use_container_width=True,
                                     disabled=current_status in ['completed', 'skipped']):
                            question_status[field_key]['status'] = 'skipped'
                            st.session_state.ic_question_status = question_status
                            st.rerun()

                    with col2:
                        if st.button("Auto-fill", key=f"autofill_{qtype.value}_{q_idx}", use_container_width=True,
                                     disabled=current_status in ['completed'],
                                     help="Quick auto-fill without LLM generation"):
                            # Just fetch evidence and let user review
                            with st.spinner("Fetching evidence..."):
                                try:
                                    evidence_result = evidence_retriever.find_evidence(
                                        question=field.field_text,
                                        question_type=field.question_type or QuestionType.GENERAL,
                                        collection_name=selected_collection,
                                        max_results=max_evidence,
                                        use_reranker=use_reranker
                                    )
                                    st.session_state.ic_evidence_cache[field_key] = evidence_result.evidence
                                except Exception as e:
                                    st.error(f"Evidence search failed: {e}")
                            st.rerun()

                    with col3:
                        if st.button("Generate", key=f"gen_{qtype.value}_{q_idx}", type="primary",
                                     use_container_width=True, disabled=current_status == 'completed',
                                     help="Generate response with LLM using evidence"):
                            with st.spinner("Generating response..."):
                                try:
                                    # Fetch evidence if not cached
                                    if field_key not in st.session_state.ic_evidence_cache:
                                        evidence_result = evidence_retriever.find_evidence(
                                            question=field.field_text,
                                            question_type=field.question_type or QuestionType.GENERAL,
                                            collection_name=selected_collection,
                                            max_results=max_evidence,
                                            use_reranker=use_reranker
                                        )
                                        st.session_state.ic_evidence_cache[field_key] = evidence_result.evidence

                                    evidence = st.session_state.ic_evidence_cache.get(field_key, [])

                                    # Generate response
                                    response = response_generator.generate(
                                        classified_field=field,
                                        evidence=evidence,
                                        entity_id=workspace.metadata.entity_id
                                    )

                                    question_status[field_key]['status'] = 'completed'
                                    question_status[field_key]['response'] = response.text
                                    question_status[field_key]['evidence'] = evidence
                                    st.session_state.ic_question_status = question_status
                                except Exception as e:
                                    st.error(f"Generation failed: {e}")
                            st.rerun()

                    with col4:
                        if st.button("Manual", key=f"manual_{qtype.value}_{q_idx}", use_container_width=True,
                                     disabled=current_status in ['completed', 'manual'],
                                     help="Mark for manual entry"):
                            question_status[field_key]['status'] = 'manual'
                            st.session_state.ic_question_status = question_status
                            st.rerun()

                    # Show evidence if cached
                    cached_evidence = st.session_state.ic_evidence_cache.get(field_key, [])
                    if cached_evidence:
                        with st.expander(f"Evidence ({len(cached_evidence)} sources)", expanded=False):
                            for ev_idx, evidence in enumerate(cached_evidence[:3]):
                                st.markdown(f"""
                                <div class="evidence-card">
                                    <strong>{evidence.source_doc}</strong>
                                    <span style="float:right; color:#888;">Relevance: {evidence.relevance_score:.0%}</span>
                                    <br><br>
                                    {evidence.text[:400]}{'...' if len(evidence.text) > 400 else ''}
                                </div>
                                """, unsafe_allow_html=True)

                    # Show/edit response if completed or manual
                    if current_status in ['completed', 'manual']:
                        response_text = st.text_area(
                            "Response",
                            value=status_data.get('response', ''),
                            height=150,
                            key=f"response_{qtype.value}_{q_idx}",
                            label_visibility="collapsed"
                        )

                        # Update if changed
                        if response_text != status_data.get('response', ''):
                            question_status[field_key]['response'] = response_text
                            st.session_state.ic_question_status = question_status

            st.divider()

    # ============================
    # Export / Summary
    # ============================

    if intel_fields:
        st.subheader("Export")

        completed_responses = {k: v for k, v in question_status.items()
                              if v['status'] in ['completed', 'manual'] and v.get('response')}

        if completed_responses:
            st.success(f"{len(completed_responses)} responses ready for export")

            if st.button("Export to Text", use_container_width=True):
                export_text = f"# Intelligent Completion Export\n"
                export_text += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                export_text += f"# Workspace: {selected_workspace_name}\n\n"

                for field_text, data in completed_responses.items():
                    export_text += f"## {field_text}\n\n"
                    export_text += f"{data['response']}\n\n"
                    export_text += "---\n\n"

                st.download_button(
                    "Download Export",
                    data=export_text,
                    file_name=f"intelligent_completion_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("Complete some questions to enable export")

# Footer
st.divider()
st.caption("v2.0.0 | Interactive human-in-the-loop workflow with evidence-backed responses")
