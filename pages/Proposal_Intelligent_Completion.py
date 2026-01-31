"""
Proposal Intelligent Completion
Version: 2.9.0
Date: 2026-01-23

Purpose: Interactive two-tier intelligent proposal completion workflow.
- Tier 1: Auto-complete simple fields from entity profile
- Tier 2: Interactive human-in-the-loop responses for substantive questions

Key Features:
- Strict field extraction to focus on real substantive questions
- Questions grouped by type in collapsible sections
- Clean editorial design with refined typography
- Per-question actions: Edit / Generate (no Skip needed)
- Evidence panel with knowledge collection search
- Human-in-the-loop regeneration with hints to refine responses
- Save As You Go: Automatic persistence of workflow state
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

# Persistence models for "Save As You Go"
from cortex_engine.ic_persistence_model import (
    ICCompletionState,
    PersistedClassifiedField,
    PersistedQuestionState,
    PersistedEvidence,
    PersistedFieldTier,
    PersistedQuestionType,
    evidence_to_persisted,
    persisted_to_evidence,
    classified_field_to_persisted,
    persisted_to_classified_field
)

st.set_page_config(
    page_title="Intelligent Completion - Cortex Suite",
    page_icon="🧠",
    layout="wide"
)

# Load config
config = ConfigManager().get_config()
db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))

# Initialize managers (use session state for persistence, but reload collection_manager fresh)
if 'ic_workspace_manager' not in st.session_state:
    st.session_state.ic_workspace_manager = WorkspaceManager(Path(db_path) / "workspaces")
    st.session_state.ic_entity_manager = EntityProfileManager(Path(db_path))
    st.session_state.ic_field_classifier = FieldClassifier()
    st.session_state.ic_llm = LLMInterface(model="qwen2.5:72b-instruct-q4_K_M")
    st.session_state.ic_chunker = DocumentChunker(target_chunk_size=4000, max_chunk_size=6000)

workspace_manager = st.session_state.ic_workspace_manager
entity_manager = st.session_state.ic_entity_manager
field_classifier = st.session_state.ic_field_classifier
llm = st.session_state.ic_llm
chunker = st.session_state.ic_chunker

# Always reload collection_manager to get fresh data (fixes stale collection counts)
collection_manager = WorkingCollectionManager()

# Question type display names (no emojis in headers - cleaner look)
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

# ============================
# REFINED EDITORIAL CSS
# ============================
st.markdown("""
<style>
    /* Import refined typography */
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=DM+Sans:wght@400;500;600&display=swap');

    /* Base styling */
    .stApp {
        background: #FAFAF8;
    }

    /* Main header - editorial style */
    .main-header {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2.25rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .main-subtitle {
        font-family: 'DM Sans', -apple-system, sans-serif;
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 2rem;
    }

    /* Progress bar container */
    .progress-summary {
        display: flex;
        gap: 2rem;
        padding: 1rem 0;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 1.5rem;
    }

    .progress-item {
        font-family: 'DM Sans', sans-serif;
    }

    .progress-number {
        font-size: 1.75rem;
        font-weight: 600;
        color: #2D5F4F;
        line-height: 1;
    }

    .progress-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* Section accordion headers */
    .section-header {
        font-family: 'DM Sans', sans-serif;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.875rem 0;
        cursor: pointer;
        border-bottom: 1px solid #e5e5e5;
        transition: all 0.15s ease;
    }

    .section-header:hover {
        background: #f5f5f3;
        margin: 0 -1rem;
        padding: 0.875rem 1rem;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-count {
        font-size: 0.8rem;
        font-weight: 500;
        color: #888;
        background: #f0f0ee;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
    }

    .section-count-done {
        background: #E8F5E9;
        color: #2D5F4F;
    }

    /* Question cards - minimal */
    .question-card {
        padding: 1.25rem;
        margin: 0.75rem 0;
        background: white;
        border: 1px solid #e8e8e6;
        border-radius: 6px;
        transition: border-color 0.15s ease;
    }

    .question-card:hover {
        border-color: #ccc;
    }

    .question-card.completed {
        border-left: 3px solid #2D5F4F;
    }

    .question-card.editing {
        border-left: 3px solid #D4A853;
    }

    .question-number {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    .question-text {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.05rem;
        color: #1a1a1a;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }

    .question-meta {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        color: #888;
    }

    /* Status indicators - subtle */
    .status-badge {
        font-family: 'DM Sans', sans-serif;
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .status-pending {
        background: #f5f5f3;
        color: #888;
    }

    .status-completed {
        background: #E8F5E9;
        color: #2D5F4F;
    }

    .status-editing {
        background: #FEF7E6;
        color: #B8860B;
    }

    /* Evidence cards - refined */
    .evidence-card {
        background: #FAFAF8;
        border-left: 2px solid #2D5F4F;
        padding: 0.875rem 1rem;
        margin: 0.5rem 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.875rem;
        line-height: 1.6;
        color: #444;
    }

    .evidence-source {
        font-size: 0.75rem;
        font-weight: 600;
        color: #2D5F4F;
        margin-bottom: 0.5rem;
    }

    .evidence-score {
        float: right;
        font-size: 0.7rem;
        color: #888;
    }

    /* Action buttons - compact and refined */
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.9rem !important;
        border-radius: 4px !important;
        transition: all 0.15s ease !important;
    }

    /* Primary action (Generate) */
    .stButton > button[kind="primary"] {
        background: #2D5F4F !important;
        border: none !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #234A3D !important;
    }

    /* Secondary action (Edit) */
    .stButton > button[kind="secondary"] {
        background: white !important;
        border: 1px solid #ddd !important;
        color: #333 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #f5f5f3 !important;
        border-color: #ccc !important;
    }

    /* Text areas - cleaner */
    .stTextArea textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        border: 1px solid #e0e0de !important;
        border-radius: 4px !important;
        padding: 0.875rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #2D5F4F !important;
        box-shadow: 0 0 0 1px #2D5F4F !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #555 !important;
        background: transparent !important;
        border: none !important;
    }

    /* Word count */
    .word-count {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.75rem;
        color: #888;
        text-align: right;
        margin-top: 0.25rem;
    }

    /* Sidebar refinements */
    section[data-testid="stSidebar"] {
        background: #f8f8f6;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #555 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metrics - cleaner */
    [data-testid="stMetricValue"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2D5F4F !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #e5e5e5;
        margin: 1.5rem 0;
    }

    /* Container borders - subtle */
    [data-testid="stVerticalBlock"] > div:has(> .stContainer) {
        border-color: #e8e8e6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Intelligent Completion</div>', unsafe_allow_html=True)
st.warning("**This page has been superseded by Proposal Manager.** Use the new consolidated workflow for a better experience.")
st.page_link("pages/13_Proposal_Manager.py", label="Go to Proposal Manager", icon="📋")
st.divider()
st.markdown('<div class="main-subtitle">Review and complete substantive proposal questions with AI assistance</div>', unsafe_allow_html=True)

# ============================
# SESSION STATE INITIALIZATION
# ============================

if 'ic_classified_fields' not in st.session_state:
    st.session_state.ic_classified_fields = None
    st.session_state.ic_auto_complete_fields = []
    st.session_state.ic_intelligent_fields = []
    st.session_state.ic_questions_by_type = {}
    st.session_state.ic_question_status = {}
    st.session_state.ic_current_question_idx = 0
    st.session_state.ic_evidence_cache = {}

# Track which workspace was last loaded to detect changes
if 'ic_last_loaded_workspace' not in st.session_state:
    st.session_state.ic_last_loaded_workspace = None

# Track expanded sections
if 'ic_expanded_sections' not in st.session_state:
    st.session_state.ic_expanded_sections = set()


# ============================
# PERSISTENCE HELPER FUNCTIONS
# ============================

def load_ic_state_from_workspace(workspace_id: str) -> bool:
    """Load saved IC state from workspace into session state."""
    saved_state = workspace_manager.get_ic_completion_state(workspace_id)

    if not saved_state:
        return False

    try:
        persisted_fields = saved_state.get('classified_fields', [])
        if not persisted_fields:
            return False

        classified_fields = []
        for pf_dict in persisted_fields:
            pf = PersistedClassifiedField(**pf_dict)
            classified_fields.append(persisted_to_classified_field(pf))

        st.session_state.ic_classified_fields = classified_fields

        auto_indices = saved_state.get('auto_complete_field_indices', [])
        intel_indices = saved_state.get('intelligent_field_indices', [])

        st.session_state.ic_auto_complete_fields = [
            classified_fields[i] for i in auto_indices if i < len(classified_fields)
        ]
        st.session_state.ic_intelligent_fields = [
            classified_fields[i] for i in intel_indices if i < len(classified_fields)
        ]

        questions_by_type_raw = saved_state.get('questions_by_type', {})
        questions_by_type = {}
        for qtype_value, indices in questions_by_type_raw.items():
            questions_by_type[qtype_value] = [
                classified_fields[i] for i in indices if i < len(classified_fields)
            ]
        st.session_state.ic_questions_by_type = questions_by_type

        question_status_raw = saved_state.get('question_status', {})
        question_status = {}
        for field_text, qs_dict in question_status_raw.items():
            evidence_list = []
            for ev_dict in qs_dict.get('evidence', []):
                pe = PersistedEvidence(**ev_dict)
                evidence_list.append(persisted_to_evidence(pe))

            question_status[field_text] = {
                'status': qs_dict.get('status', 'pending'),
                'response': qs_dict.get('response', ''),
                'evidence': evidence_list,
                'confidence': qs_dict.get('confidence'),
                'collection_name': qs_dict.get('collection_name'),
                'creativity_level': qs_dict.get('creativity_level')
            }
        st.session_state.ic_question_status = question_status

        evidence_cache_raw = saved_state.get('evidence_cache', {})
        evidence_cache = {}
        for field_text, ev_list in evidence_cache_raw.items():
            evidence_cache[field_text] = [
                persisted_to_evidence(PersistedEvidence(**ev_dict))
                for ev_dict in ev_list
            ]
        st.session_state.ic_evidence_cache = evidence_cache

        return True

    except Exception as e:
        st.warning(f"Could not restore saved progress: {e}")
        return False


def save_ic_state_to_workspace(workspace_id: str) -> bool:
    """Save current IC session state to workspace."""
    if not st.session_state.ic_classified_fields:
        return False

    try:
        classified_fields = st.session_state.ic_classified_fields
        field_to_index = {f.field_text: i for i, f in enumerate(classified_fields)}

        persisted_fields = [
            classified_field_to_persisted(f).model_dump()
            for f in classified_fields
        ]

        auto_indices = [
            field_to_index[f.field_text]
            for f in st.session_state.ic_auto_complete_fields
            if f.field_text in field_to_index
        ]
        intel_indices = [
            field_to_index[f.field_text]
            for f in st.session_state.ic_intelligent_fields
            if f.field_text in field_to_index
        ]

        questions_by_type_indices = {}
        for qtype_value, fields in st.session_state.ic_questions_by_type.items():
            questions_by_type_indices[qtype_value] = [
                field_to_index[f.field_text]
                for f in fields
                if f.field_text in field_to_index
            ]

        question_status_persisted = {}
        for field_text, qs in st.session_state.ic_question_status.items():
            evidence_persisted = [
                evidence_to_persisted(ev).model_dump()
                for ev in qs.get('evidence', [])
            ]
            question_status_persisted[field_text] = {
                'status': qs.get('status', 'pending'),
                'response': qs.get('response', ''),
                'evidence': evidence_persisted,
                'confidence': qs.get('confidence'),
                'collection_name': qs.get('collection_name'),
                'creativity_level': qs.get('creativity_level')
            }

        evidence_cache_persisted = {}
        for field_text, ev_list in st.session_state.ic_evidence_cache.items():
            evidence_cache_persisted[field_text] = [
                evidence_to_persisted(ev).model_dump()
                for ev in ev_list
            ]

        state = {
            'workspace_id': workspace_id,
            'created_at': datetime.now().isoformat(),
            'classified_fields': persisted_fields,
            'auto_complete_field_indices': auto_indices,
            'intelligent_field_indices': intel_indices,
            'questions_by_type': questions_by_type_indices,
            'question_status': question_status_persisted,
            'evidence_cache': evidence_cache_persisted
        }

        return workspace_manager.save_ic_completion_state(workspace_id, state)

    except Exception as e:
        st.warning(f"Could not save progress: {e}")
        return False


def clear_ic_state():
    """Clear all IC session state (for re-extract)."""
    st.session_state.ic_classified_fields = None
    st.session_state.ic_auto_complete_fields = []
    st.session_state.ic_intelligent_fields = []
    st.session_state.ic_questions_by_type = {}
    st.session_state.ic_question_status = {}
    st.session_state.ic_evidence_cache = {}
    st.session_state.ic_expanded_sections = set()


# ============================
# SIDEBAR: Configuration
# ============================

with st.sidebar:
    st.markdown("### Configuration")

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

    # Auto-load saved state when workspace changes
    if st.session_state.ic_last_loaded_workspace != workspace.metadata.workspace_id:
        clear_ic_state()
        if load_ic_state_from_workspace(workspace.metadata.workspace_id):
            st.toast("Restored previous progress", icon="")
        st.session_state.ic_last_loaded_workspace = workspace.metadata.workspace_id

    # Check entity is bound
    if not workspace.metadata.entity_id:
        st.error("No entity bound to workspace")
        st.stop()

    # Show entity info
    entity = entity_manager.get_entity_profile(workspace.metadata.entity_id)
    if entity:
        st.caption(f"Entity: {getattr(entity, 'company_name', workspace.metadata.entity_id)}")

    st.divider()

    # Knowledge collection selection
    st.markdown("#### Evidence Source")

    collections = collection_manager.get_collection_names()
    collection_options = ["-- Entire Knowledge Base --"] + (collections if collections else [])

    selected_option = st.selectbox(
        "Collection",
        options=collection_options,
        key="ic_collection_select",
        label_visibility="collapsed"
    )

    if selected_option == "-- Entire Knowledge Base --":
        selected_collection = None
        st.caption("Searching entire knowledge base")
    else:
        selected_collection = selected_option
        doc_count = len(collection_manager.get_doc_ids_by_name(selected_collection))
        st.caption(f"{doc_count} documents")

    st.divider()

    # Generation settings
    st.markdown("#### Generation")

    creativity_level = st.select_slider(
        "Style",
        options=[0, 1, 2],
        value=1,
        format_func=lambda x: {0: "Factual", 1: "Balanced", 2: "Creative"}[x]
    )

    temperature_map = {0: 0.3, 1: 0.7, 2: 1.0}
    generation_temperature = temperature_map[creativity_level]

    # Advanced options
    with st.expander("Advanced", expanded=False):
        max_evidence = st.slider(
            "Evidence per question",
            min_value=2,
            max_value=10,
            value=5
        )

        use_reranker = st.checkbox(
            "Neural reranker",
            value=True,
            help="Better precision, slower"
        )

        use_strict_filter = st.checkbox(
            "Strict filtering",
            value=True,
            help="Only substantive questions"
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
# STEP 1: Field Extraction
# ============================

fields_already_extracted = (
    st.session_state.ic_classified_fields is not None and
    len(st.session_state.ic_classified_fields) > 0
)

if not fields_already_extracted:
    st.markdown("### Extract Questions")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption("Scan the document to identify substantive questions requiring responses.")
    with col2:
        if st.button("Extract", type="primary"):
            with st.spinner("Analyzing document..."):
                chunks = chunker.create_chunks(document_text)
                completable_chunks = chunker.filter_completable_chunks(chunks)

                classified_fields = []

                for chunk in completable_chunks:
                    field_patterns = [
                        r'^([A-Za-z][A-Za-z\s\(\)\']+):\s*$',
                        r'^([A-Z][^?]+\?)\s*$',
                        r'((?:Please\s+)?(?:provide|describe|detail|outline|explain)\s+[^.]{20,}\.)',
                        r'(How\s+(?:will|would|do|can)\s+you\s+[^?]+\?)',
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
                                    if not any(cf.field_text == classified.field_text for cf in classified_fields):
                                        classified_fields.append(classified)
                                break

                auto_fields = [f for f in classified_fields if f.tier == FieldTier.AUTO_COMPLETE]
                intel_fields = [f for f in classified_fields if f.tier == FieldTier.INTELLIGENT]

                questions_by_type = defaultdict(list)
                for field in intel_fields:
                    qtype = field.question_type or QuestionType.GENERAL
                    questions_by_type[qtype.value].append(field)

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

                save_ic_state_to_workspace(workspace.metadata.workspace_id)
                st.rerun()

else:
    # ============================
    # Show Results
    # ============================

    auto_fields = st.session_state.ic_auto_complete_fields
    intel_fields = st.session_state.ic_intelligent_fields
    questions_by_type = st.session_state.ic_questions_by_type
    question_status = st.session_state.ic_question_status

    # Progress summary - clean metrics
    total_questions = len(intel_fields)
    completed_count = sum(1 for s in question_status.values() if s['status'] == 'completed')
    editing_count = sum(1 for s in question_status.values() if s['status'] == 'editing')
    pending_count = total_questions - completed_count - editing_count

    # Progress display
    st.markdown(f"""
    <div class="progress-summary">
        <div class="progress-item">
            <div class="progress-number">{completed_count}</div>
            <div class="progress-label">Completed</div>
        </div>
        <div class="progress-item">
            <div class="progress-number">{editing_count}</div>
            <div class="progress-label">In Progress</div>
        </div>
        <div class="progress-item">
            <div class="progress-number">{pending_count}</div>
            <div class="progress-label">Remaining</div>
        </div>
        <div class="progress-item">
            <div class="progress-number">{len(auto_fields)}</div>
            <div class="progress-label">Auto-filled</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Re-extract option (subtle)
    with st.expander("Extraction Settings", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Found {len(st.session_state.ic_classified_fields)} fields total")
        with col2:
            if st.button("Re-extract", type="secondary"):
                clear_ic_state()
                workspace_manager.clear_ic_completion_state(workspace.metadata.workspace_id)
                st.rerun()

    st.divider()

    # Initialize components
    evidence_retriever = EvidenceRetriever(db_path)
    llm.temperature = generation_temperature
    response_generator = ResponseGenerator(llm, entity_manager)

    # ============================
    # Questions by Type (Accordion Sections)
    # ============================

    for qtype in QuestionType:
        qtype_key = qtype.value
        if qtype_key not in questions_by_type or not questions_by_type[qtype_key]:
            continue

        type_questions = questions_by_type[qtype_key]
        display_name = QTYPE_DISPLAY.get(qtype, qtype.value.title())

        # Count status for this type
        type_completed = sum(1 for q in type_questions if question_status.get(q.field_text, {}).get('status') == 'completed')
        type_editing = sum(1 for q in type_questions if question_status.get(q.field_text, {}).get('status') == 'editing')
        type_total = len(type_questions)

        # Determine if section should be expanded
        section_expanded = qtype_key in st.session_state.ic_expanded_sections

        # Section header with expander
        count_class = "section-count-done" if type_completed == type_total else "section-count"

        with st.expander(f"**{display_name}** — {type_completed}/{type_total} complete", expanded=section_expanded):
            # Track expansion state
            if qtype_key not in st.session_state.ic_expanded_sections:
                st.session_state.ic_expanded_sections.add(qtype_key)

            for q_idx, field in enumerate(type_questions):
                field_key = field.field_text
                status_data = question_status.get(field_key, {'status': 'pending', 'response': '', 'evidence': []})
                current_status = status_data['status']

                # Determine card style based on status
                card_class = ""
                if current_status == 'completed':
                    card_class = "completed"
                elif current_status == 'editing':
                    card_class = "editing"

                # Question card
                word_limit_text = f'Word limit: {field.word_limit}' if field.word_limit else ''
                card_html = f'<div class="question-card {card_class}"><div class="question-number">Question {q_idx + 1}</div><div class="question-text">{field.field_text}</div><div class="question-meta">{word_limit_text}<span class="status-badge status-{current_status}" style="float:right;">{current_status.upper()}</span></div></div>'
                st.markdown(card_html, unsafe_allow_html=True)

                # Action area
                if current_status == 'pending':
                    # Compact action buttons with creativity dropdown
                    btn_col1, creat_col, btn_col2, spacer = st.columns([1, 1.2, 1, 2])

                    with btn_col1:
                        if st.button("Edit", key=f"edit_{qtype_key}_{q_idx}", type="secondary"):
                            question_status[field_key]['status'] = 'editing'
                            st.session_state.ic_question_status = question_status
                            try:
                                evidence_result = evidence_retriever.find_evidence(
                                    question=field.field_text,
                                    question_type=field.question_type or QuestionType.GENERAL,
                                    collection_name=selected_collection,
                                    max_results=max_evidence,
                                    use_reranker=use_reranker
                                )
                                st.session_state.ic_evidence_cache[field_key] = evidence_result.evidence
                            except Exception:
                                pass
                            save_ic_state_to_workspace(workspace.metadata.workspace_id)
                            st.rerun()

                    with creat_col:
                        q_creativity = st.selectbox(
                            "Style",
                            options=["Factual", "Balanced", "Creative"],
                            index=1,  # Default to Balanced
                            key=f"creat_{qtype_key}_{q_idx}",
                            label_visibility="collapsed"
                        )
                        q_temp = {"Factual": 0.3, "Balanced": 0.7, "Creative": 1.0}[q_creativity]

                    with btn_col2:
                        if st.button("Generate", key=f"gen_{qtype_key}_{q_idx}", type="primary"):
                            with st.spinner("Generating..."):
                                try:
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

                                    # Apply per-question creativity
                                    llm.temperature = q_temp

                                    response = response_generator.generate(
                                        classified_field=field,
                                        evidence=evidence,
                                        entity_id=workspace.metadata.entity_id
                                    )

                                    question_status[field_key]['status'] = 'completed'
                                    question_status[field_key]['response'] = response.text
                                    question_status[field_key]['evidence'] = evidence
                                    question_status[field_key]['creativity_level'] = q_creativity
                                    st.session_state.ic_question_status = question_status
                                    save_ic_state_to_workspace(workspace.metadata.workspace_id)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Generation failed: {e}")

                elif current_status in ['completed', 'editing']:
                    # Show response editor
                    response_text = st.text_area(
                        "Response",
                        value=status_data.get('response', ''),
                        height=180,
                        key=f"response_{qtype_key}_{q_idx}",
                        label_visibility="collapsed"
                    )

                    # Update if changed
                    if response_text != status_data.get('response', ''):
                        question_status[field_key]['response'] = response_text
                        question_status[field_key]['status'] = 'completed'
                        st.session_state.ic_question_status = question_status
                        save_ic_state_to_workspace(workspace.metadata.workspace_id)

                    # Word count and actions row
                    wc_col, action_col1, action_col2, action_col3 = st.columns([2, 1, 1, 1])

                    with wc_col:
                        word_count = len(response_text.split()) if response_text else 0
                        limit_info = f" / {field.word_limit}" if field.word_limit else ""
                        st.caption(f"{word_count} words{limit_info}")

                    with action_col1:
                        st.download_button(
                            "Export",
                            data=f"# {field.field_text}\n\n{response_text}",
                            file_name=f"response_{q_idx + 1}.txt",
                            mime="text/plain",
                            key=f"export_{qtype_key}_{q_idx}"
                        )

                    # Toggle for showing evidence
                    show_evidence_key = f"show_ev_{qtype_key}_{q_idx}"
                    if show_evidence_key not in st.session_state:
                        st.session_state[show_evidence_key] = False

                    with action_col2:
                        if st.button("Evidence", key=f"ev_btn_{qtype_key}_{q_idx}", type="secondary"):
                            st.session_state[show_evidence_key] = not st.session_state[show_evidence_key]
                            st.rerun()

                    # Toggle for refinement
                    show_refine_key = f"show_ref_{qtype_key}_{q_idx}"
                    if show_refine_key not in st.session_state:
                        st.session_state[show_refine_key] = False

                    with action_col3:
                        if st.button("Refine", key=f"ref_btn_{qtype_key}_{q_idx}", type="secondary"):
                            st.session_state[show_refine_key] = not st.session_state[show_refine_key]
                            st.rerun()

                    # Evidence panel (conditionally shown)
                    cached_evidence = st.session_state.ic_evidence_cache.get(field_key, [])
                    if st.session_state[show_evidence_key] and cached_evidence:
                        st.markdown("---")
                        st.caption(f"**Evidence** ({len(cached_evidence)} sources)")
                        for ev in cached_evidence[:3]:
                            ev_text = ev.text[:350] + ('...' if len(ev.text) > 350 else '')
                            ev_html = f'<div class="evidence-card"><div class="evidence-source">{ev.source_doc}<span class="evidence-score">{ev.relevance_score:.0%} relevant</span></div>{ev_text}</div>'
                            st.markdown(ev_html, unsafe_allow_html=True)

                    # Refinement panel (conditionally shown)
                    if st.session_state[show_refine_key]:
                        st.markdown("---")
                        st.caption("**Refine with AI Guidance**")
                        hint_text = st.text_area(
                            "Guidance",
                            placeholder="e.g., Focus on ISO certification, add specific metrics...",
                            key=f"hint_{qtype_key}_{q_idx}",
                            height=80,
                            label_visibility="collapsed"
                        )

                        if st.button("Regenerate", key=f"regen_{qtype_key}_{q_idx}", type="primary"):
                            with st.spinner("Regenerating..."):
                                try:
                                    evidence_result = evidence_retriever.find_evidence(
                                        question=field.field_text,
                                        question_type=field.question_type or QuestionType.GENERAL,
                                        collection_name=selected_collection,
                                        max_results=max_evidence,
                                        use_reranker=use_reranker
                                    )
                                    evidence = evidence_result.evidence
                                    st.session_state.ic_evidence_cache[field_key] = evidence

                                    previous_response = DraftResponse(
                                        question=field.field_text,
                                        question_type=field.question_type or QuestionType.GENERAL,
                                        text=status_data.get('response', ''),
                                        evidence_used=evidence,
                                        confidence=0.5,
                                        word_count=len(status_data.get('response', '').split()),
                                        needs_review=True,
                                        placeholders=[],
                                        generation_time=0,
                                        metadata={
                                            'entity_id': workspace.metadata.entity_id,
                                            'word_limit': field.word_limit
                                        }
                                    )

                                    guidance = hint_text if hint_text else "Improve with more specific details from evidence."
                                    new_response = response_generator.regenerate(
                                        previous_response=previous_response,
                                        entity_id=workspace.metadata.entity_id,
                                        additional_guidance=guidance,
                                        new_evidence=evidence
                                    )

                                    question_status[field_key]['response'] = new_response.text
                                    question_status[field_key]['confidence'] = new_response.confidence
                                    st.session_state.ic_question_status = question_status
                                    save_ic_state_to_workspace(workspace.metadata.workspace_id)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Regeneration failed: {e}")

                st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # ============================
    # Export Section
    # ============================

    st.divider()

    completed_responses = {k: v for k, v in question_status.items()
                          if v['status'] == 'completed' and v.get('response')}

    if completed_responses:
        exp_col1, exp_col2 = st.columns([3, 1])
        with exp_col1:
            st.caption(f"{len(completed_responses)} responses ready for export")
        with exp_col2:
            export_text = f"# Intelligent Completion Export\n"
            export_text += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            export_text += f"# Workspace: {selected_workspace_name}\n\n"

            for field_text, data in completed_responses.items():
                export_text += f"## {field_text}\n\n"
                export_text += f"{data['response']}\n\n"
                export_text += "---\n\n"

            st.download_button(
                "Export All",
                data=export_text,
                file_name=f"proposal_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                type="primary"
            )

# Footer
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
st.caption("v2.9.0 — Progress saved automatically")
