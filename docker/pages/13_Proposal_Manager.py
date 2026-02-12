"""
Proposal Manager - Consolidated Proposal Workflow
Version: 1.0.0
Date: 2026-02-01

Purpose: Unified 3-phase proposal workflow replacing Proposal Workspace,
Chunk Review V2, and Intelligent Completion pages.

Phase 1: Setup - Create workspace, upload tender, auto-process
Phase 2: Work - Template Fields (Tier 1) + AI Assisted (Tier 2) in parallel tabs
Phase 3: Review & Export - Completeness dashboard, comparison, export
"""

import streamlit as st
import sys
import os
import re
import time
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

st.set_page_config(
    page_title="Proposal Manager - Cortex Suite",
    page_icon="📋",
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.workspace_model import WorkspaceState, ChunkProgress, MentionBinding
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.document_processor import DocumentProcessor
from cortex_engine.document_chunker import DocumentChunker
from cortex_engine.field_classifier import (
    FieldClassifier, FieldTier, QuestionType, ClassifiedField
)
from cortex_engine.evidence_retriever import EvidenceRetriever, Evidence
from cortex_engine.response_generator import ResponseGenerator, DraftResponse
from cortex_engine.llm_interface import LLMInterface
from cortex_engine.config_manager import ConfigManager
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.markup_engine import MarkupEngine
from cortex_engine.proposal_export_engine import ProposalExportEngine, CompletionReport
from cortex_engine.ic_persistence_model import (
    PersistedClassifiedField,
    PersistedEvidence,
    persisted_to_classified_field,
    persisted_to_evidence,
    classified_field_to_persisted,
    evidence_to_persisted
)
from cortex_engine.utils import convert_windows_to_wsl_path, resolve_db_root_path, get_logger

logger = get_logger(__name__)

# ============================================
# CSS
# ============================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=DM+Sans:wght@400;500;600&display=swap');

    .phase-indicator {
        display: flex;
        justify-content: center;
        gap: 0;
        margin: 1rem 0 2rem 0;
        font-family: 'DM Sans', sans-serif;
    }
    .phase-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.2rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #888;
        background: #f5f5f3;
        border: 1px solid #e5e5e3;
    }
    .phase-step:first-child { border-radius: 6px 0 0 6px; }
    .phase-step:last-child { border-radius: 0 6px 6px 0; }
    .phase-step.active {
        color: white;
        background: #2D5F4F;
        border-color: #2D5F4F;
        font-weight: 600;
    }
    .phase-step.done {
        color: #2D5F4F;
        background: #E8F5E9;
        border-color: #C8E6C9;
    }
    .phase-arrow {
        color: #ccc;
        font-size: 1rem;
        padding: 0 0.25rem;
    }

    .progress-number {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #2D5F4F;
        line-height: 1;
    }
    .progress-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    .question-card {
        padding: 1.25rem;
        margin: 0.75rem 0;
        background: white;
        border: 1px solid #e8e8e6;
        border-radius: 6px;
    }
    .question-card.completed { border-left: 3px solid #2D5F4F; }
    .question-card.editing { border-left: 3px solid #D4A853; }

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

    .evidence-card {
        background: #FAFAF8;
        border-left: 2px solid #2D5F4F;
        padding: 0.875rem 1rem;
        margin: 0.5rem 0;
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

    .field-row {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0ee;
    }
    .field-label {
        font-weight: 500;
        color: #555;
        min-width: 200px;
    }
    .field-value {
        color: #1a1a1a;
        flex: 1;
    }
    .field-status {
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================
# INITIALIZATION
# ============================================

def init_managers():
    """Initialize all managers in session state."""
    if 'pm_initialized' not in st.session_state:
        config = ConfigManager().get_config()
        raw_db_path = config.get('ai_database_path', '')
        resolved_root = resolve_db_root_path(raw_db_path)
        if resolved_root:
            db_path = str(resolved_root)
        elif os.path.exists('/.dockerenv'):
            db_path = raw_db_path
        else:
            db_path = convert_windows_to_wsl_path(raw_db_path)

        st.session_state.pm_workspace_manager = WorkspaceManager(Path(db_path) / "workspaces")
        st.session_state.pm_entity_manager = EntityProfileManager(Path(db_path))
        st.session_state.pm_field_classifier = FieldClassifier()
        st.session_state.pm_llm = LLMInterface(model="qwen2.5:72b-instruct-q4_K_M")
        st.session_state.pm_chunker = DocumentChunker(target_chunk_size=4000, max_chunk_size=6000)
        st.session_state.pm_db_path = db_path
        st.session_state.pm_export_engine = ProposalExportEngine(
            st.session_state.pm_workspace_manager,
            st.session_state.pm_entity_manager
        )
        st.session_state.pm_initialized = True

    return (
        st.session_state.pm_workspace_manager,
        st.session_state.pm_entity_manager,
        st.session_state.pm_field_classifier,
        st.session_state.pm_llm,
        st.session_state.pm_chunker,
        st.session_state.pm_db_path,
        st.session_state.pm_export_engine
    )


workspace_mgr, entity_mgr, field_classifier, llm, chunker, db_path, export_engine = init_managers()

# Always reload collection_manager fresh
collection_manager = WorkingCollectionManager()

# Session state defaults
if 'pm_workspace_id' not in st.session_state:
    st.session_state.pm_workspace_id = None
if 'pm_phase' not in st.session_state:
    st.session_state.pm_phase = 1
if 'pm_ic_fields' not in st.session_state:
    st.session_state.pm_ic_fields = None
if 'pm_ic_question_status' not in st.session_state:
    st.session_state.pm_ic_question_status = {}
if 'pm_ic_questions_by_type' not in st.session_state:
    st.session_state.pm_ic_questions_by_type = {}
if 'pm_ic_auto_fields' not in st.session_state:
    st.session_state.pm_ic_auto_fields = []
if 'pm_ic_intel_fields' not in st.session_state:
    st.session_state.pm_ic_intel_fields = []
if 'pm_ic_evidence_cache' not in st.session_state:
    st.session_state.pm_ic_evidence_cache = {}
if 'pm_ic_loaded_for' not in st.session_state:
    st.session_state.pm_ic_loaded_for = None


# ============================================
# IC PERSISTENCE HELPERS
# ============================================

def load_ic_state(workspace_id: str) -> bool:
    """Load IC state from workspace into session state."""
    saved = workspace_mgr.get_ic_completion_state(workspace_id)
    if not saved:
        return False
    try:
        persisted_fields = saved.get('classified_fields', [])
        if not persisted_fields:
            return False

        classified_fields = []
        for pf_dict in persisted_fields:
            pf = PersistedClassifiedField(**pf_dict)
            classified_fields.append(persisted_to_classified_field(pf))

        st.session_state.pm_ic_fields = classified_fields

        auto_indices = saved.get('auto_complete_field_indices', [])
        intel_indices = saved.get('intelligent_field_indices', [])
        st.session_state.pm_ic_auto_fields = [classified_fields[i] for i in auto_indices if i < len(classified_fields)]
        st.session_state.pm_ic_intel_fields = [classified_fields[i] for i in intel_indices if i < len(classified_fields)]

        questions_by_type = {}
        for qtype_value, indices in saved.get('questions_by_type', {}).items():
            questions_by_type[qtype_value] = [classified_fields[i] for i in indices if i < len(classified_fields)]
        st.session_state.pm_ic_questions_by_type = questions_by_type

        question_status = {}
        for field_text, qs_dict in saved.get('question_status', {}).items():
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
        st.session_state.pm_ic_question_status = question_status

        evidence_cache = {}
        for field_text, ev_list in saved.get('evidence_cache', {}).items():
            evidence_cache[field_text] = [persisted_to_evidence(PersistedEvidence(**ev_dict)) for ev_dict in ev_list]
        st.session_state.pm_ic_evidence_cache = evidence_cache

        return True
    except Exception as e:
        logger.error(f"Failed to load IC state: {e}")
        return False


def save_ic_state(workspace_id: str) -> bool:
    """Save current IC session state to workspace."""
    if not st.session_state.pm_ic_fields:
        return False
    try:
        classified_fields = st.session_state.pm_ic_fields
        field_to_index = {f.field_text: i for i, f in enumerate(classified_fields)}

        persisted_fields = [classified_field_to_persisted(f).model_dump() for f in classified_fields]

        auto_indices = [field_to_index[f.field_text] for f in st.session_state.pm_ic_auto_fields if f.field_text in field_to_index]
        intel_indices = [field_to_index[f.field_text] for f in st.session_state.pm_ic_intel_fields if f.field_text in field_to_index]

        questions_by_type_indices = {}
        for qtype_value, fields in st.session_state.pm_ic_questions_by_type.items():
            questions_by_type_indices[qtype_value] = [field_to_index[f.field_text] for f in fields if f.field_text in field_to_index]

        question_status_persisted = {}
        for field_text, qs in st.session_state.pm_ic_question_status.items():
            evidence_persisted = [evidence_to_persisted(ev).model_dump() for ev in qs.get('evidence', [])]
            question_status_persisted[field_text] = {
                'status': qs.get('status', 'pending'),
                'response': qs.get('response', ''),
                'evidence': evidence_persisted,
                'confidence': qs.get('confidence'),
                'collection_name': qs.get('collection_name'),
                'creativity_level': qs.get('creativity_level')
            }

        evidence_cache_persisted = {}
        for field_text, ev_list in st.session_state.pm_ic_evidence_cache.items():
            evidence_cache_persisted[field_text] = [evidence_to_persisted(ev).model_dump() for ev in ev_list]

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
        return workspace_mgr.save_ic_completion_state(workspace_id, state)
    except Exception as e:
        logger.error(f"Failed to save IC state: {e}")
        return False


def clear_ic_state():
    """Clear IC session state."""
    st.session_state.pm_ic_fields = None
    st.session_state.pm_ic_auto_fields = []
    st.session_state.pm_ic_intel_fields = []
    st.session_state.pm_ic_questions_by_type = {}
    st.session_state.pm_ic_question_status = {}
    st.session_state.pm_ic_evidence_cache = {}


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.header("Proposal Manager")

    workspaces = workspace_mgr.list_workspaces()

    if workspaces:
        workspace_options = {"-- Create New --": None}
        workspace_options.update({
            ws.metadata.workspace_name: ws.metadata.workspace_id
            for ws in workspaces
        })

        # Find default index
        default_idx = 0
        if st.session_state.pm_workspace_id:
            for idx, (name, ws_id) in enumerate(workspace_options.items()):
                if ws_id == st.session_state.pm_workspace_id:
                    default_idx = idx
                    break

        selected_name = st.selectbox(
            "Workspace",
            options=list(workspace_options.keys()),
            index=default_idx,
            key="pm_ws_select"
        )

        selected_ws_id = workspace_options[selected_name]
        if selected_ws_id != st.session_state.pm_workspace_id:
            st.session_state.pm_workspace_id = selected_ws_id
            if selected_ws_id:
                # Determine phase based on workspace state
                ws = workspace_mgr.get_workspace(selected_ws_id)
                doc_path = ws.workspace_path / "documents" / "tender_original.txt"
                if not doc_path.exists():
                    st.session_state.pm_phase = 1
                else:
                    st.session_state.pm_phase = 2
            else:
                st.session_state.pm_phase = 1
            clear_ic_state()
            st.session_state.pm_ic_loaded_for = None
            st.rerun()
    else:
        st.info("No workspaces yet.")
        st.session_state.pm_workspace_id = None

    st.divider()

    # Progress dashboard
    if st.session_state.pm_workspace_id:
        ws = workspace_mgr.get_workspace(st.session_state.pm_workspace_id)
        if ws:
            report = export_engine.analyze_completeness(st.session_state.pm_workspace_id)

            st.markdown("**Progress**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tier 1", f"{report.tier1_completed}/{report.tier1_total}")
            with col2:
                st.metric("Tier 2", f"{report.tier2_completed}/{report.tier2_total}")

            if report.total_fields > 0:
                st.progress(report.completion_percentage / 100)
                st.caption(f"{report.completion_percentage:.0f}% complete")

            st.divider()

            # Phase navigation
            st.markdown("**Jump to Phase**")
            if st.button("1. Setup", use_container_width=True, key="sb_phase1"):
                st.session_state.pm_phase = 1
                st.rerun()
            doc_path = ws.workspace_path / "documents" / "tender_original.txt"
            if st.button("2. Work", use_container_width=True, disabled=not doc_path.exists(), key="sb_phase2"):
                st.session_state.pm_phase = 2
                st.rerun()
            if st.button("3. Review & Export", use_container_width=True, disabled=not doc_path.exists(), key="sb_phase3"):
                st.session_state.pm_phase = 3
                st.rerun()


# ============================================
# PHASE INDICATOR
# ============================================

phase = st.session_state.pm_phase

def phase_class(n):
    if n == phase:
        return "active"
    elif n < phase:
        return "done"
    return ""

st.markdown(f"""
<div class="phase-indicator">
    <div class="phase-step {phase_class(1)}">1. Setup</div>
    <div class="phase-step {phase_class(2)}">2. Work</div>
    <div class="phase-step {phase_class(3)}">3. Review & Export</div>
</div>
""", unsafe_allow_html=True)


# ============================================
# PHASE 1: SETUP
# ============================================

if phase == 1:
    if st.session_state.pm_workspace_id is None:
        # CREATE NEW WORKSPACE
        st.subheader("Create New Workspace")

        with st.form("pm_create_workspace"):
            col1, col2 = st.columns(2)
            with col1:
                tender_name = st.text_input("Tender Name*", placeholder="Department of Digital Services - Consulting")
                tender_reference = st.text_input("RFT Reference", placeholder="RFT12345")
            with col2:
                workspace_name = st.text_input("Workspace Name*", placeholder="RFT12345 - Digital Services")
                created_by = st.text_input("Your Email", placeholder="user@example.com")

            st.subheader("Entity Profile")
            entities = entity_mgr.list_entity_profiles()
            if entities:
                entity_options = {e.entity_name: e.entity_id for e in entities}
                entity_name = st.selectbox("Select Entity Profile*", options=list(entity_options.keys()))
                entity_id = entity_options[entity_name]
            else:
                st.warning("No entity profiles found. Create one in Entity Profile Manager first.")
                entity_id = None
                entity_name = None

            submit = st.form_submit_button("Create Workspace", type="primary")

            if submit:
                if not all([tender_name, workspace_name, entity_id]):
                    st.error("Please fill in all required fields (*)")
                else:
                    try:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        entity_slug = entity_id.lower().replace(" ", "_")
                        tender_slug = tender_reference.lower().replace(" ", "_") if tender_reference else "tender"
                        ws_id = f"workspace_{tender_slug}_{entity_slug}_{date_str}"

                        workspace_mgr.create_workspace(
                            workspace_id=ws_id,
                            workspace_name=workspace_name,
                            tender_name=tender_name,
                            tender_reference=tender_reference,
                            created_by=created_by
                        )
                        workspace_mgr.bind_entity(ws_id, entity_id, entity_name)
                        st.session_state.pm_workspace_id = ws_id
                        st.success(f"Created workspace: {workspace_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating workspace: {e}")

    else:
        # EXISTING WORKSPACE SETUP
        ws = workspace_mgr.get_workspace(st.session_state.pm_workspace_id)

        st.subheader(f"Setup: {ws.metadata.workspace_name}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Tender: {ws.metadata.tender_name}")
        with col2:
            st.caption(f"Entity: {ws.metadata.entity_name or 'Not bound'}")
        with col3:
            st.caption(f"Created: {ws.metadata.created_at.strftime('%Y-%m-%d')}")

        st.divider()

        # Document upload
        doc_path = ws.workspace_path / "documents" / "tender_original.txt"

        if not doc_path.exists():
            st.markdown("### Upload Tender Document")
            st.info("Upload the tender document to begin processing.")

            uploaded_file = st.file_uploader(
                "Upload Tender Document",
                type=['txt', 'docx', 'pdf'],
                key="pm_file_upload"
            )

            if uploaded_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)

                    with st.spinner("Processing document..."):
                        text = DocumentProcessor.process_document(tmp_path)
                        doc_path.parent.mkdir(parents=True, exist_ok=True)
                        doc_path.write_text(text, encoding='utf-8')

                        ws.metadata.original_filename = uploaded_file.name
                        ws.metadata.document_type = Path(uploaded_file.name).suffix.replace('.', '')
                        workspace_mgr._save_workspace(ws)

                    # Auto-process: chunk + classify fields
                    with st.spinner("Extracting fields and classifying..."):
                        chunks = chunker.create_chunks(text)
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
                                            classified = field_classifier.classify(field_text, context=chunk.title, strict_filter=True)
                                            if not any(cf.field_text == classified.field_text for cf in classified_fields):
                                                classified_fields.append(classified)
                                        break

                        auto_fields = [f for f in classified_fields if f.tier == FieldTier.AUTO_COMPLETE]
                        intel_fields = [f for f in classified_fields if f.tier == FieldTier.INTELLIGENT]

                        questions_by_type = defaultdict(list)
                        for f in intel_fields:
                            qtype = f.question_type or QuestionType.GENERAL
                            questions_by_type[qtype.value].append(f)

                        question_status = {}
                        for f in intel_fields:
                            question_status[f.field_text] = {'status': 'pending', 'response': '', 'evidence': []}

                        st.session_state.pm_ic_fields = classified_fields
                        st.session_state.pm_ic_auto_fields = auto_fields
                        st.session_state.pm_ic_intel_fields = intel_fields
                        st.session_state.pm_ic_questions_by_type = dict(questions_by_type)
                        st.session_state.pm_ic_question_status = question_status
                        st.session_state.pm_ic_evidence_cache = {}
                        st.session_state.pm_ic_loaded_for = st.session_state.pm_workspace_id

                        save_ic_state(st.session_state.pm_workspace_id)

                    # Also initialize chunk mode for mention analysis
                    ws = workspace_mgr.get_workspace(st.session_state.pm_workspace_id)
                    ws.metadata.chunk_mode_enabled = True
                    ws.metadata.total_chunks = len(completable_chunks)
                    ws.metadata.current_chunk_id = 1
                    ws.metadata.chunks_reviewed = 0
                    ws.metadata.analysis_status = "pending"
                    ws.chunks.clear()
                    for chunk in completable_chunks:
                        ws.chunks.append(ChunkProgress(
                            chunk_id=chunk.chunk_id,
                            title=chunk.title,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            status="pending",
                            mentions_found=0,
                            mentions_approved=0
                        ))
                    workspace_mgr._save_workspace(ws)

                    st.success(f"Uploaded and processed: {uploaded_file.name}")
                    st.info(f"Found {len(auto_fields)} template fields (Tier 1) and {len(intel_fields)} substantive questions (Tier 2)")
                    st.session_state.pm_phase = 2
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    logger.error(f"Document processing failed: {e}", exc_info=True)
        else:
            st.success(f"Document uploaded: {ws.metadata.original_filename}")

            with st.expander("View Document Preview"):
                document_text = doc_path.read_text(encoding='utf-8')
                st.text_area("Document", value=document_text[:5000] + ("..." if len(document_text) > 5000 else ""), height=250, disabled=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Replace Document", key="pm_replace_doc"):
                    doc_path.unlink()
                    clear_ic_state()
                    st.session_state.pm_ic_loaded_for = None
                    st.rerun()
            with col2:
                if st.button("Proceed to Phase 2", type="primary", key="pm_goto_phase2"):
                    st.session_state.pm_phase = 2
                    st.rerun()

        # Workspace management
        with st.expander("Workspace Management"):
            st.caption(f"ID: {st.session_state.pm_workspace_id}")
            if st.button("Delete Workspace", key="pm_delete_ws"):
                st.session_state.pm_confirm_delete = True
            if st.session_state.get('pm_confirm_delete', False):
                st.warning("Are you sure? This cannot be undone.")
                if st.button("Yes, Delete", type="primary", key="pm_confirm_del"):
                    workspace_mgr.delete_workspace(st.session_state.pm_workspace_id)
                    st.session_state.pm_workspace_id = None
                    st.session_state.pm_confirm_delete = False
                    st.session_state.pm_phase = 1
                    clear_ic_state()
                    st.rerun()


# ============================================
# PHASE 2: WORK
# ============================================

elif phase == 2:
    ws = workspace_mgr.get_workspace(st.session_state.pm_workspace_id)
    if not ws:
        st.error("Workspace not found")
        st.stop()

    # Load IC state if not already loaded for this workspace
    if st.session_state.pm_ic_loaded_for != st.session_state.pm_workspace_id:
        clear_ic_state()
        if load_ic_state(st.session_state.pm_workspace_id):
            st.toast("Restored previous progress")
        st.session_state.pm_ic_loaded_for = st.session_state.pm_workspace_id

    st.subheader(f"Work: {ws.metadata.workspace_name}")

    tab1, tab2 = st.tabs(["Template Fields (Tier 1)", "AI Assisted (Tier 2)"])

    # ========================================
    # TAB 1: TEMPLATE FIELDS (TIER 1)
    # ========================================

    with tab1:
        entity = entity_mgr.get_entity_profile(ws.metadata.entity_id) if ws.metadata.entity_id else None

        if not entity:
            st.warning("No entity profile bound to this workspace.")
            st.stop()

        # Get approved mentions and their resolved values
        mentions = ws.mentions
        approved_count = sum(1 for m in mentions if m.approved)
        total_mentions = len(mentions)

        # If no mentions yet (batch analysis not run), offer to run it
        if total_mentions == 0 and ws.metadata.analysis_status == "pending":
            st.info("No template fields extracted yet. Run batch analysis to identify @mentions in the document.")

            if st.button("Run Batch Analysis", type="primary", key="pm_run_batch"):
                markup_engine = MarkupEngine(entity_mgr, llm)

                doc_path = ws.workspace_path / "documents" / "tender_original.txt"
                document_text = doc_path.read_text(encoding='utf-8')
                chunks = chunker.create_chunks(document_text)
                completable_chunks = chunker.filter_completable_chunks(chunks)

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.markdown(f"Analyzing chunk {current}/{total}...")

                ws.metadata.analysis_status = "analyzing"
                workspace_mgr._save_workspace(ws)

                results = markup_engine.analyze_all_chunks_batch(completable_chunks, ws.metadata.entity_id, update_progress)

                all_mentions = []
                for chunk_id, chunk_mentions in results.items():
                    all_mentions.extend(chunk_mentions)
                    cp = next((c for c in ws.chunks if c.chunk_id == chunk_id), None)
                    if cp:
                        cp.mentions_found = len(chunk_mentions)

                ws = workspace_mgr.add_mention_bindings(ws.metadata.workspace_id, all_mentions)
                ws.metadata.analysis_status = "complete"
                ws.metadata.total_mentions_found = len(all_mentions)
                workspace_mgr._save_workspace(ws)

                st.success(f"Found {len(all_mentions)} template fields")
                st.rerun()

        elif total_mentions > 0:
            # Progress
            st.progress(approved_count / total_mentions if total_mentions > 0 else 0)
            st.caption(f"{approved_count}/{total_mentions} fields approved")

            # Auto-fill all button
            if st.button("Auto-Fill All from Entity Profile", type="primary", key="pm_autofill"):
                filled = 0
                for mention in mentions:
                    if mention.approved or mention.rejected or mention.ignored:
                        continue
                    # Try to resolve from entity profile
                    value = export_engine.resolve_profile_field(entity, mention.field_path)
                    if value:
                        workspace_mgr.update_mention_binding(
                            ws.metadata.workspace_id,
                            mention.mention_text,
                            approved=True,
                            resolved_value=value,
                            chunk_id=mention.chunk_id,
                            location=mention.location
                        )
                        filled += 1
                if filled > 0:
                    st.success(f"Auto-filled {filled} fields")
                    st.rerun()
                else:
                    st.info("No fields could be auto-filled. Manual review may be needed.")

            st.divider()

            # Group mentions by section/type
            mention_groups = defaultdict(list)
            for mention in mentions:
                # Group by field_path prefix
                prefix = mention.field_path.split('.')[0] if '.' in mention.field_path else "other"
                mention_groups[prefix].append(mention)

            for group_name, group_mentions in mention_groups.items():
                group_approved = sum(1 for m in group_mentions if m.approved or m.rejected or m.ignored)
                group_total = len(group_mentions)
                label = group_name.replace('_', ' ').title()

                with st.expander(f"**{label}** ({group_approved}/{group_total} done)", expanded=(group_approved < group_total)):
                    for idx, mention in enumerate(group_mentions):
                        col1, col2, col3, col4 = st.columns([2, 3, 1, 1])

                        with col1:
                            st.markdown(f"**{mention.mention_text}**")
                            st.caption(mention.field_path)

                        with col2:
                            resolved = mention.resolved_value
                            if not resolved:
                                resolved = export_engine.resolve_profile_field(entity, mention.field_path)

                            if mention.approved:
                                st.success(resolved or "(approved, no value)")
                            elif mention.rejected:
                                st.warning("Rejected")
                            elif mention.ignored:
                                st.info("Skipped")
                            else:
                                st.text(resolved or "(unresolved)")

                        with col3:
                            if not mention.approved and not mention.rejected and not mention.ignored:
                                if st.button("Approve", key=f"pm_t1_approve_{group_name}_{idx}"):
                                    val = resolved or ""
                                    workspace_mgr.update_mention_binding(
                                        ws.metadata.workspace_id,
                                        mention.mention_text,
                                        approved=True,
                                        resolved_value=val,
                                        chunk_id=mention.chunk_id,
                                        location=mention.location
                                    )
                                    st.rerun()

                        with col4:
                            if not mention.approved and not mention.rejected and not mention.ignored:
                                if st.button("Skip", key=f"pm_t1_skip_{group_name}_{idx}"):
                                    workspace_mgr.update_mention_binding(
                                        ws.metadata.workspace_id,
                                        mention.mention_text,
                                        ignored=True,
                                        chunk_id=mention.chunk_id,
                                        location=mention.location
                                    )
                                    st.rerun()
        else:
            st.info("No template fields found. Upload a document first or run batch analysis.")

    # ========================================
    # TAB 2: AI ASSISTED (TIER 2)
    # ========================================

    with tab2:
        if not st.session_state.pm_ic_fields:
            st.info("Fields not yet extracted. Upload a document in Phase 1 to auto-extract, or click Extract below.")

            if st.button("Extract Fields", type="primary", key="pm_extract_fields"):
                doc_path = ws.workspace_path / "documents" / "tender_original.txt"
                if not doc_path.exists():
                    st.error("No document found. Upload in Phase 1.")
                    st.stop()

                with st.spinner("Analyzing document..."):
                    document_text = doc_path.read_text(encoding='utf-8')
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
                                        classified = field_classifier.classify(field_text, context=chunk.title, strict_filter=True)
                                        if not any(cf.field_text == classified.field_text for cf in classified_fields):
                                            classified_fields.append(classified)
                                    break

                    auto_fields = [f for f in classified_fields if f.tier == FieldTier.AUTO_COMPLETE]
                    intel_fields = [f for f in classified_fields if f.tier == FieldTier.INTELLIGENT]

                    questions_by_type = defaultdict(list)
                    for f in intel_fields:
                        qtype = f.question_type or QuestionType.GENERAL
                        questions_by_type[qtype.value].append(f)

                    question_status = {}
                    for f in intel_fields:
                        question_status[f.field_text] = {'status': 'pending', 'response': '', 'evidence': []}

                    st.session_state.pm_ic_fields = classified_fields
                    st.session_state.pm_ic_auto_fields = auto_fields
                    st.session_state.pm_ic_intel_fields = intel_fields
                    st.session_state.pm_ic_questions_by_type = dict(questions_by_type)
                    st.session_state.pm_ic_question_status = question_status
                    st.session_state.pm_ic_evidence_cache = {}
                    st.session_state.pm_ic_loaded_for = st.session_state.pm_workspace_id

                    save_ic_state(st.session_state.pm_workspace_id)
                    st.rerun()
        else:
            # Sidebar-like config inline
            intel_fields = st.session_state.pm_ic_intel_fields
            questions_by_type = st.session_state.pm_ic_questions_by_type
            question_status = st.session_state.pm_ic_question_status

            total_q = len(intel_fields)
            completed_q = sum(1 for s in question_status.values() if s['status'] == 'completed')

            # Config row
            config_col1, config_col2, config_col3 = st.columns([2, 2, 1])

            with config_col1:
                collections = collection_manager.get_collection_names()
                collection_options = ["-- Entire Knowledge Base --"] + (collections if collections else [])
                selected_option = st.selectbox("Evidence Source", options=collection_options, key="pm_ic_collection")
                selected_collection = None if selected_option == "-- Entire Knowledge Base --" else selected_option

            with config_col2:
                creativity_level = st.select_slider(
                    "Style",
                    options=[0, 1, 2],
                    value=1,
                    format_func=lambda x: {0: "Factual", 1: "Balanced", 2: "Creative"}[x],
                    key="pm_ic_creativity"
                )
                temperature_map = {0: 0.3, 1: 0.7, 2: 1.0}

            with config_col3:
                max_evidence = st.number_input("Evidence per Q", min_value=2, max_value=10, value=5, key="pm_ic_max_ev")

            # Progress
            st.progress(completed_q / total_q if total_q > 0 else 0)
            st.caption(f"{completed_q}/{total_q} questions completed")

            # Re-extract option
            with st.expander("Extraction Settings"):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.caption(f"Found {len(st.session_state.pm_ic_fields)} fields total")
                with c2:
                    if st.button("Re-extract", key="pm_reextract"):
                        clear_ic_state()
                        workspace_mgr.clear_ic_completion_state(st.session_state.pm_workspace_id)
                        st.session_state.pm_ic_loaded_for = None
                        st.rerun()

            st.divider()

            # Initialize evidence retriever and response generator
            evidence_retriever = EvidenceRetriever(db_path)
            llm.temperature = temperature_map[creativity_level]
            response_generator = ResponseGenerator(llm, entity_mgr)

            # Question type display names
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

            # Render questions by type
            for qtype in QuestionType:
                qtype_key = qtype.value
                if qtype_key not in questions_by_type or not questions_by_type[qtype_key]:
                    continue

                type_questions = questions_by_type[qtype_key]
                display_name = QTYPE_DISPLAY.get(qtype, qtype.value.title())

                type_completed = sum(1 for q in type_questions if question_status.get(q.field_text, {}).get('status') == 'completed')
                type_total = len(type_questions)

                with st.expander(f"**{display_name}** -- {type_completed}/{type_total} complete", expanded=(type_completed < type_total)):
                    for q_idx, fld in enumerate(type_questions):
                        field_key = fld.field_text
                        status_data = question_status.get(field_key, {'status': 'pending', 'response': '', 'evidence': []})
                        current_status = status_data['status']

                        card_class = "completed" if current_status == 'completed' else ("editing" if current_status == 'editing' else "")
                        word_limit_text = f'Word limit: {fld.word_limit}' if fld.word_limit else ''

                        st.markdown(f'<div class="question-card {card_class}"><div class="question-number">Question {q_idx + 1}</div><div class="question-text">{fld.field_text}</div><div style="font-size:0.8rem;color:#888;">{word_limit_text}<span style="float:right;font-size:0.7rem;font-weight:600;text-transform:uppercase;">{current_status}</span></div></div>', unsafe_allow_html=True)

                        if current_status == 'pending':
                            btn_col1, creat_col, btn_col2, spacer = st.columns([1, 1.2, 1, 2])

                            with btn_col1:
                                if st.button("Edit", key=f"pm_edit_{qtype_key}_{q_idx}", type="secondary"):
                                    question_status[field_key]['status'] = 'editing'
                                    st.session_state.pm_ic_question_status = question_status
                                    try:
                                        ev_result = evidence_retriever.find_evidence(
                                            question=fld.field_text,
                                            question_type=fld.question_type or QuestionType.GENERAL,
                                            collection_name=selected_collection,
                                            max_results=max_evidence,
                                            use_reranker=True
                                        )
                                        st.session_state.pm_ic_evidence_cache[field_key] = ev_result.evidence
                                    except Exception:
                                        pass
                                    save_ic_state(st.session_state.pm_workspace_id)
                                    st.rerun()

                            with creat_col:
                                q_creativity = st.selectbox(
                                    "Style",
                                    options=["Factual", "Balanced", "Creative"],
                                    index=1,
                                    key=f"pm_creat_{qtype_key}_{q_idx}",
                                    label_visibility="collapsed"
                                )
                                q_temp = {"Factual": 0.3, "Balanced": 0.7, "Creative": 1.0}[q_creativity]

                            with btn_col2:
                                if st.button("Generate", key=f"pm_gen_{qtype_key}_{q_idx}", type="primary"):
                                    with st.spinner("Generating..."):
                                        try:
                                            if field_key not in st.session_state.pm_ic_evidence_cache:
                                                ev_result = evidence_retriever.find_evidence(
                                                    question=fld.field_text,
                                                    question_type=fld.question_type or QuestionType.GENERAL,
                                                    collection_name=selected_collection,
                                                    max_results=max_evidence,
                                                    use_reranker=True
                                                )
                                                st.session_state.pm_ic_evidence_cache[field_key] = ev_result.evidence

                                            evidence = st.session_state.pm_ic_evidence_cache.get(field_key, [])
                                            llm.temperature = q_temp

                                            response = response_generator.generate(
                                                classified_field=fld,
                                                evidence=evidence,
                                                entity_id=ws.metadata.entity_id
                                            )

                                            question_status[field_key]['status'] = 'completed'
                                            question_status[field_key]['response'] = response.text
                                            question_status[field_key]['evidence'] = evidence
                                            question_status[field_key]['creativity_level'] = q_creativity
                                            st.session_state.pm_ic_question_status = question_status
                                            save_ic_state(st.session_state.pm_workspace_id)
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Generation failed: {e}")

                        elif current_status in ['completed', 'editing']:
                            response_text = st.text_area(
                                "Response",
                                value=status_data.get('response', ''),
                                height=180,
                                key=f"pm_resp_{qtype_key}_{q_idx}",
                                label_visibility="collapsed"
                            )

                            if response_text != status_data.get('response', ''):
                                question_status[field_key]['response'] = response_text
                                question_status[field_key]['status'] = 'completed'
                                st.session_state.pm_ic_question_status = question_status
                                save_ic_state(st.session_state.pm_workspace_id)

                            wc_col, act_col1, act_col2, act_col3 = st.columns([2, 1, 1, 1])

                            with wc_col:
                                wc = len(response_text.split()) if response_text else 0
                                limit_info = f" / {fld.word_limit}" if fld.word_limit else ""
                                st.caption(f"{wc} words{limit_info}")

                            with act_col1:
                                st.download_button(
                                    "Export",
                                    data=f"# {fld.field_text}\n\n{response_text}",
                                    file_name=f"response_{q_idx + 1}.txt",
                                    mime="text/plain",
                                    key=f"pm_exp_{qtype_key}_{q_idx}"
                                )

                            # Evidence toggle
                            show_ev_key = f"pm_show_ev_{qtype_key}_{q_idx}"
                            if show_ev_key not in st.session_state:
                                st.session_state[show_ev_key] = False

                            with act_col2:
                                if st.button("Evidence", key=f"pm_ev_btn_{qtype_key}_{q_idx}", type="secondary"):
                                    st.session_state[show_ev_key] = not st.session_state[show_ev_key]
                                    st.rerun()

                            # Refine toggle
                            show_ref_key = f"pm_show_ref_{qtype_key}_{q_idx}"
                            if show_ref_key not in st.session_state:
                                st.session_state[show_ref_key] = False

                            with act_col3:
                                if st.button("Refine", key=f"pm_ref_btn_{qtype_key}_{q_idx}", type="secondary"):
                                    st.session_state[show_ref_key] = not st.session_state[show_ref_key]
                                    st.rerun()

                            # Evidence panel
                            cached_ev = st.session_state.pm_ic_evidence_cache.get(field_key, [])
                            if st.session_state[show_ev_key] and cached_ev:
                                st.markdown("---")
                                st.caption(f"**Evidence** ({len(cached_ev)} sources)")
                                for ev in cached_ev[:3]:
                                    ev_text = ev.text[:350] + ('...' if len(ev.text) > 350 else '')
                                    st.markdown(f'<div class="evidence-card"><div class="evidence-source">{ev.source_doc} ({ev.relevance_score:.0%})</div>{ev_text}</div>', unsafe_allow_html=True)

                            # Refine panel
                            if st.session_state[show_ref_key]:
                                st.markdown("---")
                                st.caption("**Refine with AI Guidance**")
                                hint_text = st.text_area(
                                    "Guidance",
                                    placeholder="e.g., Focus on ISO certification, add specific metrics...",
                                    key=f"pm_hint_{qtype_key}_{q_idx}",
                                    height=80,
                                    label_visibility="collapsed"
                                )

                                if st.button("Regenerate", key=f"pm_regen_{qtype_key}_{q_idx}", type="primary"):
                                    with st.spinner("Regenerating..."):
                                        try:
                                            ev_result = evidence_retriever.find_evidence(
                                                question=fld.field_text,
                                                question_type=fld.question_type or QuestionType.GENERAL,
                                                collection_name=selected_collection,
                                                max_results=max_evidence,
                                                use_reranker=True
                                            )
                                            evidence = ev_result.evidence
                                            st.session_state.pm_ic_evidence_cache[field_key] = evidence

                                            previous_response = DraftResponse(
                                                question=fld.field_text,
                                                question_type=fld.question_type or QuestionType.GENERAL,
                                                text=status_data.get('response', ''),
                                                evidence_used=evidence,
                                                confidence=0.5,
                                                word_count=len(status_data.get('response', '').split()),
                                                needs_review=True,
                                                placeholders=[],
                                                generation_time=0,
                                                metadata={'entity_id': ws.metadata.entity_id, 'word_limit': fld.word_limit}
                                            )

                                            guidance = hint_text if hint_text else "Improve with more specific details from evidence."
                                            new_response = response_generator.regenerate(
                                                previous_response=previous_response,
                                                entity_id=ws.metadata.entity_id,
                                                additional_guidance=guidance,
                                                new_evidence=evidence
                                            )

                                            question_status[field_key]['response'] = new_response.text
                                            question_status[field_key]['confidence'] = new_response.confidence
                                            st.session_state.pm_ic_question_status = question_status
                                            save_ic_state(st.session_state.pm_workspace_id)
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Regeneration failed: {e}")

                        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Phase navigation
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Setup", key="pm_back_setup"):
            st.session_state.pm_phase = 1
            st.rerun()
    with col2:
        if st.button("Proceed to Review & Export", type="primary", key="pm_goto_phase3"):
            st.session_state.pm_phase = 3
            st.rerun()


# ============================================
# PHASE 3: REVIEW & EXPORT
# ============================================

elif phase == 3:
    ws = workspace_mgr.get_workspace(st.session_state.pm_workspace_id)
    if not ws:
        st.error("Workspace not found")
        st.stop()

    st.subheader(f"Review & Export: {ws.metadata.workspace_name}")

    # Completeness dashboard
    report = export_engine.analyze_completeness(st.session_state.pm_workspace_id)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall", f"{report.completion_percentage:.0f}%")
    with col2:
        st.metric("Tier 1 Fields", f"{report.tier1_completed}/{report.tier1_total}")
    with col3:
        st.metric("Tier 2 Questions", f"{report.tier2_completed}/{report.tier2_total}")
    with col4:
        st.metric("Quality Flags", len(report.quality_flags))

    st.progress(report.completion_percentage / 100)

    # Missing items
    pending_items = [i for i in report.items if i.status == "pending"]
    if pending_items:
        with st.expander(f"Missing Items ({len(pending_items)})", expanded=True):
            for item in pending_items:
                tier_label = "Tier 1" if item.tier == "tier1" else "Tier 2"
                st.markdown(f"- **[{tier_label}]** {item.field_text}")

    # Quality flags
    if report.quality_flags:
        with st.expander(f"Quality Flags ({len(report.quality_flags)})"):
            for flag in report.quality_flags:
                st.warning(flag)

    st.divider()

    # Side-by-side comparison
    st.markdown("### Document Comparison")

    doc_path = ws.workspace_path / "documents" / "tender_original.txt"
    if doc_path.exists():
        original_text = doc_path.read_text(encoding='utf-8')

        entity = entity_mgr.get_entity_profile(ws.metadata.entity_id) if ws.metadata.entity_id else None

        # Get IC responses
        ic_responses = {}
        ic_state = workspace_mgr.get_ic_completion_state(st.session_state.pm_workspace_id)
        if ic_state:
            for field_text, qs in ic_state.get('question_status', {}).items():
                if qs.get('status') == 'completed' and qs.get('response'):
                    ic_responses[field_text] = qs['response']

        completed_text = export_engine.build_completed_document(
            original_text, ws.mentions, ic_responses, entity
        )

        view_col1, view_col2 = st.columns(2)
        with view_col1:
            st.markdown("**Original**")
            st.text_area("original", value=original_text[:5000], height=400, disabled=True, label_visibility="collapsed")
        with view_col2:
            st.markdown("**Completed**")
            st.text_area("completed", value=completed_text[:5000], height=400, disabled=True, label_visibility="collapsed")

    st.divider()

    # Export options
    st.markdown("### Export")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        export_format = st.selectbox("Format", ["Markdown", "DOCX"], key="pm_export_fmt")
    with exp_col2:
        include_citations = st.checkbox("Include citations", value=False, key="pm_export_cite")
    with exp_col3:
        flag_incomplete = st.checkbox("Flag incomplete items", value=True, key="pm_export_flag")

    if export_format == "Markdown":
        md_content = export_engine.generate_export_markdown(
            st.session_state.pm_workspace_id,
            include_citations=include_citations,
            flag_incomplete=flag_incomplete
        )
        st.download_button(
            "Download Markdown",
            data=md_content,
            file_name=f"proposal_{ws.metadata.workspace_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            type="primary",
            key="pm_dl_md"
        )
    else:
        docx_bytes = export_engine.generate_export_docx(
            st.session_state.pm_workspace_id,
            include_citations=include_citations,
            flag_incomplete=flag_incomplete
        )
        if docx_bytes:
            st.download_button(
                "Download DOCX",
                data=docx_bytes,
                file_name=f"proposal_{ws.metadata.workspace_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                key="pm_dl_docx"
            )
        else:
            st.warning("DOCX export requires python-docx. Install with: pip install python-docx")

    # Back button
    st.divider()
    if st.button("Back to Work", key="pm_back_work"):
        st.session_state.pm_phase = 2
        st.rerun()


# Footer
st.divider()
st.caption("v1.0.0 | Proposal Manager - Consolidated proposal workflow")
