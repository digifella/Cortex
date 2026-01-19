"""
Proposal Intelligent Completion
Version: 1.0.0
Date: 2026-01-19

Purpose: Two-tier intelligent proposal completion workflow.
- Tier 1: Auto-complete simple fields from entity profile
- Tier 2: Generate substantive responses using knowledge collection + LLM

Key Features:
- Field classification (auto-complete vs intelligent)
- Knowledge collection selection for evidence
- Draft response generation with evidence panel
- Confidence scoring and regeneration support
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

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
from cortex_engine.response_generator import (
    ResponseGenerator, DraftResponse, BatchResponseGenerator
)

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

    .confidence-high {
        color: #2E7D32;
    }

    .confidence-medium {
        color: #F57C00;
    }

    .confidence-low {
        color: #C62828;
    }

    .evidence-card {
        background: #FAFAFA;
        border-left: 3px solid #2D5F4F;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .auto-field-row {
        display: flex;
        padding: 0.5rem 0;
        border-bottom: 1px solid #E0E0E0;
    }

    .field-label {
        font-weight: 500;
        color: #555;
        min-width: 200px;
    }

    .field-value {
        color: #2D5F4F;
        font-family: monospace;
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
    .qtype-general { background: #ECEFF1; color: #546E7A; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🧠 Intelligent Proposal Completion</div>', unsafe_allow_html=True)
st.caption("Two-tier processing: auto-complete simple fields, generate intelligent responses for substantive questions")

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
        st.error("❌ No entity bound to workspace")
        st.stop()

    # Show entity info
    entity = entity_manager.get_entity_profile(workspace.metadata.entity_id)
    if entity:
        st.success(f"✅ Entity: {getattr(entity, 'company_name', workspace.metadata.entity_id)}")

    st.divider()

    # Knowledge collection selection
    st.subheader("📚 Knowledge Collection")
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
            st.caption(f"📄 {doc_count} documents in collection")

    st.divider()

    # Processing options
    st.subheader("⚙️ Options")
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
    st.error("❌ No document file found in workspace")
    st.stop()

# Load document
try:
    with open(doc_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
except Exception as e:
    st.error(f"❌ Error reading document: {str(e)}")
    st.stop()

# ============================
# STEP 1: Field Extraction & Classification
# ============================

if 'ic_classified_fields' not in st.session_state:
    st.session_state.ic_classified_fields = None
    st.session_state.ic_auto_complete_fields = []
    st.session_state.ic_intelligent_fields = []
    st.session_state.ic_draft_responses = []

# Extract fields button
st.subheader("Step 1: Extract & Classify Fields")

col1, col2 = st.columns([3, 1])

with col1:
    st.info("""
    This will scan the document for fillable fields and classify them:
    - **Auto-Complete**: Simple fields (name, ABN, address) filled from entity profile
    - **Intelligent**: Substantive questions requiring evidence-based responses
    """)

with col2:
    if st.button("🔍 Extract Fields", type="primary", use_container_width=True):
        with st.spinner("Extracting and classifying fields..."):
            # Extract fields using document chunker to find completable sections
            chunks = chunker.create_chunks(document_text)
            completable_chunks = chunker.filter_completable_chunks(chunks)

            # For each chunk, extract field-like content and classify
            classified_fields = []

            for chunk in completable_chunks:
                # Look for field patterns in chunk content
                import re

                # Patterns that indicate fillable fields
                field_patterns = [
                    # Label: [blank] patterns
                    r'^([A-Za-z][A-Za-z\s\(\)]+):\s*$',
                    # Question patterns
                    r'^([A-Z][^?]+\?)\s*$',
                    # "Please provide" patterns
                    r'((?:Please\s+)?(?:provide|describe|detail|outline|explain)[^.]+\.)',
                    # Field with placeholder
                    r'^([A-Za-z][A-Za-z\s]+):\s*[\[\<\_]',
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
                                    context=chunk.title
                                )
                                # Avoid duplicates
                                if not any(cf.field_text == classified.field_text for cf in classified_fields):
                                    classified_fields.append(classified)
                            break

            # Separate by tier
            auto_fields = [f for f in classified_fields if f.tier == FieldTier.AUTO_COMPLETE]
            intel_fields = [f for f in classified_fields if f.tier == FieldTier.INTELLIGENT]

            st.session_state.ic_classified_fields = classified_fields
            st.session_state.ic_auto_complete_fields = auto_fields
            st.session_state.ic_intelligent_fields = intel_fields

            st.rerun()

# Show classification results
if st.session_state.ic_classified_fields:
    auto_fields = st.session_state.ic_auto_complete_fields
    intel_fields = st.session_state.ic_intelligent_fields

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fields", len(st.session_state.ic_classified_fields))
    col2.metric("Auto-Complete", len(auto_fields), help="Simple fields from entity profile")
    col3.metric("Intelligent", len(intel_fields), help="Need evidence-based responses")

    st.divider()

    # ============================
    # STEP 2: Auto-Complete Fields
    # ============================

    st.subheader("Step 2: Auto-Complete Fields")

    if auto_fields:
        st.success(f"✅ {len(auto_fields)} fields can be auto-completed from entity profile")

        with st.expander("📋 Review Auto-Completed Fields", expanded=True):
            # Build a table of auto-completed values
            for field in auto_fields:
                col1, col2, col3 = st.columns([2, 3, 1])

                with col1:
                    st.markdown(f"**{field.field_text}**")

                with col2:
                    # Get value from entity profile
                    profile_field = field.auto_complete_mapping
                    value = "—"

                    if entity and profile_field:
                        # Try to get the value from entity
                        value = getattr(entity, profile_field, None)
                        if value is None and hasattr(entity, 'data'):
                            value = entity.data.get(profile_field)
                        if value is None:
                            value = f"[Not set: {profile_field}]"

                    st.code(str(value) if value else "—", language=None)

                with col3:
                    st.caption(f"→ {profile_field}")

    else:
        st.info("No auto-complete fields found")

    st.divider()

    # ============================
    # STEP 3: Intelligent Response Generation
    # ============================

    st.subheader("Step 3: Generate Intelligent Responses")

    if not intel_fields:
        st.info("No substantive questions found requiring intelligent completion")
    elif not selected_collection:
        st.warning("⚠️ Select a knowledge collection in the sidebar to generate responses")
    else:
        st.info(f"""
        **{len(intel_fields)} substantive questions** will be answered using:
        - Evidence from collection: **{selected_collection}**
        - LLM: qwen2.5:72b-instruct-q4_K_M
        """)

        # Generate button
        if st.button("🚀 Generate Draft Responses", type="primary", use_container_width=True):
            # Initialize evidence retriever
            evidence_retriever = EvidenceRetriever(db_path)
            response_generator = ResponseGenerator(llm, entity_manager)

            progress_bar = st.progress(0)
            status_text = st.empty()

            draft_responses = []
            total = len(intel_fields)

            for i, field in enumerate(intel_fields):
                progress = (i + 1) / total
                progress_bar.progress(progress)
                status_text.markdown(f"**Processing {i+1}/{total}:** {field.field_text[:50]}...")

                try:
                    # Retrieve evidence
                    evidence_result = evidence_retriever.find_evidence(
                        question=field.field_text,
                        question_type=field.question_type or QuestionType.GENERAL,
                        collection_name=selected_collection,
                        max_results=max_evidence,
                        use_reranker=use_reranker
                    )

                    # Generate response
                    response = response_generator.generate(
                        classified_field=field,
                        evidence=evidence_result.evidence,
                        entity_id=workspace.metadata.entity_id
                    )

                    draft_responses.append(response)

                except Exception as e:
                    st.error(f"Failed on field: {field.field_text[:50]}... - {str(e)}")
                    # Create placeholder response
                    draft_responses.append(DraftResponse(
                        question=field.field_text,
                        question_type=field.question_type or QuestionType.GENERAL,
                        text=f"[Generation failed: {str(e)}]",
                        evidence_used=[],
                        confidence=0.0,
                        word_count=0,
                        needs_review=True,
                        placeholders=["[Generation failed]"],
                        generation_time=0,
                        metadata={}
                    ))

            progress_bar.progress(1.0)
            status_text.markdown("**✅ Generation complete!**")

            st.session_state.ic_draft_responses = draft_responses
            st.rerun()

        # Show generated responses
        if st.session_state.ic_draft_responses:
            st.divider()
            st.subheader("📝 Draft Responses")

            responses = st.session_state.ic_draft_responses

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Responses", len(responses))
            col2.metric("High Confidence", sum(1 for r in responses if r.confidence >= 0.7))
            col3.metric("Needs Review", sum(1 for r in responses if r.needs_review))
            col4.metric("With Placeholders", sum(1 for r in responses if r.placeholders))

            st.divider()

            # Individual response cards
            for idx, response in enumerate(responses):
                # Determine confidence color
                if response.confidence >= 0.7:
                    conf_class = "confidence-high"
                    conf_icon = "🟢"
                elif response.confidence >= 0.4:
                    conf_class = "confidence-medium"
                    conf_icon = "🟡"
                else:
                    conf_class = "confidence-low"
                    conf_icon = "🔴"

                # Question type badge
                qtype = response.question_type.value if response.question_type else "general"
                qtype_class = f"qtype-{qtype}" if qtype in ['capability', 'methodology', 'value', 'compliance'] else "qtype-general"

                with st.container(border=True):
                    # Header row
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"""
                        <span class="question-type-badge {qtype_class}">{qtype.upper()}</span>
                        **{response.question[:80]}{'...' if len(response.question) > 80 else ''}**
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"{conf_icon} **{response.confidence:.0%}** confidence")

                    # Response text (editable)
                    edited_text = st.text_area(
                        "Draft Response",
                        value=response.text,
                        height=200,
                        key=f"response_text_{idx}",
                        label_visibility="collapsed"
                    )

                    # Placeholders warning
                    if response.placeholders:
                        st.warning(f"⚠️ Placeholders found: {', '.join(response.placeholders)}")

                    # Evidence panel
                    if response.evidence_used:
                        with st.expander(f"📚 Evidence Sources ({len(response.evidence_used)})", expanded=False):
                            for ev_idx, evidence in enumerate(response.evidence_used):
                                st.markdown(f"""
                                <div class="evidence-card">
                                    <strong>{evidence.source_doc}</strong>
                                    <span style="float:right; color:#888;">Relevance: {evidence.relevance_score:.0%}</span>
                                    <br><br>
                                    {evidence.text[:500]}{'...' if len(evidence.text) > 500 else ''}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.caption("No evidence used - response may be generic")

                    # Action buttons
                    col1, col2, col3 = st.columns([1, 1, 2])

                    with col1:
                        if st.button("✅ Approve", key=f"approve_{idx}", use_container_width=True):
                            st.success("Approved!")

                    with col2:
                        if st.button("🔄 Regenerate", key=f"regen_{idx}", use_container_width=True):
                            st.info("Regeneration not yet implemented")

                    with col3:
                        st.caption(f"📊 {response.word_count} words • ⏱️ {response.generation_time:.1f}s")

                    st.divider()

# Footer
st.divider()
st.caption("🧠 Intelligent Proposal Completion • Evidence-backed responses from your knowledge collection")
