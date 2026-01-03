"""
Proposal Workflow - Complete Tender Response System
Version: 1.0.0
Date: 2026-01-03

Purpose: Complete workflow for responding to tenders with structured data extraction
and auto-fill capabilities.

Workflow:
1. Select tender document (file picker)
2. Select source documents for extraction (file picker)
3. Extract structured data from sources
4. Match tender fields to extracted data (Phase 2)
5. Review and approve matches
6. Fill and export completed tender (Phase 3)
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import asyncio
import tempfile
import shutil

# Set page config
st.set_page_config(
    page_title="Proposal Workflow",
    page_icon="📝",
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortex_engine.entity_manager import EntityManager, EntityProfile, EntityType
from cortex_engine.tender_data_extractor import TenderDataExtractor
from cortex_engine.adaptive_model_manager import AdaptiveModelManager
from cortex_engine.config_manager import ConfigManager
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.utils import (
    convert_windows_to_wsl_path,
    get_logger
)
import chromadb
from chromadb.config import Settings as ChromaSettings
import networkx as nx
import docx

# Apply theme
apply_theme()

# Set up logging
logger = get_logger(__name__)

# Page version
PAGE_VERSION = "v1.0.0"


def initialize_session_state():
    """Initialize session state variables."""
    if 'tender_document' not in st.session_state:
        st.session_state.tender_document = None

    if 'tender_filename' not in st.session_state:
        st.session_state.tender_filename = None

    if 'source_documents' not in st.session_state:
        st.session_state.source_documents = []

    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None

    if 'selected_entity' not in st.session_state:
        st.session_state.selected_entity = None

    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = 1


def render_workflow_progress(current_step: int):
    """Render workflow progress indicator."""
    steps = [
        "1️⃣ Select Tender",
        "2️⃣ Select Sources",
        "3️⃣ Extract Data",
        "4️⃣ Match Fields",
        "5️⃣ Fill & Export"
    ]

    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current_step:
                st.success(f"✅ {step_name}")
            elif i == current_step:
                st.info(f"▶️ {step_name}")
            else:
                st.caption(f"⚪ {step_name}")


def render_step1_select_tender():
    """Step 1: Select tender document to fill out."""
    section_header("📄", "Step 1: Select Tender Document", "Choose the RFT/RFQ you need to complete")

    st.markdown("""
    Upload the tender document you need to respond to (e.g., RFT12493.docx).
    This is the document with blank fields that need to be filled.
    """)

    # Option 1: Upload tender file
    st.markdown("### 📤 Upload Tender Document")
    uploaded_tender = st.file_uploader(
        "Choose tender document (.docx)",
        type=['docx'],
        key="tender_uploader",
        help="Select the tender/RFQ/RFT document you need to complete"
    )

    if uploaded_tender:
        # Save to session state
        st.session_state.tender_filename = uploaded_tender.name

        # Load document
        try:
            tender_doc = docx.Document(uploaded_tender)
            st.session_state.tender_document = tender_doc

            # Show document info
            st.success(f"✅ Loaded: {uploaded_tender.name}")

            # Count sections
            para_count = len(tender_doc.paragraphs)
            table_count = len(tender_doc.tables)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Paragraphs", para_count)
            with col2:
                st.metric("Tables", table_count)

            # Preview
            with st.expander("📖 Preview First Few Paragraphs"):
                for i, para in enumerate(tender_doc.paragraphs[:10]):
                    if para.text.strip():
                        st.caption(f"{i+1}. {para.text[:200]}...")

            # Next step button
            if st.button("➡️ Continue to Select Sources", type="primary"):
                st.session_state.workflow_step = 2
                st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to load document: {str(e)}")
            logger.error(f"Tender document load failed: {e}", exc_info=True)

    # Option 2: Use entity's saved tender template
    st.divider()
    st.markdown("### 🏢 Or Use Entity Template")
    st.info("💡 **Coming Soon:** Select from saved tender templates in your entity profiles")


def render_step2_select_sources():
    """Step 2: Select source documents for data extraction."""
    section_header("📁", "Step 2: Select Source Documents", "Choose documents to extract company data from")

    st.markdown(f"""
    **Selected Tender:** {st.session_state.tender_filename}

    Now select the source documents containing your company information:
    - Company registration (ABN, ACN)
    - Insurance certificates
    - Team CVs/qualifications
    - Project case studies
    - References
    """)

    # Option 1: Upload source files
    st.markdown("### 📤 Upload Source Documents")
    uploaded_sources = st.file_uploader(
        "Choose source documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        key="source_uploader",
        help="Select documents containing company information to extract"
    )

    if uploaded_sources:
        st.success(f"✅ Selected {len(uploaded_sources)} source documents")

        # Show list
        with st.expander("📋 Source Documents"):
            for doc in uploaded_sources:
                st.caption(f"📄 {doc.name} ({doc.size} bytes)")

        st.session_state.source_documents = uploaded_sources

    # Option 2: Use pre-configured entity
    st.divider()
    st.markdown("### 🏢 Or Use Entity Profile")

    # Get config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    db_path = config.get('ai_database_path')

    if db_path:
        try:
            wsl_db_path = convert_windows_to_wsl_path(db_path)
            entity_manager = EntityManager(Path(wsl_db_path))
            entities = entity_manager.list_entities()

            if entities:
                entity_names = [e.entity_name for e in entities]
                selected_entity_name = st.selectbox(
                    "Select entity with pre-configured sources",
                    options=["None"] + entity_names,
                    help="Use documents from a pre-configured entity profile"
                )

                if selected_entity_name != "None":
                    entity = next(e for e in entities if e.entity_name == selected_entity_name)
                    st.session_state.selected_entity = entity

                    st.info(f"""
                    ✅ Using **{entity.entity_name}**
                    - Source folders: {len(entity.source_folders)}
                    - Source documents: {entity.source_document_count}
                    - Last extracted: {entity.last_extracted.strftime('%Y-%m-%d') if entity.last_extracted else 'Never'}
                    """)
            else:
                st.warning("No entities configured. Go to **Proposal Entity Manager** to create one.")

        except Exception as e:
            logger.warning(f"Could not load entities: {e}")
            st.warning("Could not load entity profiles")

    # Navigation
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back to Tender Selection"):
            st.session_state.workflow_step = 1
            st.rerun()

    with col2:
        can_continue = (
            (uploaded_sources and len(uploaded_sources) > 0) or
            st.session_state.selected_entity is not None
        )

        if st.button("➡️ Continue to Extract Data", type="primary", disabled=not can_continue):
            st.session_state.workflow_step = 3
            st.rerun()

        if not can_continue:
            st.caption("⚠️ Please select source documents or choose an entity")


def render_step3_extract_data():
    """Step 3: Extract structured data from source documents."""
    section_header("🔍", "Step 3: Extract Structured Data", "Extract company information from sources")

    st.markdown(f"""
    **Selected Tender:** {st.session_state.tender_filename}

    **Sources:** {len(st.session_state.source_documents) if st.session_state.source_documents else 'Entity profile'}
    """)

    # Show what will be extracted
    st.markdown("### 📊 Extraction Target")
    st.info("""
    The system will extract:
    - 🏢 Organization details (ABN, ACN, address, contact)
    - 🛡️ Insurance policies (policy numbers, coverage, expiry)
    - 🎓 Team qualifications and certifications
    - 💼 Work experience and employment history
    - 🚀 Project experience and case studies
    - 📞 Client references
    - ⭐ Organizational capabilities
    """)

    # Extract button
    if st.button("🚀 Extract Structured Data", type="primary", key="extract_data_button"):
        with st.spinner("🔍 Extracting structured data..."):
            try:
                # Get config
                config_manager = ConfigManager()
                config = config_manager.get_config()
                db_path = config.get('ai_database_path')
                wsl_db_path = convert_windows_to_wsl_path(db_path)

                # Initialize extractor
                chroma_db_path = str(Path(wsl_db_path) / "knowledge_hub_db")
                db_settings = ChromaSettings(anonymized_telemetry=False)
                chroma_client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
                collection = chroma_client.get_collection(config.get('collection_name', 'knowledge_hub'))

                # Load knowledge graph
                graph_path = Path(wsl_db_path) / "knowledge_cortex.gpickle"
                knowledge_graph = nx.read_gpickle(graph_path) if graph_path.exists() else None

                model_manager = AdaptiveModelManager()
                extractor = TenderDataExtractor(
                    vector_index=collection,
                    knowledge_graph=knowledge_graph,
                    model_manager=model_manager,
                    db_path=Path(wsl_db_path)
                )

                # Progress tracking
                progress_placeholder = st.empty()

                def progress_callback(message):
                    progress_placeholder.info(message)

                # Run extraction
                if st.session_state.selected_entity:
                    # Use entity's pre-configured documents
                    entity = st.session_state.selected_entity
                    structured_data = asyncio.run(
                        extractor.extract_all_structured_data(
                            progress_callback=progress_callback,
                            entity_id=entity.entity_id,
                            document_ids=entity.source_document_ids
                        )
                    )
                else:
                    # Use uploaded source documents
                    # TODO: Ingest uploaded documents to temp collection first
                    st.warning("⚠️ Direct file upload extraction not yet implemented. Please use entity profiles for now.")
                    return

                # Save to session state
                st.session_state.extracted_data = structured_data

                # Show results
                st.success("✅ Extraction complete!")

                stats = structured_data.summary_stats
                st.markdown("### 📊 Extraction Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Organization", "✓" if structured_data.organization else "✗")
                    st.metric("Insurances", stats.get('insurances', 0))

                with col2:
                    st.metric("Qualifications", stats.get('qualifications', 0))
                    st.metric("Work Experience", stats.get('work_experiences', 0))

                with col3:
                    st.metric("Projects", stats.get('projects', 0))
                    st.metric("References", stats.get('references', 0))

                with col4:
                    st.metric("Capabilities", stats.get('capabilities', 0))

                # Next step
                st.info("✨ Ready to match tender fields! Click 'Continue' below.")

            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
                logger.error(f"Extraction failed: {e}", exc_info=True)

    # Show extracted data if available
    if st.session_state.extracted_data:
        with st.expander("📋 View Extracted Data"):
            data = st.session_state.extracted_data

            if data.organization:
                st.markdown("**Organization:**")
                st.json({
                    "legal_name": data.organization.legal_name,
                    "abn": data.organization.abn,
                    "address": data.organization.address
                })

            if data.insurances:
                st.markdown(f"**Insurances ({len(data.insurances)}):**")
                for ins in data.insurances[:3]:
                    st.caption(f"• {ins.insurance_type.value}: {ins.policy_number} (Expires: {ins.expiry_date})")

    # Navigation
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back to Source Selection"):
            st.session_state.workflow_step = 2
            st.rerun()

    with col2:
        can_continue = st.session_state.extracted_data is not None

        if st.button("➡️ Continue to Match Fields", type="primary", disabled=not can_continue):
            st.session_state.workflow_step = 4
            st.rerun()

        if not can_continue:
            st.caption("⚠️ Please extract data first")


def render_step4_match_fields():
    """Step 4: Match tender fields to extracted data."""
    section_header("🎯", "Step 4: Match Tender Fields", "Auto-match tender fields to extracted data")

    st.info("🚧 **Phase 2 - Coming Soon**")
    st.markdown("""
    This step will:
    1. Parse tender document to identify fillable fields
    2. Classify field types (ABN, insurance, qualification, etc.)
    3. Auto-match fields to extracted structured data
    4. Show confidence scores (high/medium/low)
    5. Let you review and approve matches

    **Status:** Phase 1 (Extraction) complete ✅ | Phase 2 (Matching) in development 🚧
    """)

    # Navigation
    st.divider()
    if st.button("⬅️ Back to Extraction"):
        st.session_state.workflow_step = 3
        st.rerun()


def render_step5_fill_export():
    """Step 5: Fill tender and export."""
    section_header("✅", "Step 5: Fill & Export", "Complete tender document")

    st.info("🚧 **Phase 3 - Coming Soon**")
    st.markdown("""
    This step will:
    1. Fill tender document with matched data
    2. Maintain original formatting
    3. Allow manual edits
    4. Export completed tender document

    **Status:** Phase 3 in planning 📋
    """)


def main():
    """Main page function."""
    st.title("📝 Proposal Workflow")
    st.caption(f"Version: {PAGE_VERSION}")

    st.markdown("""
    Complete workflow for responding to tenders with structured data extraction and auto-fill.
    """)

    # Initialize session state
    initialize_session_state()

    # Show progress
    st.divider()
    render_workflow_progress(st.session_state.workflow_step)
    st.divider()

    # Render current step
    if st.session_state.workflow_step == 1:
        render_step1_select_tender()
    elif st.session_state.workflow_step == 2:
        render_step2_select_sources()
    elif st.session_state.workflow_step == 3:
        render_step3_extract_data()
    elif st.session_state.workflow_step == 4:
        render_step4_match_fields()
    elif st.session_state.workflow_step == 5:
        render_step5_fill_export()

    # Reset button
    st.divider()
    if st.button("🔄 Start Over"):
        st.session_state.workflow_step = 1
        st.session_state.tender_document = None
        st.session_state.source_documents = []
        st.session_state.extracted_data = None
        st.session_state.selected_entity = None
        st.rerun()


if __name__ == "__main__":
    main()
