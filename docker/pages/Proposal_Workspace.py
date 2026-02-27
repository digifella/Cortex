"""
Proposal Workspace - Workspace Creation and Management
Version: 2.0.0
Date: 2026-01-20

Purpose: Create and manage proposal workspaces. Upload tender documents and bind entity profiles.
After setup, use Chunk Review and Intelligent Completion pages for the actual proposal work.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import tempfile

st.set_page_config(
    page_title="Proposal Workspace",
    page_icon="📝",
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.workspace_model import WorkspaceState
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.document_processor import DocumentProcessor
from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path, resolve_db_root_path, get_logger

logger = get_logger(__name__)
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB

# ============================================
# INITIALIZATION
# ============================================

def initialize_managers():
    """Initialize all managers."""
    if 'workspace_manager' not in st.session_state:
        config = ConfigManager().get_config()
        raw_db_path = config.get('ai_database_path', '')
        resolved_root = resolve_db_root_path(raw_db_path)
        db_path = str(resolved_root) if resolved_root else convert_windows_to_wsl_path(raw_db_path)

        workspaces_path = Path(db_path) / "workspaces"
        st.session_state.workspace_manager = WorkspaceManager(workspaces_path)
        st.session_state.entity_manager = EntityProfileManager(Path(db_path))

    return (
        st.session_state.workspace_manager,
        st.session_state.entity_manager
    )

# ============================================
# MAIN UI
# ============================================

st.title("Proposal Workspace")
st.warning("**This page has been superseded by Proposal Manager.** Use the new consolidated workflow for a better experience.")
st.page_link("pages/13_Proposal_Manager.py", label="Go to Proposal Manager", icon="📋")
st.divider()
st.markdown("Create workspaces, upload tender documents, and bind entity profiles")

workspace_manager, entity_manager = initialize_managers()

# ============================================
# SIDEBAR: WORKSPACE SELECTION
# ============================================

with st.sidebar:
    st.header("Workspaces")

    workspaces = workspace_manager.list_workspaces()

    if workspaces:
        workspace_options = {
            f"{ws.metadata.workspace_name}": ws.metadata.workspace_id
            for ws in workspaces
        }

        default_index = 0
        if 'selected_workspace_id' in st.session_state and st.session_state.selected_workspace_id:
            for idx, (name, ws_id) in enumerate(workspace_options.items(), start=1):
                if ws_id == st.session_state.selected_workspace_id:
                    default_index = idx
                    break

        selected_name = st.selectbox(
            "Select Workspace",
            options=["-- Create New --"] + list(workspace_options.keys()),
            index=default_index
        )

        if selected_name != "-- Create New --":
            selected_workspace_id = workspace_options[selected_name]
            st.session_state.selected_workspace_id = selected_workspace_id
        else:
            selected_workspace_id = None
            st.session_state.selected_workspace_id = None
    else:
        st.info("No workspaces yet. Create your first one!")
        selected_workspace_id = None
        st.session_state.selected_workspace_id = None

    st.markdown("---")

    # Quick stats for selected workspace
    if selected_workspace_id:
        workspace = workspace_manager.get_workspace(selected_workspace_id)
        st.metric("Entity", workspace.metadata.entity_name or "Not bound")
        if workspace.metadata.original_filename:
            st.success(f"Document: {workspace.metadata.original_filename[:20]}...")
        else:
            st.warning("No document uploaded")

# ============================================
# MAIN CONTENT
# ============================================

if selected_workspace_id is None:
    # CREATE NEW WORKSPACE
    st.subheader("Create New Workspace")

    with st.form("create_workspace_form"):
        col1, col2 = st.columns(2)

        with col1:
            tender_name = st.text_input(
                "Tender Name*",
                placeholder="Department of Digital Services - Consulting",
                help="Full name of the tender/RFT"
            )

            tender_reference = st.text_input(
                "RFT Reference",
                placeholder="RFT12345",
                help="RFT/tender reference number"
            )

        with col2:
            workspace_name = st.text_input(
                "Workspace Name*",
                placeholder="RFT12345 - Digital Services",
                help="Short name for this workspace"
            )

            created_by = st.text_input(
                "Your Email",
                placeholder="user@example.com",
                help="Your email address"
            )

        # Entity selection
        st.subheader("Entity Profile")

        entities = entity_manager.list_entity_profiles()

        if entities:
            entity_options = {e.entity_name: e.entity_id for e in entities}
            entity_name = st.selectbox(
                "Select Entity Profile*",
                options=list(entity_options.keys()),
                help="Entity profile to use for this tender"
            )
            entity_id = entity_options[entity_name]
        else:
            st.warning("No entity profiles found. Create one in Entity Profile Manager first.")
            entity_id = None
            entity_name = None

        submit = st.form_submit_button("Create Workspace", type="primary")

        if submit:
            try:
                if not all([tender_name, workspace_name, entity_id]):
                    st.error("Please fill in all required fields (*)")
                else:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    entity_slug = entity_id.lower().replace(" ", "_")
                    tender_slug = tender_reference.lower().replace(" ", "_") if tender_reference else "tender"
                    workspace_id = f"workspace_{tender_slug}_{entity_slug}_{date_str}"

                    workspace = workspace_manager.create_workspace(
                        workspace_id=workspace_id,
                        workspace_name=workspace_name,
                        tender_name=tender_name,
                        tender_reference=tender_reference,
                        created_by=created_by
                    )

                    workspace = workspace_manager.bind_entity(
                        workspace_id=workspace_id,
                        entity_id=entity_id,
                        entity_name=entity_name
                    )

                    st.session_state.selected_workspace_id = workspace_id
                    st.success(f"Created workspace: {workspace_name}")
                    st.rerun()

            except Exception as e:
                st.error(f"Error creating workspace: {e}")
                logger.error(f"Failed to create workspace: {e}", exc_info=True)

else:
    # MANAGE EXISTING WORKSPACE
    workspace = workspace_manager.get_workspace(selected_workspace_id)

    # Workspace Header
    st.subheader(f"{workspace.metadata.workspace_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Tender: {workspace.metadata.tender_name}")
    with col2:
        st.caption(f"Entity: {workspace.metadata.entity_name or 'Not bound'}")
    with col3:
        st.caption(f"Created: {workspace.metadata.created_at.strftime('%Y-%m-%d')}")

    st.divider()

    # ========================================
    # STEP 1: DOCUMENT UPLOAD
    # ========================================

    st.subheader("Step 1: Upload Tender Document")

    doc_path = workspace.workspace_path / "documents" / "tender_original.txt"

    if not doc_path.exists():
        st.info("Upload the tender document to begin")

        uploaded_file = st.file_uploader(
            "Upload Tender Document",
            type=['txt', 'docx', 'pdf'],
            help="Upload the original tender/RFT document"
        )

        if uploaded_file:
            if int(getattr(uploaded_file, "size", 0) or 0) > MAX_UPLOAD_BYTES:
                st.error("Selected file exceeds the 1GB upload limit.")
            else:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)

                    with st.spinner("Processing document..."):
                        text = DocumentProcessor.process_document(tmp_path)
                        doc_path.parent.mkdir(parents=True, exist_ok=True)
                        doc_path.write_text(text, encoding='utf-8')

                        workspace.metadata.original_filename = uploaded_file.name
                        workspace.metadata.document_type = Path(uploaded_file.name).suffix.replace('.', '')
                        workspace_manager._save_workspace(workspace)

                    st.success(f"Uploaded: {uploaded_file.name}")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    logger.error(f"Failed to process document: {e}", exc_info=True)
    else:
        st.success(f"Document uploaded: {workspace.metadata.original_filename}")

        with st.expander("View Document Preview"):
            document_text = doc_path.read_text(encoding='utf-8')
            st.text_area(
                "Document Content",
                value=document_text[:5000] + ("..." if len(document_text) > 5000 else ""),
                height=300,
                disabled=True
            )

        if st.button("Replace Document"):
            doc_path.unlink()
            st.rerun()

    st.divider()

    # ========================================
    # STEP 2: NEXT STEPS - NAVIGATION
    # ========================================

    st.subheader("Step 2: Work on Your Proposal")

    if not doc_path.exists():
        st.warning("Upload a document first to proceed")
    else:
        st.markdown("""
        Your workspace is ready. Choose your next step:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Chunk Review
            Review extracted sections from your tender document.
            Navigate by section type (Company, Personnel, Project).

            **Use when:** You want to review what was extracted from the tender document.
            """)
            st.page_link("pages/Proposal_Chunk_Review_V2.py", label="Go to Chunk Review", icon="📋")

        with col2:
            st.markdown("""
            ### Intelligent Completion
            Interactive completion of substantive questions.
            Uses evidence from your knowledge collections.

            **Use when:** You want to generate responses to tender questions.
            """)
            st.page_link("pages/Proposal_Intelligent_Completion.py", label="Go to Intelligent Completion", icon="🧠")

        st.divider()

        # Knowledge Collection reminder
        st.info("""
        **Tip:** For best results with Intelligent Completion, create a Working Collection
        in Collection Management with relevant evidence documents (case studies, CVs, capability statements).
        """)

    st.divider()

    # ========================================
    # WORKSPACE MANAGEMENT
    # ========================================

    with st.expander("Workspace Management"):
        st.caption(f"Workspace ID: {workspace.metadata.workspace_id}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Delete Workspace", type="secondary"):
                st.session_state.confirm_delete = True

        with col2:
            if st.session_state.get('confirm_delete', False):
                st.warning("Are you sure? This cannot be undone.")
                if st.button("Yes, Delete", type="primary"):
                    try:
                        workspace_manager.delete_workspace(selected_workspace_id)
                        st.session_state.selected_workspace_id = None
                        st.session_state.confirm_delete = False
                        st.success("Workspace deleted")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

# Footer
st.divider()
st.caption("v2.0.0 | Proposal Workspace - Create and manage tender response workspaces")
