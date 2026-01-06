"""
Proposal Workspace - Mention-Based Tender Response System
Version: 1.0.0
Date: 2026-01-05

Purpose: Complete workflow for tender responses using @mention system.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import tempfile

# Set page config
st.set_page_config(
    page_title="Proposal Workspace",
    page_icon="📝",
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import theme and components
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.ui_components import render_version_footer

# Apply theme
apply_theme()

# Import workspace system
from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.workspace_model import WorkspaceState, MentionBinding
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.document_processor import DocumentProcessor
from cortex_engine.markup_engine import MarkupEngine
from cortex_engine.llm_interface import LLMInterface
from cortex_engine.mention_parser import MentionParser
from cortex_engine.field_substitution_engine import FieldSubstitutionEngine
from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger

logger = get_logger(__name__)

# ============================================
# INITIALIZATION
# ============================================

def initialize_managers():
    """Initialize all managers."""
    if 'workspace_manager' not in st.session_state:
        config = ConfigManager().get_config()
        db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))

        workspaces_path = Path(db_path) / "workspaces"
        st.session_state.workspace_manager = WorkspaceManager(workspaces_path)
        st.session_state.entity_manager = EntityProfileManager(Path(db_path))
        st.session_state.llm = LLMInterface(model="mistral-small3.2")

    return (
        st.session_state.workspace_manager,
        st.session_state.entity_manager,
        st.session_state.llm
    )

# ============================================
# MAIN UI
# ============================================

st.title("📝 Proposal Workspace")
st.markdown("Create, manage, and complete tender responses using the @mention system")
st.markdown("---")

workspace_manager, entity_manager, llm = initialize_managers()

# ============================================
# SIDEBAR: WORKSPACE SELECTION
# ============================================

with st.sidebar:
    st.header("📂 Workspaces")

    # List workspaces
    workspaces = workspace_manager.list_workspaces()

    if workspaces:
        workspace_options = {
            f"{ws.metadata.workspace_name} ({ws.get_state_display()})": ws.metadata.workspace_id
            for ws in workspaces
        }

        # Determine default selection
        default_index = 0
        if 'selected_workspace_id' in st.session_state and st.session_state.selected_workspace_id:
            # Try to find the previously selected workspace
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
        st.info("No workspaces yet. Create your first one below!")
        selected_workspace_id = None
        st.session_state.selected_workspace_id = None

    st.markdown("---")

    # Quick stats for selected workspace
    if selected_workspace_id:
        workspace = workspace_manager.get_workspace(selected_workspace_id)

        st.metric("State", workspace.get_state_display())
        st.metric("Progress", f"{workspace.get_progress_percentage():.0f}%")
        st.metric("Mentions", workspace.metadata.total_mentions)
        st.metric("Approved", workspace.metadata.approved_mentions)

# ============================================
# MAIN CONTENT AREA
# ============================================

if selected_workspace_id is None:
    # CREATE NEW WORKSPACE
    section_header("➕", "Create New Workspace", "Start a new tender response")

    with st.form("create_workspace_form"):
        st.subheader("Workspace Details")

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
                placeholder="paul@longboardfella.com.au",
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
            st.warning("⚠️ No entity profiles found. Create one in Entity Profile Manager first.")
            entity_id = None
            entity_name = None

        submit = st.form_submit_button("✅ Create Workspace", type="primary")

        if submit:
            try:
                if not all([tender_name, workspace_name, entity_id]):
                    st.error("Please fill in all required fields (*)")
                else:
                    # Generate workspace ID
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    entity_slug = entity_id.lower().replace(" ", "_")
                    tender_slug = tender_reference.lower().replace(" ", "_") if tender_reference else "tender"
                    workspace_id = f"workspace_{tender_slug}_{entity_slug}_{date_str}"

                    # Create workspace
                    workspace = workspace_manager.create_workspace(
                        workspace_id=workspace_id,
                        workspace_name=workspace_name,
                        tender_name=tender_name,
                        tender_reference=tender_reference,
                        created_by=created_by
                    )

                    # Bind entity
                    workspace = workspace_manager.bind_entity(
                        workspace_id=workspace_id,
                        entity_id=entity_id,
                        entity_name=entity_name
                    )

                    # Automatically select the newly created workspace
                    st.session_state.selected_workspace_id = workspace_id

                    st.success(f"✅ Created workspace: {workspace_name}")
                    st.rerun()

            except Exception as e:
                st.error(f"Error creating workspace: {e}")
                logger.error(f"Failed to create workspace: {e}", exc_info=True)

else:
    # MANAGE EXISTING WORKSPACE
    workspace = workspace_manager.get_workspace(selected_workspace_id)

    # Tabs for different workflow stages
    tab_overview, tab_document, tab_markup, tab_review, tab_generate, tab_export = st.tabs([
        "📊 Overview",
        "📄 Document",
        "🤖 Markup",
        "👁️ Review",
        "✨ Generate",
        "📦 Export"
    ])

    # ========================================
    # TAB: OVERVIEW
    # ========================================

    with tab_overview:
        section_header("📊", "Workspace Overview", workspace.metadata.workspace_name)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("State", workspace.get_state_display())
            st.metric("Created", workspace.metadata.created_at.strftime("%Y-%m-%d"))

        with col2:
            st.metric("Progress", f"{workspace.get_progress_percentage():.0f}%")
            st.metric("Entity", workspace.metadata.entity_name or "Not bound")

        with col3:
            st.metric("Total Mentions", workspace.metadata.total_mentions)
            st.metric("Approved", workspace.metadata.approved_mentions)

        with col4:
            st.metric("Generated", workspace.metadata.generated_mentions)
            st.metric("Rejected", workspace.metadata.rejected_mentions)

        st.markdown("---")

        # Workspace details
        with st.expander("📋 Workspace Details", expanded=True):
            st.write(f"**Workspace ID:** {workspace.metadata.workspace_id}")
            st.write(f"**Tender Name:** {workspace.metadata.tender_name}")
            if workspace.metadata.tender_reference:
                st.write(f"**RFT Reference:** {workspace.metadata.tender_reference}")
            st.write(f"**Created By:** {workspace.metadata.created_by or 'Unknown'}")
            st.write(f"**Last Updated:** {workspace.metadata.updated_at.strftime('%Y-%m-%d %H:%M')}")

        # Git history
        if workspace.workspace_path:
            with st.expander("📜 Version History"):
                from cortex_engine.workspace_git import WorkspaceGit

                try:
                    git = WorkspaceGit(workspace.workspace_path)
                    commits = git.get_log(limit=10)

                    for commit in commits:
                        st.write(f"**{commit['date'][:10]}** - {commit['message']}")
                        st.caption(f"by {commit['author']}")
                        st.markdown("---")
                except Exception as e:
                    st.warning(f"Could not load git history: {e}")

    # ========================================
    # TAB: DOCUMENT
    # ========================================

    with tab_document:
        section_header("📄", "Tender Document", "Upload and manage tender document")

        # Check if document exists
        doc_path = workspace.workspace_path / "documents" / "tender_original.txt"

        if not doc_path.exists():
            st.info("📤 Upload the tender document to begin")

            uploaded_file = st.file_uploader(
                "Upload Tender Document",
                type=['txt', 'docx', 'pdf'],
                help="Upload the original tender/RFT document"
            )

            if uploaded_file:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)

                    # Process document
                    with st.spinner("Processing document..."):
                        text = DocumentProcessor.process_document(tmp_path)

                        # Save to workspace
                        doc_path.write_text(text, encoding='utf-8')

                        # Update workspace metadata
                        workspace.metadata.original_filename = uploaded_file.name
                        workspace.metadata.document_type = Path(uploaded_file.name).suffix.replace('.', '')
                        workspace_manager._save_workspace(workspace)

                        # Git commit
                        from cortex_engine.workspace_git import WorkspaceGit
                        git = WorkspaceGit(workspace.workspace_path)
                        git.commit_changes(f"Uploaded tender document: {uploaded_file.name}")

                    st.success(f"✅ Uploaded and processed: {uploaded_file.name}")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    logger.error(f"Failed to process document: {e}", exc_info=True)

        else:
            st.success(f"✅ Document uploaded: {workspace.metadata.original_filename}")

            # Display document
            with st.expander("📄 View Document", expanded=False):
                document_text = doc_path.read_text(encoding='utf-8')
                st.text_area(
                    "Document Content",
                    value=document_text,
                    height=400,
                    disabled=True
                )

            # Option to replace
            if st.button("🔄 Replace Document"):
                doc_path.unlink()
                st.rerun()

    # ========================================
    # TAB: MARKUP
    # ========================================

    with tab_markup:
        section_header("🤖", "Auto-Markup", "Suggest @mention placements")

        # Check if document exists
        doc_path = workspace.workspace_path / "documents" / "tender_original.txt"

        if not doc_path.exists():
            st.warning("⚠️ Please upload a tender document first")
        elif len(workspace.mentions) > 0:
            st.info(f"✅ Markup complete: {len(workspace.mentions)} mentions suggested")

            # Show summary
            st.write("**Mention Summary:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                simple_count = sum(1 for m in workspace.mentions if m.mention_type == 'simple')
                st.metric("Simple Fields", simple_count)

            with col2:
                structured_count = sum(1 for m in workspace.mentions if m.mention_type == 'structured')
                st.metric("Structured Fields", structured_count)

            with col3:
                narrative_count = sum(1 for m in workspace.mentions if m.mention_type == 'narrative')
                st.metric("Narrative Sections", narrative_count)

            if st.button("🔄 Re-run Markup"):
                try:
                    with st.spinner("Re-analyzing document with enhanced LLM filtering..."):
                        # Clear existing mentions
                        workspace.mentions = []
                        workspace_manager._save_workspace(workspace)

                        # Load document
                        document_text = doc_path.read_text(encoding='utf-8')

                        # Create markup engine
                        markup_engine = MarkupEngine(entity_manager, llm)

                        # Analyze document with new LLM-based filtering
                        mentions = markup_engine.analyze_document(
                            document_text,
                            workspace.metadata.entity_id
                        )

                        # Add to workspace
                        workspace = workspace_manager.add_mention_bindings(
                            workspace.metadata.workspace_id,
                            mentions
                        )

                        # Update state (only if not already in MARKUP_SUGGESTED)
                        if workspace.metadata.state != WorkspaceState.MARKUP_SUGGESTED:
                            workspace = workspace_manager.update_workspace_state(
                                workspace.metadata.workspace_id,
                                WorkspaceState.MARKUP_SUGGESTED
                            )

                    st.success(f"✅ Re-analyzed with LLM filtering: {len(mentions)} @mention placements suggested")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error during re-markup: {e}")
                    logger.error(f"Re-markup failed: {e}", exc_info=True)

        else:
            st.info("🤖 Run auto-markup to analyze the document and suggest @mention placements")

            if st.button("▶️ Run Auto-Markup", type="primary"):
                try:
                    with st.spinner("Analyzing document..."):
                        # Load document
                        document_text = doc_path.read_text(encoding='utf-8')

                        # Create markup engine
                        markup_engine = MarkupEngine(entity_manager, llm)

                        # Analyze document
                        mentions = markup_engine.analyze_document(
                            document_text,
                            workspace.metadata.entity_id
                        )

                        # Add to workspace
                        workspace = workspace_manager.add_mention_bindings(
                            workspace.metadata.workspace_id,
                            mentions
                        )

                        # Update state (only if not already in MARKUP_SUGGESTED)
                        if workspace.metadata.state != WorkspaceState.MARKUP_SUGGESTED:
                            workspace = workspace_manager.update_workspace_state(
                                workspace.metadata.workspace_id,
                                WorkspaceState.MARKUP_SUGGESTED
                            )

                    st.success(f"✅ Suggested {len(mentions)} @mention placements")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error during markup: {e}")
                    logger.error(f"Markup failed: {e}", exc_info=True)

    # ========================================
    # TAB: REVIEW
    # ========================================

    with tab_review:
        section_header("👁️", "Review Mentions", "Approve or reject suggested @mentions")

        if len(workspace.mentions) == 0:
            st.warning("⚠️ No mentions to review yet. Run auto-markup first.")
        else:
            pending = workspace.get_pending_mentions()

            if len(pending) == 0:
                st.success(f"✅ All mentions reviewed! ({workspace.metadata.approved_mentions} approved, {workspace.metadata.rejected_mentions} rejected)")

                if workspace.metadata.state == WorkspaceState.MARKUP_SUGGESTED:
                    if st.button("Continue to Content Generation"):
                        workspace = workspace_manager.update_workspace_state(
                            workspace.metadata.workspace_id,
                            WorkspaceState.MARKUP_REVIEWED
                        )
                        st.rerun()
            else:
                # Load document text for context
                doc_path = workspace.workspace_path / "documents" / "tender_original.txt"

                if doc_path.exists():
                    document_text = doc_path.read_text(encoding='utf-8')

                    # Use new context-rich review interface
                    from cortex_engine.review_ui.mention_review_streamlit import render_mention_review

                    render_mention_review(
                        workspace=workspace,
                        workspace_manager=workspace_manager,
                        entity_profile_manager=st.session_state.entity_manager,
                        document_text=document_text,
                        mentions=workspace.mentions
                    )
                else:
                    # Fallback to simple list if document not available
                    st.info(f"📋 {len(pending)} mentions pending review")

                    for idx, mention in enumerate(pending):
                        with st.expander(f"📌 {mention.mention_text} ({mention.mention_type})"):
                            st.write(f"**Location:** {mention.location}")
                            st.write(f"**Field Path:** {mention.field_path}")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if st.button("✅ Approve", key=f"approve_{idx}_{mention.mention_text}"):
                                    workspace_manager.update_mention_binding(
                                        workspace.metadata.workspace_id,
                                        mention.mention_text,
                                        approved=True
                                    )
                                    st.rerun()

                            with col2:
                                if st.button("❌ Reject", key=f"reject_{idx}_{mention.mention_text}"):
                                    workspace_manager.update_mention_binding(
                                        workspace.metadata.workspace_id,
                                        mention.mention_text,
                                        rejected=True
                                    )
                                    st.rerun()

    # ========================================
    # TAB: GENERATE
    # ========================================

    with tab_generate:
        section_header("✨", "Generate Content", "LLM generation for complex mentions")

        # Check if we have mentions that need LLM generation
        llm_mentions = workspace.get_llm_mentions()

        if len(workspace.mentions) == 0:
            st.warning("⚠️ No mentions yet. Complete the markup and review steps first.")
        elif len(llm_mentions) == 0:
            st.success("✅ No mentions require LLM generation")

            if workspace.metadata.state == WorkspaceState.MARKUP_REVIEWED:
                if st.button("Continue to Export"):
                    workspace = workspace_manager.update_workspace_state(
                        workspace.metadata.workspace_id,
                        WorkspaceState.CONTENT_GENERATED
                    )
                    st.rerun()
        else:
            st.info(f"🤖 {len(llm_mentions)} mentions require LLM content generation")

            # Show mentions that need generation
            for gen_idx, mention in enumerate(llm_mentions):
                with st.expander(f"✨ {mention.mention_text} ({mention.mention_type})"):
                    st.write(f"**Location:** {mention.location}")
                    st.write(f"**Field Path:** {mention.field_path}")

                    # Determine generation type
                    generation_type = "unknown"
                    if "@cv[" in mention.mention_text:
                        generation_type = "cv"
                        person_id = mention.mention_text.split('[')[1].split(']')[0]

                        st.write(f"**Type:** CV Generation")
                        st.write(f"**Person:** {person_id}")

                        # Get person data
                        team_member = entity_manager.get_team_member(
                            workspace.metadata.entity_id,
                            person_id
                        )

                        if team_member:
                            # Show preview of data
                            with st.expander("📋 Source Data"):
                                st.write(f"**Name:** {team_member.full_name}")
                                st.write(f"**Role:** {team_member.role}")
                                st.write(f"**Qualifications:** {len(team_member.qualifications)}")
                                st.write(f"**Experience:** {len(team_member.experience)}")

                    elif "@project_summary[" in mention.mention_text:
                        generation_type = "project_summary"
                        project_id = mention.mention_text.split('[')[1].split(']')[0]

                        st.write(f"**Type:** Project Summary")
                        st.write(f"**Project:** {project_id}")

                        # Get project data
                        project = entity_manager.get_project(
                            workspace.metadata.entity_id,
                            project_id
                        )

                        if project:
                            with st.expander("📋 Source Data"):
                                st.write(f"**Project:** {project.project_name}")
                                st.write(f"**Client:** {project.client}")
                                st.write(f"**Value:** ${project.financials.contract_value:,.0f}")

                    elif "@reference[" in mention.mention_text:
                        generation_type = "reference"
                        reference_id = mention.mention_text.split('[')[1].split(']')[0]

                        st.write(f"**Type:** Reference")
                        st.write(f"**Reference:** {reference_id}")

                    if st.button("▶️ Generate Content", key=f"gen_{gen_idx}_{mention.mention_text}"):
                        try:
                            with st.spinner("Generating content..."):
                                # Create content generation engine
                                from cortex_engine.content_generator import ContentGenerator

                                generator = ContentGenerator(entity_manager, llm)

                                # Generate content
                                import time
                                start_time = time.time()

                                generated_content = generator.generate_content(
                                    mention,
                                    workspace.metadata.entity_id,
                                    generation_type
                                )

                                generation_time = time.time() - start_time

                                # Update mention with generated content
                                workspace_manager.update_mention_binding(
                                    workspace.metadata.workspace_id,
                                    mention.mention_text,
                                    resolved_value=generated_content
                                )

                                # Log generation
                                from cortex_engine.workspace_model import GenerationLog
                                log = GenerationLog(
                                    mention_text=mention.mention_text,
                                    generation_type=generation_type,
                                    prompt="Auto-generated from entity data",
                                    model=llm.model,
                                    temperature=llm.temperature,
                                    generated_content=generated_content,
                                    generation_time=generation_time
                                )
                                workspace_manager.add_generation_log(
                                    workspace.metadata.workspace_id,
                                    log
                                )

                            st.success("✅ Content generated!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Generation failed: {e}")
                            logger.error(f"Content generation failed: {e}", exc_info=True)

            # Check if all generated
            if all(m.resolved_value for m in workspace.mentions if m.requires_llm and m.approved):
                st.success("✅ All content generated!")

                if st.button("Continue to Export", type="primary"):
                    workspace = workspace_manager.update_workspace_state(
                        workspace.metadata.workspace_id,
                        WorkspaceState.CONTENT_GENERATED
                    )
                    st.rerun()

    # ========================================
    # TAB: EXPORT
    # ========================================

    with tab_export:
        section_header("📦", "Export Document", "Generate final tender document")

        # Check if ready to export
        if workspace.metadata.state not in [WorkspaceState.CONTENT_GENERATED, WorkspaceState.DRAFT_READY, WorkspaceState.EXPORTED]:
            st.warning("⚠️ Complete the Generate step first")
        else:
            st.success("✅ Ready to export!")

            # Load original document
            doc_path = workspace.workspace_path / "documents" / "tender_original.txt"

            if doc_path.exists():
                document_text = doc_path.read_text(encoding='utf-8')

                # Build replacements dictionary
                replacements = {}

                # Use field substitution engine to resolve all mentions
                parser = MentionParser()
                engine = FieldSubstitutionEngine(entity_manager)

                # Find all mentions in document
                mentions_in_doc = parser.parse_all(document_text)

                with st.spinner("Resolving all mentions..."):
                    for parsed_mention in mentions_in_doc:
                        # Try to resolve from workspace first (for generated content)
                        workspace_mention = next(
                            (m for m in workspace.mentions if m.mention_text == parsed_mention.raw_text),
                            None
                        )

                        if workspace_mention and workspace_mention.resolved_value:
                            # Use generated content
                            replacements[parsed_mention.raw_text] = workspace_mention.resolved_value
                        else:
                            # Resolve from entity profile
                            result = engine.resolve(parsed_mention, workspace.metadata.entity_id)

                            if result.success and not result.requires_llm:
                                replacements[parsed_mention.raw_text] = result.value

                st.write(f"**Resolved:** {len(replacements)} mentions")

                # Preview
                with st.expander("👁️ Preview Final Document", expanded=True):
                    # Apply replacements
                    filled_text = document_text
                    for mention, value in replacements.items():
                        filled_text = filled_text.replace(mention, value)

                    st.text_area(
                        "Final Document",
                        value=filled_text,
                        height=400,
                        disabled=True
                    )

                # Export options
                st.subheader("Export Options")

                col1, col2 = st.columns(2)

                with col1:
                    export_format = st.selectbox(
                        "Format",
                        options=['txt', 'docx'],
                        help="Choose export format"
                    )

                with col2:
                    export_filename = st.text_input(
                        "Filename",
                        value=f"tender_final_{datetime.now().strftime('%Y%m%d')}",
                        help="Filename without extension"
                    )

                if st.button("📦 Export Document", type="primary"):
                    try:
                        with st.spinner("Generating final document..."):
                            # Replace all mentions
                            filled_text = document_text
                            for mention, value in replacements.items():
                                filled_text = filled_text.replace(mention, value)

                            # Save to exports directory
                            export_path = workspace.workspace_path / "exports" / f"{export_filename}.{export_format}"

                            success = DocumentProcessor.save_document_with_mentions(
                                filled_text,
                                export_path,
                                export_format
                            )

                            if success:
                                # Update workspace state
                                workspace = workspace_manager.update_workspace_state(
                                    workspace.metadata.workspace_id,
                                    WorkspaceState.EXPORTED
                                )

                                st.success(f"✅ Exported: {export_path.name}")

                                # Provide download
                                with open(export_path, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Download Document",
                                        data=f.read(),
                                        file_name=export_path.name,
                                        mime='application/octet-stream'
                                    )
                            else:
                                st.error("Export failed")

                    except Exception as e:
                        st.error(f"Export failed: {e}")
                        logger.error(f"Export failed: {e}", exc_info=True)

            else:
                st.error("Original document not found")

# Footer
st.markdown("---")
render_version_footer()
