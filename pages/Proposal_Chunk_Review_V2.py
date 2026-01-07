"""
Proposal Chunk Review V2
Version: 2.0.0
Date: 2026-01-07

Purpose: Professional batch-and-review workflow for tender document completion.
Key Features:
- Automatic batch analysis of all chunks
- Clean, compact navigation
- Inline mention editing
- Multi-session support
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.markup_engine import MarkupEngine
from cortex_engine.document_chunker import DocumentChunker
from cortex_engine.workspace_model import ChunkProgress, WorkspaceState
from cortex_engine.llm_interface import LLMInterface
from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path

st.set_page_config(
    page_title="Chunk Review V2 - Cortex Suite",
    page_icon="📑",
    layout="wide"
)

# Load config
config = ConfigManager().get_config()
db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))

# Initialize managers
workspace_manager = WorkspaceManager(Path(db_path) / "workspaces")
entity_manager = EntityProfileManager(Path(db_path))
llm = LLMInterface(model="qwen2.5:72b-instruct-q4_K_M")
markup_engine = MarkupEngine(entity_manager, llm)
chunker = DocumentChunker(target_chunk_size=4000, max_chunk_size=6000)

# Professional CSS
st.markdown("""
<style>
    /* Clean, professional design */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2D5F4F;
        margin-bottom: 0.5rem;
    }

    .progress-container {
        background: #F5F4F2;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .progress-bar {
        background: #E5E3DF;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .progress-fill {
        background: linear-gradient(90deg, #2D5F4F 0%, #3A7A66 100%);
        height: 100%;
        transition: width 0.3s ease;
    }

    .mention-card {
        background: white;
        border: 1px solid #E5E3DF;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        transition: box-shadow 0.2s;
    }

    .mention-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .mention-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .mention-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #C85D3C;
    }

    .mention-details {
        font-size: 0.875rem;
        color: #666;
        line-height: 1.6;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-analyzing {
        background: #FFF4E6;
        color: #C85D3C;
    }

    .status-complete {
        background: #E8F5F1;
        color: #2D5F4F;
    }

    .status-pending {
        background: #F5F4F2;
        color: #999;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📑 Tender Document Review</div>', unsafe_allow_html=True)
st.caption("Professional batch analysis and review workflow")

# Workspace selection
st.subheader("Select Workspace")

workspaces = workspace_manager.list_workspaces()
if not workspaces:
    st.warning("No workspaces found. Create one in Proposal Workspace.")
    st.stop()

workspace_options = {
    f"{ws.metadata.workspace_name} ({ws.metadata.workspace_id})": ws
    for ws in workspaces
}

selected_name = st.selectbox(
    "Workspace",
    options=list(workspace_options.keys()),
    key="workspace_select_v2",
    label_visibility="collapsed"
)

workspace = workspace_options[selected_name]

# Verify entity bound and document uploaded
if not workspace.metadata.entity_id:
    st.error("❌ No entity profile bound to this workspace. Please bind an entity first.")
    st.stop()

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

# Initialize chunk mode if needed
if not workspace.metadata.chunk_mode_enabled:
    st.info("🔄 Initializing chunk-based review mode...")

    with st.spinner("Creating document chunks..."):
        # Create chunks
        chunks = chunker.create_chunks(document_text)
        completable_chunks = chunker.filter_completable_chunks(chunks)

        # Initialize workspace
        workspace.metadata.chunk_mode_enabled = True
        workspace.metadata.total_chunks = len(completable_chunks)
        workspace.metadata.current_chunk_id = 1
        workspace.metadata.chunks_reviewed = 0
        workspace.metadata.analysis_status = "pending"

        # Create chunk progress entries
        workspace.chunks.clear()
        for chunk in completable_chunks:
            workspace.chunks.append(ChunkProgress(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                status="pending",
                mentions_found=0,
                mentions_approved=0
            ))

        workspace_manager._save_workspace(workspace)
        st.session_state.document_chunks = completable_chunks
        st.rerun()

# Load chunks
if 'document_chunks' not in st.session_state:
    chunks = chunker.create_chunks(document_text)
    completable_chunks = chunker.filter_completable_chunks(chunks)
    st.session_state.document_chunks = completable_chunks

chunks = st.session_state.document_chunks

# ============================
# BATCH ANALYSIS AUTO-START
# ============================

if workspace.metadata.analysis_status == "pending":
    st.info("🤖 **Batch Analysis Ready** - Click to analyze all chunks automatically")

    if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
        workspace.metadata.analysis_status = "analyzing"
        workspace.metadata.analysis_started_at = datetime.now()
        workspace.metadata.analysis_progress = 0
        workspace_manager._save_workspace(workspace)
        st.rerun()

elif workspace.metadata.analysis_status == "analyzing":
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown("### 🔬 Analyzing Document...")

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Batch analysis with progress
    def update_progress(current, total):
        progress = int((current / total) * 100)
        progress_bar.progress(progress / 100)
        status_text.markdown(f"**Analyzing chunk {current}/{total}** ({progress}%)")

        # Update workspace progress
        workspace.metadata.analysis_progress = progress
        workspace_manager._save_workspace(workspace)

    # Run batch analysis
    results = markup_engine.analyze_all_chunks_batch(chunks, workspace.metadata.entity_id, update_progress)

    # Save all mentions
    all_mentions = []
    for chunk_id, mentions in results.items():
        # Update chunk progress
        chunk_progress = next((cp for cp in workspace.chunks if cp.chunk_id == chunk_id), None)
        if chunk_progress:
            chunk_progress.mentions_found = len(mentions)
        all_mentions.extend(mentions)

    # Add all mentions to workspace
    workspace = workspace_manager.add_mention_bindings(workspace.metadata.workspace_id, all_mentions)

    # Mark analysis complete
    workspace.metadata.analysis_status = "complete"
    workspace.metadata.analysis_completed_at = datetime.now()
    workspace.metadata.analysis_progress = 100
    workspace.metadata.total_mentions_found = len(all_mentions)
    workspace_manager._save_workspace(workspace)

    st.markdown('</div>', unsafe_allow_html=True)
    st.success(f"✅ **Analysis Complete!** Found {len(all_mentions)} mentions across {len(chunks)} chunks")
    time.sleep(2)
    st.rerun()

elif workspace.metadata.analysis_status == "complete":
    # Show completion summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Mentions", workspace.metadata.total_mentions_found)
    with col2:
        st.metric("Chunks Reviewed", f"{workspace.metadata.chunks_reviewed}/{workspace.metadata.total_chunks}")
    with col3:
        progress_pct = int((workspace.metadata.chunks_reviewed / workspace.metadata.total_chunks) * 100) if workspace.metadata.total_chunks > 0 else 0
        st.metric("Review Progress", f"{progress_pct}%")

    # Progress bar
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress_pct}%"></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ============================
    # COMPACT NAVIGATION
    # ============================

    st.subheader("Review Mentions")

    # Current chunk
    current_chunk_id = workspace.metadata.current_chunk_id or 1
    current_chunk = next((c for c in chunks if c.chunk_id == current_chunk_id), None)
    chunk_progress = next((cp for cp in workspace.chunks if cp.chunk_id == current_chunk_id), None)

    if not current_chunk:
        st.error("Chunk not found")
        st.stop()

    # Navigation controls
    col_prev, col_jump, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.button("← Prev", disabled=(current_chunk_id == 1), use_container_width=True):
            workspace.metadata.current_chunk_id = current_chunk_id - 1
            workspace_manager._save_workspace(workspace)
            st.rerun()

    with col_jump:
        chunk_options = []
        for cp in workspace.chunks:
            status = "✅" if cp.status == "reviewed" else "⏳"
            chunk_options.append(f"{status} Chunk {cp.chunk_id}: {cp.title[:40]}...")

        selected_idx = st.selectbox(
            "Jump to chunk",
            range(len(chunk_options)),
            index=current_chunk_id - 1,
            format_func=lambda i: chunk_options[i],
            label_visibility="collapsed"
        )

        if selected_idx + 1 != current_chunk_id:
            workspace.metadata.current_chunk_id = selected_idx + 1
            workspace_manager._save_workspace(workspace)
            st.rerun()

    with col_next:
        if st.button("Next →", disabled=(current_chunk_id == workspace.metadata.total_chunks), use_container_width=True):
            workspace.metadata.current_chunk_id = current_chunk_id + 1
            workspace_manager._save_workspace(workspace)
            st.rerun()

    st.divider()

    # Chunk title
    st.markdown(f"### Chunk {current_chunk.chunk_id}: {current_chunk.title}")
    st.caption(f"Lines {current_chunk.start_line}-{current_chunk.end_line} • {current_chunk.char_count} characters")

    # Get mentions for this chunk
    chunk_mentions = [m for m in workspace.mentions if m.chunk_id == current_chunk_id]
    pending_mentions = [m for m in chunk_mentions if not m.approved and not m.rejected and not m.ignored]

    # ============================
    # MENTION REVIEW CARDS
    # ============================

    if pending_mentions:
        st.markdown(f"**{len(pending_mentions)} mentions to review:**")

        for idx, mention in enumerate(pending_mentions):
            # Mention card
            st.markdown(f"""
            <div class="mention-card">
                <div class="mention-header">
                    <span class="mention-text">{mention.mention_text}</span>
                </div>
                <div class="mention-details">
                    <strong>Field:</strong> {mention.field_path}<br>
                    <strong>Location:</strong> {mention.location}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("✅ Approve", key=f"approve_{mention.mention_text}_{current_chunk_id}_{idx}", use_container_width=True):
                    workspace = workspace_manager.update_mention_binding(
                        workspace.metadata.workspace_id,
                        mention.mention_text,
                        approved=True,
                        chunk_id=current_chunk_id,
                        location=mention.location
                    )

                    if chunk_progress:
                        chunk_progress.mentions_approved += 1
                    workspace_manager._save_workspace(workspace)
                    st.rerun()

            with col2:
                if st.button("❌ Reject", key=f"reject_{mention.mention_text}_{current_chunk_id}_{idx}", use_container_width=True):
                    workspace = workspace_manager.update_mention_binding(
                        workspace.metadata.workspace_id,
                        mention.mention_text,
                        rejected=True,
                        chunk_id=current_chunk_id,
                        location=mention.location
                    )
                    workspace_manager._save_workspace(workspace)
                    st.rerun()

            with col3:
                if st.button("✏️ Edit Binding", key=f"edit_{mention.mention_text}_{current_chunk_id}_{idx}", use_container_width=True):
                    st.session_state.editing_mention = mention
                    st.rerun()

        # Check if all mentions reviewed - auto-advance
        if len(pending_mentions) == 0 and chunk_progress and chunk_progress.status != "reviewed":
            chunk_progress.status = "reviewed"
            chunk_progress.reviewed_at = datetime.now()
            workspace.metadata.chunks_reviewed += 1
            workspace.metadata.last_reviewed_chunk_id = current_chunk_id
            workspace.metadata.last_reviewed_at = datetime.now()
            workspace_manager._save_workspace(workspace)

            if current_chunk_id < workspace.metadata.total_chunks:
                st.success("✅ Chunk complete! Advancing to next chunk...")
                workspace.metadata.current_chunk_id = current_chunk_id + 1
                workspace_manager._save_workspace(workspace)
                time.sleep(1)
                st.rerun()
            else:
                st.balloons()
                st.success("🎉 All chunks completed!")

    else:
        if chunk_progress and chunk_progress.mentions_found > 0:
            st.success("✅ All mentions in this chunk have been reviewed!")
        else:
            st.info("ℹ️ No mentions found in this chunk")

        # Auto-mark as reviewed if not already
        if chunk_progress and chunk_progress.status != "reviewed":
            chunk_progress.status = "reviewed"
            chunk_progress.reviewed_at = datetime.now()
            workspace.metadata.chunks_reviewed += 1
            workspace_manager._save_workspace(workspace)

# ============================
# EDIT DIALOG (if editing)
# ============================

if 'editing_mention' in st.session_state:
    mention = st.session_state.editing_mention

    with st.expander("✏️ Edit Mention Binding", expanded=True):
        st.markdown("### Edit Binding Details")

        new_mention_text = st.text_input(
            "Mention Text",
            value=mention.mention_text,
            help="The @mention text that appears in the document"
        )

        new_field_path = st.text_input(
            "Field Path",
            value=mention.field_path,
            help="Path to field in entity profile (e.g., company.legal_name)"
        )

        new_resolved_value = st.text_area(
            "Resolved Value (optional)",
            value=mention.resolved_value or "",
            help="Pre-fill the value for this mention"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                # Call edit_mention_binding
                try:
                    workspace = workspace_manager.edit_mention_binding(
                        workspace.metadata.workspace_id,
                        mention.mention_text,
                        mention.chunk_id,
                        mention.location,
                        new_mention_text=new_mention_text if new_mention_text != mention.mention_text else None,
                        new_field_path=new_field_path if new_field_path != mention.field_path else None,
                        new_resolved_value=new_resolved_value if new_resolved_value != (mention.resolved_value or "") else None
                    )
                    st.success("✅ Mention binding updated!")
                    del st.session_state.editing_mention
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating mention: {str(e)}")

        with col2:
            if st.button("Cancel", use_container_width=True):
                del st.session_state.editing_mention
                st.rerun()

# Export button (when all chunks reviewed)
if workspace.metadata.chunks_reviewed == workspace.metadata.total_chunks:
    st.divider()
    if st.button("📤 Export Final Document", type="primary", use_container_width=True):
        st.success("✅ Export functionality will be implemented in next phase")
        st.info("All chunks reviewed! Ready to generate final document.")
