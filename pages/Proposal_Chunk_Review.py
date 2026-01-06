"""
Proposal Chunk Review
Version: 1.0.0
Date: 2026-01-06

Purpose: Chunk-based document review interface for large tender documents.
Provides systematic, manageable review workflow.
"""

import streamlit as st
import sys
from pathlib import Path

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
from datetime import datetime

st.set_page_config(
    page_title="Chunk Review - Cortex Suite",
    page_icon="📑",
    layout="wide"
)

# Load config (matching Proposal_Workspace.py initialization)
config = ConfigManager().get_config()
db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))

# Initialize managers
workspace_manager = WorkspaceManager(Path(db_path) / "workspaces")
entity_manager = EntityProfileManager(Path(db_path))

# Use a powerful model for chunk analysis - qwen2.5:72b is excellent for document understanding
llm = LLMInterface(model="qwen2.5:72b-instruct-q4_K_M")
markup_engine = MarkupEngine(entity_manager, llm)
chunker = DocumentChunker(target_chunk_size=4000, max_chunk_size=6000)

# Custom CSS for chunk navigator
st.markdown("""
<style>
    /* Chunk Progress Bar */
    .chunk-progress {
        background: #E5E3DF;
        height: 12px;
        border-radius: 6px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .chunk-progress-fill {
        background: linear-gradient(90deg, #2D5F4F 0%, #C85D3C 100%);
        height: 100%;
        transition: width 0.6s ease;
    }

    /* Chunk Grid */
    .chunk-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
        gap: 0.5rem;
        margin: 1.5rem 0;
    }

    .chunk-card {
        aspect-ratio: 1;
        border: 2px solid #E5E3DF;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
        background: white;
        padding: 0.5rem;
        text-align: center;
    }

    .chunk-card:hover {
        border-color: #C85D3C;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .chunk-card.reviewed {
        background: #2D5F4F;
        border-color: #2D5F4F;
        color: white;
    }

    .chunk-card.current {
        border-color: #C85D3C;
        border-width: 3px;
        box-shadow: 0 0 0 4px rgba(200, 93, 60, 0.1);
    }

    .chunk-card.pending {
        border-style: dashed;
        border-color: #B89968;
    }

    .chunk-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .chunk-status {
        font-size: 0.625rem;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Mention highlight */
    .mention-highlight {
        background: #FFF4E6;
        border-bottom: 2px solid #C85D3C;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📑 Chunk-Based Document Review")
st.markdown("Systematically review large tender documents in manageable chunks")

# Sidebar - Workspace Selection
with st.sidebar:
    st.header("Select Workspace")

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
        key="workspace_select"
    )

    workspace = workspace_options[selected_name]

    st.divider()
    st.metric("Entity", workspace.metadata.entity_name or "Not bound")
    st.metric("Total Mentions", workspace.metadata.total_mentions)

    if workspace.metadata.chunk_mode_enabled:
        st.metric("Chunks Reviewed", f"{workspace.metadata.chunks_reviewed}/{workspace.metadata.total_chunks}")

    st.divider()
    st.caption("🧠 **Analysis Model**")
    st.caption("qwen2.5:72b-instruct-q4_K_M")
    st.caption("(47 GB - High capability)")

# Check if entity bound
if not workspace.metadata.entity_id:
    st.error("❌ Please bind an entity profile to this workspace first (Proposal Workspace page)")
    st.stop()

# Find document file (could be original filename or tender_original.txt)
documents_dir = workspace.workspace_path / "documents"
doc_path = None

# Try original filename first
if (documents_dir / workspace.metadata.original_filename).exists():
    doc_path = documents_dir / workspace.metadata.original_filename
# Try tender_original.txt (common converted name)
elif (documents_dir / "tender_original.txt").exists():
    doc_path = documents_dir / "tender_original.txt"
# Try any .txt file in documents directory
else:
    txt_files = list(documents_dir.glob("*.txt"))
    if txt_files:
        doc_path = txt_files[0]

if not doc_path or not doc_path.exists():
    st.error(f"❌ Document file not found in {documents_dir}")
    st.error(f"Expected: {workspace.metadata.original_filename} or tender_original.txt")
    st.stop()

# Load document content
try:
    with open(doc_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    st.info(f"📄 Loaded document: {doc_path.name}")
except Exception as e:
    st.error(f"❌ Error reading document: {str(e)}")
    st.stop()

# Initialize chunk mode if not already
if not workspace.metadata.chunk_mode_enabled:
    st.info("🔄 Initializing chunk-based review mode...")

    with st.spinner("Creating document chunks..."):
        # Clear old mentions from non-chunk workflow (they have chunk_id=None)
        old_mentions_count = len([m for m in workspace.mentions if m.chunk_id is None])
        if old_mentions_count > 0:
            workspace.mentions = [m for m in workspace.mentions if m.chunk_id is not None]
            st.warning(f"🧹 Cleared {old_mentions_count} old mentions from previous workflow")

        # Create chunks
        chunks = chunker.create_chunks(document_text)

        # Filter to completable chunks only (skip personnel sections)
        completable_chunks = chunker.filter_completable_chunks(chunks)

        st.success(f"✅ Created {len(completable_chunks)} reviewable chunks (filtered from {len(chunks)} total)")

        # Initialize chunk progress in workspace
        workspace.metadata.chunk_mode_enabled = True
        workspace.metadata.total_chunks = len(completable_chunks)
        workspace.metadata.current_chunk_id = 1
        workspace.metadata.chunks_reviewed = 0

        # Create chunk progress entries
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

        # Save workspace
        workspace_manager._save_workspace(workspace)
        workspace_manager._save_bindings(workspace)

        # Store chunks in session state
        st.session_state.document_chunks = completable_chunks

        st.rerun()

# Clean up orphaned mentions from old workflow (chunk_id=None)
orphaned_mentions = [m for m in workspace.mentions if m.chunk_id is None]
if orphaned_mentions and workspace.metadata.chunk_mode_enabled:
    st.warning(f"🧹 Found {len(orphaned_mentions)} orphaned mentions from old workflow. Clearing...")
    workspace.mentions = [m for m in workspace.mentions if m.chunk_id is not None]
    workspace_manager._save_workspace(workspace)
    workspace_manager._save_bindings(workspace)
    st.rerun()

# Load chunks from workspace or session state
if 'document_chunks' not in st.session_state:
    # Recreate chunks (they're not persisted, only progress is)
    chunks = chunker.create_chunks(document_text)
    completable_chunks = chunker.filter_completable_chunks(chunks)
    st.session_state.document_chunks = completable_chunks

chunks = st.session_state.document_chunks

# Check if chunk IDs match between workspace and session chunks
# This handles cases where chunks were created with old numbering scheme
workspace_chunk_ids = {cp.chunk_id for cp in workspace.chunks}
session_chunk_ids = {c.chunk_id for c in chunks}

if workspace_chunk_ids != session_chunk_ids and workspace.metadata.chunks_reviewed == 0:
    # Chunk IDs don't match and no progress made - reinitialize
    st.warning("🔄 Chunk numbering mismatch detected. Reinitializing chunks...")

    # Clear old chunks
    workspace.chunks.clear()
    workspace.metadata.current_chunk_id = 1

    # Reinitialize with new chunk IDs
    for chunk in chunks:
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
    # Don't call st.rerun() - just continue with corrected data

# Get current chunk
current_chunk_id = workspace.metadata.current_chunk_id or 1
current_chunk = next((c for c in chunks if c.chunk_id == current_chunk_id), chunks[0])

# Progress Section
st.header("Review Progress")

# Progress bar
progress_pct = (workspace.metadata.chunks_reviewed / workspace.metadata.total_chunks) * 100 if workspace.metadata.total_chunks > 0 else 0

progress_html = f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;"><span style="font-weight: 600;">Progress</span><span style="color: #C85D3C; font-weight: 600;">{workspace.metadata.chunks_reviewed} of {workspace.metadata.total_chunks} chunks reviewed</span></div><div class="chunk-progress"><div class="chunk-progress-fill" style="width: {progress_pct}%"></div></div>'
st.markdown(progress_html, unsafe_allow_html=True)

# Chunk grid
st.markdown("### Chunks")

chunk_grid_html = '<div class="chunk-grid">'
for chunk_progress in workspace.chunks:
    status_class = ""
    status_text = "Pending"

    if chunk_progress.chunk_id == current_chunk_id:
        status_class = "current"
        status_text = "Current"
    elif chunk_progress.status == "reviewed":
        status_class = "reviewed"
        status_text = "Done"
    else:
        status_class = "pending"

    # Build HTML on single line to avoid formatting issues
    chunk_grid_html += f'<div class="chunk-card {status_class}"><span class="chunk-number">{chunk_progress.chunk_id}</span><span class="chunk-status">{status_text}</span></div>'

chunk_grid_html += '</div>'
st.markdown(chunk_grid_html, unsafe_allow_html=True)

st.divider()

# Current Chunk Review
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"Chunk {current_chunk.chunk_id}: {current_chunk.title}")
    st.caption(f"Lines {current_chunk.start_line}-{current_chunk.end_line} • {current_chunk.char_count} characters")

    # Analyze button
    chunk_progress = next((cp for cp in workspace.chunks if cp.chunk_id == current_chunk_id), None)

    if chunk_progress and chunk_progress.mentions_found == 0:
        if st.button("🤖 Analyze This Chunk", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing chunk {current_chunk.chunk_id} with LLM..."):
                # Analyze chunk
                mentions = markup_engine.analyze_chunk(current_chunk, workspace.metadata.entity_id)

                # Add to workspace
                workspace = workspace_manager.add_mention_bindings(workspace.metadata.workspace_id, mentions)

                # Update chunk progress
                chunk_progress.mentions_found = len(mentions)
                chunk_progress.status = "analyzing"
                workspace_manager._save_workspace(workspace)

                st.success(f"✅ Found {len(mentions)} mentions in this chunk")
                st.rerun()

    # Display chunk content
    st.markdown("### Content")

    # Highlight any mentions in the content
    display_content = current_chunk.content

    # Get mentions for this chunk
    chunk_mentions = [m for m in workspace.mentions if m.chunk_id == current_chunk_id]

    for mention in chunk_mentions:
        if mention.mention_text in display_content:
            display_content = display_content.replace(
                mention.mention_text,
                f'<span class="mention-highlight">{mention.mention_text}</span>'
            )

    # Escape HTML in content but preserve our mention highlights
    import html
    # First escape everything
    safe_content = html.escape(display_content)
    # Then unescape our highlight spans
    safe_content = safe_content.replace('&lt;span class=&quot;mention-highlight&quot;&gt;', '<span class="mention-highlight">')
    safe_content = safe_content.replace('&lt;/span&gt;', '</span>')
    # Preserve line breaks
    safe_content = safe_content.replace('\n', '<br>')

    st.markdown(f'<div style="background: white; padding: 2rem; border-radius: 8px; border: 1px solid #E5E3DF; line-height: 1.8;">{safe_content}</div>', unsafe_allow_html=True)

with col2:
    st.header("Mentions Found")

    # Get pending mentions for this chunk
    pending_mentions = [m for m in chunk_mentions if not m.approved and not m.rejected and not m.ignored]

    if pending_mentions:
        for idx, mention in enumerate(pending_mentions):
            with st.expander(f"📌 {mention.mention_text}", expanded=idx==0):
                st.write(f"**Type:** {mention.mention_type}")
                st.write(f"**Field:** `{mention.field_path}`")
                st.write(f"**Location:** {mention.location}")

                col_approve, col_reject = st.columns(2)

                with col_approve:
                    if st.button("✅ Approve", key=f"approve_{mention.mention_text}_{current_chunk_id}", use_container_width=True):
                        workspace = workspace_manager.update_mention_binding(
                            workspace.metadata.workspace_id,
                            mention.mention_text,
                            approved=True
                        )

                        # Update chunk progress
                        chunk_progress.mentions_approved += 1
                        workspace_manager._save_workspace(workspace)

                        st.rerun()

                with col_reject:
                    if st.button("❌ Reject", key=f"reject_{mention.mention_text}_{current_chunk_id}", use_container_width=True):
                        workspace = workspace_manager.update_mention_binding(
                            workspace.metadata.workspace_id,
                            mention.mention_text,
                            rejected=True
                        )

                        workspace_manager._save_workspace(workspace)
                        st.rerun()
    else:
        if chunk_progress and chunk_progress.mentions_found > 0:
            st.success("✅ All mentions reviewed!")

            # Auto-mark chunk as reviewed when all mentions processed
            if chunk_progress.status != "reviewed":
                chunk_progress.status = "reviewed"
                chunk_progress.reviewed_at = datetime.now()
                workspace.metadata.chunks_reviewed += 1
                workspace_manager._save_workspace(workspace)
                st.info("✨ Chunk marked as complete!")
        elif chunk_progress and chunk_progress.mentions_found == 0:
            # No mentions found - auto-mark as reviewed
            if chunk_progress.status != "reviewed":
                chunk_progress.status = "reviewed"
                chunk_progress.reviewed_at = datetime.now()
                workspace.metadata.chunks_reviewed += 1
                workspace_manager._save_workspace(workspace)
                st.success("✅ No mentions found - chunk marked as complete!")
        else:
            st.info("Click 'Analyze This Chunk' to find mentions")

# Navigation
st.divider()

col_prev, col_next, col_complete = st.columns([1, 1, 1])

with col_prev:
    if st.button("⬅️ Previous Chunk", disabled=(current_chunk_id == 1), use_container_width=True):
        workspace.metadata.current_chunk_id = current_chunk_id - 1
        workspace_manager._save_workspace(workspace)
        st.rerun()

with col_next:
    if st.button("Next Chunk ➡️", disabled=(current_chunk_id == workspace.metadata.total_chunks), use_container_width=True):
        # Just navigate to next chunk (marking as reviewed happens automatically above)
        workspace.metadata.current_chunk_id = current_chunk_id + 1
        workspace_manager._save_workspace(workspace)
        st.rerun()

with col_complete:
    all_reviewed = workspace.metadata.chunks_reviewed == workspace.metadata.total_chunks
    if st.button("📤 Export Final Document", disabled=not all_reviewed, type="primary" if all_reviewed else "secondary", use_container_width=True):
        st.success("✅ Export functionality will be implemented in next phase")
        st.info("All chunks reviewed! Ready to export final document.")

# Keyboard shortcuts hint
st.caption("💡 **Tip:** Use ← → arrow keys to navigate chunks (coming soon)")

# Footer
st.markdown("---")
st.caption("Chunk-Based Review v1.0.0 • Systematic tender document processing")
