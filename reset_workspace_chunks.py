#!/usr/bin/env python3
"""
Reset workspace chunk mode to force re-chunking with new classification logic.
"""
from pathlib import Path
from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path

# Load config
config = ConfigManager().get_config()
db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))
workspaces_path = Path(db_path) / "workspaces"

# Initialize manager
manager = WorkspaceManager(workspaces_path)

# Get the DHA workspace
workspace_id = "workspace_rft12493_longboardfella_consulting_2026-01-06"
workspace = manager.get_workspace(workspace_id)

if not workspace:
    print(f"❌ Workspace not found: {workspace_id}")
    exit(1)

print(f"📋 Current workspace state:")
print(f"   Chunk mode enabled: {workspace.metadata.chunk_mode_enabled}")
print(f"   Total chunks: {workspace.metadata.total_chunks}")
print(f"   Chunks reviewed: {workspace.metadata.chunks_reviewed}")
print(f"   Chunks in list: {len(workspace.chunks)}")

# Reset chunk mode to force re-initialization
workspace.metadata.chunk_mode_enabled = False
workspace.metadata.total_chunks = 0
workspace.metadata.chunks_reviewed = 0
workspace.metadata.current_chunk_id = None
workspace.chunks = []

manager._save_workspace(workspace)

print(f"\n✅ Reset complete!")
print(f"   Chunk mode disabled - will re-initialize on next page load")
print(f"   All chunks cleared")
print(f"\nNext steps:")
print(f"   1. Kill Streamlit: pkill -9 streamlit")
print(f"   2. Restart Streamlit: streamlit run Cortex_Suite.py")
print(f"   3. Go to Proposal Chunk Review page")
print(f"   4. Select the DHA workspace")
print(f"   5. It will re-chunk with new classification logic")
print(f"   6. You should see ~27 chunks instead of 2!")
