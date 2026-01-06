"""
Workspace Manager
Version: 2.0.0
Date: 2026-01-05

Purpose: Manage tender response workspaces with mention-based proposal system.
"""

import yaml
import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .workspace_model import (
    Workspace,
    WorkspaceMetadata,
    WorkspaceConfig,
    WorkspaceState,
    MentionBinding,
    GenerationLog,
    ApprovalRecord
)
from .workspace_git import WorkspaceGit, WorkspaceGitOperations
from .entity_profile_manager import EntityProfileManager
from .entity_profile_schema import CustomField
from .utils import get_logger

logger = get_logger(__name__)


class WorkspaceManager:
    """Manager for tender response workspaces."""

    def __init__(self, workspaces_base_path: Path):
        """
        Initialize workspace manager.

        Args:
            workspaces_base_path: Base directory for all workspaces
        """
        self.base_path = Path(workspaces_base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"WorkspaceManager initialized at {self.base_path}")

    def create_workspace(
        self,
        workspace_id: str,
        workspace_name: str,
        tender_name: str,
        tender_reference: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Workspace:
        """
        Create new workspace.

        Args:
            workspace_id: Unique identifier (e.g., "workspace_RFT12345_companyname_2026-01-05")
            workspace_name: Human-readable name
            tender_name: Tender/RFT name
            tender_reference: Optional RFT reference number
            created_by: Optional creator email

        Returns:
            Workspace object

        Example:
            >>> manager = WorkspaceManager(Path("/workspaces"))
            >>> ws = manager.create_workspace(
            ...     workspace_id="workspace_RFT12345_longboardfella_2026-01-05",
            ...     workspace_name="RFT12345 - Digital Services",
            ...     tender_name="Department of Digital Services - Consulting",
            ...     tender_reference="RFT12345",
            ...     created_by="paul@longboardfella.com.au"
            ... )
        """
        workspace_dir = self.base_path / workspace_id

        if workspace_dir.exists():
            raise ValueError(f"Workspace already exists: {workspace_id}")

        # Create workspace structure
        workspace_dir.mkdir(parents=True)
        (workspace_dir / "documents").mkdir()
        (workspace_dir / "exports").mkdir()

        # Create metadata
        metadata = WorkspaceMetadata(
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            tender_name=tender_name,
            tender_reference=tender_reference,
            original_filename="",  # Will be set when document uploaded
            created_by=created_by,
            state=WorkspaceState.CREATED
        )

        # Create workspace
        workspace = Workspace(
            metadata=metadata,
            config=WorkspaceConfig(),
            workspace_path=workspace_dir
        )

        # Save metadata
        self._save_workspace(workspace)

        # Initialize git
        git = WorkspaceGit(workspace_dir)
        git.commit_changes(f"Workspace created: {workspace_name}")

        logger.info(f"Created workspace: {workspace_id}")

        return workspace

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """
        Get workspace by ID.

        Args:
            workspace_id: Workspace identifier

        Returns:
            Workspace or None if not found
        """
        workspace_dir = self.base_path / workspace_id
        metadata_path = workspace_dir / "metadata.yaml"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            data = yaml.safe_load(f)

        # Load mention bindings
        bindings_path = workspace_dir / "field_bindings.yaml"
        mentions = []
        if bindings_path.exists():
            with open(bindings_path, 'r') as f:
                bindings_data = yaml.safe_load(f) or []
                mentions = [MentionBinding(**m) for m in bindings_data]

        # Load generation logs
        logs_path = workspace_dir / "generation_log.json"
        logs = []
        if logs_path.exists():
            with open(logs_path, 'r') as f:
                logs_data = json.load(f)
                logs = [GenerationLog(**log) for log in logs_data]

        # Load approval records
        approvals_path = workspace_dir / "approval_status.yaml"
        approvals = []
        if approvals_path.exists():
            with open(approvals_path, 'r') as f:
                approvals_data = yaml.safe_load(f) or []
                approvals = [ApprovalRecord(**a) for a in approvals_data]

        workspace = Workspace(
            metadata=WorkspaceMetadata(**data['metadata']),
            config=WorkspaceConfig(**data.get('config', {})),
            mentions=mentions,
            generation_logs=logs,
            approval_records=approvals,
            workspace_path=workspace_dir
        )

        return workspace

    def list_workspaces(self, state: Optional[WorkspaceState] = None) -> List[Workspace]:
        """
        List all workspaces, optionally filtered by state.

        Args:
            state: Optional state filter

        Returns:
            List of workspaces
        """
        workspaces = []

        for workspace_dir in self.base_path.iterdir():
            if not workspace_dir.is_dir() or workspace_dir.name.startswith('_'):
                continue

            workspace = self.get_workspace(workspace_dir.name)
            if workspace:
                if state is None or workspace.metadata.state == state:
                    workspaces.append(workspace)

        # Sort by updated_at desc
        workspaces.sort(key=lambda w: w.metadata.updated_at, reverse=True)

        return workspaces

    def update_workspace_state(
        self,
        workspace_id: str,
        new_state: WorkspaceState,
        user_email: Optional[str] = None
    ) -> Workspace:
        """
        Update workspace state.

        Args:
            workspace_id: Workspace identifier
            new_state: New state
            user_email: Optional user email for git commit

        Returns:
            Updated workspace
        """
        workspace = self.get_workspace(workspace_id)

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        if not workspace.can_transition_to(new_state):
            raise ValueError(
                f"Invalid state transition: {workspace.metadata.state.value} -> {new_state.value}"
            )

        workspace.metadata.state = new_state
        workspace.metadata.updated_at = datetime.now()

        self._save_workspace(workspace)

        # Git commit
        git = WorkspaceGit(workspace.workspace_path)
        git.commit_changes(
            f"State transition: {new_state.value}",
            author=f"{user_email} <{user_email}>" if user_email else None
        )

        logger.info(f"Updated workspace {workspace_id} to state: {new_state.value}")

        return workspace

    def bind_entity(
        self,
        workspace_id: str,
        entity_id: str,
        entity_name: str
    ) -> Workspace:
        """
        Bind entity profile to workspace.

        Args:
            workspace_id: Workspace identifier
            entity_id: Entity profile ID
            entity_name: Entity display name

        Returns:
            Updated workspace
        """
        workspace = self.get_workspace(workspace_id)

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        workspace.metadata.entity_id = entity_id
        workspace.metadata.entity_name = entity_name
        workspace.metadata.updated_at = datetime.now()

        self._save_workspace(workspace)

        # Git commit
        git = WorkspaceGit(workspace.workspace_path)
        WorkspaceGitOperations.on_entity_bound(git, entity_name)

        logger.info(f"Bound entity {entity_id} to workspace {workspace_id}")

        return workspace

    def add_mention_bindings(
        self,
        workspace_id: str,
        mentions: List[MentionBinding]
    ) -> Workspace:
        """
        Add mention bindings to workspace.

        Args:
            workspace_id: Workspace identifier
            mentions: List of mention bindings

        Returns:
            Updated workspace
        """
        workspace = self.get_workspace(workspace_id)

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        workspace.mentions.extend(mentions)
        workspace.metadata.total_mentions = len(workspace.mentions)
        workspace.metadata.updated_at = datetime.now()

        self._save_workspace(workspace)
        self._save_bindings(workspace)

        logger.info(f"Added {len(mentions)} mention bindings to workspace {workspace_id}")

        return workspace

    def update_mention_binding(
        self,
        workspace_id: str,
        mention_text: str,
        approved: Optional[bool] = None,
        rejected: Optional[bool] = None,
        ignored: Optional[bool] = None,
        resolved_value: Optional[str] = None
    ) -> Workspace:
        """
        Update mention binding status.

        Args:
            workspace_id: Workspace identifier
            mention_text: Mention text to update
            approved: Set approval status
            rejected: Set rejection status
            ignored: Set ignored status (not relevant)
            resolved_value: Set resolved value

        Returns:
            Updated workspace
        """
        workspace = self.get_workspace(workspace_id)

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        # Find mention
        mention = next((m for m in workspace.mentions if m.mention_text == mention_text), None)

        if not mention:
            raise ValueError(f"Mention not found: {mention_text}")

        # Update status
        if approved is not None:
            mention.approved = approved
            if approved:
                mention.reviewed_at = datetime.now()

        if rejected is not None:
            mention.rejected = rejected
            if rejected:
                mention.reviewed_at = datetime.now()

        if ignored is not None:
            mention.ignored = ignored
            if ignored:
                mention.reviewed_at = datetime.now()

        if resolved_value is not None:
            mention.resolved_value = resolved_value
            mention.generated_at = datetime.now()

        # Update counts
        workspace.metadata.approved_mentions = sum(1 for m in workspace.mentions if m.approved)
        workspace.metadata.rejected_mentions = sum(1 for m in workspace.mentions if m.rejected)
        workspace.metadata.ignored_mentions = sum(1 for m in workspace.mentions if m.ignored)
        workspace.metadata.generated_mentions = sum(1 for m in workspace.mentions if m.resolved_value)
        workspace.metadata.updated_at = datetime.now()

        self._save_workspace(workspace)
        self._save_bindings(workspace)

        return workspace

    def replace_mention_with_custom_field(
        self,
        workspace_id: str,
        mention_text: str,
        custom_field_name: str,
        custom_field_value: str,
        custom_field_description: Optional[str],
        entity_profile_manager: EntityProfileManager
    ) -> Workspace:
        """
        Replace a mention with a custom field and save to entity profile.

        Args:
            workspace_id: Workspace identifier
            mention_text: Mention text to replace
            custom_field_name: Name for the custom field
            custom_field_value: Value for the custom field
            custom_field_description: Optional description
            entity_profile_manager: Entity profile manager instance

        Returns:
            Updated workspace
        """
        workspace = self.get_workspace(workspace_id)

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        # Get entity ID from workspace
        if not workspace.metadata.entity_id:
            raise ValueError(f"Workspace has no entity bound: {workspace_id}")

        # Load entity profile
        entity_profile = entity_profile_manager.get_entity_profile(workspace.metadata.entity_id)

        if not entity_profile:
            raise ValueError(f"Entity profile not found: {workspace.metadata.entity_id}")

        # Add custom field to entity profile
        custom_field = entity_profile.add_custom_field(
            field_name=custom_field_name,
            field_value=custom_field_value,
            description=custom_field_description
        )

        # Save updated profile
        entity_profile_manager._save_profile(entity_profile)

        logger.info(f"Added custom field '{custom_field_name}' to entity {workspace.metadata.entity_id}")

        # Find and update the mention
        mention = next((m for m in workspace.mentions if m.mention_text == mention_text), None)

        if mention:
            # Mark original mention as rejected
            mention.rejected = True
            mention.reviewed_at = datetime.now()

        # Create new mention binding for the custom field
        new_mention = MentionBinding(
            mention_text=f"@{custom_field.field_name}",
            field_path=f"custom_fields.{custom_field.field_name}",
            location=mention.location if mention else "Custom",
            mention_type="simple",
            approved=True,  # Auto-approve custom fields
            resolved_value=custom_field.field_value,
            reviewed_at=datetime.now()
        )

        workspace.mentions.append(new_mention)

        # Update counts
        workspace.metadata.approved_mentions = sum(1 for m in workspace.mentions if m.approved)
        workspace.metadata.rejected_mentions = sum(1 for m in workspace.mentions if m.rejected)
        workspace.metadata.total_mentions = len(workspace.mentions)
        workspace.metadata.updated_at = datetime.now()

        self._save_workspace(workspace)
        self._save_bindings(workspace)

        # Git commit
        git_ops = WorkspaceGit(workspace)
        git_ops.commit(f"Replace mention with custom field: {custom_field.field_name}")

        logger.info(f"Replaced mention '{mention_text}' with custom field '@{custom_field.field_name}'")

        return workspace

    def add_generation_log(
        self,
        workspace_id: str,
        log: GenerationLog
    ) -> Workspace:
        """
        Add generation log entry.

        Args:
            workspace_id: Workspace identifier
            log: Generation log entry

        Returns:
            Updated workspace
        """
        workspace = self.get_workspace(workspace_id)

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        workspace.generation_logs.append(log)
        workspace.metadata.updated_at = datetime.now()

        self._save_workspace(workspace)
        self._save_generation_logs(workspace)

        return workspace

    def add_approval_record(
        self,
        workspace_id: str,
        record: ApprovalRecord
    ) -> Workspace:
        """
        Add approval record.

        Args:
            workspace_id: Workspace identifier
            record: Approval record

        Returns:
            Updated workspace
        """
        workspace = self.get_workspace(workspace_id)

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        workspace.approval_records.append(record)
        workspace.metadata.updated_at = datetime.now()

        self._save_workspace(workspace)
        self._save_approval_records(workspace)

        return workspace

    def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete workspace (moves to archive).

        Args:
            workspace_id: Workspace identifier

        Returns:
            True if deleted
        """
        workspace_dir = self.base_path / workspace_id

        if not workspace_dir.exists():
            return False

        # Move to archive
        archive_dir = self.base_path / "_archive"
        archive_dir.mkdir(exist_ok=True)

        archived_path = archive_dir / f"{workspace_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(workspace_dir), str(archived_path))

        logger.info(f"Archived workspace: {workspace_id}")

        return True

    # ========================================
    # PRIVATE METHODS
    # ========================================

    def _save_workspace(self, workspace: Workspace):
        """Save workspace metadata."""
        metadata_path = workspace.workspace_path / "metadata.yaml"

        data = {
            'metadata': workspace.metadata.model_dump(mode='json'),
            'config': workspace.config.model_dump(mode='json')
        }

        with open(metadata_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _save_bindings(self, workspace: Workspace):
        """Save mention bindings."""
        bindings_path = workspace.workspace_path / "field_bindings.yaml"

        data = [m.model_dump(mode='json') for m in workspace.mentions]

        with open(bindings_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _save_generation_logs(self, workspace: Workspace):
        """Save generation logs."""
        logs_path = workspace.workspace_path / "generation_log.json"

        data = [log.model_dump(mode='json') for log in workspace.generation_logs]

        with open(logs_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_approval_records(self, workspace: Workspace):
        """Save approval records."""
        approvals_path = workspace.workspace_path / "approval_status.yaml"

        data = [record.model_dump(mode='json') for record in workspace.approval_records]

        with open(approvals_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
