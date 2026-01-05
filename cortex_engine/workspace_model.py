"""
Workspace Model
Version: 1.0.0
Date: 2026-01-05

Purpose: Data models for tender response workspaces.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
from pathlib import Path


class WorkspaceState(str, Enum):
    """Workspace workflow states."""
    CREATED = "created"                      # Workspace created, document uploaded
    MARKUP_SUGGESTED = "markup_suggested"    # LLM has suggested @mentions
    MARKUP_REVIEWED = "markup_reviewed"      # Human has reviewed suggestions
    ENTITY_BOUND = "entity_bound"           # Entity profile bound to workspace
    CONTENT_GENERATED = "content_generated"  # LLM content generated
    DRAFT_READY = "draft_ready"             # Draft document ready for review
    IN_REVIEW = "in_review"                 # Under human review
    APPROVED = "approved"                   # Final approval granted
    EXPORTED = "exported"                   # Final document exported


class MentionBinding(BaseModel):
    """Binding of a mention to entity data."""
    mention_text: str = Field(description="Original @mention text (e.g., '@companyname')")
    mention_type: str = Field(description="Type: simple, structured, content_gen, narrative")
    field_path: str = Field(description="Field path in entity (e.g., 'company.legal_name')")
    location: str = Field(description="Location in document (section name or page number)")

    # Review status
    suggested_by_llm: bool = Field(default=True, description="Was this suggested by LLM?")
    approved: bool = Field(default=False, description="Has human approved this mention?")
    modified: bool = Field(default=False, description="Did human modify this mention?")
    rejected: bool = Field(default=False, description="Did human reject this mention?")

    # Resolution
    requires_llm: bool = Field(default=False, description="Requires LLM generation?")
    resolved_value: Optional[str] = Field(default=None, description="Resolved value (if already generated)")
    generation_prompt: Optional[str] = Field(default=None, description="Prompt used for LLM generation")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = Field(default=None)
    generated_at: Optional[datetime] = Field(default=None)


class WorkspaceMetadata(BaseModel):
    """Workspace metadata."""
    workspace_id: str = Field(description="Unique workspace identifier")
    workspace_name: str = Field(description="Human-readable workspace name")

    # Entity binding
    entity_id: Optional[str] = Field(default=None, description="Bound entity profile ID")
    entity_name: Optional[str] = Field(default=None, description="Entity display name")

    # Document info
    tender_name: str = Field(description="Name of tender/RFT")
    tender_reference: Optional[str] = Field(default=None, description="RFT reference number")
    original_filename: str = Field(description="Original uploaded filename")
    document_type: str = Field(default="docx", description="Document type: docx, pdf, txt")

    # Workflow state
    state: WorkspaceState = Field(default=WorkspaceState.CREATED)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Participants
    created_by: Optional[str] = Field(default=None, description="Email of creator")
    reviewers: List[str] = Field(default_factory=list, description="Emails of reviewers")
    approvers: List[str] = Field(default_factory=list, description="Emails of approvers")

    # Progress tracking
    total_mentions: int = Field(default=0, description="Total @mentions in document")
    approved_mentions: int = Field(default=0, description="Approved @mentions")
    rejected_mentions: int = Field(default=0, description="Rejected @mentions")
    generated_mentions: int = Field(default=0, description="LLM-generated @mentions")

    # Tags and metadata
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(default=None)


class WorkspaceConfig(BaseModel):
    """Workspace configuration."""
    # LLM settings
    llm_model: str = Field(default="mistral-small3.2", description="Model for content generation")
    llm_temperature: float = Field(default=0.7, description="Temperature for generation")

    # Generation preferences
    cv_format: str = Field(default="detailed", description="CV format: brief, detailed, comprehensive")
    project_format: str = Field(default="summary", description="Project format: brief, summary, detailed")
    reference_format: str = Field(default="formatted", description="Reference format: simple, formatted, testimonial")

    # Document preferences
    include_toc: bool = Field(default=True, description="Include table of contents")
    page_numbers: bool = Field(default=True, description="Include page numbers")

    # Approval workflow
    require_approval: bool = Field(default=True, description="Require final approval")
    min_approvers: int = Field(default=1, description="Minimum number of approvers")


class GenerationLog(BaseModel):
    """Log entry for LLM generation."""
    mention_text: str
    generation_type: str = Field(description="Type: cv, project_summary, reference, etc.")
    prompt: str
    model: str
    temperature: float
    generated_content: str
    tokens_used: Optional[int] = Field(default=None)
    generation_time: float = Field(description="Time taken in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    approved: bool = Field(default=False)
    modified: bool = Field(default=False)


class ApprovalRecord(BaseModel):
    """Record of approval/review."""
    reviewer_email: str
    reviewer_name: Optional[str] = Field(default=None)
    action: str = Field(description="approved, rejected, requested_changes")
    section: Optional[str] = Field(default=None, description="Specific section reviewed")
    feedback: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)


class Workspace(BaseModel):
    """Complete workspace object."""
    metadata: WorkspaceMetadata
    config: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    mentions: List[MentionBinding] = Field(default_factory=list)
    generation_logs: List[GenerationLog] = Field(default_factory=list)
    approval_records: List[ApprovalRecord] = Field(default_factory=list)

    # Directory path (not stored in YAML)
    workspace_path: Optional[Path] = Field(default=None, exclude=True)

    def get_state_display(self) -> str:
        """Get human-readable state."""
        state_display = {
            WorkspaceState.CREATED: "📝 Created",
            WorkspaceState.MARKUP_SUGGESTED: "🤖 Markup Suggested",
            WorkspaceState.MARKUP_REVIEWED: "👁️ Markup Reviewed",
            WorkspaceState.ENTITY_BOUND: "🔗 Entity Bound",
            WorkspaceState.CONTENT_GENERATED: "✨ Content Generated",
            WorkspaceState.DRAFT_READY: "📄 Draft Ready",
            WorkspaceState.IN_REVIEW: "🔍 In Review",
            WorkspaceState.APPROVED: "✅ Approved",
            WorkspaceState.EXPORTED: "📦 Exported"
        }
        return state_display.get(self.metadata.state, str(self.metadata.state))

    def get_progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.metadata.total_mentions == 0:
            return 0.0

        # Weight different stages
        weights = {
            'approved': 0.4,
            'generated': 0.6
        }

        approved_pct = (self.metadata.approved_mentions / self.metadata.total_mentions) * weights['approved']
        generated_pct = (self.metadata.generated_mentions / self.metadata.total_mentions) * weights['generated']

        return (approved_pct + generated_pct) * 100

    def get_pending_mentions(self) -> List[MentionBinding]:
        """Get mentions pending review."""
        return [m for m in self.mentions if not m.approved and not m.rejected]

    def get_llm_mentions(self) -> List[MentionBinding]:
        """Get mentions requiring LLM generation."""
        return [m for m in self.mentions if m.requires_llm and m.approved and not m.resolved_value]

    def can_transition_to(self, new_state: WorkspaceState) -> bool:
        """Check if transition to new state is valid."""
        valid_transitions = {
            WorkspaceState.CREATED: [WorkspaceState.MARKUP_SUGGESTED],
            WorkspaceState.MARKUP_SUGGESTED: [WorkspaceState.MARKUP_REVIEWED],
            WorkspaceState.MARKUP_REVIEWED: [WorkspaceState.ENTITY_BOUND],
            WorkspaceState.ENTITY_BOUND: [WorkspaceState.CONTENT_GENERATED],
            WorkspaceState.CONTENT_GENERATED: [WorkspaceState.DRAFT_READY],
            WorkspaceState.DRAFT_READY: [WorkspaceState.IN_REVIEW],
            WorkspaceState.IN_REVIEW: [WorkspaceState.APPROVED, WorkspaceState.MARKUP_REVIEWED],
            WorkspaceState.APPROVED: [WorkspaceState.EXPORTED],
            WorkspaceState.EXPORTED: []
        }

        return new_state in valid_transitions.get(self.metadata.state, [])
