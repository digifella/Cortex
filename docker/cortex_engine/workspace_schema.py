"""
Workspace Schema - Tender Workspace Data Models
Version: 1.0.0
Date: 2026-01-03

Purpose: Pydantic models for tender workspace management.
Workspaces are per-tender collections that contain tender-specific context.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pathlib import Path


class WorkspaceStatus(str, Enum):
    """Status of a tender workspace."""
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    FIELD_MATCHING = "field_matching"
    READY_TO_FILL = "ready_to_fill"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class DocumentSource(str, Enum):
    """Source type for documents in workspace."""
    TENDER_DOCUMENT = "tender_document"
    ENTITY_DATA = "entity_data"
    USER_RESEARCH = "user_research"
    USER_NOTES = "user_notes"
    CUSTOM_NARRATIVE = "custom_narrative"
    ADDITIONAL_UPLOAD = "additional_upload"


class WorkspaceDocument(BaseModel):
    """A document or chunk in the workspace collection."""
    doc_id: str = Field(description="Unique document ID in workspace collection")
    source_type: DocumentSource
    content: str = Field(description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    added_date: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class FieldMapping(BaseModel):
    """Mapping between a tender field and source data."""
    field_id: str = Field(description="Unique field identifier")
    field_location: str = Field(description="Location in tender (e.g., 'Table 3, Row 5')")
    field_description: str = Field(description="What the field is asking for")
    field_type: Optional[str] = Field(None, description="Classified type (ABN, insurance, etc.)")

    matched_data: Optional[str] = Field(None, description="Data to fill this field")
    data_source: Optional[str] = Field(None, description="Where the data came from")
    confidence: Optional[float] = Field(None, description="Match confidence 0-1")

    user_approved: bool = Field(default=False)
    user_override: Optional[str] = Field(None, description="User's manual override")

    class Config:
        use_enum_values = True


class WorkspaceMetadata(BaseModel):
    """Metadata for a tender workspace."""
    workspace_id: str = Field(description="Unique workspace ID")
    workspace_name: str = Field(description="Human-readable name")

    # Tender info
    tender_id: str = Field(description="Tender identifier (e.g., RFT12493)")
    tender_filename: str = Field(description="Original tender document filename")
    tender_uploaded_date: datetime = Field(default_factory=datetime.now)

    # Entity info
    entity_id: Optional[str] = Field(None, description="Associated entity ID")
    entity_name: Optional[str] = Field(None, description="Entity name")

    # Status
    status: WorkspaceStatus = Field(default=WorkspaceStatus.CREATED)
    created_date: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    completed_date: Optional[datetime] = None

    # Collections
    collection_name: str = Field(description="ChromaDB collection name for this workspace")

    # Data snapshots
    entity_snapshot_file: Optional[str] = Field(None, description="Path to entity JSON snapshot")
    field_mappings_file: Optional[str] = Field(None, description="Path to field mappings JSON")

    # Statistics
    document_count: int = Field(default=0)
    field_count: int = Field(default=0)
    matched_field_count: int = Field(default=0)

    # Export history
    export_history: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        use_enum_values = True

    @property
    def is_active(self) -> bool:
        """Check if workspace is actively being worked on."""
        return self.status in [
            WorkspaceStatus.CREATED,
            WorkspaceStatus.IN_PROGRESS,
            WorkspaceStatus.FIELD_MATCHING,
            WorkspaceStatus.READY_TO_FILL
        ]

    @property
    def age_days(self) -> int:
        """Days since workspace creation."""
        return (datetime.now() - self.created_date).days

    @property
    def completion_percentage(self) -> float:
        """Percentage of fields matched."""
        if self.field_count == 0:
            return 0.0
        return (self.matched_field_count / self.field_count) * 100


class WorkspaceSnapshot(BaseModel):
    """Complete snapshot of workspace state for export/backup."""
    metadata: WorkspaceMetadata
    documents: List[WorkspaceDocument]
    field_mappings: List[FieldMapping]
    entity_data_snapshot: Optional[Dict[str, Any]] = None

    def to_json_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        return {
            "metadata": self.metadata.model_dump(),
            "documents": [doc.model_dump() for doc in self.documents],
            "field_mappings": [fm.model_dump() for fm in self.field_mappings],
            "entity_data_snapshot": self.entity_data_snapshot
        }
