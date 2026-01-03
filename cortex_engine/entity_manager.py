"""
Entity Manager - Manage Organizational Entities for Tender Responses
Version: 1.0.0
Date: 2026-01-03

Purpose: Manage multiple organizational entities (e.g., longboardfella, Deakin, Escient)
with separate structured data profiles. Each entity has selected source documents
and its own extracted structured knowledge.

Key Features:
- Create/manage entity profiles
- Link entities to specific KB folders/documents
- Track extraction status per entity
- Selective extraction from chosen documents only
- Support multiple entities with separate data
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Type of organizational entity."""
    MY_COMPANY = "my_company"           # Primary trading entities
    CLIENT = "client"                   # Client organizations
    PARTNER = "partner"                 # Partnership entities
    SUBSIDIARY = "subsidiary"           # Subsidiary companies
    OTHER = "other"


class ExtractionStatus(str, Enum):
    """Extraction status for an entity."""
    NEVER = "never"                     # Never extracted
    EXTRACTING = "extracting"           # Currently extracting
    COMPLETE = "complete"               # Extraction complete
    STALE = "stale"                     # Needs re-extraction (>30 days)
    ERROR = "error"                     # Extraction failed


class EntityProfile(BaseModel):
    """Profile for an organizational entity."""

    # Identity
    entity_id: str = Field(description="Unique identifier (slug)")
    entity_name: str = Field(description="Display name")
    entity_type: EntityType = Field(description="Type of entity")
    description: Optional[str] = Field(None, description="User description")

    # Source selection
    source_folders: List[str] = Field(default_factory=list, description="KB folder paths to extract from")
    source_document_ids: List[str] = Field(default_factory=list, description="Specific KB document IDs")
    include_subfolders: bool = Field(True, description="Include documents from subfolders")

    # Extraction tracking
    last_extracted: Optional[datetime] = Field(None, description="When extraction last ran")
    extraction_status: ExtractionStatus = Field(ExtractionStatus.NEVER, description="Current status")
    extraction_error: Optional[str] = Field(None, description="Last error message if failed")

    # Data reference
    structured_data_file: str = Field(description="Path to structured knowledge JSON")

    # Metadata
    created_date: datetime = Field(default_factory=datetime.now)
    modified_date: datetime = Field(default_factory=datetime.now)

    # Statistics
    source_document_count: int = Field(0, description="Number of source documents selected")
    data_completeness: Dict[str, bool] = Field(
        default_factory=dict,
        description="Completeness flags: organization, insurances, qualifications, etc."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "source_folders": self.source_folders,
            "source_document_ids": self.source_document_ids,
            "include_subfolders": self.include_subfolders,
            "last_extracted": self.last_extracted.isoformat() if self.last_extracted else None,
            "extraction_status": self.extraction_status.value,
            "extraction_error": self.extraction_error,
            "structured_data_file": self.structured_data_file,
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "source_document_count": self.source_document_count,
            "data_completeness": self.data_completeness
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityProfile":
        """Load from dictionary."""
        # Parse datetime fields
        if isinstance(data.get('last_extracted'), str):
            data['last_extracted'] = datetime.fromisoformat(data['last_extracted'])
        if isinstance(data.get('created_date'), str):
            data['created_date'] = datetime.fromisoformat(data['created_date'])
        if isinstance(data.get('modified_date'), str):
            data['modified_date'] = datetime.fromisoformat(data['modified_date'])

        return cls(**data)

    @property
    def is_stale(self) -> bool:
        """Check if extraction is stale (>30 days)."""
        if not self.last_extracted or self.extraction_status == ExtractionStatus.NEVER:
            return False
        age_days = (datetime.now() - self.last_extracted).days
        return age_days > 30

    @property
    def age_days(self) -> Optional[int]:
        """Get age of extraction in days."""
        if not self.last_extracted:
            return None
        return (datetime.now() - self.last_extracted).days


class EntityManager:
    """
    Manages organizational entities and their structured data profiles.
    Stores entity metadata and tracks extraction status.
    """

    def __init__(self, db_path: Path):
        """
        Initialize entity manager.

        Args:
            db_path: Database root path (stores entities.json and structured_data/)
        """
        self.db_path = Path(db_path)
        self.entities_file = self.db_path / "entities.json"
        self.structured_data_dir = self.db_path / "structured_data"

        # Ensure directories exist
        self.structured_data_dir.mkdir(parents=True, exist_ok=True)

        # Load entities
        self.entities: Dict[str, EntityProfile] = self._load_entities()

    def _load_entities(self) -> Dict[str, EntityProfile]:
        """Load all entities from storage."""
        try:
            if not self.entities_file.exists():
                return {}

            with open(self.entities_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entities = {}
            for entity_dict in data.get('entities', []):
                try:
                    entity = EntityProfile.from_dict(entity_dict)
                    entities[entity.entity_id] = entity
                except Exception as e:
                    logger.error(f"Failed to load entity: {e}")

            logger.info(f"Loaded {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Failed to load entities: {e}")
            return {}

    def _save_entities(self):
        """Save all entities to storage."""
        try:
            data = {
                "version": "1.0.0",
                "last_modified": datetime.now().isoformat(),
                "entities": [entity.to_dict() for entity in self.entities.values()]
            }

            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Saved {len(self.entities)} entities")

        except Exception as e:
            logger.error(f"Failed to save entities: {e}")
            raise

    def create_entity(
        self,
        entity_name: str,
        entity_type: EntityType,
        description: Optional[str] = None,
        source_folders: Optional[List[str]] = None,
        source_document_ids: Optional[List[str]] = None
    ) -> EntityProfile:
        """
        Create a new entity profile.

        Args:
            entity_name: Display name for entity
            entity_type: Type of entity
            description: Optional description
            source_folders: KB folder paths to extract from
            source_document_ids: Specific document IDs to extract from

        Returns:
            Created EntityProfile
        """
        # Generate entity ID (slug)
        entity_id = entity_name.lower().replace(' ', '_').replace('pty', '').replace('ltd', '').strip('_')
        entity_id = ''.join(c for c in entity_id if c.isalnum() or c == '_')

        # Check if entity already exists
        if entity_id in self.entities:
            raise ValueError(f"Entity with ID '{entity_id}' already exists")

        # Create structured data file path
        structured_data_file = str(self.structured_data_dir / f"{entity_id}.json")

        # Create entity profile
        entity = EntityProfile(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            description=description,
            source_folders=source_folders or [],
            source_document_ids=source_document_ids or [],
            structured_data_file=structured_data_file
        )

        # Save entity
        self.entities[entity_id] = entity
        self._save_entities()

        logger.info(f"Created entity: {entity_name} ({entity_id})")
        return entity

    def list_entities(self) -> List[EntityProfile]:
        """Get list of all entities."""
        return list(self.entities.values())

    def get_entity(self, entity_id: str) -> Optional[EntityProfile]:
        """Get specific entity by ID."""
        return self.entities.get(entity_id)

    def update_entity(
        self,
        entity_id: str,
        **updates
    ) -> EntityProfile:
        """
        Update entity profile.

        Args:
            entity_id: Entity to update
            **updates: Fields to update

        Returns:
            Updated EntityProfile
        """
        entity = self.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity '{entity_id}' not found")

        # Update fields
        for key, value in updates.items():
            if hasattr(entity, key):
                setattr(entity, key, value)

        # Update modified date
        entity.modified_date = datetime.now()

        # Save
        self._save_entities()

        logger.info(f"Updated entity: {entity_id}")
        return entity

    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete entity and its structured data.

        Args:
            entity_id: Entity to delete

        Returns:
            True if deleted, False if not found
        """
        entity = self.entities.get(entity_id)
        if not entity:
            return False

        # Delete structured data file
        try:
            structured_file = Path(entity.structured_data_file)
            if structured_file.exists():
                structured_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete structured data file: {e}")

        # Remove entity
        del self.entities[entity_id]
        self._save_entities()

        logger.info(f"Deleted entity: {entity_id}")
        return True

    def update_extraction_status(
        self,
        entity_id: str,
        status: ExtractionStatus,
        error: Optional[str] = None
    ):
        """Update extraction status for an entity."""
        entity = self.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity '{entity_id}' not found")

        entity.extraction_status = status
        entity.extraction_error = error

        if status == ExtractionStatus.COMPLETE:
            entity.last_extracted = datetime.now()

        entity.modified_date = datetime.now()
        self._save_entities()

    def get_entities_by_type(self, entity_type: EntityType) -> List[EntityProfile]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_stale_entities(self) -> List[EntityProfile]:
        """Get entities with stale extractions (>30 days)."""
        return [e for e in self.entities.values() if e.is_stale]

    def update_data_completeness(
        self,
        entity_id: str,
        completeness: Dict[str, bool]
    ):
        """
        Update data completeness flags for an entity.

        Args:
            entity_id: Entity to update
            completeness: Dict of category -> has_data flags
                         e.g., {"organization": True, "insurances": True, "qualifications": False}
        """
        entity = self.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity '{entity_id}' not found")

        entity.data_completeness = completeness
        entity.modified_date = datetime.now()
        self._save_entities()

    def get_entity_structured_data_path(self, entity_id: str) -> Optional[Path]:
        """Get path to structured data file for an entity."""
        entity = self.entities.get(entity_id)
        if not entity:
            return None
        return Path(entity.structured_data_file)
