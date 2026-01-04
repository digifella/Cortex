"""
Workspace Manager - Tender Workspace Management
Version: 1.0.0
Date: 2026-01-03

Purpose: Manage tender workspaces with ChromaDB collections.
Each workspace is a per-tender working environment with:
- ChromaDB collection for semantic search
- JSON snapshots of entity data
- Field mappings
- User additions (notes, research, narratives)
"""

import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings as ChromaSettings

from .workspace_schema import (
    WorkspaceMetadata,
    WorkspaceStatus,
    WorkspaceDocument,
    DocumentSource,
    FieldMapping,
    WorkspaceSnapshot
)
from .utils import get_logger

logger = get_logger(__name__)


class WorkspaceManager:
    """Manages tender workspaces with ChromaDB collections."""

    def __init__(self, db_path: Path):
        """
        Initialize workspace manager.

        Args:
            db_path: Path to AI database directory
        """
        self.db_path = Path(db_path)
        self.workspaces_dir = self.db_path / "workspaces"
        self.workspaces_dir.mkdir(exist_ok=True)

        # ChromaDB setup
        self.chroma_db_path = str(self.db_path / "knowledge_hub_db")
        db_settings = ChromaSettings(anonymized_telemetry=False)
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=db_settings
        )

        logger.info(f"WorkspaceManager initialized at {self.workspaces_dir}")

    def create_workspace(
        self,
        tender_id: str,
        tender_filename: str,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None
    ) -> WorkspaceMetadata:
        """
        Create a new tender workspace.

        Args:
            tender_id: Tender identifier (e.g., "RFT12493")
            tender_filename: Original tender document filename
            entity_id: Optional entity ID
            entity_name: Optional entity name

        Returns:
            WorkspaceMetadata: Created workspace metadata
        """
        # Generate workspace ID
        date_str = datetime.now().strftime("%Y-%m-%d")
        entity_slug = entity_id.lower().replace(" ", "_") if entity_id else "noentity"
        workspace_id = f"workspace_{tender_id}_{entity_slug}_{date_str}"

        # Collection name (ChromaDB naming rules)
        collection_name = workspace_id.replace("-", "_").lower()

        # Create workspace directory
        workspace_dir = self.workspaces_dir / workspace_id
        workspace_dir.mkdir(exist_ok=True)

        # Create ChromaDB collection
        try:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "workspace_id": workspace_id,
                    "tender_id": tender_id,
                    "hnsw:space": "cosine"
                }
            )
            logger.info(f"Created ChromaDB collection: {collection_name}")
        except Exception as e:
            # Collection might already exist
            logger.warning(f"Collection {collection_name} might exist: {e}")
            collection = self.chroma_client.get_collection(collection_name)

        # Create workspace metadata
        metadata = WorkspaceMetadata(
            workspace_id=workspace_id,
            workspace_name=f"{tender_id} - {entity_name or 'No Entity'}",
            tender_id=tender_id,
            tender_filename=tender_filename,
            entity_id=entity_id,
            entity_name=entity_name,
            collection_name=collection_name,
            status=WorkspaceStatus.CREATED
        )

        # Save metadata
        self._save_metadata(workspace_id, metadata)

        logger.info(f"Created workspace: {workspace_id}")
        return metadata

    def get_workspace(self, workspace_id: str) -> Optional[WorkspaceMetadata]:
        """
        Get workspace metadata by ID.

        Args:
            workspace_id: Workspace ID

        Returns:
            WorkspaceMetadata or None if not found
        """
        metadata_file = self.workspaces_dir / workspace_id / "metadata.json"
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)

                # Convert string dates back to datetime objects
                for date_field in ['created_date', 'last_modified', 'completed_date', 'tender_uploaded_date']:
                    if date_field in data and data[date_field] and isinstance(data[date_field], str):
                        from datetime import datetime as dt
                        data[date_field] = dt.fromisoformat(data[date_field].replace('Z', '+00:00'))

                return WorkspaceMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load workspace {workspace_id}: {e}")
            return None

    def list_workspaces(self, include_archived: bool = False) -> List[WorkspaceMetadata]:
        """
        List all workspaces.

        Args:
            include_archived: Include archived workspaces

        Returns:
            List of workspace metadata
        """
        workspaces = []

        for workspace_dir in self.workspaces_dir.iterdir():
            if workspace_dir.is_dir():
                metadata = self.get_workspace(workspace_dir.name)
                if metadata:
                    if include_archived or metadata.status != WorkspaceStatus.ARCHIVED:
                        workspaces.append(metadata)

        return sorted(workspaces, key=lambda w: w.created_date, reverse=True)

    def update_workspace_status(
        self,
        workspace_id: str,
        status: WorkspaceStatus
    ) -> bool:
        """
        Update workspace status.

        Args:
            workspace_id: Workspace ID
            status: New status

        Returns:
            True if successful
        """
        metadata = self.get_workspace(workspace_id)
        if not metadata:
            return False

        metadata.status = status
        metadata.last_modified = datetime.now()

        if status == WorkspaceStatus.COMPLETED:
            metadata.completed_date = datetime.now()

        self._save_metadata(workspace_id, metadata)
        return True

    def add_document_to_workspace(
        self,
        workspace_id: str,
        content: str,
        source_type: DocumentSource,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> bool:
        """
        Add a document/chunk to workspace collection.

        Args:
            workspace_id: Workspace ID
            content: Document text content
            source_type: Source type (tender doc, entity data, user notes, etc.)
            metadata: Optional metadata dict
            doc_id: Optional custom document ID

        Returns:
            True if successful
        """
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            logger.error(f"Workspace {workspace_id} not found")
            return False

        try:
            collection = self.chroma_client.get_collection(workspace.collection_name)

            # Generate doc ID if not provided
            if not doc_id:
                doc_count = collection.count()
                doc_id = f"{source_type.value}_{doc_count + 1}"

            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata["source_type"] = source_type.value
            doc_metadata["added_date"] = datetime.now().isoformat()

            # Generate embedding using our embedding service
            from .embedding_service import embed_query
            content_embedding = embed_query(content)

            # Add to collection with custom embedding
            collection.add(
                documents=[content],
                metadatas=[doc_metadata],
                embeddings=[content_embedding],
                ids=[doc_id]
            )

            # Update workspace stats
            workspace.document_count = collection.count()
            workspace.last_modified = datetime.now()
            self._save_metadata(workspace_id, workspace)

            logger.info(f"Added document {doc_id} to workspace {workspace_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document to workspace {workspace_id}: {e}")
            return False

    def add_entity_snapshot(
        self,
        workspace_id: str,
        entity_data: Dict[str, Any]
    ) -> bool:
        """
        Save entity data snapshot to workspace.

        Args:
            workspace_id: Workspace ID
            entity_data: Entity structured data

        Returns:
            True if successful
        """
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False

        workspace_dir = self.workspaces_dir / workspace_id
        snapshot_file = workspace_dir / "entity_snapshot.json"

        try:
            with open(snapshot_file, 'w') as f:
                json.dump(entity_data, f, indent=2)

            # Update metadata
            workspace.entity_snapshot_file = str(snapshot_file)
            workspace.last_modified = datetime.now()
            self._save_metadata(workspace_id, workspace)

            logger.info(f"Saved entity snapshot to {snapshot_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save entity snapshot: {e}")
            return False

    def get_entity_snapshot(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """
        Load entity data snapshot from workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            Entity data dict or None
        """
        workspace = self.get_workspace(workspace_id)
        if not workspace or not workspace.entity_snapshot_file:
            return None

        try:
            with open(workspace.entity_snapshot_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load entity snapshot: {e}")
            return None

    def save_field_mappings(
        self,
        workspace_id: str,
        field_mappings: List[FieldMapping]
    ) -> bool:
        """
        Save field mappings to workspace.

        Args:
            workspace_id: Workspace ID
            field_mappings: List of field mappings

        Returns:
            True if successful
        """
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False

        workspace_dir = self.workspaces_dir / workspace_id
        mappings_file = workspace_dir / "field_mappings.json"

        try:
            mappings_data = [fm.model_dump() for fm in field_mappings]
            with open(mappings_file, 'w') as f:
                json.dump(mappings_data, f, indent=2)

            # Update metadata
            workspace.field_mappings_file = str(mappings_file)
            workspace.field_count = len(field_mappings)
            workspace.matched_field_count = sum(
                1 for fm in field_mappings if fm.matched_data is not None
            )
            workspace.last_modified = datetime.now()
            self._save_metadata(workspace_id, workspace)

            logger.info(f"Saved {len(field_mappings)} field mappings")
            return True

        except Exception as e:
            logger.error(f"Failed to save field mappings: {e}")
            return False

    def search_workspace(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        source_type_filter: Optional[DocumentSource] = None
    ) -> List[Dict[str, Any]]:
        """
        Search workspace collection with semantic search.

        Args:
            workspace_id: Workspace ID
            query: Search query
            top_k: Number of results
            source_type_filter: Optional filter by source type

        Returns:
            List of search results
        """
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return []

        try:
            # Import embedding service
            from .embedding_service import embed_query

            collection = self.chroma_client.get_collection(workspace.collection_name)

            # Get query embedding
            query_embedding = embed_query(query)

            # Build where clause
            where_clause = None
            if source_type_filter:
                where_clause = {"source_type": source_type_filter.value}

            # Query collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else None
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search workspace {workspace_id}: {e}")
            return []

    def archive_workspace(self, workspace_id: str) -> bool:
        """
        Archive a workspace (keep data but mark as archived).

        Args:
            workspace_id: Workspace ID

        Returns:
            True if successful
        """
        return self.update_workspace_status(workspace_id, WorkspaceStatus.ARCHIVED)

    def delete_workspace(self, workspace_id: str) -> bool:
        """
        Permanently delete a workspace and its collection.

        Args:
            workspace_id: Workspace ID

        Returns:
            True if successful
        """
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False

        try:
            # Delete ChromaDB collection
            self.chroma_client.delete_collection(workspace.collection_name)
            logger.info(f"Deleted collection {workspace.collection_name}")

            # Delete workspace directory
            workspace_dir = self.workspaces_dir / workspace_id
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)
                logger.info(f"Deleted workspace directory {workspace_dir}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete workspace {workspace_id}: {e}")
            return False

    def cleanup_old_workspaces(self, age_days: int = 180) -> int:
        """
        Delete archived workspaces older than age_days.

        Args:
            age_days: Delete workspaces older than this (default 180 days)

        Returns:
            Number of workspaces deleted
        """
        deleted_count = 0
        workspaces = self.list_workspaces(include_archived=True)

        for workspace in workspaces:
            if (workspace.status == WorkspaceStatus.ARCHIVED and
                workspace.age_days > age_days):
                if self.delete_workspace(workspace.workspace_id):
                    deleted_count += 1
                    logger.info(f"Cleaned up old workspace: {workspace.workspace_id}")

        return deleted_count

    def _save_metadata(self, workspace_id: str, metadata: WorkspaceMetadata):
        """Save workspace metadata to JSON file."""
        workspace_dir = self.workspaces_dir / workspace_id
        workspace_dir.mkdir(exist_ok=True)

        metadata_file = workspace_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)
