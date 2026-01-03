"""
KB Navigator - Browse and Select Documents from Knowledge Base
Version: 1.0.0
Date: 2026-01-03

Purpose: Navigate ChromaDB collection to browse folder structure and select
documents for entity-specific extraction. Provides search, filtering, and
hierarchical folder views.

Key Features:
- Extract folder structure from document metadata
- Search documents by name/path
- Filter by folder path
- Multi-select documents
- Get document metadata
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from collections import defaultdict
import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class KBDocument:
    """Represents a document in the knowledge base."""

    def __init__(
        self,
        doc_id: str,
        file_name: str,
        file_path: str,
        folder_path: str,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.doc_id = doc_id
        self.file_name = file_name
        self.file_path = file_path
        self.folder_path = folder_path
        self.document_type = document_type
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "folder_path": self.folder_path,
            "document_type": self.document_type,
            "metadata": self.metadata
        }


class FolderNode:
    """Represents a folder in the hierarchy."""

    def __init__(self, name: str, full_path: str):
        self.name = name
        self.full_path = full_path
        self.subfolders: Dict[str, 'FolderNode'] = {}
        self.documents: List[KBDocument] = []
        self.document_count = 0  # Total including subfolders

    def add_document(self, doc: KBDocument):
        """Add document to this folder."""
        self.documents.append(doc)

    def get_total_document_count(self) -> int:
        """Get total document count including subfolders."""
        count = len(self.documents)
        for subfolder in self.subfolders.values():
            count += subfolder.get_total_document_count()
        return count


class KBNavigator:
    """
    Navigate and select documents from knowledge base.
    Provides folder browsing, search, and document selection.
    """

    def __init__(self, db_path: str, collection_name: str = "knowledge_hub"):
        """
        Initialize KB navigator.

        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of ChromaDB collection
        """
        self.db_path = db_path
        self.collection_name = collection_name

        # Connect to ChromaDB
        chroma_db_path = str(Path(db_path) / "knowledge_hub_db")
        db_settings = ChromaSettings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)

        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Connected to collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to collection: {e}")
            raise

        # Cache
        self._all_documents: Optional[List[KBDocument]] = None
        self._folder_tree: Optional[FolderNode] = None

    def get_all_documents(self, force_refresh: bool = False) -> List[KBDocument]:
        """
        Get all documents from collection.

        Args:
            force_refresh: Force refresh from ChromaDB

        Returns:
            List of KBDocument objects
        """
        if self._all_documents is not None and not force_refresh:
            return self._all_documents

        try:
            # Get all documents from collection
            results = self.collection.get(
                include=["metadatas"]
            )

            documents = []
            seen_ids = set()

            if results and 'ids' in results and results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    # Skip duplicates
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)

                    metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}

                    # Extract file info from metadata
                    file_name = metadata.get('file_name', metadata.get('source', 'Unknown'))
                    file_path = metadata.get('file_path', metadata.get('source', ''))
                    document_type = metadata.get('document_type', 'Unknown')

                    # Extract folder path from file path
                    folder_path = self._extract_folder_path(file_path)

                    doc = KBDocument(
                        doc_id=doc_id,
                        file_name=file_name,
                        file_path=file_path,
                        folder_path=folder_path,
                        document_type=document_type,
                        metadata=metadata
                    )
                    documents.append(doc)

            self._all_documents = documents
            logger.info(f"Loaded {len(documents)} unique documents from KB")
            return documents

        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []

    def _extract_folder_path(self, file_path: str) -> str:
        """Extract folder path from full file path."""
        if not file_path:
            return "/"

        # Normalize path separators
        file_path = file_path.replace('\\', '/')

        # Get directory part
        path_obj = Path(file_path)
        folder = str(path_obj.parent) if path_obj.parent != Path('.') else "/"

        # Normalize
        folder = folder.replace('\\', '/')
        if not folder.startswith('/'):
            folder = '/' + folder

        return folder

    def build_folder_tree(self, force_refresh: bool = False) -> FolderNode:
        """
        Build hierarchical folder tree from documents.

        Args:
            force_refresh: Force refresh from ChromaDB

        Returns:
            Root FolderNode
        """
        if self._folder_tree is not None and not force_refresh:
            return self._folder_tree

        # Get all documents
        documents = self.get_all_documents(force_refresh=force_refresh)

        # Create root node
        root = FolderNode(name="/", full_path="/")

        # Build tree
        for doc in documents:
            # Parse folder path
            folder_parts = [p for p in doc.folder_path.split('/') if p]

            # Navigate/create folder nodes
            current_node = root
            current_path = ""

            for part in folder_parts:
                current_path += "/" + part

                if part not in current_node.subfolders:
                    current_node.subfolders[part] = FolderNode(
                        name=part,
                        full_path=current_path
                    )

                current_node = current_node.subfolders[part]

            # Add document to leaf folder
            current_node.add_document(doc)

        # Update document counts
        self._update_document_counts(root)

        self._folder_tree = root
        logger.info(f"Built folder tree with {len(root.subfolders)} top-level folders")
        return root

    def _update_document_counts(self, node: FolderNode):
        """Recursively update document counts."""
        node.document_count = len(node.documents)
        for subfolder in node.subfolders.values():
            self._update_document_counts(subfolder)
            node.document_count += subfolder.document_count

    def search_documents(
        self,
        query: str,
        search_in: str = "both"  # "name", "path", or "both"
    ) -> List[KBDocument]:
        """
        Search documents by name or path.

        Args:
            query: Search query
            search_in: Where to search ("name", "path", or "both")

        Returns:
            Matching documents
        """
        documents = self.get_all_documents()
        query_lower = query.lower()

        results = []
        for doc in documents:
            match = False

            if search_in in ("name", "both"):
                if query_lower in doc.file_name.lower():
                    match = True

            if search_in in ("path", "both"):
                if query_lower in doc.file_path.lower():
                    match = True

            if match:
                results.append(doc)

        logger.info(f"Search '{query}' found {len(results)} documents")
        return results

    def filter_by_folder(
        self,
        folder_path: str,
        include_subfolders: bool = True
    ) -> List[KBDocument]:
        """
        Get documents in a specific folder.

        Args:
            folder_path: Folder path to filter by
            include_subfolders: Include documents from subfolders

        Returns:
            Documents in folder
        """
        documents = self.get_all_documents()

        # Normalize folder path
        folder_path = folder_path.replace('\\', '/').strip('/')
        if folder_path and not folder_path.startswith('/'):
            folder_path = '/' + folder_path

        results = []
        for doc in documents:
            if include_subfolders:
                # Check if document is in folder or subfolder
                if doc.folder_path.startswith(folder_path):
                    results.append(doc)
            else:
                # Exact folder match only
                if doc.folder_path == folder_path:
                    results.append(doc)

        logger.info(f"Folder '{folder_path}' has {len(results)} documents (subfolders={include_subfolders})")
        return results

    def get_folder_node(self, folder_path: str) -> Optional[FolderNode]:
        """Get specific folder node by path."""
        tree = self.build_folder_tree()

        if not folder_path or folder_path == "/":
            return tree

        # Navigate to folder
        parts = [p for p in folder_path.strip('/').split('/') if p]
        current = tree

        for part in parts:
            if part in current.subfolders:
                current = current.subfolders[part]
            else:
                return None

        return current

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[KBDocument]:
        """Get specific documents by their IDs."""
        all_docs = self.get_all_documents()
        doc_map = {doc.doc_id: doc for doc in all_docs}

        return [doc_map[doc_id] for doc_id in doc_ids if doc_id in doc_map]

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document."""
        all_docs = self.get_all_documents()
        for doc in all_docs:
            if doc.doc_id == doc_id:
                return doc.metadata
        return None

    def get_folder_suggestions(self, prefix: str = "", limit: int = 10) -> List[str]:
        """
        Get folder path suggestions for autocomplete.

        Args:
            prefix: Folder path prefix to match
            limit: Maximum suggestions to return

        Returns:
            List of folder paths
        """
        tree = self.build_folder_tree()

        # Collect all folder paths
        all_folders = []

        def collect_folders(node: FolderNode):
            if node.full_path != "/":
                all_folders.append(node.full_path)
            for subfolder in node.subfolders.values():
                collect_folders(subfolder)

        collect_folders(tree)

        # Filter by prefix
        if prefix:
            prefix_lower = prefix.lower()
            matches = [f for f in all_folders if prefix_lower in f.lower()]
        else:
            matches = all_folders

        # Limit results
        return sorted(matches)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get KB statistics."""
        documents = self.get_all_documents()
        tree = self.build_folder_tree()

        # Document types
        type_counts = defaultdict(int)
        for doc in documents:
            type_counts[doc.document_type or "Unknown"] += 1

        # Folder depths
        def get_depth(node: FolderNode, current_depth: int = 0) -> int:
            if not node.subfolders:
                return current_depth
            return max(get_depth(sub, current_depth + 1) for sub in node.subfolders.values())

        max_depth = get_depth(tree)

        return {
            "total_documents": len(documents),
            "total_folders": len(list(self._iter_all_folders(tree))),
            "max_folder_depth": max_depth,
            "document_types": dict(type_counts)
        }

    def _iter_all_folders(self, node: FolderNode):
        """Iterate all folders in tree."""
        if node.full_path != "/":
            yield node
        for subfolder in node.subfolders.values():
            yield from self._iter_all_folders(subfolder)
