# ## File: cortex_engine/collection_manager.py
# Version: 6.0.0 (Utilities Refactor)
# Date: 2025-07-23
# Purpose: Manages CRUD operations for collections.
#          - REFACTOR (v6.0.0): Updated to use centralized utilities for path handling,
#            logging, and error handling. Removed code duplication.

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# Import centralized utilities
from .utils import convert_windows_to_wsl_path, convert_to_docker_mount_path, get_project_root, get_logger
from .utils.file_utils import get_file_hash
from .exceptions import CollectionError, PathError

# Set up logging
logger = get_logger(__name__)

# Configuration
PROJECT_ROOT = get_project_root()
# Default fallback - will be overridden by get_collections_file_path()
COLLECTIONS_FILE = str(PROJECT_ROOT / "working_collections.json")

def get_collections_file_path():
    """Get the correct path for collections file based on configured KB database path."""
    try:
        from .config_manager import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        db_path = config.get('ai_database_path')
        
        if db_path:
            # Convert to a container-visible path in Docker; otherwise WSL/posix
            safe_db_path = convert_to_docker_mount_path(db_path)
            collections_path = os.path.join(safe_db_path, "working_collections.json")
            return collections_path
    except Exception as e:
        logger.warning(f"Could not get KB database path, using project root: {e}")
    
    # Fallback to project root
    return COLLECTIONS_FILE

class WorkingCollectionManager:
    """Manages CRUD operations for working collections stored in a JSON file."""

    def __init__(self):
        self.collections_file = get_collections_file_path()
        self.collections = self._load()

    def _load(self):
        """Load collections from file with proper error handling."""
        collections = {}
        
        # Ensure the directory exists
        collections_dir = os.path.dirname(self.collections_file)
        if collections_dir and not os.path.exists(collections_dir):
            try:
                os.makedirs(collections_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create collections directory: {e}")
        
        if os.path.exists(self.collections_file):
            try:
                with open(self.collections_file, 'r') as f:
                    collections = json.load(f)
                logger.debug(f"Loaded {len(collections)} collections from {self.collections_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load collections file, starting with empty collections: {e}")
                collections = {}

        now_iso = datetime.now().isoformat()
        for name, data in collections.items():
            if isinstance(data, dict):
                if 'created_at' not in data: data['created_at'] = now_iso
                if 'modified_at' not in data: data['modified_at'] = data.get('created_at', now_iso)
            else:
                collections[name] = {"name": name, "doc_ids": data, "created_at": now_iso, "modified_at": now_iso}

        if "default" not in collections:
            collections["default"] = {"name": "default", "doc_ids": [], "created_at": now_iso, "modified_at": now_iso}
        return collections

    def _save(self):
        """Save collections to file with proper error handling."""
        try:
            # Ensure directory exists
            collections_dir = os.path.dirname(self.collections_file)
            if collections_dir and not os.path.exists(collections_dir):
                os.makedirs(collections_dir, exist_ok=True)
                
            with open(self.collections_file, 'w') as f:
                json.dump(self.collections, f, indent=4)
            logger.debug(f"Collections saved to {self.collections_file}")
            return True
        except IOError as e:
            logger.error(f"Failed to save collections: {e}")
            return False

    def get_collection_names(self) -> list:
        return list(self.collections.keys())

    def get_doc_ids_by_name(self, name: str) -> list:
        return self.collections.get(name, {}).get("doc_ids", [])

    def create_collection(self, name: str) -> bool:
        if name and name not in self.collections:
            now_iso = datetime.now().isoformat()
            self.collections[name] = {"name": name, "doc_ids": [], "created_at": now_iso, "modified_at": now_iso}
            self._save()
            return True
        return False

    def add_docs_by_id_to_collection(self, name: str, doc_ids: list):
        if not doc_ids: return
        if name not in self.collections: self.create_collection(name)
        existing_ids = set(self.collections[name].get("doc_ids", []))
        ids_added = False
        for doc_id in doc_ids:
            if doc_id not in existing_ids:
                self.collections[name]["doc_ids"].append(doc_id)
                existing_ids.add(doc_id)
                ids_added = True
        if ids_added:
            self.collections[name]["modified_at"] = datetime.now().isoformat()
            self._save()

    def remove_from_collection(self, name: str, doc_ids_to_remove: list):
        if name in self.collections:
            initial_count = len(self.collections[name].get("doc_ids", []))
            self.collections[name]["doc_ids"] = [doc_id for doc_id in self.collections[name]["doc_ids"] if doc_id not in doc_ids_to_remove]
            if len(self.collections[name]["doc_ids"]) < initial_count:
                self.collections[name]['modified_at'] = datetime.now().isoformat()
                self._save()

    def rename_collection(self, old_name: str, new_name: str):
        if old_name in self.collections and new_name not in self.collections and old_name != "default":
            self.collections[new_name] = self.collections.pop(old_name)
            self.collections[new_name]['name'] = new_name
            self.collections[new_name]['modified_at'] = datetime.now().isoformat()
            self._save()

    def delete_collection(self, name: str):
        if name in self.collections and name != "default":
            del self.collections[name]
            self._save()

    def merge_collections(self, source_name: str, dest_name: str) -> bool:
        """Merges the source collection into the destination and deletes the source."""
        if source_name not in self.collections or dest_name not in self.collections or source_name == dest_name:
            return False

        source_doc_ids = self.get_doc_ids_by_name(source_name)
        self.add_docs_by_id_to_collection(dest_name, source_doc_ids)
        self.delete_collection(source_name)
        return True

    def export_collection_files(self, name: str, output_dir: str, vector_collection) -> tuple:
        """Export files from a collection to a directory."""
        doc_ids = self.get_doc_ids_by_name(name)
        if not doc_ids: 
            logger.info(f"No documents found in collection '{name}'")
            return [], []
        
        try:
            safe_output_dir = convert_to_docker_mount_path(output_dir)
            Path(safe_output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Exporting {len(doc_ids)} documents from collection '{name}' to {safe_output_dir}")
            
            results = vector_collection.get(where={"doc_id": {"$in": doc_ids}}, include=["metadatas"])
            source_paths = set(meta['doc_posix_path'] for meta in results['metadatas'] if 'doc_posix_path' in meta)
            copied_files, failed_files = [], []
            
            for src_path_str in source_paths:
                src_path = Path(src_path_str)
                dest_path = Path(safe_output_dir) / src_path.name
                try:
                    if src_path.exists():
                        shutil.copy(src_path, dest_path)
                        copied_files.append(str(dest_path))
                        logger.debug(f"Copied {src_path} to {dest_path}")
                    else:
                        raise FileNotFoundError(f"Source file not found: {src_path}")
                except (FileNotFoundError, Exception) as e:
                    failed_files.append(f"{src_path_str} (Reason: {e})")
                    logger.warning(f"Failed to copy {src_path_str}: {e}")
            
            logger.info(f"Export completed: {len(copied_files)} files copied, {len(failed_files)} failed")
            return copied_files, failed_files
            
        except Exception as e:
            logger.error(f"Export collection failed: {e}")
            raise CollectionError(f"Failed to export collection '{name}'", str(e))

    def deduplicate_vector_store(self, vector_collection, dry_run: bool = True) -> dict:
        """
        Identify and optionally remove duplicate documents from the vector store.
        
        Args:
            vector_collection: ChromaDB collection object
            dry_run: If True, only analyze duplicates without removing them
            
        Returns:
            Dict with deduplication analysis results
        """
        try:
            logger.info("Starting vector store deduplication analysis...")
            
            # Get all documents from vector store
            all_results = vector_collection.get(include=["metadatas", "documents"])
            all_metadatas = all_results.get('metadatas', [])
            all_documents = all_results.get('documents', [])
            all_ids = all_results.get('ids', [])
            
            if not all_metadatas:
                logger.warning("No documents found in vector store")
                return {"status": "no_documents", "duplicates_found": 0, "total_documents": 0}
            
            total_docs = len(all_metadatas)
            logger.info(f"Analyzing {total_docs} documents for duplicates...")
            
            # Group documents by file hash and doc_posix_path
            file_groups = {}
            doc_id_to_metadata = {}
            
            for i, (metadata, doc_content, chroma_id) in enumerate(zip(all_metadatas, all_documents, all_ids)):
                # Create a unique key based on file hash and path
                file_hash = metadata.get('file_hash', '')
                doc_path = metadata.get('doc_posix_path', '')
                doc_id = metadata.get('doc_id', chroma_id)
                
                # Use file hash as primary key, path as fallback
                if file_hash:
                    key = f"hash_{file_hash}"
                elif doc_path:
                    key = f"path_{doc_path}"
                else:
                    key = f"content_{get_file_hash(doc_content[:1000])}"  # Hash first 1000 chars
                
                if key not in file_groups:
                    file_groups[key] = []
                
                file_groups[key].append({
                    'chroma_id': chroma_id,
                    'doc_id': doc_id,
                    'metadata': metadata,
                    'content_length': len(doc_content) if doc_content else 0,
                    'index': i
                })
                
                doc_id_to_metadata[doc_id] = metadata
            
            # Identify duplicates - groups with more than one document
            duplicate_groups = {k: v for k, v in file_groups.items() if len(v) > 1}
            duplicates_to_remove = []
            kept_documents = []
            
            for group_key, docs in duplicate_groups.items():
                # Sort by content length (descending) to keep the most complete version
                docs.sort(key=lambda x: x['content_length'], reverse=True)
                
                # Keep the first (longest) document, mark others for removal
                kept_doc = docs[0]
                kept_documents.append(kept_doc)
                
                for duplicate_doc in docs[1:]:
                    duplicates_to_remove.append(duplicate_doc)
                    logger.debug(f"Marking duplicate for removal: {duplicate_doc['doc_id']} (keeping {kept_doc['doc_id']})")
            
            duplicate_count = len(duplicates_to_remove)
            duplicate_percentage = (duplicate_count / total_docs * 100) if total_docs > 0 else 0
            
            results = {
                "status": "analysis_complete",
                "total_documents": total_docs,
                "duplicates_found": duplicate_count,
                "duplicate_percentage": duplicate_percentage,
                "unique_files": len(file_groups),
                "duplicate_groups": len(duplicate_groups),
                "duplicates_to_remove": [d['chroma_id'] for d in duplicates_to_remove],
                "dry_run": dry_run
            }
            
            logger.info(f"Deduplication analysis complete: {duplicate_count} duplicates found ({duplicate_percentage:.1f}%)")
            
            # If not dry run, actually remove the duplicates
            if not dry_run and duplicates_to_remove:
                logger.info(f"Removing {len(duplicates_to_remove)} duplicate documents...")
                chroma_ids_to_remove = [d['chroma_id'] for d in duplicates_to_remove]
                
                # Remove duplicates in batches to avoid memory issues
                batch_size = 100
                removed_count = 0
                
                for i in range(0, len(chroma_ids_to_remove), batch_size):
                    batch_ids = chroma_ids_to_remove[i:i + batch_size]
                    try:
                        vector_collection.delete(ids=batch_ids)
                        removed_count += len(batch_ids)
                        logger.debug(f"Removed batch {i//batch_size + 1}: {len(batch_ids)} documents")
                    except Exception as e:
                        logger.error(f"Failed to remove batch {i//batch_size + 1}: {e}")
                        results["removal_errors"] = results.get("removal_errors", 0) + 1
                
                results["removed_count"] = removed_count
                results["status"] = "cleanup_complete"
                logger.info(f"Deduplication cleanup complete: {removed_count} duplicates removed")
                
                # Update collection doc_ids to remove deleted documents
                self._update_collections_after_deduplication(duplicates_to_remove)
            
            return results
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return {"status": "error", "error": str(e), "total_documents": 0, "duplicates_found": 0}

    def _update_collections_after_deduplication(self, removed_documents: list):
        """Update collection doc_ids after removing duplicates."""
        try:
            removed_doc_ids = {doc['doc_id'] for doc in removed_documents}
            collections_updated = 0
            
            for collection_name, collection_data in self.collections.items():
                original_count = len(collection_data.get('doc_ids', []))
                # Remove deleted doc_ids from collection
                updated_doc_ids = [doc_id for doc_id in collection_data.get('doc_ids', []) 
                                 if doc_id not in removed_doc_ids]
                
                if len(updated_doc_ids) < original_count:
                    self.collections[collection_name]['doc_ids'] = updated_doc_ids
                    self.collections[collection_name]['modified_at'] = datetime.now().isoformat()
                    collections_updated += 1
                    logger.info(f"Updated collection '{collection_name}': {original_count} -> {len(updated_doc_ids)} documents")
            
            if collections_updated > 0:
                self._save()
                logger.info(f"Updated {collections_updated} collections after deduplication")
                
        except Exception as e:
            logger.error(f"Failed to update collections after deduplication: {e}")
    
    def clear_all_collections(self) -> dict:
        """Clear all collections except default. Returns summary of cleared collections."""
        cleared_collections = {}
        collections_to_clear = [name for name in self.collections.keys() if name != "default"]
        
        for name in collections_to_clear:
            doc_count = len(self.collections[name].get("doc_ids", []))
            cleared_collections[name] = doc_count
            del self.collections[name]
        
        # Also clear default collection documents but keep the collection
        default_doc_count = len(self.collections["default"].get("doc_ids", []))
        if default_doc_count > 0:
            self.collections["default"]["doc_ids"] = []
            self.collections["default"]["modified_at"] = datetime.now().isoformat()
            cleared_collections["default"] = default_doc_count
        
        if cleared_collections:
            self._save()
            logger.info(f"Cleared {len(cleared_collections)} collections")
        
        return cleared_collections
    
    def clear_empty_collections(self) -> dict:
        """Remove collections that have no documents. Returns summary of removed collections."""
        empty_collections = {}
        collections_to_remove = []
        
        for name, data in self.collections.items():
            if name != "default":  # Never remove default
                doc_ids = data.get("doc_ids", [])
                if not doc_ids:  # Empty collection
                    collections_to_remove.append(name)
                    empty_collections[name] = {
                        "created_at": data.get("created_at", "Unknown"),
                        "modified_at": data.get("modified_at", "Unknown")
                    }
        
        for name in collections_to_remove:
            del self.collections[name]
        
        if empty_collections:
            self._save()
            logger.info(f"Removed {len(empty_collections)} empty collections")
        
        return empty_collections
    
    def get_collections_summary(self) -> dict:
        """Get a summary of all collections with document counts."""
        summary = {}
        for name, data in self.collections.items():
            doc_count = len(data.get("doc_ids", []))
            summary[name] = {
                "document_count": doc_count,
                "created_at": data.get("created_at", "Unknown"),
                "modified_at": data.get("modified_at", "Unknown"),
                "is_empty": doc_count == 0
            }
        return summary
