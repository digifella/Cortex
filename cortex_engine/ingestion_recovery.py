"""
Ingestion Recovery and Repair System

This module provides robust recovery mechanisms for failed or interrupted ingestion processes.
It can detect orphaned documents, repair broken collections, and salvage partial ingests.

Version: 1.0.0
Date: 2025-08-03
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import chromadb
from chromadb.config import Settings as ChromaSettings

from cortex_engine.utils import convert_windows_to_wsl_path, get_logger
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.config import COLLECTION_NAME, INGESTED_FILES_LOG

logger = get_logger(__name__)

class IngestionRecoveryManager:
    """
    Manages recovery and repair of ingestion processes.
    """
    
    def __init__(self, db_path: str):
        """Initialize the recovery manager with database path."""
        self.db_path = db_path
        self.wsl_db_path = convert_windows_to_wsl_path(db_path)
        self.chroma_db_path = os.path.join(self.wsl_db_path, "knowledge_hub_db")
        self.ingested_log_path = os.path.join(self.chroma_db_path, INGESTED_FILES_LOG)
        self.collection_mgr = WorkingCollectionManager()
        
        # Recovery state file
        self.recovery_state_path = os.path.join(self.chroma_db_path, "recovery_state.json")
        
        logger.info(f"IngestionRecoveryManager initialized for {self.chroma_db_path}")
    
    def analyze_ingestion_state(self) -> Dict[str, Any]:
        """
        Analyze the current state of ingestion and identify any issues.
        
        Returns:
            Dict containing analysis results and recommended actions
        """
        try:
            logger.info("Analyzing ingestion state...")
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "database_path": self.chroma_db_path,
                "issues_found": [],
                "recovery_actions": [],
                "statistics": {},
                "orphaned_documents": [],
                "collection_inconsistencies": [],
                "recommendations": []
            }
            
            # Check if database directory exists
            if not os.path.exists(self.chroma_db_path):
                analysis["issues_found"].append("Database directory does not exist")
                analysis["recovery_actions"].append("Initialize database")
                return analysis
            
            # Load ingested files log
            ingested_files = self._load_ingested_files_log()
            analysis["statistics"]["ingested_files_count"] = len(ingested_files)
            
            # Check ChromaDB collection
            chroma_docs = self._get_chromadb_documents()
            analysis["statistics"]["chromadb_docs_count"] = len(chroma_docs)
            
            # Find orphaned documents (in log but not in ChromaDB)
            orphaned_docs = self._find_orphaned_documents(ingested_files, chroma_docs)
            analysis["orphaned_documents"] = orphaned_docs
            analysis["statistics"]["orphaned_count"] = len(orphaned_docs)
            
            # Check collection consistency
            collection_issues = self._check_collection_consistency(chroma_docs)
            analysis["collection_inconsistencies"] = collection_issues
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis)
            analysis["recommendations"] = recommendations
            
            # Check for recent ingestion activity
            recent_activity = self._check_recent_activity(ingested_files)
            analysis["recent_activity"] = recent_activity
            
            logger.info(f"Analysis complete: {len(orphaned_docs)} orphaned docs, {len(collection_issues)} collection issues")
            return analysis
            
        except Exception as e:
            logger.error(f"Ingestion state analysis failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "issues_found": ["Analysis failed"],
                "recovery_actions": ["Manual investigation required"]
            }
    
    def _load_ingested_files_log(self) -> Dict[str, Any]:
        """Load the ingested files log."""
        try:
            if os.path.exists(self.ingested_log_path):
                with open(self.ingested_log_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Ingested files log not found")
                return {}
        except Exception as e:
            logger.error(f"Failed to load ingested files log: {e}")
            return {}
    
    def _get_chromadb_documents(self) -> List[str]:
        """Get all document IDs from ChromaDB."""
        try:
            db_settings = ChromaSettings(anonymized_telemetry=False)
            db = chromadb.PersistentClient(path=self.chroma_db_path, settings=db_settings)
            collection = db.get_or_create_collection(COLLECTION_NAME)
            
            # Get all document IDs
            results = collection.get(include=["metadatas"])
            if results and "metadatas" in results:
                doc_ids = []
                for metadata in results["metadatas"]:
                    if metadata and "doc_id" in metadata:
                        doc_ids.append(metadata["doc_id"])
                return doc_ids
            return []
            
        except Exception as e:
            logger.error(f"Failed to get ChromaDB documents: {e}")
            return []
    
    def _find_orphaned_documents(self, ingested_files: Dict[str, Any], 
                                chroma_docs: List[str]) -> List[Dict[str, Any]]:
        """Find documents that are in the log but not in ChromaDB."""
        orphaned = []
        chroma_doc_set = set(chroma_docs)
        
        for file_path, metadata in ingested_files.items():
            try:
                # Extract doc_id from metadata
                doc_id = None
                if isinstance(metadata, dict):
                    doc_id = metadata.get("doc_id")
                elif isinstance(metadata, str):
                    doc_id = metadata  # Old format
                
                if doc_id and doc_id not in chroma_doc_set:
                    orphaned.append({
                        "file_path": file_path,
                        "doc_id": doc_id,
                        "metadata": metadata,
                        "file_name": os.path.basename(file_path)
                    })
            except Exception as e:
                logger.warning(f"Error processing ingested file entry {file_path}: {e}")
        
        return orphaned
    
    def _check_collection_consistency(self, chroma_docs: List[str]) -> List[Dict[str, Any]]:
        """Check for inconsistencies in working collections."""
        issues = []
        
        try:
            collections = self.collection_mgr.get_collection_names()
            chroma_doc_set = set(chroma_docs)
            
            for collection_name in collections:
                collection_docs = set(self.collection_mgr.get_doc_ids_by_name(collection_name))
                
                # Find documents in collection but not in ChromaDB
                missing_from_chromadb = collection_docs - chroma_doc_set
                if missing_from_chromadb:
                    issues.append({
                        "type": "missing_from_chromadb",
                        "collection": collection_name,
                        "missing_docs": list(missing_from_chromadb),
                        "count": len(missing_from_chromadb)
                    })
                
                # Check for empty collections
                if len(collection_docs) == 0:
                    issues.append({
                        "type": "empty_collection",
                        "collection": collection_name,
                        "count": 0
                    })
        
        except Exception as e:
            logger.error(f"Collection consistency check failed: {e}")
            issues.append({
                "type": "check_failed",
                "error": str(e)
            })
        
        return issues
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recovery recommendations based on analysis."""
        recommendations = []
        
        # Check for orphaned documents
        if analysis["statistics"].get("orphaned_count", 0) > 0:
            recommendations.append({
                "priority": "high",
                "action": "recover_orphaned_documents",
                "description": f"Recover {analysis['statistics']['orphaned_count']} orphaned documents",
                "details": "Documents exist in ingestion log but not in ChromaDB. They can be recovered."
            })
        
        # Check for collection issues
        if analysis["collection_inconsistencies"]:
            recommendations.append({
                "priority": "medium",
                "action": "fix_collection_inconsistencies",
                "description": "Fix collection inconsistencies",
                "details": "Some collections reference documents that don't exist in ChromaDB."
            })
        
        # Check for recent failed ingestion
        recent_activity = analysis.get("recent_activity", {})
        if recent_activity.get("recent_files_count", 0) > 0 and analysis["statistics"].get("orphaned_count", 0) > 0:
            recommendations.append({
                "priority": "high",
                "action": "create_recovery_collection",
                "description": "Create collection from recent ingestion",
                "details": f"Create a collection with {recent_activity['recent_files_count']} recently ingested files."
            })
        
        return recommendations
    
    def _check_recent_activity(self, ingested_files: Dict[str, Any]) -> Dict[str, Any]:
        """Check for recent ingestion activity."""
        cutoff_time = datetime.now() - timedelta(hours=6)  # Last 6 hours
        recent_files = []
        
        for file_path, metadata in ingested_files.items():
            try:
                timestamp_str = None
                if isinstance(metadata, dict):
                    timestamp_str = metadata.get("timestamp")
                
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp > cutoff_time:
                        recent_files.append({
                            "file_path": file_path,
                            "timestamp": timestamp_str,
                            "doc_id": metadata.get("doc_id") if isinstance(metadata, dict) else metadata
                        })
            except Exception as e:
                logger.warning(f"Error checking timestamp for {file_path}: {e}")
        
        return {
            "recent_files_count": len(recent_files),
            "recent_files": recent_files[:10],  # Show first 10
            "cutoff_time": cutoff_time.isoformat()
        }
    
    def recover_orphaned_documents(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Recover documents that are in the ingestion log but missing from ChromaDB.
        
        Args:
            collection_name: Optional name for collection to create with recovered docs
            
        Returns:
            Dict containing recovery results
        """
        try:
            logger.info("Starting orphaned document recovery...")
            
            # Analyze current state
            analysis = self.analyze_ingestion_state()
            orphaned_docs = analysis.get("orphaned_documents", [])
            
            if not orphaned_docs:
                return {
                    "status": "success",
                    "message": "No orphaned documents found",
                    "recovered_count": 0
                }
            
            # Extract document IDs
            doc_ids = [doc["doc_id"] for doc in orphaned_docs]
            
            recovery_result = {
                "status": "success",
                "orphaned_found": len(orphaned_docs),
                "recovered_count": len(doc_ids),
                "collection_created": False,
                "collection_name": collection_name,
                "doc_ids": doc_ids
            }
            
            # Create collection if requested
            if collection_name:
                if self.collection_mgr.create_collection(collection_name):
                    logger.info(f"Created recovery collection: {collection_name}")
                    recovery_result["collection_created"] = True
                else:
                    logger.info(f"Collection {collection_name} already exists")
                
                # Add documents to collection
                self.collection_mgr.add_docs_by_id_to_collection(collection_name, doc_ids)
                
                # Verify addition
                added_docs = self.collection_mgr.get_doc_ids_by_name(collection_name)
                recovery_result["documents_in_collection"] = len(added_docs)
                
                logger.info(f"Added {len(added_docs)} documents to collection {collection_name}")
            
            # Save recovery state
            self._save_recovery_state(recovery_result)
            
            logger.info(f"Recovery complete: {len(doc_ids)} documents recovered")
            return recovery_result
            
        except Exception as e:
            logger.error(f"Document recovery failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "recovered_count": 0
            }
    
    def auto_repair_collections(self) -> Dict[str, Any]:
        """
        Automatically repair collection inconsistencies.
        
        Returns:
            Dict containing repair results
        """
        try:
            logger.info("Starting automatic collection repair...")
            
            analysis = self.analyze_ingestion_state()
            inconsistencies = analysis.get("collection_inconsistencies", [])
            
            repair_results = {
                "status": "success",
                "repairs_performed": [],
                "collections_cleaned": 0,
                "invalid_refs_removed": 0
            }
            
            for issue in inconsistencies:
                if issue["type"] == "missing_from_chromadb":
                    collection_name = issue["collection"]
                    missing_docs = issue["missing_docs"]
                    
                    # Remove invalid document references
                    self.collection_mgr.remove_from_collection(collection_name, missing_docs)
                    
                    repair_results["repairs_performed"].append({
                        "collection": collection_name,
                        "action": "removed_invalid_refs",
                        "count": len(missing_docs)
                    })
                    repair_results["invalid_refs_removed"] += len(missing_docs)
                    
                    logger.info(f"Removed {len(missing_docs)} invalid references from {collection_name}")
                
                elif issue["type"] == "empty_collection":
                    collection_name = issue["collection"]
                    # Note: We don't auto-delete empty collections as they might be intentional
                    repair_results["repairs_performed"].append({
                        "collection": collection_name,
                        "action": "noted_empty",
                        "count": 0
                    })
            
            repair_results["collections_cleaned"] = len([r for r in repair_results["repairs_performed"] if r["action"] == "removed_invalid_refs"])
            
            logger.info(f"Collection repair complete: {repair_results['collections_cleaned']} collections cleaned")
            return repair_results
            
        except Exception as e:
            logger.error(f"Collection repair failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "repairs_performed": []
            }
    
    def create_recovery_collection_from_recent(self, collection_name: str, 
                                             hours_back: int = 6) -> Dict[str, Any]:
        """
        Create a collection from recently ingested files.
        
        Args:
            collection_name: Name for the new collection
            hours_back: How many hours back to look for recent files
            
        Returns:
            Dict containing operation results
        """
        try:
            logger.info(f"Creating recovery collection '{collection_name}' from last {hours_back} hours")
            
            # Load ingested files
            ingested_files = self._load_ingested_files_log()
            
            # Find recent files
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_doc_ids = []
            
            for file_path, metadata in ingested_files.items():
                try:
                    timestamp_str = None
                    doc_id = None
                    
                    if isinstance(metadata, dict):
                        timestamp_str = metadata.get("timestamp")
                        doc_id = metadata.get("doc_id")
                    elif isinstance(metadata, str):
                        doc_id = metadata
                    
                    if timestamp_str and doc_id:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp > cutoff_time:
                            recent_doc_ids.append(doc_id)
                    elif doc_id and not timestamp_str:
                        # No timestamp, assume it's recent if no timestamp info
                        recent_doc_ids.append(doc_id)
                        
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")
            
            if not recent_doc_ids:
                return {
                    "status": "success",
                    "message": f"No files found from last {hours_back} hours",
                    "collection_created": False,
                    "documents_added": 0
                }
            
            # Create collection
            collection_created = self.collection_mgr.create_collection(collection_name)
            
            # Add documents
            self.collection_mgr.add_docs_by_id_to_collection(collection_name, recent_doc_ids)
            
            # Verify
            added_docs = self.collection_mgr.get_doc_ids_by_name(collection_name)
            
            result = {
                "status": "success",
                "collection_name": collection_name,
                "collection_created": collection_created,
                "documents_found": len(recent_doc_ids),
                "documents_added": len(added_docs),
                "hours_back": hours_back
            }
            
            logger.info(f"Recovery collection created: {len(added_docs)} documents added to '{collection_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Recovery collection creation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "collection_created": False,
                "documents_added": 0
            }
    
    def _save_recovery_state(self, recovery_data: Dict[str, Any]):
        """Save recovery state for future reference."""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "recovery_data": recovery_data
            }
            
            with open(self.recovery_state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Recovery state saved to {self.recovery_state_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save recovery state: {e}")
    
    def get_recovery_history(self) -> Dict[str, Any]:
        """Get history of recovery operations."""
        try:
            if os.path.exists(self.recovery_state_path):
                with open(self.recovery_state_path, 'r') as f:
                    return json.load(f)
            else:
                return {"message": "No recovery history found"}
                
        except Exception as e:
            logger.error(f"Failed to load recovery history: {e}")
            return {"error": str(e)}