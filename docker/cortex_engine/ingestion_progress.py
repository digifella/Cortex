"""
Ingestion Progress Management

This module provides persistent progress tracking for long-running ingestion processes,
allowing recovery from interruptions and providing real-time status updates.

Version: 1.0.0
Date: 2025-08-03
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

from cortex_engine.utils import get_logger

logger = get_logger(__name__)

class IngestionProgressTracker:
    """
    Tracks and persists ingestion progress to enable recovery from interruptions.
    """
    
    def __init__(self, db_path: str, session_id: Optional[str] = None):
        """Initialize progress tracker."""
        self.db_path = db_path
        self.session_id = session_id or f"ingest_{int(time.time())}"
        
        # Progress file path
        self.progress_dir = os.path.join(db_path, "ingestion_progress")
        os.makedirs(self.progress_dir, exist_ok=True)
        self.progress_file = os.path.join(self.progress_dir, f"{self.session_id}.json")
        
        # Progress state
        self.progress = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "initializing",
            "total_files": 0,
            "processed_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "current_file": None,
            "processed_list": [],
            "failed_list": [],
            "collection_name": None,
            "estimated_completion": None,
            "errors": []
        }
        
        # Auto-save settings
        self.auto_save_interval = 5  # seconds
        self.auto_save_thread = None
        self.auto_save_enabled = False
        
        logger.info(f"IngestionProgressTracker initialized: {self.session_id}")
    
    def start_session(self, total_files: int, collection_name: Optional[str] = None):
        """Start a new ingestion session."""
        self.progress.update({
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "total_files": total_files,
            "collection_name": collection_name
        })
        
        self.save_progress()
        self.start_auto_save()
        
        logger.info(f"Ingestion session started: {total_files} files, collection: {collection_name}")
    
    def update_current_file(self, file_path: str):
        """Update the currently processing file."""
        self.progress.update({
            "current_file": file_path,
            "last_update": datetime.now().isoformat()
        })
    
    def mark_file_completed(self, file_path: str, doc_id: str, success: bool = True, error: Optional[str] = None):
        """Mark a file as completed (successfully or with error)."""
        self.progress["processed_files"] += 1
        self.progress["last_update"] = datetime.now().isoformat()
        
        if success:
            self.progress["successful_files"] += 1
            self.progress["processed_list"].append({
                "file_path": file_path,
                "doc_id": doc_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            self.progress["failed_files"] += 1
            self.progress["failed_list"].append({
                "file_path": file_path,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
            
            if error:
                self.progress["errors"].append({
                    "file_path": file_path,
                    "error": error,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Update estimated completion
        if self.progress["processed_files"] > 0:
            elapsed_time = (datetime.now() - datetime.fromisoformat(self.progress["start_time"])).total_seconds()
            avg_time_per_file = elapsed_time / self.progress["processed_files"]
            remaining_files = self.progress["total_files"] - self.progress["processed_files"]
            estimated_remaining_seconds = remaining_files * avg_time_per_file
            
            estimated_completion = datetime.now().timestamp() + estimated_remaining_seconds
            self.progress["estimated_completion"] = datetime.fromtimestamp(estimated_completion).isoformat()
    
    def complete_session(self, final_status: str = "completed"):
        """Complete the ingestion session."""
        self.progress.update({
            "status": final_status,
            "end_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat()
        })
        
        self.stop_auto_save()
        self.save_progress()
        
        logger.info(f"Ingestion session completed: {final_status}, {self.progress['successful_files']}/{self.progress['total_files']} successful")
    
    def save_progress(self):
        """Save current progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def load_progress(self) -> bool:
        """Load progress from file if it exists."""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
                logger.info(f"Progress loaded: {self.progress['processed_files']}/{self.progress['total_files']} files")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return False
    
    def start_auto_save(self):
        """Start automatic progress saving in background thread."""
        if not self.auto_save_enabled:
            self.auto_save_enabled = True
            self.auto_save_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
            self.auto_save_thread.start()
    
    def stop_auto_save(self):
        """Stop automatic progress saving."""
        self.auto_save_enabled = False
        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=2)
    
    def _auto_save_worker(self):
        """Background worker for automatic progress saving."""
        while self.auto_save_enabled:
            try:
                time.sleep(self.auto_save_interval)
                if self.auto_save_enabled:
                    self.save_progress()
            except Exception as e:
                logger.error(f"Auto-save error: {e}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress."""
        completion_percentage = 0
        if self.progress["total_files"] > 0:
            completion_percentage = (self.progress["processed_files"] / self.progress["total_files"]) * 100
        
        return {
            "session_id": self.session_id,
            "status": self.progress["status"],
            "completion_percentage": completion_percentage,
            "processed_files": self.progress["processed_files"],
            "total_files": self.progress["total_files"],
            "successful_files": self.progress["successful_files"],
            "failed_files": self.progress["failed_files"],
            "current_file": self.progress.get("current_file"),
            "estimated_completion": self.progress.get("estimated_completion"),
            "last_update": self.progress["last_update"],
            "error_count": len(self.progress.get("errors", []))
        }
    
    def get_detailed_progress(self) -> Dict[str, Any]:
        """Get detailed progress information."""
        return self.progress.copy()
    
    def cleanup_old_sessions(self, days_old: int = 7):
        """Clean up old progress files."""
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            for file_path in os.listdir(self.progress_dir):
                if file_path.endswith('.json'):
                    full_path = os.path.join(self.progress_dir, file_path)
                    if os.path.getmtime(full_path) < cutoff_time:
                        os.remove(full_path)
                        logger.info(f"Cleaned up old progress file: {file_path}")
        except Exception as e:
            logger.error(f"Progress cleanup failed: {e}")
    
    @staticmethod
    def find_incomplete_sessions(db_path: str) -> List[Dict[str, Any]]:
        """Find incomplete ingestion sessions that can be resumed."""
        progress_dir = os.path.join(db_path, "ingestion_progress")
        incomplete_sessions = []
        
        if not os.path.exists(progress_dir):
            return incomplete_sessions
        
        try:
            for file_path in os.listdir(progress_dir):
                if file_path.endswith('.json'):
                    full_path = os.path.join(progress_dir, file_path)
                    try:
                        with open(full_path, 'r') as f:
                            progress = json.load(f)
                        
                        if progress.get("status") == "running":
                            incomplete_sessions.append({
                                "session_id": progress.get("session_id"),
                                "start_time": progress.get("start_time"),
                                "total_files": progress.get("total_files", 0),
                                "processed_files": progress.get("processed_files", 0),
                                "collection_name": progress.get("collection_name"),
                                "last_update": progress.get("last_update"),
                                "progress_file": full_path
                            })
                    except Exception as e:
                        logger.warning(f"Error reading progress file {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error finding incomplete sessions: {e}")
        
        return incomplete_sessions
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.complete_session("failed")
        else:
            self.complete_session("completed")
        return False