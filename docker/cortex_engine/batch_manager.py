#!/usr/bin/env python3
"""
Batch Processing Manager with Pause/Resume Functionality
Handles long-running ingestion processes with crash recovery
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

class BatchState:
    """Manages batch processing state with pause/resume capability"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.state_file = self.db_path / "batch_state.json"
        self.processed_log = self.db_path / "knowledge_hub_db" / "ingested_files.log"
        
    def create_batch(self, file_paths: List[str], scan_config: Optional[Dict] = None, chunk_size: Optional[int] = None, auto_pause_after_chunks: Optional[int] = None) -> str:
        """Create a new batch processing job with optional scan configuration and chunking"""
        batch_id = str(uuid.uuid4())
        
        # Handle chunked processing
        chunks = []
        if chunk_size and len(file_paths) > chunk_size:
            # Split files into chunks
            for i in range(0, len(file_paths), chunk_size):
                chunk = file_paths[i:i + chunk_size]
                chunks.append(chunk)
            logger.info(f"Created {len(chunks)} chunks of max {chunk_size} files each")
        else:
            chunks = [file_paths]  # Single chunk
        
        state = {
            "batch_id": batch_id,
            "total_files": len(file_paths),
            "files_remaining": file_paths.copy(),
            "files_completed": 0,
            "current_phase": "analyze",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "paused": False,
            "error_count": 0,
            "errors": [],
            "scan_config": scan_config or {},  # Store original scan configuration
            "chunked_processing": chunk_size is not None,
            "chunk_size": chunk_size,
            "chunks": chunks,
            "current_chunk": 0,
            "current_chunk_files": chunks[0] if chunks else [],
            "auto_pause_after_chunks": auto_pause_after_chunks,
            "chunks_processed_in_session": 0,
            "current_chunk_progress": 0  # Track documents processed in current chunk
        }
        
        self._save_state(state)
        logger.info(f"Created new batch {batch_id} with {len(file_paths)} files{f' in {len(chunks)} chunks' if len(chunks) > 1 else ''}{f' (auto-pause after {auto_pause_after_chunks} chunks)' if auto_pause_after_chunks else ''}")
        return batch_id
    
    def resume_or_create_batch(self, file_paths: List[str], scan_config: Optional[Dict] = None, chunk_size: Optional[int] = None) -> Tuple[str, List[str], int]:
        """Resume existing batch or create new one. Returns (batch_id, files_to_process, completed_count)"""
        
        # Check for existing batch
        existing_state = self.load_state()
        if existing_state:
            logger.info(f"Found existing batch {existing_state['batch_id']}")
            
            # Filter out files that have been processed since batch started
            processed_files = self._get_processed_files()
            original_remaining = existing_state.get('files_remaining', [])
            
            # Remove processed files from remaining list
            still_remaining = [f for f in original_remaining if f not in processed_files]
            
            # Update state with current progress
            completed_count = existing_state['total_files'] - len(still_remaining)
            existing_state['files_remaining'] = still_remaining
            existing_state['files_completed'] = completed_count
            existing_state['last_updated'] = datetime.now().isoformat()
            existing_state['paused'] = False
            
            if still_remaining:
                self._save_state(existing_state)
                logger.info(f"Resuming batch: {completed_count}/{existing_state['total_files']} completed, {len(still_remaining)} remaining")
                return existing_state['batch_id'], still_remaining, completed_count
            else:
                logger.info("All files in existing batch have been processed")
                self.clear_batch()
        
        # No existing batch or it's completed - create new one
        # Remove already processed files from the new batch
        processed_files = self._get_processed_files()
        unprocessed_files = [f for f in file_paths if f not in processed_files]
        
        if not unprocessed_files:
            logger.info("All files have already been processed")
            return "", [], len(file_paths)
        
        logger.info(f"Creating new batch: {len(file_paths) - len(unprocessed_files)} files already processed, {len(unprocessed_files)} new files")
        batch_id = self.create_batch(unprocessed_files, scan_config, chunk_size)
        return batch_id, unprocessed_files, len(file_paths) - len(unprocessed_files)
    
    def update_progress(self, completed_file: str):
        """Update batch progress when a file is completed"""
        state = self.load_state()
        if not state:
            return
            
        if completed_file in state['files_remaining']:
            state['files_remaining'].remove(completed_file)
            state['files_completed'] += 1
            
            # Update chunk progress if using chunked processing
            if state.get('chunked_processing', False):
                current_chunk_files = state.get('current_chunk_files', [])
                if completed_file in current_chunk_files:
                    # Remove from current chunk files and update chunk progress
                    current_chunk_files.remove(completed_file)
                    state['current_chunk_files'] = current_chunk_files
                    state['current_chunk_progress'] = state.get('current_chunk_progress', 0) + 1
            
            state['last_updated'] = datetime.now().isoformat()
            self._save_state(state)
            
            logger.info(f"Progress: {state['files_completed']}/{state['total_files']} completed")
    
    def record_error(self, file_path: str, error_msg: str):
        """Record an error for a file"""
        state = self.load_state()
        if not state:
            return
            
        state['error_count'] += 1
        state['errors'].append({
            "file": file_path,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        state['last_updated'] = datetime.now().isoformat()
        
        # Remove from remaining files even if it errored
        if file_path in state['files_remaining']:
            state['files_remaining'].remove(file_path)
            
        self._save_state(state)
    
    def pause_batch(self):
        """Mark batch as paused"""
        state = self.load_state()
        if state:
            state['paused'] = True
            state['last_updated'] = datetime.now().isoformat()
            self._save_state(state)
            logger.info(f"Batch {state['batch_id']} paused")
    
    def is_paused(self) -> bool:
        """Check if batch is currently paused"""
        state = self.load_state()
        return state.get('paused', False) if state else False
    
    def clear_batch(self):
        """Clear batch state (when completed or cancelled)"""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("Batch state cleared")
    
    def load_state(self) -> Optional[Dict]:
        """Load current batch state"""
        if not self.state_file.exists():
            return None
            
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load batch state: {e}")
            return None
    
    def get_scan_config(self) -> Optional[Dict]:
        """Get the stored scan configuration from active batch"""
        state = self.load_state()
        if state:
            return state.get('scan_config', {})
        return None
    
    def get_current_chunk_files(self) -> List[str]:
        """Get files for the current chunk being processed"""
        state = self.load_state()
        if not state:
            return []
            
        if state.get('chunked_processing', False):
            return state.get('current_chunk_files', [])
        else:
            # Non-chunked processing - return remaining files
            return state.get('files_remaining', [])
    
    def advance_to_next_chunk(self) -> bool:
        """Move to the next chunk. Returns True if there are more chunks, False if completed"""
        state = self.load_state()
        if not state or not state.get('chunked_processing', False):
            return False
            
        current_chunk = state.get('current_chunk', 0)
        chunks = state.get('chunks', [])
        
        # Increment chunks processed in current session
        state['chunks_processed_in_session'] = state.get('chunks_processed_in_session', 0) + 1
        
        # Check if we should auto-pause
        auto_pause_after = state.get('auto_pause_after_chunks')
        if auto_pause_after and state['chunks_processed_in_session'] >= auto_pause_after:
            logger.info(f"Auto-pausing after {state['chunks_processed_in_session']} chunks")
            state['paused'] = True
            state['last_updated'] = datetime.now().isoformat()
            self._save_state(state)
            return False  # Don't advance, just pause
        
        if current_chunk + 1 < len(chunks):
            # Move to next chunk
            state['current_chunk'] = current_chunk + 1
            state['current_chunk_files'] = chunks[current_chunk + 1]
            state['current_chunk_progress'] = 0  # Reset progress for new chunk
            state['last_updated'] = datetime.now().isoformat()
            self._save_state(state)
            
            logger.info(f"Advanced to chunk {current_chunk + 2}/{len(chunks)}")
            return True
        else:
            # No more chunks
            return False
    
    def start_new_session(self):
        """Reset the session chunk counter for resuming after auto-pause"""
        state = self.load_state()
        if state:
            state['chunks_processed_in_session'] = 0
            state['paused'] = False
            # Reset current chunk progress when starting new session
            state['current_chunk_progress'] = 0
            state['last_updated'] = datetime.now().isoformat()
            self._save_state(state)
            logger.info("Started new processing session")
    
    def should_auto_pause(self) -> bool:
        """Check if the batch should auto-pause after current chunk"""
        state = self.load_state()
        if not state:
            return False
            
        auto_pause_after = state.get('auto_pause_after_chunks')
        if not auto_pause_after:
            return False
            
        chunks_in_session = state.get('chunks_processed_in_session', 0)
        return chunks_in_session >= auto_pause_after
    
    def is_chunked_processing(self) -> bool:
        """Check if this batch uses chunked processing"""
        state = self.load_state()
        return state.get('chunked_processing', False) if state else False
    
    def get_status(self) -> Dict:
        """Get current batch status for UI display"""
        state = self.load_state()
        if not state:
            return {"active": False}
            
        # Calculate chunk info
        chunk_info = {}
        if state.get('chunked_processing', False):
            chunks = state.get('chunks', [])
            current_chunk = state.get('current_chunk', 0)
            current_chunk_progress = state.get('current_chunk_progress', 0)
            chunk_size = state.get('chunk_size', 0)
            current_chunk_files_remaining = len(state.get('current_chunk_files', []))
            
            # Calculate documents processed in current chunk
            docs_in_current_chunk = current_chunk_progress
            docs_remaining_in_chunk = current_chunk_files_remaining
            total_docs_in_chunk = docs_in_current_chunk + docs_remaining_in_chunk
            
            chunk_info = {
                "is_chunked": True,
                "total_chunks": len(chunks),
                "current_chunk": current_chunk + 1,  # 1-based for display
                "chunk_size": chunk_size,
                "current_chunk_files": current_chunk_files_remaining,
                "auto_pause_after_chunks": state.get('auto_pause_after_chunks'),
                "chunks_processed_in_session": state.get('chunks_processed_in_session', 0),
                "current_chunk_progress": docs_in_current_chunk,
                "current_chunk_total": total_docs_in_chunk,
                "chunk_progress_percent": round((docs_in_current_chunk / total_docs_in_chunk) * 100, 1) if total_docs_in_chunk > 0 else 0
            }
        else:
            chunk_info = {"is_chunked": False}
        
        return {
            "active": True,
            "batch_id": state['batch_id'],
            "total_files": state['total_files'],
            "completed": state['files_completed'],
            "remaining": len(state['files_remaining']),
            "error_count": state['error_count'],
            "paused": state.get('paused', False),
            "started_at": state['started_at'],
            "last_updated": state['last_updated'],
            "progress_percent": round((state['files_completed'] / state['total_files']) * 100, 1) if state['total_files'] > 0 else 0,
            "has_scan_config": bool(state.get('scan_config')),
            **chunk_info
        }
    
    def _save_state(self, state: Dict):
        """Save batch state to file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save batch state: {e}")
    
    def _get_processed_files(self) -> set:
        """Get set of already processed files from ingested_files.log"""
        if not self.processed_log.exists():
            return set()
            
        try:
            with open(self.processed_log, 'r') as f:
                data = json.load(f)
                return set(data.keys())
        except (json.JSONDecodeError, IOError):
            return set()