import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils.logging_utils import get_logger


logger = get_logger(__name__)


class BatchState:
    """Lightweight batch manager for Docker distribution.
    Persists simple state to knowledge_hub_db/batch_state.json.
    Implements only the API used by docker/pages/2_Knowledge_Ingest.py.
    """

    def __init__(self, db_path: str):
        # db_path is expected to be container-visible
        self.db_path = db_path
        self.chroma_dir = Path(self.db_path) / "knowledge_hub_db"
        self.state_path = self.chroma_dir / "batch_state.json"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def _default_state(self) -> Dict[str, Any]:
        return {
            "active": False,
            "paused": False,
            "created_at": None,
            "updated_at": None,
            "files_total": 0,
            "files_remaining": [],
            "files_completed": 0,
            "scan_config": {},
            "chunk_size": None,
            "current_chunk": 1,
            "total_chunks": 1,
        }

    def load_state(self) -> Dict[str, Any]:
        try:
            if self.state_path.exists():
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load batch state: {e}")
        return self._default_state()

    def _save_state(self, state: Dict[str, Any]) -> None:
        try:
            state["updated_at"] = datetime.now().isoformat()
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save batch state: {e}")

    def get_status(self) -> Dict[str, Any]:
        s = self.load_state()
        return {
            "active": s.get("active", False),
            "paused": s.get("paused", False),
            "files_total": s.get("files_total", 0),
            "files_completed": s.get("files_completed", 0),
            "current_chunk": s.get("current_chunk", 1),
            "total_chunks": s.get("total_chunks", 1),
        }

    def clear_batch(self) -> None:
        try:
            if self.state_path.exists():
                self.state_path.unlink()
        except Exception:
            # Overwrite with default if unlink fails
            self._save_state(self._default_state())

    def pause_batch(self) -> None:
        s = self.load_state()
        s["paused"] = True
        self._save_state(s)

    def start_new_session(self) -> None:
        s = self._default_state()
        s["created_at"] = datetime.now().isoformat()
        self._save_state(s)

    def resume_or_create_batch(
        self,
        candidate_files: List[str],
        scan_config: Dict[str, Any],
        chunk_size: Optional[int] = None,
    ) -> Tuple[str, List[str], int]:
        """Resume existing batch if active, otherwise create a new one.
        Returns (batch_id, files_to_process, completed_count)
        """
        s = self.load_state()
        if s.get("active") and s.get("files_remaining"):
            files_remaining = s.get("files_remaining", [])
            completed = s.get("files_completed", 0)
            return (s.get("created_at", "batch"), files_remaining, completed)

        # Create new batch
        total = len(candidate_files)
        s = self._default_state()
        s.update({
            "active": True,
            "paused": False,
            "created_at": datetime.now().isoformat(),
            "files_total": total,
            "files_remaining": candidate_files,
            "files_completed": 0,
            "scan_config": scan_config or {},
            "chunk_size": chunk_size,
            "current_chunk": 1,
            "total_chunks": 1,
        })
        self._save_state(s)
        return (s["created_at"], candidate_files, 0)

    def create_batch(
        self,
        remaining_files: List[str],
        scan_config: Dict[str, Any],
        chunk_size: Optional[int] = None,
        auto_pause_chunks: Optional[int] = None,
    ) -> None:
        total = len(remaining_files)
        s = self._default_state()
        s.update({
            "active": True,
            "paused": False,
            "created_at": datetime.now().isoformat(),
            "files_total": total,
            "files_remaining": remaining_files,
            "files_completed": 0,
            "scan_config": scan_config or {},
            "chunk_size": chunk_size,
            "current_chunk": 1,
            "total_chunks": 1,
        })
        self._save_state(s)

    def get_scan_config(self) -> Dict[str, Any]:
        return self.load_state().get("scan_config", {})

    # Chunking helpers (no-op/simple for docker shim)
    def is_chunked_processing(self) -> bool:
        return False

    def get_current_chunk_files(self) -> List[str]:
        return []

