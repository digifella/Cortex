"""
Shared queue monitor state store for worker telemetry and Streamlit UI.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class QueueMonitorStore:
    """Simple JSON-backed state store with atomic writes."""

    def __init__(self, state_path: Path):
        self.state_path = Path(state_path)
        self._lock = threading.Lock()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state(self._initial_state())

    @staticmethod
    def _initial_state() -> Dict[str, Any]:
        return {
            "updated_at": _utc_now_iso(),
            "worker": {},
            "jobs": {},
            "events": [],
        }

    def _read_state(self) -> Dict[str, Any]:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            payload = json.loads(raw) if raw.strip() else {}
            if not isinstance(payload, dict):
                return self._initial_state()
            payload.setdefault("worker", {})
            payload.setdefault("jobs", {})
            payload.setdefault("events", [])
            payload.setdefault("updated_at", _utc_now_iso())
            return payload
        except Exception:
            return self._initial_state()

    def _write_state(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = _utc_now_iso()
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="queue_monitor_", suffix=".json", dir=str(self.state_path.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=True, indent=2, sort_keys=True)
            os.replace(tmp_path, self.state_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._read_state()

    def set_worker(self, **fields: Any) -> None:
        with self._lock:
            state = self._read_state()
            worker = dict(state.get("worker") or {})
            worker.update(fields)
            worker["updated_at"] = _utc_now_iso()
            state["worker"] = worker
            self._write_state(state)

    def upsert_job(self, job_id: int, **fields: Any) -> None:
        with self._lock:
            state = self._read_state()
            jobs = dict(state.get("jobs") or {})
            key = str(job_id)
            item = dict(jobs.get(key) or {})
            item.update(fields)
            item.setdefault("job_id", job_id)
            item.setdefault("created_at", _utc_now_iso())
            item["updated_at"] = _utc_now_iso()
            jobs[key] = item
            state["jobs"] = jobs
            self._write_state(state)

    def append_event(self, message: str, level: str = "info", job_id: Optional[int] = None) -> None:
        with self._lock:
            state = self._read_state()
            events: List[Dict[str, Any]] = list(state.get("events") or [])
            events.append(
                {
                    "ts": _utc_now_iso(),
                    "level": level.lower().strip() or "info",
                    "job_id": job_id,
                    "message": str(message),
                }
            )
            # Keep file bounded.
            state["events"] = events[-1000:]
            self._write_state(state)

    def request_cancel(self, job_id: int, requested_by: str = "operator") -> None:
        self.upsert_job(
            job_id,
            cancel_requested=True,
            cancel_requested_at=_utc_now_iso(),
            cancel_requested_by=requested_by,
        )
        self.append_event("Cancellation requested", level="warning", job_id=job_id)

    def clear_job(self, job_id: int) -> None:
        with self._lock:
            state = self._read_state()
            jobs = dict(state.get("jobs") or {})
            jobs.pop(str(job_id), None)
            state["jobs"] = jobs
            self._write_state(state)

    def clear_events(self) -> None:
        with self._lock:
            state = self._read_state()
            state["events"] = []
            self._write_state(state)

    def is_cancel_requested(self, job_id: int) -> bool:
        with self._lock:
            state = self._read_state()
            jobs = dict(state.get("jobs") or {})
            item = dict(jobs.get(str(job_id)) or {})
            return bool(item.get("cancel_requested", False))
