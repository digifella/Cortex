from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

from .config_manager import ConfigManager
from .utils import get_logger
from .utils.path_utils import resolve_db_root_path

logger = get_logger(__name__)


@dataclass(frozen=True)
class MaintenanceTask:
    name: str
    frequency: str
    runner: Callable[[], Dict[str, Any]]


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _parse_clock_hhmm(value: str) -> tuple[int, int]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Clock value is required")
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid clock value: {value}")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid clock value: {value}")
    return hour, minute


def within_window(now: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    start_hour, start_minute = _parse_clock_hhmm(start_hhmm)
    end_hour, end_minute = _parse_clock_hhmm(end_hhmm)
    current_minutes = now.hour * 60 + now.minute
    start_minutes = start_hour * 60 + start_minute
    end_minutes = end_hour * 60 + end_minute
    if start_minutes <= end_minutes:
        return start_minutes <= current_minutes <= end_minutes
    return current_minutes >= start_minutes or current_minutes <= end_minutes


def _week_marker(now: datetime) -> str:
    iso_year, iso_week, _ = now.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _resolve_db_root(db_path: Optional[str] = None) -> Path:
    candidate = resolve_db_root_path(db_path)
    if candidate:
        return candidate

    configured = ConfigManager().get_config().get("ai_database_path")
    resolved = resolve_db_root_path(configured)
    if resolved:
        return resolved
    raise ValueError("Could not resolve Cortex database root path")


class CortexMaintenanceRunner:
    def __init__(self, db_path: Optional[str] = None):
        self.db_root = _resolve_db_root(db_path)
        self.maintenance_dir = self.db_root / "maintenance"
        self.reports_dir = self.maintenance_dir / "reports"
        self.state_path = self.maintenance_dir / "runner_state.json"
        self.maintenance_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load maintenance state: %s", exc)
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _collections_file_for_db(self) -> Path:
        return self.db_root / "working_collections.json"

    def _collection_manager_for_db(self):
        from .collection_manager import WorkingCollectionManager

        manager = WorkingCollectionManager()
        manager.collections_file = str(self._collections_file_for_db())
        manager.collections = manager._load()
        return manager

    def _build_tasks(self, profile: str, real_dedup: bool, keep_backups: int) -> List[MaintenanceTask]:
        daily_tasks = [
            MaintenanceTask("stakeholder_graph_rebuild", "daily", self._task_stakeholder_graph_rebuild),
            MaintenanceTask("cleanup_orphaned_log_entries", "daily", self._task_cleanup_orphaned_log_entries),
            MaintenanceTask("auto_repair_collections", "daily", self._task_auto_repair_collections),
            MaintenanceTask(
                "vector_dedup_analysis" if not real_dedup else "vector_dedup_cleanup",
                "daily" if not real_dedup else "weekly",
                lambda: self._task_vector_dedup(dry_run=not real_dedup),
            ),
        ]
        weekly_tasks = [
            MaintenanceTask("verify_latest_backup", "weekly", self._task_verify_latest_backup),
            MaintenanceTask(
                "cleanup_old_backups",
                "weekly",
                lambda: self._task_cleanup_old_backups(keep_count=keep_backups),
            ),
        ]
        if profile == "daily":
            return daily_tasks
        if profile == "weekly":
            return daily_tasks + weekly_tasks
        if profile == "full":
            return daily_tasks + weekly_tasks
        raise ValueError(f"Unsupported maintenance profile: {profile}")

    def _profile_due(self, state: Dict[str, Any], profile: str, now: datetime) -> bool:
        marker = now.date().isoformat() if profile == "daily" else _week_marker(now)
        state_key = "daily" if profile == "daily" else "weekly"
        return str((state.get(state_key) or {}).get("marker") or "") != marker

    def _mark_profile_ran(self, state: Dict[str, Any], profile: str, now: datetime, report_path: Path) -> None:
        marker = now.date().isoformat() if profile == "daily" else _week_marker(now)
        state_key = "daily" if profile == "daily" else "weekly"
        state[state_key] = {
            "marker": marker,
            "completed_at": now.isoformat(),
            "report_path": str(report_path),
        }

    def _task_stakeholder_graph_rebuild(self) -> Dict[str, Any]:
        from .stakeholder_signal_store import StakeholderSignalStore

        result = StakeholderSignalStore(base_path=self.db_root / "stakeholder_intel").rebuild_stakeholder_graph()
        return {"status": "success", **result}

    def _task_cleanup_orphaned_log_entries(self) -> Dict[str, Any]:
        from .ingestion_recovery import IngestionRecoveryManager

        result = IngestionRecoveryManager(str(self.db_root)).cleanup_orphaned_log_entries()
        return result if isinstance(result, dict) else {"status": "success", "result": result}

    def _task_auto_repair_collections(self) -> Dict[str, Any]:
        from .ingestion_recovery import IngestionRecoveryManager

        result = IngestionRecoveryManager(str(self.db_root)).auto_repair_collections()
        return result if isinstance(result, dict) else {"status": "success", "result": result}

    def _task_vector_dedup(self, dry_run: bool = True) -> Dict[str, Any]:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from .config import COLLECTION_NAME

        chroma_path = self.db_root / "knowledge_hub_db"
        if not chroma_path.exists():
            return {"status": "no_database", "dry_run": dry_run}

        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            return {"status": "no_collection", "collection_name": COLLECTION_NAME, "dry_run": dry_run}

        result = self._collection_manager_for_db().deduplicate_vector_store(collection, dry_run=dry_run)
        result["dry_run"] = dry_run
        return result

    def _task_verify_latest_backup(self) -> Dict[str, Any]:
        from .sync_backup_manager import SyncBackupManager

        manager = SyncBackupManager(str(self.db_root))
        backups = manager.list_backups()
        if not backups:
            return {"status": "no_backups"}
        backups.sort(key=lambda item: item.creation_time, reverse=True)
        latest = backups[0]
        ok = manager.verify_backup_integrity(latest.backup_id)
        return {
            "status": "success" if ok else "failed",
            "backup_id": latest.backup_id,
            "creation_time": latest.creation_time,
            "integrity_ok": bool(ok),
        }

    def _task_cleanup_old_backups(self, keep_count: int = 10) -> Dict[str, Any]:
        from .sync_backup_manager import SyncBackupManager

        manager = SyncBackupManager(str(self.db_root))
        deleted_count = manager.cleanup_old_backups(keep_count=keep_count)
        return {"status": "success", "deleted_count": deleted_count, "keep_count": keep_count}

    def run(
        self,
        profile: str = "daily",
        *,
        window_start: str = "11:00",
        window_end: str = "14:00",
        enforce_window: bool = False,
        run_if_due: bool = False,
        force: bool = False,
        real_dedup: bool = False,
        keep_backups: int = 10,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        started_at = now or _now_local()
        state = self._load_state()
        report: Dict[str, Any] = {
            "status": "ok",
            "profile": profile,
            "db_root": str(self.db_root),
            "started_at": started_at.isoformat(),
            "window_start": window_start,
            "window_end": window_end,
            "enforce_window": bool(enforce_window),
            "run_if_due": bool(run_if_due),
            "real_dedup": bool(real_dedup),
            "tasks": [],
        }

        if enforce_window and not within_window(started_at, window_start, window_end):
            report["status"] = "skipped_window"
            report["reason"] = "outside_configured_window"
            return report

        if run_if_due and not force:
            if profile == "daily" and not self._profile_due(state, "daily", started_at):
                report["status"] = "skipped_already_ran"
                report["reason"] = "daily_maintenance_already_completed"
                return report
            if profile in {"weekly", "full"} and not self._profile_due(state, "weekly", started_at):
                report["status"] = "skipped_already_ran"
                report["reason"] = "weekly_maintenance_already_completed"
                return report

        failures = 0
        for task in self._build_tasks(profile, real_dedup=real_dedup, keep_backups=keep_backups):
            task_started = perf_counter()
            try:
                result = task.runner()
                status = str(result.get("status") or "success")
                if status in {"error", "failed"}:
                    failures += 1
            except Exception as exc:
                failures += 1
                result = {"status": "error", "error": str(exc)}
            report["tasks"].append(
                {
                    "name": task.name,
                    "frequency": task.frequency,
                    "duration_seconds": round(perf_counter() - task_started, 3),
                    "result": result,
                }
            )

        finished_at = _now_local()
        report["finished_at"] = finished_at.isoformat()
        report["task_count"] = len(report["tasks"])
        report["failure_count"] = failures
        if failures:
            report["status"] = "completed_with_errors"

        report_path = self.reports_dir / f"maintenance_{started_at.strftime('%Y%m%d_%H%M%S')}_{profile}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)

        if report["status"] in {"ok", "completed_with_errors"}:
            self._mark_profile_ran(state, "daily", finished_at, report_path)
            if profile in {"weekly", "full"}:
                self._mark_profile_ran(state, "weekly", finished_at, report_path)
            self._save_state(state)

        return report
