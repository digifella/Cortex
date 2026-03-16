from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from cortex_engine.maintenance_runner import CortexMaintenanceRunner, MaintenanceTask, within_window


def test_within_window_handles_same_day_range():
    now = datetime.fromisoformat("2026-03-16T12:15:00+11:00")
    assert within_window(now, "11:00", "14:00") is True
    assert within_window(now, "13:00", "14:00") is False


def test_within_window_handles_overnight_range():
    late = datetime.fromisoformat("2026-03-16T23:30:00+11:00")
    early = datetime.fromisoformat("2026-03-17T01:15:00+11:00")
    midday = datetime.fromisoformat("2026-03-17T12:00:00+11:00")
    assert within_window(late, "22:00", "02:00") is True
    assert within_window(early, "22:00", "02:00") is True
    assert within_window(midday, "22:00", "02:00") is False


def test_run_skips_when_outside_window(tmp_path):
    runner = CortexMaintenanceRunner(db_path=str(tmp_path))
    report = runner.run(
        profile="daily",
        enforce_window=True,
        now=datetime.fromisoformat("2026-03-16T09:30:00+11:00"),
    )
    assert report["status"] == "skipped_window"
    assert report["reason"] == "outside_configured_window"


def test_run_if_due_skips_when_daily_already_completed(tmp_path):
    runner = CortexMaintenanceRunner(db_path=str(tmp_path))
    runner.state_path.write_text(
        json.dumps(
            {
                "daily": {
                    "marker": "2026-03-16",
                    "completed_at": "2026-03-16T11:05:00+11:00",
                }
            }
        ),
        encoding="utf-8",
    )

    report = runner.run(
        profile="daily",
        run_if_due=True,
        now=datetime.fromisoformat("2026-03-16T12:00:00+11:00"),
    )

    assert report["status"] == "skipped_already_ran"
    assert report["reason"] == "daily_maintenance_already_completed"


def test_run_executes_tasks_and_persists_report_and_state(tmp_path, monkeypatch):
    runner = CortexMaintenanceRunner(db_path=str(tmp_path))

    monkeypatch.setattr(
        runner,
        "_build_tasks",
        lambda profile, real_dedup, keep_backups: [
            MaintenanceTask("task_a", "daily", lambda: {"status": "success", "count": 2}),
            MaintenanceTask("task_b", "daily", lambda: {"status": "success", "count": 3}),
        ],
    )

    report = runner.run(
        profile="daily",
        run_if_due=True,
        now=datetime.fromisoformat("2026-03-16T11:30:00+11:00"),
    )

    assert report["status"] == "ok"
    assert report["task_count"] == 2
    assert report["failure_count"] == 0
    assert runner.reports_dir.joinpath(Path(report["report_path"]).name).exists()

    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    assert state["daily"]["marker"] == "2026-03-16"


def test_weekly_run_marks_weekly_state(tmp_path, monkeypatch):
    runner = CortexMaintenanceRunner(db_path=str(tmp_path))

    monkeypatch.setattr(
        runner,
        "_build_tasks",
        lambda profile, real_dedup, keep_backups: [
            MaintenanceTask("task_weekly", "weekly", lambda: {"status": "success"}),
        ],
    )

    runner.run(
        profile="weekly",
        run_if_due=True,
        now=datetime.fromisoformat("2026-03-16T11:45:00+11:00"),
    )

    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    assert state["weekly"]["marker"] == "2026-W12"
