from __future__ import annotations

from cortex_engine.queue_monitor import QueueMonitorStore


def test_append_event_with_metadata_and_clear_by_status(tmp_path):
    state_path = tmp_path / "queue_monitor_state.json"
    store = QueueMonitorStore(state_path)

    store.upsert_job(1, status="completed", message="done")
    store.upsert_job(2, status="failed", message="bad")
    store.upsert_job(3, status="processing", message="running")

    removed = store.clear_jobs_by_status(["completed", "failed"])
    assert removed == 2

    state = store.get_state()
    jobs = dict(state.get("jobs") or {})
    assert set(jobs.keys()) == {"3"}

    store.append_event(
        "Progress update",
        level="info",
        job_id=3,
        stage="processing",
        progress_pct=42,
        source="worker.progress",
    )
    state = store.get_state()
    events = list(state.get("events") or [])
    assert len(events) == 1
    ev = events[0]
    assert ev["job_id"] == 3
    assert ev["stage"] == "processing"
    assert ev["progress_pct"] == 42
    assert ev["source"] == "worker.progress"
