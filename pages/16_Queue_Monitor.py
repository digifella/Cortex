"""
Queue Monitor
Live visibility into local queue worker telemetry with cancel/clear controls.
"""

from __future__ import annotations

import os
import json
import fcntl
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

import requests
import streamlit as st

from cortex_engine.queue_monitor import QueueMonitorStore
from cortex_engine.version_config import VERSION_STRING


ROOT = Path(__file__).resolve().parent.parent
PRIVATE_QUEUE_PATH = Path("/home/longboardfella/.nemoclaw/private-knowledge-queue.json")
PRIVATE_QUEUE_LOCK = Path("/home/longboardfella/.nemoclaw/private-knowledge-queue.lock")
PRIVATE_QUEUE_RUNNER = Path("/home/longboardfella/nemoclaw-private-knowledge-queue-runner.py")
PRIVATE_QUEUE_MANUAL_LOG = Path("/home/longboardfella/.nemoclaw/logs/private-knowledge-queue-manual.log")
PRIVATE_BATCH_DIR = Path("/home/longboardfella/.nemoclaw/private-knowledge-batches")
PRIVATE_SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".txt"}

st.set_page_config(page_title="Queue Monitor", layout="wide", page_icon="📡")


def _load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _resolve_monitor_state_path() -> Path:
    env_path = ROOT / "worker" / "config.env"
    env_vars = _load_env_file(env_path)
    raw = (
        os.environ.get("QUEUE_MONITOR_STATE_PATH")
        or env_vars.get("QUEUE_MONITOR_STATE_PATH")
        or str(ROOT / "worker" / "tmp" / "queue_monitor_state.json")
    )
    return Path(raw)


def _queue_api_config() -> Dict[str, str]:
    env_path = ROOT / "worker" / "config.env"
    env_vars = _load_env_file(env_path)
    return {
        "server_url": (os.environ.get("QUEUE_SERVER_URL") or env_vars.get("QUEUE_SERVER_URL") or "").strip(),
        "secret_key": (os.environ.get("QUEUE_SECRET_KEY") or env_vars.get("QUEUE_SECRET_KEY") or "").strip(),
    }


def _send_remote_cancel(job_id: int, reason: str) -> str:
    cfg = _queue_api_config()
    if not cfg["server_url"] or not cfg["secret_key"]:
        return "Queue API config missing (QUEUE_SERVER_URL / QUEUE_SECRET_KEY)."

    session = requests.Session()
    session.headers.update({"X-Queue-Key": cfg["secret_key"]})
    params = {"action": "fail", "id": str(job_id)}
    data = {"error": reason[:5000]}

    try:
        resp = session.post(cfg["server_url"], params=params, data=data, timeout=30)
        if resp.status_code == 403:
            params["key"] = cfg["secret_key"]
            resp = session.post(cfg["server_url"], params=params, data=data, timeout=30)
        resp.raise_for_status()
        return f"Remote queue updated for job {job_id}."
    except Exception as e:
        return f"Remote queue update failed for job {job_id}: {e}"


def _sort_jobs(jobs: Dict[str, Any]):
    values = list((jobs or {}).values())
    return sorted(values, key=lambda x: x.get("updated_at", ""), reverse=True)


def _parse_iso(ts: str) -> datetime | None:
    text = str(ts or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _is_stale(ts: str, stale_seconds: int = 300) -> bool:
    dt = _parse_iso(ts)
    if not dt:
        return False
    return (datetime.now(timezone.utc) - dt).total_seconds() > stale_seconds


def _local_now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return cleaned or "source"


def _normalise_source_path(raw_path: str) -> Path:
    text = str(raw_path or "").strip().strip('"').strip("'")
    if len(text) >= 3 and text[1:3] in {":\\", ":/"}:
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(text).expanduser()


def _source_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".git", ".obsidian"} for part in path.parts):
            continue
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() in PRIVATE_SUPPORTED_EXTS:
            files.append(path)
    return files


def _load_private_queue() -> Dict[str, Any]:
    try:
        raw = PRIVATE_QUEUE_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        payload["entries"] = []
    return payload


def _write_private_queue(payload: Dict[str, Any]) -> None:
    PRIVATE_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_QUEUE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PRIVATE_QUEUE_LOCK.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Private queue runner is active; try again after the current job finishes.") from exc

        payload["updated_at"] = _local_now_iso()
        fd, tmp_name = tempfile.mkstemp(
            prefix="private-knowledge-queue-",
            suffix=".json",
            dir=str(PRIVATE_QUEUE_PATH.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, PRIVATE_QUEUE_PATH)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _entry_label(entry: Dict[str, Any]) -> str:
    return f"{entry.get('branch_name', '[unnamed]')} [{entry.get('status', 'pending')}]"


def _find_private_entry(entries: list[Dict[str, Any]], branch_name: str) -> tuple[int, Dict[str, Any]] | tuple[None, None]:
    for idx, entry in enumerate(entries):
        if str(entry.get("branch_name", "")) == branch_name:
            return idx, entry
    return None, None


def _insert_private_entries(new_entries: list[Dict[str, Any]], *, priority: bool) -> str:
    if not new_entries:
        return "No entries to add."

    queue = _load_private_queue()
    entries = list(queue.get("entries") or [])
    existing = {str(entry.get("branch_name", "")) for entry in entries}
    duplicates = [entry["branch_name"] for entry in new_entries if str(entry.get("branch_name", "")) in existing]
    if duplicates:
        raise ValueError(f"Queue already has branch name(s): {', '.join(duplicates[:5])}")

    if priority:
        insert_at = next(
            (i for i, item in enumerate(entries) if str(item.get("status", "pending")).lower() == "pending"),
            0,
        )
        entries[insert_at:insert_at] = new_entries
    else:
        entries.extend(new_entries)
    queue["entries"] = entries
    _write_private_queue(queue)
    return f"Added {len(new_entries)} private knowledge queue entr{'y' if len(new_entries) == 1 else 'ies'}."


def _build_queue_entries_for_source(
    *,
    source_root: Path,
    branch_name: str,
    max_files: int,
    notes: str,
    split_batches: bool,
    priority: bool,
) -> list[Dict[str, Any]]:
    files = _source_files(source_root)
    if not files:
        raise ValueError(f"No supported files found under {source_root}. Supported: {', '.join(sorted(PRIVATE_SUPPORTED_EXTS))}")

    now = _local_now_iso()
    common = {
        "source_root": str(source_root),
        "max_files": int(max_files),
        "notes": notes.strip() or "Added from Cortex Queue Monitor",
        "status": "pending",
        "created_at": now,
        "created_by": "cortex_queue_monitor",
        "seed_file_count": len(files),
    }

    if not split_batches or len(files) <= max_files:
        return [{
            **common,
            "branch_name": branch_name,
            "operator_action": "enqueue_priority" if priority else "enqueue",
        }]

    PRIVATE_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx in range(0, len(files), max_files):
        batch_no = idx // max_files + 1
        batch_files = files[idx:idx + max_files]
        batch_name = f"{branch_name}-batch-{batch_no:02d}"
        batch_path = PRIVATE_BATCH_DIR / f"{batch_name}.txt"
        batch_path.write_text(
            "\n".join(str(path) for path in batch_files) + "\n",
            encoding="utf-8",
        )
        entries.append({
            **common,
            "branch_name": batch_name,
            "file_list": str(batch_path),
            "last_file_count": len(batch_files),
            "notes": f"{common['notes']} Batch {batch_no} of {(len(files) - 1) // max_files + 1}.",
            "operator_action": "enqueue_priority_batch" if priority else "enqueue_batch",
        })
    return entries


def _mutate_private_entry(branch_name: str, action: str) -> str:
    queue = _load_private_queue()
    entries = list(queue.get("entries") or [])
    idx, entry = _find_private_entry(entries, branch_name)
    if entry is None or idx is None:
        raise ValueError(f"Queue entry not found: {branch_name}")

    status = str(entry.get("status", "pending")).lower()
    if action == "delete":
        if status == "running":
            raise ValueError("Refusing to delete a running job.")
        entries.pop(idx)
        queue["entries"] = entries
        _write_private_queue(queue)
        return f"Deleted {branch_name}."

    if action == "pause":
        if status == "running":
            raise ValueError("Refusing to pause a running job from the UI.")
        entry["status"] = "paused"
        entry["last_summary"] = "Paused by operator from Cortex Queue Monitor"
        entry["paused_at"] = _local_now_iso()
        _write_private_queue(queue)
        return f"Paused {branch_name}."

    if action in {"retry", "promote"}:
        entry["status"] = "pending"
        entry["last_error"] = ""
        entry["runner_pid"] = None
        entry["operator_updated_at"] = _local_now_iso()
        entry["operator_action"] = action
        if action == "promote":
            entries.pop(idx)
            insert_at = next(
                (i for i, item in enumerate(entries) if str(item.get("status", "pending")).lower() == "pending"),
                0,
            )
            entries.insert(insert_at, entry)
            queue["entries"] = entries
            _write_private_queue(queue)
            return f"Promoted {branch_name} to the front of the pending queue."
        _write_private_queue(queue)
        return f"Set {branch_name} to pending."

    raise ValueError(f"Unsupported action: {action}")


def _run_private_queue_now() -> str:
    if not PRIVATE_QUEUE_RUNNER.exists():
        return f"Runner missing: {PRIVATE_QUEUE_RUNNER}"
    PRIVATE_QUEUE_MANUAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PRIVATE_QUEUE_MANUAL_LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_local_now_iso()}] manual run requested from Cortex Queue Monitor\n")
        proc = subprocess.Popen(
            ["python3", str(PRIVATE_QUEUE_RUNNER)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd="/home/longboardfella",
            start_new_session=True,
        )
    return f"Started private queue runner as PID {proc.pid}. Log: {PRIVATE_QUEUE_MANUAL_LOG}"


def _tail_text(path: Path, limit: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[could not read {path}: {exc}]"
    return text[-limit:]


def render_private_knowledge_queue() -> None:
    st.header("Private Knowledge Queue")
    st.caption("Monitor and manage the cron-driven OneDrive-to-private-vault import queue.")

    q_top_left, q_top_mid, q_top_right = st.columns([2, 1, 1])
    with q_top_left:
        st.code(f"Queue: {PRIVATE_QUEUE_PATH}")
        st.code(f"Manual run log: {PRIVATE_QUEUE_MANUAL_LOG}")
    with q_top_mid:
        if st.button("Refresh Private Queue", use_container_width=True):
            st.rerun()
    with q_top_right:
        if st.button("Run Next Pending Now", type="primary", use_container_width=True):
            st.info(_run_private_queue_now())

    queue = _load_private_queue()
    entries = list(queue.get("entries") or [])
    counts = Counter(str(e.get("status", "pending")).lower() for e in entries)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total", len(entries))
    m2.metric("Pending", counts.get("pending", 0))
    m3.metric("Running", counts.get("running", 0))
    m4.metric("Done", counts.get("done", 0))
    m5.metric("Paused", counts.get("paused", 0))
    m6.metric("Failed", counts.get("failed", 0))

    with st.expander("Add Knowledge Source", expanded=False):
        st.caption("Add a folder of PDFs/DOCX/PPTX/TXT files to the private vault import queue.")
        with st.form("private_queue_add_source"):
            source_raw = st.text_input(
                "Source folder",
                placeholder=r"C:\Users\paul\OneDrive - VentraIP Australia\01_Clients\Active\Example\_Knowledge",
                help="Windows C:\\ paths are converted to /mnt/c/... automatically.",
            )
            branch_raw = st.text_input(
                "Branch name",
                placeholder="Leave blank to generate from the folder name",
            )
            max_files = st.number_input("Files per job", min_value=1, max_value=100, value=20, step=1)
            notes = st.text_input("Notes", value="Added from Cortex Queue Monitor")
            c_split, c_priority, c_run = st.columns(3)
            with c_split:
                split_batches = st.checkbox("Split large source into batches", value=True)
            with c_priority:
                priority = st.checkbox("Add to front of queue", value=True)
            with c_run:
                run_now = st.checkbox("Run next job now", value=False)
            submitted = st.form_submit_button("Add Source", type="primary", use_container_width=True)

        if submitted:
            try:
                source_root = _normalise_source_path(source_raw)
                if not source_root.is_dir():
                    raise ValueError(f"Source folder not found: {source_root}")
                branch_name = _slugify(branch_raw or source_root.name)
                new_entries = _build_queue_entries_for_source(
                    source_root=source_root.resolve(),
                    branch_name=branch_name,
                    max_files=int(max_files),
                    notes=notes,
                    split_batches=bool(split_batches),
                    priority=bool(priority),
                )
                st.success(_insert_private_entries(new_entries, priority=bool(priority)))
                st.info(f"Prepared {len(new_entries)} job(s) from {new_entries[0].get('seed_file_count', 0)} supported file(s).")
                if run_now:
                    st.info(_run_private_queue_now())
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    f1, f2, f3 = st.columns([1, 2, 2])
    with f1:
        status_filter = st.selectbox(
            "Status",
            ["active", "pending", "running", "failed", "paused", "done", "all"],
            key="private_queue_status_filter",
            help="Active shows pending, running, failed, and paused entries. Done jobs are hidden by default for speed.",
        )
    with f2:
        search = st.text_input("Search branch/source", value="", key="private_queue_search").strip().lower()
    with f3:
        show_columns = st.multiselect(
            "Columns",
            [
                "branch_name", "status", "last_file_count", "max_files", "duration_seconds",
                "started_at", "finished_at", "last_summary", "source_root",
            ],
            default=["branch_name", "status", "last_file_count", "duration_seconds", "finished_at", "last_summary"],
            key="private_queue_columns",
        )

    filtered = []
    for entry in entries:
        status = str(entry.get("status", "pending")).lower()
        haystack = f"{entry.get('branch_name','')} {entry.get('source_root','')} {entry.get('last_summary','')}".lower()
        if status_filter == "active" and status == "done":
            continue
        if status_filter not in {"all", "active"} and status != status_filter:
            continue
        if search and search not in haystack:
            continue
        filtered.append(entry)

    row_limit = st.slider(
        "Rows to render",
        min_value=25,
        max_value=500,
        value=min(100, max(25, len(filtered) or 25)),
        step=25,
        key="private_queue_row_limit",
    )
    visible = filtered[:row_limit]
    if len(filtered) > row_limit:
        st.caption(f"Showing {row_limit} of {len(filtered)} matching jobs. Use filters/search to narrow the list.")

    rows = [{col: entry.get(col, "") for col in show_columns} for entry in visible]
    st.dataframe(rows, use_container_width=True, hide_index=True, height=min(520, 38 + 35 * max(1, len(rows))))

    if not filtered:
        st.info("No private queue entries match the current filters.")
        return

    st.subheader("Selected Job")
    selected_branch = st.selectbox(
        "Job",
        [str(e.get("branch_name", "")) for e in visible],
        format_func=lambda value: _entry_label(next((e for e in visible if str(e.get("branch_name", "")) == value), {})),
        key="private_queue_selected_branch",
    )
    selected = next((e for e in entries if str(e.get("branch_name", "")) == selected_branch), {})
    selected_status = str(selected.get("status", "pending")).lower()

    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        if st.button("Promote", use_container_width=True, disabled=selected_status == "running"):
            st.success(_mutate_private_entry(selected_branch, "promote"))
            st.rerun()
    with a2:
        if st.button("Retry / Set Pending", use_container_width=True, disabled=selected_status == "running"):
            st.success(_mutate_private_entry(selected_branch, "retry"))
            st.rerun()
    with a3:
        if st.button("Promote + Run Now", type="primary", use_container_width=True, disabled=selected_status == "running"):
            st.success(_mutate_private_entry(selected_branch, "promote"))
            st.info(_run_private_queue_now())
            st.rerun()
    with a4:
        if st.button("Pause", use_container_width=True, disabled=selected_status == "running"):
            st.success(_mutate_private_entry(selected_branch, "pause"))
            st.rerun()
    with a5:
        confirm_delete = st.checkbox("Confirm delete", key=f"private_queue_confirm_delete_{selected_branch}")
        if st.button("Delete", use_container_width=True, disabled=(selected_status == "running" or not confirm_delete)):
            st.success(_mutate_private_entry(selected_branch, "delete"))
            st.rerun()

    with st.expander("Selected Job JSON", expanded=False):
        st.json(selected)

    log_left, log_right = st.columns([1, 1])
    with log_left:
        last_log = str(selected.get("last_log_path") or "").strip()
        if last_log:
            with st.expander("Last Job Log", expanded=False):
                st.code(_tail_text(Path(last_log), limit=6000), language="log")
    with log_right:
        with st.expander("Manual Runner Log", expanded=False):
            st.code(_tail_text(PRIVATE_QUEUE_MANUAL_LOG, limit=6000), language="log")


def main():
    st.title("Queue Monitor")
    st.caption(f"Cortex {VERSION_STRING}")
    st.markdown(
        "Monitor worker progress, inspect recent events, request cancellation, and clear local monitor entries."
    )

    view = st.radio(
        "Monitor",
        ["Private Knowledge Queue", "Website Queue Worker"],
        horizontal=True,
        key="queue_monitor_view",
    )

    if view == "Private Knowledge Queue":
        render_private_knowledge_queue()
    else:
        render_worker_queue()


def render_worker_queue() -> None:
    state_path = _resolve_monitor_state_path()
    store = QueueMonitorStore(state_path)
    state = store.get_state()

    top_left, top_mid, top_right = st.columns([2, 1, 1])
    with top_left:
        st.code(f"State file: {state_path}")
    with top_mid:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    with top_right:
        auto_refresh = st.toggle("Auto refresh (5s)", value=False, key="queue_monitor_auto_refresh")

    if auto_refresh:
        time.sleep(5)
        st.rerun()

    worker = dict(state.get("worker") or {})
    jobs = _sort_jobs(state.get("jobs") or {})
    events = list(state.get("events") or [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Worker Status", worker.get("status", "unknown"))
    col2.metric("Tracked Jobs", len(jobs))
    col3.metric("Active Jobs", sum(1 for j in jobs if j.get("status") in {"claimed", "processing", "uploading_result"}))
    col4.metric("Stale Jobs", sum(1 for j in jobs if _is_stale(j.get("updated_at", ""))))

    with st.expander("Bulk Actions", expanded=False):
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Clear Completed/Failed/Cancelled", use_container_width=True):
                removed = store.clear_jobs_by_status(["completed", "failed", "cancelled"])
                st.success(f"Cleared {removed} local job entries.")
                st.rerun()
        with b2:
            if st.button("Clear Only Completed", use_container_width=True):
                removed = store.clear_jobs_by_status(["completed"])
                st.success(f"Cleared {removed} completed job entries.")
                st.rerun()
        with b3:
            if st.button("Clear Events", use_container_width=True):
                store.clear_events()
                st.success("Cleared event log.")
                st.rerun()

    with st.expander("Worker Details", expanded=True):
        st.json(worker)

    st.subheader("Jobs")
    if not jobs:
        st.info("No jobs tracked yet.")
    else:
        for item in jobs:
            job_id = int(item.get("job_id"))
            status = str(item.get("status", "unknown"))
            title = f"#{job_id} [{status}] {item.get('input_filename', '') or item.get('job_type', '')}"
            with st.expander(title, expanded=status in {"processing", "claimed", "uploading_result"}):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.progress(max(0, min(100, int(item.get("progress_pct", 0)))) / 100.0)
                    st.write(
                        f"**Stage:** {item.get('stage', '')}  \n"
                        f"**Progress:** {item.get('progress_pct', 0)}%  \n"
                        f"**Trace:** `{item.get('trace_id', '')}`  \n"
                        f"**Scope:** `{item.get('tenant_id', 'default')}/{item.get('project_id', 'default')}`  \n"
                        f"**Message:** {item.get('message', '')}  \n"
                        f"**Updated:** {item.get('updated_at', '')}"
                    )
                    if _is_stale(item.get("updated_at", "")):
                        st.warning("No job updates for >5 minutes (stale telemetry).")
                with c2:
                    if st.button("Request Cancel", key=f"cancel_local_{job_id}", use_container_width=True):
                        store.request_cancel(job_id, requested_by="streamlit_queue_monitor")
                        st.success(f"Cancellation requested for job {job_id}.")
                        st.rerun()
                    if st.button("Clear Local Entry", key=f"clear_local_{job_id}", use_container_width=True):
                        store.clear_job(job_id)
                        st.success(f"Cleared local entry for job {job_id}.")
                        st.rerun()
                with c3:
                    if st.button("Send Website Cancel", key=f"cancel_remote_{job_id}", use_container_width=True):
                        msg = _send_remote_cancel(
                            job_id,
                            "Cancelled by operator from Queue Monitor",
                        )
                        st.info(msg)
                        st.rerun()
                    if st.button("Cancel + Clear Local", key=f"cancel_clear_{job_id}", use_container_width=True):
                        store.request_cancel(job_id, requested_by="streamlit_queue_monitor")
                        msg = _send_remote_cancel(
                            job_id,
                            "Cancelled by operator from Queue Monitor",
                        )
                        store.clear_job(job_id)
                        st.info(msg)
                        st.success(f"Cleared local entry for job {job_id}.")
                        st.rerun()
                st.json(item)

    st.subheader("Recent Events")
    ev_col1, ev_col2, ev_col3, ev_col4 = st.columns([1, 2, 2, 2])
    with ev_col1:
        level_filter = st.selectbox("Level", ["all", "error", "warning", "info"], index=0, key="qm_level_filter")
    with ev_col2:
        job_filter = st.text_input("Job ID filter", value="", key="qm_job_filter").strip()
    with ev_col3:
        max_events = st.slider("Rows", min_value=50, max_value=1000, value=250, step=50, key="qm_max_events")
    with ev_col4:
        quiet_connectivity = st.toggle("Quiet connectivity", value=True, key="qm_quiet_connectivity")

    filtered_events = list(events)
    if quiet_connectivity:
        filtered_events = [
            ev for ev in filtered_events
            if str(ev.get("source", "")).strip().lower() != "worker.connectivity"
        ]
    if level_filter != "all":
        filtered_events = [ev for ev in filtered_events if str(ev.get("level", "")).lower() == level_filter]
    if job_filter:
        filtered_events = [ev for ev in filtered_events if str(ev.get("job_id", "")) == job_filter]

    recent_events = filtered_events[-max_events:]
    if not recent_events:
        st.info("No events logged yet.")
    else:
        for ev in reversed(recent_events):
            stage_text = f" stage={ev.get('stage')}" if ev.get("stage") else ""
            pct_text = f" {ev.get('progress_pct', 0)}%" if ev.get("progress_pct") is not None else ""
            source_text = f" src={ev.get('source')}" if ev.get("source") else ""
            st.code(
                f"`{ev.get('ts','')}` "
                f"[{ev.get('level','info').upper()}] "
                f"{'#'+str(ev.get('job_id')) if ev.get('job_id') else ''} "
                f"{ev.get('message','')}{stage_text}{pct_text}{source_text}"
            )


if __name__ == "__main__":
    main()
