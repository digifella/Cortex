"""
Queue Monitor
Live visibility into local queue worker telemetry with cancel/clear controls.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any

import requests
import streamlit as st

from cortex_engine.queue_monitor import QueueMonitorStore
from cortex_engine.version_config import VERSION_STRING


ROOT = Path(__file__).resolve().parent.parent

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


def main():
    st.title("Queue Monitor")
    st.caption(f"Cortex {VERSION_STRING}")
    st.markdown(
        "Monitor worker progress, inspect recent events, request cancellation, and clear local monitor entries."
    )

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
    col4.metric("Events", len(events))

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
                    st.write(
                        f"**Stage:** {item.get('stage', '')}  \n"
                        f"**Progress:** {item.get('progress_pct', 0)}%  \n"
                        f"**Message:** {item.get('message', '')}  \n"
                        f"**Updated:** {item.get('updated_at', '')}"
                    )
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
                st.json(item)

    st.subheader("Recent Events")
    ev_col1, ev_col2 = st.columns([1, 4])
    with ev_col1:
        if st.button("Clear Events", use_container_width=True):
            store.clear_events()
            st.rerun()
    with ev_col2:
        st.caption("Most recent 250 events")
    recent_events = events[-250:]
    if not recent_events:
        st.info("No events logged yet.")
    else:
        for ev in reversed(recent_events):
            st.write(
                f"`{ev.get('ts','')}` "
                f"[{ev.get('level','info').upper()}] "
                f"{'#'+str(ev.get('job_id')) if ev.get('job_id') else ''} "
                f"{ev.get('message','')}"
            )


if __name__ == "__main__":
    main()
