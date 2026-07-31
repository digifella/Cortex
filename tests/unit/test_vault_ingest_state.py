"""Status resolution for the detached private vault ingest."""

import json
import os

import pytest

from cortex_engine.private_vault_rag import (
    start_vault_ingest,
    vault_ingest_status,
)


def _write_state(tmp_path, pid, log_text="", branch="b"):
    from cortex_engine.private_vault_rag import _pid_start_time

    log = tmp_path / "run.log"
    log.write_text(log_text, encoding="utf-8")
    state = tmp_path / "state.json"
    pid_start = _pid_start_time(pid)  # Get real starttime for the PID
    state.write_text(json.dumps({
        "pid": pid,
        "pid_start": pid_start,
        "log_path": str(log),
        "branch": branch,
        "started_at": "2026-07-31T10:00:00",
    }), encoding="utf-8")
    return state


def test_status_is_idle_when_no_state_file(tmp_path):
    assert vault_ingest_status(tmp_path / "missing.json")["state"] == "idle"


def test_status_is_running_for_live_pid_without_done_marker(tmp_path):
    state = _write_state(tmp_path, os.getpid(), "[vault-ingest] phase=textify branch=b")
    result = vault_ingest_status(state)
    assert result["state"] == "running"
    assert result["phase"] == "textify"


def test_status_is_completed_when_done_marker_rc_zero(tmp_path):
    state = _write_state(tmp_path, os.getpid(), "[vault-ingest] phase=done rc=0")
    result = vault_ingest_status(state)
    assert result["state"] == "completed"
    assert result["rc"] == 0


def test_status_is_failed_when_done_marker_nonzero(tmp_path):
    state = _write_state(tmp_path, os.getpid(), "[vault-ingest] phase=done rc=2")
    result = vault_ingest_status(state)
    assert result["state"] == "failed"
    assert result["rc"] == 2


def test_status_is_interrupted_for_dead_pid_without_marker(tmp_path):
    # PID 2^22 is above the default pid_max and is never live.
    state = _write_state(tmp_path, 4194304, "[vault-ingest] phase=textify branch=b")
    assert vault_ingest_status(state)["state"] == "interrupted"


def test_status_includes_log_tail(tmp_path):
    state = _write_state(tmp_path, os.getpid(), "line one\nline two\n[vault-ingest] phase=index")
    assert "line two" in vault_ingest_status(state)["log_tail"]


def test_start_rejects_missing_source_root(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        start_vault_ingest(tmp_path / "nope", "b", state_path=tmp_path / "s.json")


def test_start_rejects_second_run_while_one_is_live(tmp_path):
    state = _write_state(tmp_path, os.getpid(), "[vault-ingest] phase=textify")
    source = tmp_path / "src"
    source.mkdir()
    with pytest.raises(ValueError, match="already running"):
        start_vault_ingest(source, "b", state_path=state)


def test_status_is_interrupted_when_pid_start_does_not_match(tmp_path):
    """PID alive but starttime mismatch = pid was reused → interrupted."""
    state = _write_state(tmp_path, os.getpid(), "[vault-ingest] phase=textify")
    # Overwrite state with wrong pid_start (simulates reused PID)
    state.write_text(json.dumps({
        "pid": os.getpid(),
        "pid_start": "999999",  # Wrong starttime
        "log_path": str(tmp_path / "run.log"),
        "branch": "b",
        "started_at": "2026-07-31T10:00:00",
    }), encoding="utf-8")
    assert vault_ingest_status(state)["state"] == "interrupted"


def test_status_is_interrupted_when_pid_key_missing(tmp_path):
    """Missing pid key should be treated as not alive, not crash."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "log_path": str(tmp_path / "run.log"),
        "branch": "b",
        "started_at": "2026-07-31T10:00:00",
    }), encoding="utf-8")
    result = vault_ingest_status(state)
    assert result["state"] == "interrupted"


def test_status_is_interrupted_when_pid_is_null(tmp_path):
    """null pid should be treated as not alive, not crash."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "pid": None,
        "pid_start": None,
        "log_path": str(tmp_path / "run.log"),
        "branch": "b",
        "started_at": "2026-07-31T10:00:00",
    }), encoding="utf-8")
    result = vault_ingest_status(state)
    assert result["state"] == "interrupted"


def test_cancel_returns_false_when_pid_start_does_not_match(tmp_path):
    """cancel should return False and signal nothing when identity doesn't match."""
    from cortex_engine.private_vault_rag import cancel_vault_ingest

    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "pid": os.getpid(),
        "pid_start": "999999",  # Wrong starttime
        "log_path": str(tmp_path / "run.log"),
        "branch": "b",
        "started_at": "2026-07-31T10:00:00",
    }), encoding="utf-8")
    # Should return False without signaling (we can't easily verify no signal was sent,
    # but the return value confirms the identity check happened)
    assert cancel_vault_ingest(state) is False
