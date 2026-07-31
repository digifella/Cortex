"""Status resolution for the detached private vault ingest."""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from cortex_engine.private_vault_rag import (
    start_vault_ingest,
    vault_ingest_status,
)


@pytest.fixture
def live_child():
    """A real, disposable live process -- never the test runner itself."""
    proc = subprocess.Popen(["sleep", "30"])
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture
def zombie_child():
    """A real exited-but-unreaped process (/proc state 'Z'). Reaped on teardown."""
    proc = subprocess.Popen(["true"])
    # Do NOT poll()/wait() here -- that would reap it and destroy the zombie.
    for _ in range(200):
        state = _proc_state(proc.pid)
        if state == "Z":
            break
        time.sleep(0.02)
    assert _proc_state(proc.pid) == "Z", "child did not become a zombie"
    yield proc
    proc.wait()


def _proc_state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


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


def test_cancel_returns_false_when_pid_start_does_not_match(tmp_path, live_child):
    """cancel should return False and signal nothing when identity doesn't match.

    Targets a disposable child, never os.getpid(): if the identity check ever
    regresses, killpg would SIGTERM the whole process group -- pytest included.
    """
    from cortex_engine.private_vault_rag import cancel_vault_ingest

    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "pid": live_child.pid,
        "pid_start": "999999",  # Wrong starttime
        "log_path": str(tmp_path / "run.log"),
        "branch": "b",
        "started_at": "2026-07-31T10:00:00",
    }), encoding="utf-8")
    assert cancel_vault_ingest(state) is False
    # The child must be untouched: nothing was signalled.
    assert live_child.poll() is None


def test_cancel_returns_false_for_zero_pid(tmp_path):
    """pid 0 means 'our own process group' to killpg -- it must never get there."""
    from cortex_engine.private_vault_rag import cancel_vault_ingest

    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "pid": 0,
        "pid_start": None,
        "log_path": str(tmp_path / "run.log"),
        "branch": "b",
        "started_at": "2026-07-31T10:00:00",
    }), encoding="utf-8")
    assert cancel_vault_ingest(state) is False


def test_status_is_interrupted_for_zombie_pid(tmp_path, zombie_child):
    """A dead-but-unreaped child must not read as running.

    Streamlit spawns the ingest detached but stays its parent and never wait()s,
    so a finished child lingers as a zombie: /proc/<pid>/stat still exists, its
    starttime is unchanged and os.kill(pid, 0) still succeeds. Liveness has to be
    decided on the /proc state character, or the panel wedges on "running" forever
    and every new ingest is refused.
    """
    state = _write_state(tmp_path, zombie_child.pid, "[vault-ingest] phase=textify branch=b")
    assert vault_ingest_status(state)["state"] == "interrupted"


def test_start_writes_state_that_status_reads_back(tmp_path, monkeypatch):
    """Round-trip the real writer against the real reader, to a terminal state.

    Every other test here consumes a hand-rolled state payload; only this one
    proves start_vault_ingest and vault_ingest_status still agree on the schema.
    """
    from cortex_engine import private_vault_rag as pvr

    # Keep the run entirely off the real vault, manifest, state file and log dir.
    monkeypatch.setattr(pvr, "VAULT_INGEST_LOG_DIR", tmp_path / "logs")
    real_state = pvr.VAULT_INGEST_STATE
    real_state_before = real_state.stat().st_mtime if real_state.exists() else None
    real_manifest = Path(
        "/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive"
        "/.ingest/private-knowledge-manifest.json"
    )
    manifest_before = real_manifest.stat().st_mtime if real_manifest.exists() else None

    source = tmp_path / "src"
    source.mkdir()
    (source / "note.txt").write_text("a short private note", encoding="utf-8")
    state_path = tmp_path / "state.json"

    payload = start_vault_ingest(
        source, "roundtrip-branch",
        dest_root=tmp_path / "dest",
        dry_run=True,
        state_path=state_path,
    )
    try:
        assert payload["pid"] > 0
        first = vault_ingest_status(state_path)
        assert first["branch"] == "roundtrip-branch"
        assert first["state"] in {"running", "completed", "failed"}

        deadline = time.time() + 120
        while time.time() < deadline:
            result = vault_ingest_status(state_path)
            if result["state"] in {"completed", "failed", "interrupted"}:
                break
            time.sleep(0.5)
        # A dry run writes nothing, so phase 2 is skipped and rc is the ingest's.
        assert result["state"] == "completed", result["log_tail"]
        assert result["rc"] == 0
        assert "phase=skip-index reason=dry run" in result["log_tail"]
    finally:
        if payload["pid"] > 0:
            try:
                os.kill(payload["pid"], 0)
                os.kill(payload["pid"], 15)
            except OSError:
                pass

    # Nothing real was touched.
    assert (real_state.stat().st_mtime if real_state.exists() else None) == real_state_before
    assert (real_manifest.stat().st_mtime if real_manifest.exists() else None) == manifest_before
    assert not (tmp_path / "dest").exists()


def test_start_vault_ingest_forwards_file_list(tmp_path, monkeypatch):
    # start_vault_ingest logs the command it spawned as the log's first line,
    # so that line is the contract we can assert against without a real ingest.
    from cortex_engine import private_vault_rag as pvr

    monkeypatch.setattr(pvr, "VAULT_INGEST_LOG_DIR", tmp_path / "logs")
    source = tmp_path / "staging"
    source.mkdir()
    (source / "doc.txt").write_text("hello", encoding="utf-8")
    listing = source / ".filelist.txt"
    listing.write_text(f"{source / 'doc.txt'}\n", encoding="utf-8")

    started = pvr.start_vault_ingest(
        source, "upload-doc", tmp_path / "dest",
        dry_run=True, file_list=listing, state_path=tmp_path / "state.json",
    )

    try:
        first_line = Path(started["log_path"]).read_text(encoding="utf-8").splitlines()[0]
        assert "--file-list" in first_line
        assert str(listing) in first_line
    finally:
        if started["pid"] > 0:
            try:
                os.kill(started["pid"], 0)
                os.kill(started["pid"], 15)
            except OSError:
                pass
