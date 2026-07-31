# Private Vault Ingest UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a folder-path document ingest to the Cortex Suite's Vault GraphRAG page that textifies documents into the private vault and then chains the existing vault indexer, running detached so it survives closing the browser tab.

**Architecture:** A new `cortex_engine/vault_ingest.py` owns the two-phase sequence (textify → index) and is spawned once as a detached process with stdout redirected to a log file. `cortex_engine/private_vault_rag.py` gains start/status/cancel helpers backed by a small JSON state file. `pages/18_Private_Vault_GraphRAG.py` gains an ingest panel that polls status and tails the log.

**Tech Stack:** Python 3.11, Streamlit, subprocess, pytest. No new dependencies.

## Global Constraints

- Python 3.11 (the `cortex_suite/venv` interpreter). NumPy <2.0.0, spaCy 3.5.0–3.8.0 — unchanged by this work.
- Scope is the **private** vault only: `/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive`. Do not touch the public AI-Vault or wiki publish pipeline.
- The two phases run under **different interpreters**. Phase 1 needs `cortex_suite/venv/bin/python` (for `cortex_engine.textifier`); phase 2 needs `/home/longboardfella/venvs/vault-rag/bin/python3` (different chromadb pin for `vault-rag-db`). Never run one under the other's interpreter.
- No test may write to the real vault, the real manifest, or `vault-rag-db`. Every test overrides `--manifest-path` and `--dest-root` to temp paths and injects a fake process runner.
- Never hardcode a user-facing path where an existing module constant exists — reuse `PRIVATE_VAULT` from `cortex_engine.private_vault_rag`.
- Existing scripts `~/nemoclaw-private-knowledge-ingest.py` and `~/nemoclaw-vault-indexer.py` are **called, never modified**.
- Branch: `feat/private-vault-ingest-ui` (already created; the design spec is committed there as `fcb1102`).

### Reference: exact external contracts

Ingest script (`~/nemoclaw-private-knowledge-ingest.py`) — required flags `--source-root`, `--branch-name`; optional `--dest-root`, `--manifest-path`, `--limit`, `--dry-run`, `--force`, `--pdf-strategy {hybrid,docling,pymupdf}`, `--use-vision`. Final stdout line:

```
[private-ingest] done changed=N skipped=N failures=N dry_run=True|False
```

Exit code `0` when `failures == 0`, else `2`. **`2` means partial failure — `changed` may still be large.**

Indexer script (`~/nemoclaw-vault-indexer.py`) — flags `--full`, `--stats`, `--private-only`, `--public-only`.

---

### Task 1: Summary parsing and the phase-chaining policy

Pure logic, no subprocesses. This is the crux of the design's chaining table and is worth its own review gate.

**Files:**
- Create: `cortex_engine/vault_ingest.py`
- Test: `tests/unit/test_vault_ingest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `IngestSummary` (frozen dataclass, fields `changed: int`, `skipped: int`, `failures: int`, `dry_run: bool`), `parse_ingest_summary(output: str) -> IngestSummary | None`, `should_index(summary: IngestSummary | None) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vault_ingest.py`:

```python
"""Unit tests for the private vault ingest wrapper."""

import pytest

from cortex_engine.vault_ingest import (
    IngestSummary,
    parse_ingest_summary,
    should_index,
)

DONE = "[private-ingest] done changed={c} skipped={s} failures={f} dry_run={d}"


def test_parse_extracts_all_fields():
    summary = parse_ingest_summary(DONE.format(c=7, s=2, f=1, d="False"))
    assert summary == IngestSummary(changed=7, skipped=2, failures=1, dry_run=False)


def test_parse_finds_summary_among_other_output():
    output = "\n".join([
        "[private-ingest] converting a.pdf",
        "[private-ingest] ERROR: b.pdf: boom",
        DONE.format(c=1, s=0, f=1, d="False"),
    ])
    summary = parse_ingest_summary(output)
    assert summary.changed == 1
    assert summary.failures == 1


def test_parse_reads_dry_run_true():
    summary = parse_ingest_summary(DONE.format(c=3, s=0, f=0, d="True"))
    assert summary.dry_run is True


def test_parse_returns_none_when_no_summary_line():
    assert parse_ingest_summary("killed before finishing") is None


def test_index_runs_on_happy_path():
    assert should_index(IngestSummary(changed=5, skipped=0, failures=0, dry_run=False)) is True


def test_index_runs_on_partial_failure():
    # Exit code 2, but 97 files converted -- halting would strand them unindexed.
    assert should_index(IngestSummary(changed=97, skipped=0, failures=3, dry_run=False)) is True


def test_index_skipped_when_nothing_changed():
    assert should_index(IngestSummary(changed=0, skipped=12, failures=0, dry_run=False)) is False


def test_index_skipped_on_dry_run():
    # dry-run still increments `changed`, so dry_run must be checked explicitly.
    assert should_index(IngestSummary(changed=4, skipped=0, failures=0, dry_run=True)) is False


def test_index_skipped_when_summary_missing():
    assert should_index(None) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cortex_engine.vault_ingest'`

- [ ] **Step 3: Write the minimal implementation**

Create `cortex_engine/vault_ingest.py`:

```python
# ## File: cortex_engine/vault_ingest.py
# Version: v6.4.0
# Date: 2026-07-31
# Purpose: Two-phase private vault ingest -- textify documents, then index them.

from __future__ import annotations

import re
from dataclasses import dataclass

# Emitted as the final line of nemoclaw-private-knowledge-ingest.py.
SUMMARY_RE = re.compile(
    r"done changed=(\d+) skipped=(\d+) failures=(\d+) dry_run=(\w+)"
)


@dataclass(frozen=True)
class IngestSummary:
    changed: int
    skipped: int
    failures: int
    dry_run: bool


def parse_ingest_summary(output: str) -> IngestSummary | None:
    """Pull the ingest script's summary line out of its stdout, or None."""
    match = SUMMARY_RE.search(output or "")
    if not match:
        return None
    return IngestSummary(
        changed=int(match.group(1)),
        skipped=int(match.group(2)),
        failures=int(match.group(3)),
        dry_run=match.group(4).lower() == "true",
    )


def should_index(summary: IngestSummary | None) -> bool:
    """Decide whether phase 2 should run.

    A missing summary means the ingest crashed or was killed: state is unknown,
    so we do not index a partial branch as if it were complete. Partial failures
    (exit code 2) still index -- `changed` may be 97 of 100.
    """
    if summary is None:
        return False
    if summary.dry_run:
        return False
    return summary.changed > 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
cd ~/cortex_suite
git add cortex_engine/vault_ingest.py tests/unit/test_vault_ingest.py
git commit -m "feat: add ingest summary parsing and phase-chaining policy

Partial ingest failures (exit 2) still index -- halting there would strand
converted files unindexed. A missing summary line halts the chain.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Two-phase orchestration and CLI entry point

**Files:**
- Modify: `cortex_engine/vault_ingest.py`
- Test: `tests/unit/test_vault_ingest.py`

**Interfaces:**
- Consumes: `IngestSummary`, `parse_ingest_summary`, `should_index` from Task 1.
- Produces:
  - `build_ingest_command(source_root: Path, branch_name: str, dest_root: Path | None, *, pdf_strategy: str, use_vision: bool, limit: int, dry_run: bool, manifest_path: Path | None) -> list[str]`
  - `build_index_command() -> list[str]`
  - `run_ingest_then_index(source_root, branch_name, dest_root=None, *, pdf_strategy="hybrid", use_vision=False, limit=0, dry_run=False, manifest_path=None, runner=None) -> int`
  - Module constants `INGEST_SCRIPT`, `INDEXER_SCRIPT`, `CORTEX_PYTHON`, `VAULT_RAG_PYTHON`.

**Note on the injection point:** the spec described an `indexer` callable. A single `runner` hook is used instead — it covers both phases, lets tests assert *which* commands ran (so "was phase 2 invoked?" is still directly testable), and avoids two parallel injection mechanisms. Same intent, less machinery.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_vault_ingest.py`:

```python
import subprocess
from pathlib import Path

from cortex_engine.vault_ingest import (
    INDEXER_SCRIPT,
    INGEST_SCRIPT,
    build_index_command,
    build_ingest_command,
    run_ingest_then_index,
)


class FakeRunner:
    """Records commands and returns canned stdout per phase."""

    def __init__(self, ingest_stdout: str, ingest_rc: int = 0, index_rc: int = 0):
        self.ingest_stdout = ingest_stdout
        self.ingest_rc = ingest_rc
        self.index_rc = index_rc
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if str(INGEST_SCRIPT) in command:
            return subprocess.CompletedProcess(
                command, self.ingest_rc, self.ingest_stdout, ""
            )
        return subprocess.CompletedProcess(command, self.index_rc, "indexed", "")

    @property
    def ran_index(self) -> bool:
        return any(str(INDEXER_SCRIPT) in cmd for cmd in self.commands)


def test_ingest_command_includes_required_flags(tmp_path):
    command = build_ingest_command(
        tmp_path, "manuals-hifi", tmp_path / "dest",
        pdf_strategy="hybrid", use_vision=False, limit=0,
        dry_run=False, manifest_path=tmp_path / "m.json",
    )
    assert "--source-root" in command
    assert "manuals-hifi" in command
    assert str(tmp_path / "dest") in command
    assert str(tmp_path / "m.json") in command
    # Falsy options must not appear at all.
    assert "--dry-run" not in command
    assert "--use-vision" not in command
    assert "--limit" not in command


def test_ingest_command_includes_optional_flags_when_set(tmp_path):
    command = build_ingest_command(
        tmp_path, "b", None,
        pdf_strategy="docling", use_vision=True, limit=25,
        dry_run=True, manifest_path=None,
    )
    assert "--use-vision" in command
    assert "--dry-run" in command
    assert command[command.index("--limit") + 1] == "25"
    assert command[command.index("--pdf-strategy") + 1] == "docling"
    assert "--dest-root" not in command


def test_index_command_targets_private_vault_only():
    command = build_index_command()
    assert "--private-only" in command
    assert "--public-only" not in command


def test_phases_run_in_order_on_happy_path(tmp_path):
    runner = FakeRunner(DONE.format(c=5, s=0, f=0, d="False"))
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert rc == 0
    assert runner.ran_index is True
    assert str(INGEST_SCRIPT) in runner.commands[0]
    assert str(INDEXER_SCRIPT) in runner.commands[1]


def test_partial_failure_still_indexes_and_returns_nonzero(tmp_path):
    runner = FakeRunner(DONE.format(c=97, s=0, f=3, d="False"), ingest_rc=2)
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is True
    assert rc == 2


def test_zero_changed_skips_index(tmp_path):
    runner = FakeRunner(DONE.format(c=0, s=9, f=0, d="False"))
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is False
    assert rc == 0


def test_dry_run_skips_index(tmp_path):
    runner = FakeRunner(DONE.format(c=4, s=0, f=0, d="True"))
    run_ingest_then_index(tmp_path, "b", tmp_path / "d", dry_run=True, runner=runner)
    assert runner.ran_index is False


def test_crash_without_summary_halts_chain(tmp_path):
    runner = FakeRunner("segfault", ingest_rc=-9)
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is False
    assert rc != 0


def test_index_failure_surfaces_in_return_code(tmp_path):
    runner = FakeRunner(DONE.format(c=5, s=0, f=0, d="False"), index_rc=1)
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is True
    assert rc == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_ingest_command'`

- [ ] **Step 3: Write the minimal implementation**

Append to `cortex_engine/vault_ingest.py` (add `argparse`, `os`, `subprocess`, `sys`, `Path`, `Callable` to the imports at the top):

```python
HOME = Path.home()
INGEST_SCRIPT = HOME / "nemoclaw-private-knowledge-ingest.py"
INDEXER_SCRIPT = HOME / "nemoclaw-vault-indexer.py"
CORTEX_PYTHON = HOME / "cortex_suite" / "venv" / "bin" / "python"
VAULT_RAG_PYTHON = HOME / "venvs" / "vault-rag" / "bin" / "python3"


def build_ingest_command(
    source_root: Path,
    branch_name: str,
    dest_root: Path | None,
    *,
    pdf_strategy: str,
    use_vision: bool,
    limit: int,
    dry_run: bool,
    manifest_path: Path | None,
) -> list[str]:
    command = [
        str(CORTEX_PYTHON), "-u", str(INGEST_SCRIPT),
        "--source-root", str(source_root),
        "--branch-name", branch_name,
        "--pdf-strategy", pdf_strategy,
    ]
    if dest_root:
        command += ["--dest-root", str(dest_root)]
    if manifest_path:
        command += ["--manifest-path", str(manifest_path)]
    if limit:
        command += ["--limit", str(limit)]
    if use_vision:
        command.append("--use-vision")
    if dry_run:
        command.append("--dry-run")
    return command


def build_index_command() -> list[str]:
    # Phase 2 runs under the vault-rag interpreter: different chromadb pin.
    return [str(VAULT_RAG_PYTHON), "-u", str(INDEXER_SCRIPT), "--private-only"]


def _default_runner(command: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "HF_HOME": "/mnt/f/hf-home", "TOKENIZERS_PARALLELISM": "false"}
    return subprocess.run(command, capture_output=True, text=True, env=env)


def run_ingest_then_index(
    source_root: Path,
    branch_name: str,
    dest_root: Path | None = None,
    *,
    pdf_strategy: str = "hybrid",
    use_vision: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    manifest_path: Path | None = None,
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> int:
    """Textify a source branch into the private vault, then index it."""
    run = runner or _default_runner

    print(f"[vault-ingest] phase=textify branch={branch_name}", flush=True)
    ingest = run(build_ingest_command(
        source_root, branch_name, dest_root,
        pdf_strategy=pdf_strategy, use_vision=use_vision,
        limit=limit, dry_run=dry_run, manifest_path=manifest_path,
    ))
    if ingest.stdout:
        print(ingest.stdout, flush=True)
    if ingest.stderr:
        print(f"[stderr]\n{ingest.stderr}", flush=True)

    summary = parse_ingest_summary(ingest.stdout)
    if not should_index(summary):
        reason = "ingest produced no parseable summary" if summary is None else (
            "dry run" if summary.dry_run else "nothing changed"
        )
        print(f"[vault-ingest] phase=skip-index reason={reason}", flush=True)
        rc = ingest.returncode if summary is None else (2 if summary.failures else 0)
        print(f"[vault-ingest] phase=done rc={rc}", flush=True)
        return rc

    print("[vault-ingest] phase=index", flush=True)
    index = run(build_index_command())
    if index.stdout:
        print(index.stdout, flush=True)
    if index.stderr:
        print(f"[stderr]\n{index.stderr}", flush=True)

    # Index failure dominates: the branch is on disk but not searchable.
    rc = index.returncode or (2 if summary.failures else 0)
    print(f"[vault-ingest] phase=done rc={rc}", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Textify a branch into the private vault, then index it")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--dest-root", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--pdf-strategy", default="hybrid", choices=["hybrid", "docling", "pymupdf"])
    parser.add_argument("--use-vision", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_ingest_then_index(
        Path(args.source_root),
        args.branch_name,
        Path(args.dest_root) if args.dest_root else None,
        pdf_strategy=args.pdf_strategy,
        use_vision=args.use_vision,
        limit=args.limit,
        dry_run=args.dry_run,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
    )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest.py -v`
Expected: PASS, 18 passed

- [ ] **Step 5: Verify the CLI entry point wires up**

Run: `cd ~/cortex_suite && venv/bin/python -m cortex_engine.vault_ingest --help`
Expected: usage text listing `--source-root`, `--branch-name`, `--dest-root`, `--pdf-strategy`, `--use-vision`, `--limit`, `--dry-run`

- [ ] **Step 6: Commit**

```bash
cd ~/cortex_suite
git add cortex_engine/vault_ingest.py tests/unit/test_vault_ingest.py
git commit -m "feat: chain textify and index phases with a CLI entry point

Each phase runs under its own interpreter. Tests inject a fake runner so no
real process, vault write, or chromadb call happens.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Detached spawn, status, and cancel

**Files:**
- Modify: `cortex_engine/private_vault_rag.py` (append after `run_private_indexer`, currently ending line 516)
- Test: `tests/unit/test_vault_ingest_state.py`

**Interfaces:**
- Consumes: `cortex_engine.vault_ingest` (spawned as `-m`), existing `PRIVATE_VAULT` constant.
- Produces:
  - `VAULT_INGEST_STATE: Path`, `VAULT_INGEST_LOG_DIR: Path`
  - `start_vault_ingest(source_root: Path, branch_name: str, dest_root: Path | None = None, *, pdf_strategy: str = "hybrid", use_vision: bool = False, limit: int = 0, dry_run: bool = False, state_path: Path | None = None) -> dict` — returns `{"pid": int, "log_path": str, "branch": str, "started_at": str}`; raises `ValueError` on a bad source root or an already-running ingest.
  - `vault_ingest_status(state_path: Path | None = None, tail_lines: int = 40) -> dict` — returns `{"state": str, "phase": str, "branch": str, "started_at": str, "elapsed": float, "log_tail": str, "rc": int | None}` where `state` is one of `idle | running | completed | failed | interrupted`.
  - `cancel_vault_ingest(state_path: Path | None = None) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vault_ingest_state.py`:

```python
"""Status resolution for the detached private vault ingest."""

import json
import os

import pytest

from cortex_engine.private_vault_rag import (
    start_vault_ingest,
    vault_ingest_status,
)


def _write_state(tmp_path, pid, log_text="", branch="b"):
    log = tmp_path / "run.log"
    log.write_text(log_text, encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "pid": pid,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'start_vault_ingest'`

- [ ] **Step 3: Write the minimal implementation**

Append to `cortex_engine/private_vault_rag.py`. The file already imports
`datetime as dt`, `json`, `os`, `re`, `subprocess`, `sys`, `Path` and `Any` — the
**only** import to add is `signal` (used by `cancel_vault_ingest`), which goes
alphabetically between `re` and `sqlite3` at line 17:

```python
import signal
```

Then append the new code after `run_private_indexer` (currently ending line 516):

```python
VAULT_INGEST_STATE = HOME / ".nemoclaw" / "vault-ingest-ui.json"
VAULT_INGEST_LOG_DIR = HOME / ".nemoclaw" / "logs" / "private-knowledge-imports"

_PHASE_RE = re.compile(r"\[vault-ingest\] phase=(\S+)")
_DONE_RE = re.compile(r"\[vault-ingest\] phase=done rc=(-?\d+)")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return True


def start_vault_ingest(
    source_root: Path,
    branch_name: str,
    dest_root: Path | None = None,
    *,
    pdf_strategy: str = "hybrid",
    use_vision: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Spawn the two-phase ingest detached; return its pid and log path."""
    state_path = state_path or VAULT_INGEST_STATE
    source_root = Path(source_root)
    if not source_root.is_dir():
        raise ValueError(f"Source root is not a directory: {source_root}")

    current = vault_ingest_status(state_path)
    if current["state"] == "running":
        raise ValueError(f"An ingest is already running (branch {current['branch']})")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = VAULT_INGEST_LOG_DIR / f"{stamp}-{branch_name}-ui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-u", "-m", "cortex_engine.vault_ingest",
        "--source-root", str(source_root),
        "--branch-name", branch_name,
        "--pdf-strategy", pdf_strategy,
    ]
    if dest_root:
        command += ["--dest-root", str(dest_root)]
    if limit:
        command += ["--limit", str(limit)]
    if use_vision:
        command.append("--use-vision")
    if dry_run:
        command.append("--dry-run")

    env = {**os.environ, "HF_HOME": "/mnt/f/hf-home", "TOKENIZERS_PARALLELISM": "false"}
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            start_new_session=True,
        )

    payload = {
        "pid": proc.pid,
        "log_path": str(log_path),
        "branch": branch_name,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def vault_ingest_status(state_path: Path | None = None, tail_lines: int = 40) -> dict[str, Any]:
    """Resolve the state of the most recent ingest run."""
    state_path = Path(state_path) if state_path else VAULT_INGEST_STATE
    idle = {"state": "idle", "phase": "", "branch": "", "started_at": "",
            "elapsed": 0.0, "log_tail": "", "rc": None}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return idle

    log_text = ""
    try:
        log_text = Path(payload.get("log_path", "")).read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass

    done = _DONE_RE.search(log_text)
    phases = _PHASE_RE.findall(log_text)
    phase = phases[-1] if phases else ""

    if done:
        rc = int(done.group(1))
        state = "completed" if rc == 0 else "failed"
    elif _pid_alive(int(payload.get("pid", 0))):
        state = "running"
        rc = None
    else:
        state = "interrupted"
        rc = None

    elapsed = 0.0
    try:
        started = dt.datetime.fromisoformat(payload.get("started_at", ""))
        elapsed = (dt.datetime.now() - started).total_seconds()
    except Exception:
        pass

    return {
        "state": state,
        "phase": phase,
        "branch": payload.get("branch", ""),
        "started_at": payload.get("started_at", ""),
        "elapsed": elapsed,
        "log_tail": "\n".join(log_text.splitlines()[-tail_lines:]),
        "rc": rc,
    }


def cancel_vault_ingest(state_path: Path | None = None) -> bool:
    """SIGTERM the detached ingest's process group. Safe: the manifest commits per file."""
    state_path = Path(state_path) if state_path else VAULT_INGEST_STATE
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except Exception:
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        return False
    return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest_state.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Run the whole new suite together**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest.py tests/unit/test_vault_ingest_state.py -v`
Expected: PASS, 26 passed

- [ ] **Step 6: Commit**

```bash
cd ~/cortex_suite
git add cortex_engine/private_vault_rag.py tests/unit/test_vault_ingest_state.py
git commit -m "feat: detached spawn, status resolution and cancel for vault ingest

State lives in ~/.nemoclaw/vault-ingest-ui.json; logs reuse the queue runner's
log dir with a -ui suffix. A dead pid with no done marker reports interrupted,
which is what a WSL restart mid-run looks like.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Ingest panel on the Vault GraphRAG page

**Files:**
- Modify: `pages/18_Private_Vault_GraphRAG.py` (add `_ingest_panel()`; call it from `main()` at line 224-229)

**Interfaces:**
- Consumes: `start_vault_ingest`, `vault_ingest_status`, `cancel_vault_ingest`, `PRIVATE_VAULT` from `cortex_engine.private_vault_rag`; `convert_windows_to_wsl_path` from `cortex_engine.utils`.
- Produces: nothing consumed by later tasks.

This task is UI wiring with no unit test — it is verified by running the app. Streamlit page functions are not unit-testable without a harness this repo does not have, and the logic they call is already covered by Tasks 1–3.

- [ ] **Step 1: Extend the imports**

`PRIVATE_VAULT` is already imported. Add the three new names to the
`from cortex_engine.private_vault_rag import (...)` block at lines 10-20, and add
`pathlib.Path` plus the path-conversion helper (the page currently imports neither):

```python
from cortex_engine.private_vault_rag import (
    PRIVATE_VAULT,
    PUBLIC_VAULT,
    build_vault_graph,
    cancel_vault_ingest,
    load_vault_note,
    markdown_for_streamlit,
    run_vault_indexer,
    search_vault,
    start_vault_ingest,
    vault_graph_stats,
    vault_index_stats,
    vault_ingest_status,
)
from cortex_engine.utils import convert_windows_to_wsl_path
```

and alongside the existing `import subprocess` at line 6:

```python
from pathlib import Path
```

- [ ] **Step 2: Add the panel function**

Insert `_ingest_panel()` immediately before `def _maintenance_panel():` (line 57):

```python
def _ingest_panel():
    status = vault_ingest_status()
    running = status["state"] == "running"

    with st.expander("Ingest documents", expanded=running):
        st.caption(
            "Convert PDF, DOCX, PPTX and TXT files into private vault markdown, "
            "then index them. Runs in the background -- you can close this tab."
        )

        source_raw = st.text_input(
            "Source folder",
            placeholder=r"C:\Users\paul\Documents\Manuals  or  /mnt/c/...",
            disabled=running,
        )
        source_path = convert_windows_to_wsl_path(source_raw.strip()) if source_raw.strip() else ""

        default_branch = Path(source_path).name.lower().replace(" ", "-") if source_path else ""
        col1, col2 = st.columns(2)
        branch = col1.text_input("Branch name", value=default_branch, disabled=running)
        dest = col2.text_input(
            "Destination (relative to vault)",
            value=f"30 Resources/Imported Knowledge/{branch}" if branch else "",
            disabled=running,
        )

        col3, col4, col5, col6 = st.columns(4)
        strategy = col3.selectbox("PDF strategy", ["hybrid", "docling", "pymupdf"], disabled=running)
        limit = col4.number_input("File limit (0 = all)", min_value=0, value=0, step=10, disabled=running)
        use_vision = col5.checkbox("Describe images", value=False, disabled=running)
        dry_run = col6.checkbox("Dry run", value=False, disabled=running)

        if source_path and not Path(source_path).is_dir():
            st.error(f"Not a directory: `{source_path}`")

        if st.button("Ingest", type="primary", disabled=running or not (source_path and branch)):
            try:
                started = start_vault_ingest(
                    Path(source_path), branch.strip(),
                    PRIVATE_VAULT / dest.strip() if dest.strip() else None,
                    pdf_strategy=strategy, use_vision=use_vision,
                    limit=int(limit), dry_run=dry_run,
                )
                st.success(f"Started (pid {started['pid']}). Log: `{started['log_path']}`")
            except ValueError as exc:
                st.error(str(exc))
            st.rerun()

        if status["state"] != "idle":
            _render_ingest_status(status)


def _render_ingest_status(status):
    label = {
        "running": "Running",
        "completed": "Completed",
        "failed": "Finished with errors",
        "interrupted": "Interrupted (process died -- rerun to resume)",
    }.get(status["state"], status["state"])

    col1, col2, col3 = st.columns([2, 2, 1])
    col1.metric("Status", label)
    col2.metric("Phase", status["phase"] or "-")
    col3.metric("Elapsed", f"{status['elapsed'] / 60:.1f} min")

    if status["state"] == "running":
        if st.button("Cancel", key="vault_ingest_cancel"):
            if cancel_vault_ingest():
                st.warning("Cancel signal sent. Converted files are kept -- rerun to resume.")
            else:
                st.error("Could not signal the process.")
            st.rerun()
        st.caption("Refresh the page to update progress.")

    if status["log_tail"]:
        st.code(status["log_tail"])
```

- [ ] **Step 3: Call the panel from `main()`**

Change `main()` (lines 224-229) so the panel renders between the status cards and maintenance:

```python
def main():
    st.title("Vault GraphRAG")
    st.caption(f"Local public/private vault search via NemoClaw RAG and Cortex graph helpers. Cortex {VERSION_STRING}.")
    _status_cards()
    _ingest_panel()
    _maintenance_panel()
    _search_panel()
```

- [ ] **Step 4: Verify the page imports cleanly**

Run: `cd ~/cortex_suite && venv/bin/python -c "import ast,sys; ast.parse(open('pages/18_Private_Vault_GraphRAG.py').read()); print('syntax ok')"`
Expected: `syntax ok`

Run: `cd ~/cortex_suite && venv/bin/python -c "from cortex_engine.private_vault_rag import start_vault_ingest, vault_ingest_status, cancel_vault_ingest; from cortex_engine.utils import convert_windows_to_wsl_path; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 5: Verify in the running app**

Run: `cd ~/cortex_suite && venv/bin/streamlit run Cortex_Suite.py`
Then open the Vault GraphRAG page and confirm: the "Ingest documents" expander renders; a nonexistent source folder shows the "Not a directory" error and leaves Ingest disabled; a real folder with **Dry run** ticked starts a run, the status shows `phase=textify` then `phase=skip-index reason=dry run`, and the log tail is populated.

- [ ] **Step 6: Commit**

```bash
cd ~/cortex_suite
git add pages/18_Private_Vault_GraphRAG.py
git commit -m "feat: add document ingest panel to Vault GraphRAG page

Folder-path input with Windows path conversion, background run with live log
tail, elapsed timer and cancel.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Version bump, changelog, and Docker sync

**Files:**
- Modify: `cortex_engine/version_config.py`
- Modify: `pages/18_Private_Vault_GraphRAG.py` (header comment lines 2-3)
- Modify: `CHANGELOG.md` (via the version manager)
- Modify: `docker/cortex_engine/`, `docker/pages/` (copies)

- [ ] **Step 1: Bump the version**

In `cortex_engine/version_config.py` set `CORTEX_VERSION = "6.4.0"` (feature increment from 6.3.3) and update `VERSION_METADATA`:

```python
VERSION_METADATA = {
    "release_date": "2026-07-31",
    "release_name": "Private Vault Ingest UI",
    "description": "Folder-path document ingest into the private vault from the Vault GraphRAG page",
    "new_features": [
        "Ingest documents panel on the Vault GraphRAG page: folder path in, vault markdown out",
        "Textify and index chained in one background run that survives closing the browser tab",
        "Live log tail, elapsed timer and cancel for in-flight ingests",
    ],
    "improvements": [
        "Partial ingest failures now still index the files that converted successfully",
    ],
    "bug_fixes": [],
}
```

- [ ] **Step 2: Update the page header**

In `pages/18_Private_Vault_GraphRAG.py` change lines 2-3 to:

```python
# Version: v6.4.0
# Date: 2026-07-31
```

- [ ] **Step 3: Run the version sync**

```bash
cd ~/cortex_suite
venv/bin/python scripts/version_manager.py --sync-all
venv/bin/python scripts/version_manager.py --update-changelog
venv/bin/python scripts/version_manager.py --check
```
Expected: `--check` reports version consistency across components with no mismatches.

- [ ] **Step 4: Sync the Docker distribution**

```bash
cd ~/cortex_suite
cp cortex_engine/version_config.py cortex_engine/vault_ingest.py cortex_engine/private_vault_rag.py docker/cortex_engine/
cp pages/18_Private_Vault_GraphRAG.py docker/pages/
```

- [ ] **Step 5: Run the full test suite**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit -q`
Expected: all tests pass, including the 26 new ones. If pre-existing failures appear that are unrelated to this work, report them rather than fixing them here.

- [ ] **Step 6: Commit**

```bash
cd ~/cortex_suite
git add -A
git commit -m "release: Version 6.4.0 - Private Vault Ingest UI

Adds a folder-path document ingest to the Vault GraphRAG page that textifies
into the private vault and chains the existing indexer, running detached.

- Version consistency verified
- Docker distribution updated

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Do not modify** `~/nemoclaw-private-knowledge-ingest.py` or `~/nemoclaw-vault-indexer.py`. The cron queue runner depends on their current behaviour.
- The `~/venvs/vault-rag/bin/python3` symlink points at `/usr/bin/python3`, not a real venv interpreter — that is expected and is how `run_vault_indexer` already invokes it.
- `dry_run` still increments `changed` in the ingest script, which is exactly why `should_index` checks `dry_run` before `changed`.
- If `pytest` is unavailable in the venv, install it there (`venv/bin/pip install pytest`) rather than falling back to the system Python — the tests import `cortex_engine`.
