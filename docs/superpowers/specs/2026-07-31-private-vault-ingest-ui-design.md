# Private Vault Ingest UI — Design

**Date:** 2026-07-31
**Status:** Approved (pending spec review)

## Purpose

Give the Cortex Suite a UI for getting PDFs and other documents into the private
vault. Today the Streamlit app owns the *second* half of that job only:
`pages/18_Private_Vault_GraphRAG.py` can run the vault indexer and rebuild the
graph, but nothing in the app converts source documents into vault markdown. That
step exists solely as a headless script (`~/nemoclaw-private-knowledge-ingest.py`)
driven either by hand or by the `*/15 0-5,14-23` queue-runner cron.

The cron path is not expedient for ad-hoc work: it runs one branch per firing, is
dormant between 06:00 and 14:00 (power window), and auto-pauses any branch over
`max_files` (default 10) instead of processing it. This design adds the missing
textify step to the existing page and chains the already-built indexer onto it, so
a folder of manuals becomes searchable vault content in one action.

Scope is the **private** vault (`Vault_OneDrive`). The public AI-Vault wiki
pipeline is untouched.

## Approach

**One detached wrapper process, spawned once, that runs both phases in sequence.**

The two phases run under *different Python interpreters*:

| Phase | Script | Interpreter | Why |
|---|---|---|---|
| 1. Textify | `~/nemoclaw-private-knowledge-ingest.py` | `cortex_suite/venv` | needs `cortex_engine.textifier` (docling/OCR) |
| 2. Index | `~/nemoclaw-vault-indexer.py` | `venvs/vault-rag/bin/python3` | different chromadb pin for `vault-rag-db` |

`run_vault_indexer()` in `private_vault_rag.py` already handles that interpreter
split for phase 2. Rather than have Streamlit orchestrate two subprocesses and
hold the chain state in `session_state`, a new module owns the sequence and is
spawned once, detached, with stdout redirected to a log file.

This buys three things: the run survives closing the browser tab; the chaining
logic is a plain function that unit-tests without Streamlit; and the same entry
point works headless from the CLI.

Rejected alternatives:

- **Popen + reader-thread + `session_state` queue** — the pattern in
  `pages/2_Knowledge_Ingest.py`. The reader thread dies with the browser session,
  so a refresh loses all output. Unacceptable for runs measured in hours.
- **Blocking `st.status`** — matches the existing "Run indexer" button, but holds
  the tab hostage for the duration.
- **Write into `private-knowledge-queue.json`** — zero new execution code and a
  proven path, but inherits the power window and the `max_files` auto-pause.

## Components

### New module: `cortex_engine/vault_ingest.py`

```python
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
    indexer: Callable[[], subprocess.CompletedProcess] | None = None,
) -> int
```

Runs phase 1, decides whether phase 2 should run, runs it, returns an exit code.
Prints `[vault-ingest] phase=textify` / `phase=index` / `phase=done` markers so the
UI can report progress from the log alone.

`manifest_path` and `dest_root` are passthroughs to the ingest script's existing
`--manifest-path` / `--dest-root` flags; tests set both to temp paths so no test
ever writes to the real vault or manifest. `indexer` is an injection point so
tests can assert phase 2 was or wasn't invoked without touching `vault-rag-db`.

An argparse `__main__` exposes the same options for CLI use.

#### Phase-chaining policy

The ingest script prints a summary line and uses a meaningful exit code:

```
[private-ingest] done changed=N skipped=N failures=N dry_run=X
```

It returns `0` when `failures == 0` and `2` when *any* file failed — note `2` is a
**partial** failure: `changed` may still be 97 of 100. The chain therefore keys off
the parsed summary, not the exit code alone:

| Condition | Phase 2 | Rationale |
|---|---|---|
| summary parsed, `dry_run=True` | skip | nothing was written |
| summary parsed, `changed == 0` | skip | nothing new to index |
| summary parsed, `changed > 0`, `failures == 0` | **run** | happy path |
| summary parsed, `changed > 0`, `failures > 0` | **run**, surface failure count | halting would strand converted files unindexed |
| no summary line (crash/kill/timeout) | halt | state unknown; do not index a partial branch as if complete |

### Changes to `cortex_engine/private_vault_rag.py`

Three functions, alongside the existing vault helpers:

- `start_vault_ingest(...) -> dict` — validates the source root, spawns
  `sys.executable -m cortex_engine.vault_ingest` with `start_new_session=True` and
  stdout/stderr to a log file, writes run state, returns `{pid, log_path, branch}`.
  `sys.executable` is the cortex venv interpreter, since Streamlit itself runs
  under it — the same interpreter phase 1 requires. Phase 2's separate vault-rag
  interpreter is resolved inside the wrapper, not here.
- `vault_ingest_status() -> dict` — reads run state, resolves liveness, returns
  `{state, phase, branch, started_at, elapsed, log_tail}`.
- `cancel_vault_ingest() -> bool` — `SIGTERM` to the process group.

Run state lives in `~/.nemoclaw/vault-ingest-ui.json`; logs go to the existing
`~/.nemoclaw/logs/private-knowledge-imports/` directory used by the queue runner,
named `YYYYmmdd-HHMMSS-<branch>-ui.log` (the `-ui` suffix keeps UI runs visually
distinct from cron runs in a directory listing).

`state` is one of `idle | running | completed | failed | interrupted`.
`interrupted` is the dead-PID-without-completion-marker case — the same condition
the queue runner handles in `retry_interrupted_running_entries`. It matters here
because WSL restarts (one happened at 09:00 on 2026-07-31) kill in-flight runs.

### Changes to `pages/18_Private_Vault_GraphRAG.py`

A new `_ingest_panel()`, rendered above the existing Maintenance expander:

- **Source folder** — text input, passed through `convert_windows_to_wsl_path()`
  so a pasted Windows path works. Validated before spawn.
- **Branch name** — text input, defaulted from the source folder's basename.
- **Destination root** — text input pre-filled
  `30 Resources/Imported Knowledge/<branch>`, editable per run.
- Secondary controls: PDF strategy (`hybrid`/`docling`/`pymupdf`), vision
  descriptions checkbox, file limit, dry-run checkbox.
- **Ingest** button, disabled while a run is active.
- While running: phase, elapsed time, tailed log, and a **Cancel** button.

The panel polls `vault_ingest_status()` on rerun rather than holding a thread.

## Data flow

```
folder path (Windows or WSL)
   │  convert_windows_to_wsl_path() + exists check
   ▼
start_vault_ingest()  ──► ~/.nemoclaw/vault-ingest-ui.json  (pid, log, branch)
   │  spawn detached
   ▼
python -m cortex_engine.vault_ingest        stdout ──► …-ui.log
   ├─ phase 1: nemoclaw-private-knowledge-ingest.py   (cortex venv)
   │     └─ writes .md into Vault_OneDrive, updates private-knowledge-manifest.json
   └─ phase 2: nemoclaw-vault-indexer.py --private-only   (vault-rag venv)
         └─ updates vault-rag-db private collection
   ▼
page tails log ──► status cards show new chunk count
```

## Error handling

- **Source root missing or not a directory** — inline error, no process spawned.
- **No supported files found** — phase 1 reports `changed=0`; phase 2 skipped with
  "nothing to index" rather than starting chromadb for no reason.
- **Partial / total ingest failure** — per the chaining policy table above.
- **Stale PID** — dead process with no completion marker reports `interrupted` and
  offers a rerun. Rerun is cheap: the manifest (993 entries, keyed on source SHA)
  skips unchanged files.
- **Pathological document** — a single file once hung docling for 7.5 hours and
  blocked the kdocs ingest, which is why that script carries a 1200s per-batch
  timeout. A whole-folder call has no natural batch boundary, so rather than a
  run-wide timeout that would abort a legitimately slow 400-page manual, the UI
  surfaces elapsed time and offers Cancel. Cancel is safe because the manifest
  commits per file — completed work is kept and a rerun resumes.
- **Concurrent runs** — `start_vault_ingest()` refuses if state is `running` with a
  live PID. Two ingests writing the same manifest would race.

## Testing

Unit tests in `tests/unit/test_vault_ingest.py` (matching where the other engine
unit tests live), all against temp directories with
`--manifest-path` and `--dest-root` overridden, and the indexer injected as a
mock — no test touches the real vault or `vault-rag-db`:

- happy path: source dir with one small document → markdown written, indexer called
- `changed == 0` → indexer **not** called
- `dry_run=True` → indexer **not** called
- partial failure (`changed>0, failures>0`) → indexer **is** called
- crash / unparseable output → indexer **not** called
- status resolution: live PID → `running`; dead PID without marker → `interrupted`;
  dead PID with marker → `completed`

Manual verification: dry-run against a real manuals folder, then a live run,
confirming the private chunk count on the status cards increases.

## Version management

Per the project workflow this is a feature increment (6.3.3 → 6.4.0):

1. Update `CORTEX_VERSION` and `VERSION_METADATA` in `cortex_engine/version_config.py`
2. `python scripts/version_manager.py --sync-all`
3. `python scripts/version_manager.py --update-changelog`
4. `python scripts/version_manager.py --check`
5. Update the `Version:`/`Date:` header in `pages/18_Private_Vault_GraphRAG.py`
6. Sync the Docker distribution (`docker/cortex_engine/`, `docker/pages/`)

## Out of scope

- File upload — folder path only.
- The public AI-Vault / wiki publish pipeline.
- Replacing or modifying the queue-runner cron; it keeps working unchanged.
- Spreadsheet formats — the ingest script's `SUPPORTED_EXTS` is
  `.pdf/.docx/.pptx/.txt` and stays as-is.
