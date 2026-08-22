# P: Drive Photo Organise + Dedupe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **READ THE "Agent Working Agreement" SECTION BELOW BEFORE DOING ANYTHING.** It is mandatory, not advisory. Read the progress log first, append to it after every step, and stop-and-report rather than improvising. Assume you may be terminated mid-task without warning.

**Goal:** Build a resumable, reviewable, non-destructive pipeline that deduplicates ~764k photo files on the `P:` drive and files the survivors into `YYYY/YYYY-MM/` folders named from EXIF capture time.

**Architecture:** Index-first. One read-only pass builds a SQLite index on `C:`; every later decision is a query against that index producing a reviewable CSV plan; a separate executor applies an approved plan and journals every operation. Dedupe runs before organise. Nothing is ever deleted — only moved.

**Tech Stack:** Python 3.11 stdlib only (`sqlite3`, `hashlib`, `os.scandir`, `csv`, `subprocess`), plus `exiftool` 12.85 as a subprocess. `PIL` + `imagehash` are required **only** for tiers 2–3. Executes under Windows `C:\Python311\python.exe`; tests run under the WSL cortex venv.

**Spec:** `docs/superpowers/specs/2026-08-21-p-drive-photo-organise-dedupe-design.md`

## Global Constraints

- **Every filesystem path goes through `win_long()`** which applies the `\\?\` prefix on Windows. A missing prefix silently skips deep files — the primary data-loss risk in this project.
- **Halt, never skip.** An `OSError` reading a scope root aborts the stage with a non-zero exit. A stage must never record an unreadable folder as empty.
- **Dry-run by default.** Every command that writes requires an explicit `--apply`.
- **Nothing is ever deleted.** Files are moved. The only irreversible act is the user emptying `_DUPES` by hand.
- **Stdlib only; no `cortex_engine` imports.** The tool runs under Windows Python 3.11, which does not have the cortex venv.
- **RAW and JPG are never paired as duplicates**, at any tier.
- **Sidecars (`.xmp`, `.aae`) never move independently** of their parent image.
- **`*_original` files are skipped at walk time**, unconditionally.
- **EXIF only for dates.** `mtime` is never used as a capture date.
- Naming: `YYYY-MM-DD HH-MM-SS-<CameraModel>.<ext>`, no star-rating suffix. No model → drop the trailing hyphen.
- Target layout: `P:\<YYYY>\<YYYY-MM>\`. Undated → `P:\_UNDATED\<top_folder>\<rel_dir>\`.

---

## Agent Working Agreement — MANDATORY

**This section binds every agent that touches this plan.** A run can die at any
moment: a crash, a dropped drive, or a context/token budget being exhausted
mid-task. Work must always be resumable by a *different* agent starting cold,
with no memory of what came before.

### The progress log is the source of truth

Append-only, one line per completed step, at `docs/superpowers/plans/2026-08-21-p-drive-photo-organise-dedupe.progress.log`:

```
2026-08-21T14:03:11Z  T3  step2  DONE   pytest failed as expected (ModuleNotFoundError)
2026-08-21T14:05:02Z  T3  step3  DONE   wrote scripts/photo_archive/walk.py
2026-08-21T14:06:40Z  T3  step4  DONE   7 passed
2026-08-21T14:07:12Z  T3  step5  DONE   commit 4a1c9de
2026-08-21T14:07:12Z  T3  TASK   DONE
```

**The timestamp MUST come from the clock, never from your head.** Get it with
`date -u +%Y-%m-%dT%H:%M:%SZ` and append using a shell redirect, e.g.:

```bash
L=docs/superpowers/plans/2026-08-21-p-drive-photo-organise-dedupe.progress.log
printf '%s  T3  step4  DONE  7 passed in 0.11s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $L
```

Task 1's agent invented tidy 30-second-apart timestamps that were 12 hours off
the real clock. Guessed times make the log non-monotonic once a later agent
appends real ones, which destroys exactly the crash-forensics and
resume-ordering the log exists for. Never type a timestamp by hand.

Rules, without exception:

1. **Read the log FIRST, before doing anything else.** The last `DONE` line
   tells you where the previous agent stopped. Resume from the next step. Never
   restart a task from the beginning because it looks unfinished.
2. **Append after every single step**, immediately — not batched at the end of a
   task. State must never be more than one step stale, because you do not get a
   warning before you run out of context.
3. **Never rewrite or delete existing log lines.** Append only. If you discover
   an earlier line was wrong, append a `CORRECTION` line rather than editing it.
4. **Log failures too**, with the actual error: `T5 step4 FAIL 2 failed — <first assertion>`.
   A missing line is indistinguishable from a crash; an explicit `FAIL` is information.
5. **If you stop for any reason** — blocked, out of scope, uncertain — append
   `T<n> BLOCKED <one-line reason>` before you stop. Never stop silently.

### Commits are the durable checkpoint

Each task ends in a commit. `git log --oneline` is the second source of truth
and must agree with the progress log. On resume, cross-check both: if the log
says a task is done but there is no commit, the commit step did not complete —
verify the working tree, then commit.

Commit only files belonging to your task. **Never `git add -A` or `git add .`** —
this working tree routinely carries unrelated user files, per the repo's
CLAUDE.md.

### Verification is never assumed

Never write `DONE` for a test step without having actually run the command and
read its output. Paste the real result into the log line. If you did not run it,
it is not done — evidence before assertions, always.

### Scope discipline

If the code does not match what the plan describes, a command fails after one
reasonable retry, or the task needs files outside its **Files:** list — **stop
and report**. Do not improvise a fix, and do not widen scope. Append a
`BLOCKED` line and hand back.

### Model assignment

Use the least capable model that can do the task reliably. These tasks carry
complete literal code and tests, so most are transcribe-verify-commit work.

| Task | Model | Why |
|---|---|---|
| 1 — paths + config | `haiku` | Pure constants and small pure functions |
| 2 — db schema | `haiku` | Literal SQL and thin wrappers |
| 3 — walk | `sonnet` | Traversal, sidecar linking, halt-on-error semantics |
| 4 — exif | `sonnet` | `-stay_open` subprocess protocol is easy to get subtly wrong |
| 5 — hashing | `sonnet` | Two-phase logic with a real correctness consequence |
| 6 — dupes + keeper | `sonnet` | Ordering semantics; a wrong keeper loses the better copy |
| 7 — journal + undo | `sonnet` | Reverse replay and hash refusal are the safety net |
| 8 — quarantine | `sonnet` | Moves real files; collision handling must be exact |
| 9 — organise | `sonnet` | Naming rules and sidecar-follows-parent coupling |
| 10 — CLI + README | `haiku` | Mechanical wiring of existing functions |

Review, integration, and the decision to run anything against `P:` stay with
the orchestrator. Subagent reports are **leads, not facts**: reopen the cited
files and confirm the test output before accepting a task as complete.

---

## File Structure

All new code lives in `cortex_suite/scripts/photo_archive/`:

| File | Responsibility |
|---|---|
| `paths.py` | `\\?\` long-path handling; cross-platform no-op so tests run on Linux |
| `config.py` | Scope roots, exclusions, meaningful-folder list, extension→kind mapping |
| `db.py` | SQLite schema, connection, row upsert, state transitions |
| `walk.py` | Stage 1 — walk and stat |
| `exif.py` | Stage 2 — `exiftool -stay_open` batch reader, date precedence |
| `hashing.py` | Stage 3 — partial and full hashing |
| `dupes.py` | Stage 4 — tier grouping, keeper rule, CSV plan output |
| `journal.py` | Append-only CSV journal and reverse-replay undo |
| `execute.py` | Stages 5–6 — quarantine move, hash-verified evacuation |
| `organise.py` | Stages 7–8 — provenance stamp, rename and move |
| `cli.py` | Subcommand dispatch |

Tests live in `cortex_suite/tests/unit/test_photo_archive_*.py`, run with the WSL cortex venv:
`venv/bin/python -m pytest tests/unit/test_photo_archive_*.py -v`

---

### Task 1: Long-path handling and scope configuration

**Files:**
- Create: `scripts/photo_archive/__init__.py` (empty)
- Create: `scripts/photo_archive/paths.py`
- Create: `scripts/photo_archive/config.py`
- Test: `tests/unit/test_photo_archive_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `win_long(path: str) -> str`, `strip_long(path: str) -> str`, `SCOPE_ROOTS: list[str]`, `HARD_EXCLUDE: set[str]`, `MEANINGFUL_FOLDERS: set[str]`, `kind_for(ext: str) -> str`, `is_skipped_file(name: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_config.py
import os
import pytest
from scripts.photo_archive.paths import win_long, strip_long
from scripts.photo_archive import config


def test_win_long_is_identity_off_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert win_long("/mnt/p/foo") == "/mnt/p/foo"


def test_win_long_prefixes_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os.path, "abspath", lambda p: p)
    assert win_long(r"P:\foo") == "\\\\?\\P:\\foo"


def test_win_long_is_idempotent(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os.path, "abspath", lambda p: p)
    once = win_long(r"P:\foo")
    assert win_long(once) == once


def test_strip_long_removes_prefix():
    assert strip_long("\\\\?\\P:\\foo") == "P:\\foo"
    assert strip_long("P:\\foo") == "P:\\foo"


def test_original_files_are_skipped():
    assert config.is_skipped_file("x.tif_original") is True
    assert config.is_skipped_file("x.dng_original") is True
    assert config.is_skipped_file("x.tif") is False


def test_old_files_are_not_skipped():
    # .old goes through the normal hash pipeline; no special case.
    assert config.is_skipped_file("x.jpg.old") is False


def test_kind_classification():
    assert config.kind_for(".jpg") == "image"
    assert config.kind_for(".raf") == "raw"
    assert config.kind_for(".dng") == "raw"
    assert config.kind_for(".mov") == "video"
    assert config.kind_for(".xmp") == "sidecar"
    assert config.kind_for(".txt") == "other"


def test_scope_and_exclusions_are_disjoint():
    assert len(config.SCOPE_ROOTS) == 12
    assert not (set(config.SCOPE_ROOTS) & config.HARD_EXCLUDE)


def test_lr_catalog_folders_are_hard_excluded():
    assert "New LR Catalog" in config.HARD_EXCLUDE
    assert "LR Backups" in config.HARD_EXCLUDE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.photo_archive'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/paths.py
"""Windows long-path handling.

Paths over 260 characters need the \\?\ prefix even when LongPathsEnabled=1.
A PowerShell pass without it undercounted one P: folder by 47 GB, so every
filesystem call in this package goes through win_long().

On non-Windows these are identity functions, which is what lets the test
suite run under WSL.
"""
import os

LONG_PREFIX = "\\\\?\\"


def win_long(path: str) -> str:
    if os.name != "nt" or path.startswith(LONG_PREFIX):
        return path
    return LONG_PREFIX + os.path.abspath(path)


def strip_long(path: str) -> str:
    return path[len(LONG_PREFIX):] if path.startswith(LONG_PREFIX) else path
```

```python
# scripts/photo_archive/config.py
"""Scope, exclusions and file classification for the P: drive pipeline."""

DRIVE = "P:"

SCOPE_ROOTS = [
    "0 and 1 star photos originals and dupes",
    "Backup Consolidated Photos",
    "3+ Star_RAW_Backups",
    "Catalogued Photos_Predig and Preraw backups_19 Aug 2026",
    "family_Randoms",
    "2024-RAW",
    "Fully Tagged Photos",
    "Google Drive Photos",
    "SMS_Photos",
    "Carey 50 yr reunion",
    "Cremorne",
    "250922 Bahnreise Schweiz (Glacier+Zermatt)",
]

# Touching these breaks a live Lightroom catalog or is not photo data.
HARD_EXCLUDE = {
    "New LR Catalog",
    "LR Backups",
    "$RECYCLE.BIN",
    "System Volume Information",
}

# Directory names skipped anywhere in the tree, at any depth.
EXCLUDE_DIR_NAMES = {
    "Music DRM Quarantine 2026-08-12",
    "_DUPES",
    "_UNDATED",
}

# Folder names whose name carries meaning worth preserving as a keyword.
# Explicit list, never a heuristic. Stage 7 prints in-scope folders absent
# from this set so it can be extended before running with --apply.
MEANINGFUL_FOLDERS = {
    "Carey 50 yr reunion",
    "Cremorne",
    "250922 Bahnreise Schweiz (Glacier+Zermatt)",
    "SMS_Photos",
    "Google Drive Photos",
}

RAW_EXT = {".raf", ".nef", ".cr2", ".dng", ".arw", ".orf", ".rw2"}
STILL_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".bmp", ".gif"}
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mpg", ".mpeg", ".m4v"}
SIDECAR_EXT = {".xmp", ".aae"}

# Tiers 2-3 decode pixels; RAW is excluded because decoding 70,671 RAF files
# would cost more than the rest of the pipeline combined.
DECODABLE_EXT = STILL_EXT


def kind_for(ext: str) -> str:
    ext = ext.lower()
    if ext in RAW_EXT:
        return "raw"
    if ext in STILL_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in SIDECAR_EXT:
        return "sidecar"
    return "other"


def is_skipped_file(name: str) -> bool:
    """The user relocates *_original files separately; never touch them."""
    return name.endswith("_original")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_config.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/__init__.py scripts/photo_archive/paths.py \
        scripts/photo_archive/config.py tests/unit/test_photo_archive_config.py
git commit -m "feat(photo_archive): long-path handling and scope config"
```

---

### Task 2: SQLite index schema

**Files:**
- Create: `scripts/photo_archive/db.py`
- Test: `tests/unit/test_photo_archive_db.py`

**Interfaces:**
- Consumes: nothing
- Produces: `connect(db_path: str) -> sqlite3.Connection`, `init_schema(conn) -> None`, `upsert_file(conn, **fields) -> int`, `set_state(conn, file_id: int, state: str) -> None`, `STATES: frozenset`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_db.py
import sqlite3
import pytest
from scripts.photo_archive import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _row(**over):
    base = dict(path=r"P:\a\b.jpg", top_folder="a", rel_dir="", filename="b.jpg",
                ext=".jpg", size=100, mtime=1.0, kind="image")
    base.update(over)
    return base


def test_upsert_returns_id_and_is_idempotent(conn):
    first = db.upsert_file(conn, **_row())
    second = db.upsert_file(conn, **_row())
    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


def test_upsert_updates_size_on_rescan(conn):
    db.upsert_file(conn, **_row())
    db.upsert_file(conn, **_row(size=222))
    assert conn.execute("SELECT size FROM files").fetchone()[0] == 222


def test_default_state_is_walked(conn):
    db.upsert_file(conn, **_row())
    assert conn.execute("SELECT state FROM files").fetchone()[0] == "walked"


def test_set_state_rejects_unknown_state(conn):
    fid = db.upsert_file(conn, **_row())
    with pytest.raises(ValueError):
        db.set_state(conn, fid, "banana")


def test_set_state_accepts_known_state(conn):
    fid = db.upsert_file(conn, **_row())
    db.set_state(conn, fid, "hashed")
    assert conn.execute("SELECT state FROM files").fetchone()[0] == "hashed"


def test_path_is_unique(conn):
    db.upsert_file(conn, **_row())
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


def test_sidecar_links_to_parent(conn):
    parent = db.upsert_file(conn, **_row())
    db.upsert_file(conn, **_row(path=r"P:\a\b.xmp", filename="b.xmp",
                                ext=".xmp", kind="sidecar", sidecar_of=parent))
    got = conn.execute("SELECT sidecar_of FROM files WHERE ext='.xmp'").fetchone()[0]
    assert got == parent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_db.py -v`
Expected: FAIL with `ImportError: cannot import name 'db'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/db.py
"""SQLite index for the P: drive pipeline.

Lives on C:, never on P: — a drive dropout must not cost the index.
"""
import sqlite3

STATES = frozenset({
    "walked", "exif_read", "hashed",
    "planned_dupe", "quarantined", "evacuated",
    "stamped", "organised", "undated", "error",
})

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id             INTEGER PRIMARY KEY,
    path           TEXT UNIQUE NOT NULL,
    top_folder     TEXT NOT NULL,
    rel_dir        TEXT NOT NULL,
    filename       TEXT NOT NULL,
    ext            TEXT NOT NULL,
    size           INTEGER NOT NULL,
    mtime          REAL NOT NULL,
    kind           TEXT NOT NULL,
    partial_hash   TEXT,
    sha256         TEXT,
    pixel_hash     TEXT,
    percept_hash   TEXT,
    exif_dt        TEXT,
    exif_dt_source TEXT,
    camera_model   TEXT,
    sidecar_of     INTEGER REFERENCES files(id),
    state          TEXT NOT NULL DEFAULT 'walked'
);
CREATE INDEX IF NOT EXISTS idx_size  ON files(size);
CREATE INDEX IF NOT EXISTS idx_sha   ON files(sha256);
CREATE INDEX IF NOT EXISTS idx_dt    ON files(exif_dt);
CREATE INDEX IF NOT EXISTS idx_state ON files(state);
CREATE INDEX IF NOT EXISTS idx_part  ON files(partial_hash);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_file(conn: sqlite3.Connection, **fields) -> int:
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "path")
    conn.execute(
        f"INSERT INTO files ({cols}) VALUES ({marks}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}",
        tuple(fields.values()),
    )
    return conn.execute(
        "SELECT id FROM files WHERE path = ?", (fields["path"],)
    ).fetchone()[0]


def set_state(conn: sqlite3.Connection, file_id: int, state: str) -> None:
    if state not in STATES:
        raise ValueError(f"unknown state: {state}")
    conn.execute("UPDATE files SET state = ? WHERE id = ?", (state, file_id))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_db.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/db.py tests/unit/test_photo_archive_db.py
git commit -m "feat(photo_archive): SQLite index schema"
```

---

### Task 3: Stage 1 — walk and stat

**Files:**
- Create: `scripts/photo_archive/walk.py`
- Test: `tests/unit/test_photo_archive_walk.py`

**Interfaces:**
- Consumes: `win_long`, `config.*`, `db.upsert_file`
- Produces: `walk_scope(conn, drive_root: str, scope_roots: list[str]) -> dict[str, int]`, `ScopeUnreadable(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_walk.py
import os
import pytest
from scripts.photo_archive import db, walk


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "drive"
    scope = root / "family_Randoms"
    (scope / "sub").mkdir(parents=True)
    (scope / "a.jpg").write_bytes(b"x" * 10)
    (scope / "a.xmp").write_text("<xmp/>")
    (scope / "b.tif_original").write_bytes(b"y" * 10)
    (scope / "sub" / "c.raf").write_bytes(b"z" * 10)
    (root / "New LR Catalog").mkdir()
    (root / "New LR Catalog" / "cat.lrcat").write_bytes(b"n")
    return root


def test_walk_records_images_and_sidecars(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    names = {r["filename"] for r in conn.execute("SELECT filename FROM files")}
    assert names == {"a.jpg", "a.xmp", "c.raf"}


def test_walk_skips_original_files(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    rows = conn.execute(
        "SELECT COUNT(*) FROM files WHERE filename LIKE '%_original'").fetchone()[0]
    assert rows == 0


def test_walk_never_enters_hard_excluded_root(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    rows = conn.execute(
        "SELECT COUNT(*) FROM files WHERE ext = '.lrcat'").fetchone()[0]
    assert rows == 0


def test_walk_links_sidecar_to_parent(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    side = conn.execute("SELECT sidecar_of FROM files WHERE ext='.xmp'").fetchone()[0]
    parent = conn.execute("SELECT id FROM files WHERE filename='a.jpg'").fetchone()[0]
    assert side == parent


def test_walk_records_rel_dir(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    rel = conn.execute("SELECT rel_dir FROM files WHERE filename='c.raf'").fetchone()[0]
    assert rel == "sub"


def test_missing_scope_root_raises_not_silently_zero(conn, tree):
    with pytest.raises(walk.ScopeUnreadable):
        walk.walk_scope(conn, str(tree), ["No Such Folder"])


def test_walk_is_idempotent(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_walk.py -v`
Expected: FAIL with `ImportError: cannot import name 'walk'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/walk.py
"""Stage 1 - walk and stat. Reads no file contents."""
import os

from .paths import win_long
from . import config, db


class ScopeUnreadable(Exception):
    """A configured scope root could not be read.

    Raised rather than recording zero files. A run that cannot see a folder
    must fail loudly: silently indexing a folder as empty is how the photo
    index lost 1990/1991 without an error.
    """


def walk_scope(conn, drive_root: str, scope_roots: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for top in scope_roots:
        top_path = os.path.join(drive_root, top)
        if not os.path.isdir(win_long(top_path)):
            raise ScopeUnreadable(f"scope root missing or unreadable: {top_path}")
        counts[top] = _walk_one(conn, top_path, top)
        conn.commit()
    return counts


def _walk_one(conn, top_path: str, top: str) -> int:
    n = 0
    pending_sidecars: list[tuple[int, str]] = []
    stems: dict[tuple[str, str], int] = {}

    for dirpath, dirnames, filenames in os.walk(win_long(top_path)):
        dirnames[:] = [
            d for d in dirnames
            if d not in config.HARD_EXCLUDE and d not in config.EXCLUDE_DIR_NAMES
        ]
        rel_dir = os.path.relpath(dirpath, win_long(top_path))
        rel_dir = "" if rel_dir == "." else rel_dir

        for fn in filenames:
            if config.is_skipped_file(fn):
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            ext = os.path.splitext(fn)[1].lower()
            kind = config.kind_for(ext)
            fid = db.upsert_file(
                conn, path=full, top_folder=top, rel_dir=rel_dir,
                filename=fn, ext=ext, size=st.st_size,
                mtime=st.st_mtime, kind=kind,
            )
            stem = os.path.splitext(fn)[0]
            if kind == "sidecar":
                pending_sidecars.append((fid, os.path.join(rel_dir, stem)))
            else:
                stems.setdefault((rel_dir, stem), fid)
            n += 1

    for fid, relstem in pending_sidecars:
        rel_dir, stem = os.path.split(relstem)
        parent = stems.get((rel_dir, stem))
        if parent is not None:
            conn.execute("UPDATE files SET sidecar_of = ? WHERE id = ?", (parent, fid))
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_walk.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/walk.py tests/unit/test_photo_archive_walk.py
git commit -m "feat(photo_archive): stage 1 walk and stat with halt-on-unreadable-root"
```

---

### Task 4: Stage 2 — EXIF date and camera model

**Files:**
- Create: `scripts/photo_archive/exif.py`
- Test: `tests/unit/test_photo_archive_exif.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure parsing plus a subprocess wrapper)
- Produces: `pick_datetime(tags: dict) -> tuple[str | None, str | None]`, `sanitise_model(model: str) -> str`, `ExifReader(exiftool: str = "exiftool")` with `.read_many(paths: list[str]) -> dict[str, dict]` and `.close()`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_exif.py
from scripts.photo_archive import exif


def test_prefers_subsec_over_datetimeoriginal():
    dt, src = exif.pick_datetime({
        "SubSecDateTimeOriginal": "2019:01:01 15:41:12.45",
        "DateTimeOriginal": "2019:01:01 15:41:12",
        "CreateDate": "2001:01:01 00:00:00",
    })
    assert dt == "2019-01-01T15:41:12"
    assert src == "SubSecDateTimeOriginal"


def test_falls_back_to_createdate():
    dt, src = exif.pick_datetime({"CreateDate": "2005:07:04 08:09:10"})
    assert dt == "2005-07-04T08:09:10"
    assert src == "CreateDate"


def test_zero_date_is_rejected():
    # exiftool returns this for files with a null date field; it is not a date.
    dt, src = exif.pick_datetime({"DateTimeOriginal": "0000:00:00 00:00:00"})
    assert dt is None and src is None


def test_missing_tags_give_none():
    assert exif.pick_datetime({}) == (None, None)


def test_malformed_date_is_rejected():
    assert exif.pick_datetime({"DateTimeOriginal": "not a date"}) == (None, None)


def test_model_sanitised_for_filesystem():
    assert exif.sanitise_model("X-T5") == "X-T5"
    assert exif.sanitise_model("HP pstc5200") == "HP pstc5200"
    assert exif.sanitise_model("Canon/EOS:60D") == "Canon-EOS-60D"
    assert exif.sanitise_model("  X100V  ") == "X100V"


def test_empty_model_returns_empty_string():
    assert exif.sanitise_model("") == ""
    assert exif.sanitise_model(None) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_exif.py -v`
Expected: FAIL with `ImportError: cannot import name 'exif'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/exif.py
"""Stage 2 - EXIF capture date and camera model via exiftool -stay_open.

Per-file exiftool invocations run about 7s each on these drives; batch mode
is the only workable approach at this scale.
"""
import json
import os
import re
import subprocess

DATE_TAGS = ("SubSecDateTimeOriginal", "DateTimeOriginal", "CreateDate")
_DATE_RE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def pick_datetime(tags: dict) -> tuple[str | None, str | None]:
    for tag in DATE_TAGS:
        raw = tags.get(tag)
        if not raw:
            continue
        m = _DATE_RE.match(str(raw))
        if not m:
            continue
        y, mo, d, h, mi, s = m.groups()
        if y == "0000" or mo == "00" or d == "00":
            continue
        return f"{y}-{mo}-{d}T{h}:{mi}:{s}", tag
    return None, None


def sanitise_model(model) -> str:
    if not model:
        return ""
    cleaned = _UNSAFE.sub("-", str(model)).strip()
    return re.sub(r"-{2,}", "-", cleaned).strip("-. ")


class ExifReader:
    """Persistent exiftool process. Always close() it."""

    def __init__(self, exiftool: str = "exiftool"):
        self.proc = subprocess.Popen(
            [exiftool, "-stay_open", "True", "-@", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
        )

    def read_many(self, paths: list[str]) -> dict[str, dict]:
        if not paths:
            return {}
        args = ["-json", "-fast2", "-charset", "filename=utf8",
                "-DateTimeOriginal", "-SubSecDateTimeOriginal",
                "-CreateDate", "-Model", "-Make"]
        args += paths
        self.proc.stdin.write("\n".join(args) + "\n-execute\n")
        self.proc.stdin.flush()

        chunks = []
        for line in self.proc.stdout:
            if line.strip() == "{ready}":
                break
            chunks.append(line)
        try:
            records = json.loads("".join(chunks) or "[]")
        except json.JSONDecodeError:
            return {}
        return {rec.get("SourceFile", ""): rec for rec in records}

    def close(self) -> None:
        try:
            self.proc.stdin.write("-stay_open\nFalse\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_exif.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/exif.py tests/unit/test_photo_archive_exif.py
git commit -m "feat(photo_archive): stage 2 EXIF date precedence and model sanitising"
```

---

### Task 5: Stage 3 — size-bucketed hashing

**Files:**
- Create: `scripts/photo_archive/hashing.py`
- Test: `tests/unit/test_photo_archive_hashing.py`

**Interfaces:**
- Consumes: `win_long`
- Produces: `partial_hash(path: str, size: int, chunk: int = 65536) -> str`, `full_hash(path: str, bufsize: int = 1048576) -> str`, `hash_candidates(conn) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_hashing.py
import pytest
from scripts.photo_archive import db, hashing


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _add(conn, path, size, content=None):
    if content is not None:
        with open(path, "wb") as fh:
            fh.write(content)
    return db.upsert_file(conn, path=str(path), top_folder="t", rel_dir="",
                          filename="f", ext=".jpg", size=size, mtime=1.0,
                          kind="image")


def test_partial_hash_differs_on_different_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"A" * 200000)
    b.write_bytes(b"B" * 200000)
    assert hashing.partial_hash(str(a), 200000) != hashing.partial_hash(str(b), 200000)


def test_partial_hash_stable_for_same_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"A" * 200000)
    b.write_bytes(b"A" * 200000)
    assert hashing.partial_hash(str(a), 200000) == hashing.partial_hash(str(b), 200000)


def test_partial_hash_catches_tail_difference(tmp_path):
    # Files identical for the first 64KB must still be distinguished.
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"A" * 200000 + b"END1")
    b.write_bytes(b"A" * 200000 + b"END2")
    assert hashing.partial_hash(str(a), 200004) != hashing.partial_hash(str(b), 200004)


def test_full_hash_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "a"
    p.write_bytes(b"hello world")
    assert hashing.full_hash(str(p)) == hashlib.sha256(b"hello world").hexdigest()


def test_unique_size_is_never_hashed(conn, tmp_path):
    _add(conn, tmp_path / "solo.jpg", 999, b"S" * 999)
    stats = hashing.hash_candidates(conn)
    assert stats["partial_hashed"] == 0
    assert conn.execute(
        "SELECT partial_hash FROM files").fetchone()[0] is None


def test_colliding_sizes_get_partial_then_full(conn, tmp_path):
    _add(conn, tmp_path / "x.jpg", 100, b"X" * 100)
    _add(conn, tmp_path / "y.jpg", 100, b"X" * 100)
    stats = hashing.hash_candidates(conn)
    assert stats["partial_hashed"] == 2
    assert stats["full_hashed"] == 2
    shas = [r[0] for r in conn.execute("SELECT sha256 FROM files")]
    assert shas[0] == shas[1] is not None


def test_same_size_different_content_skips_full_hash(conn, tmp_path):
    _add(conn, tmp_path / "x.jpg", 100, b"X" * 100)
    _add(conn, tmp_path / "y.jpg", 100, b"Y" * 100)
    stats = hashing.hash_candidates(conn)
    assert stats["partial_hashed"] == 2
    assert stats["full_hashed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_hashing.py -v`
Expected: FAIL with `ImportError: cannot import name 'hashing'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/hashing.py
"""Stage 3 - hashing, restricted to size-collision groups.

A file whose size is globally unique cannot be a byte-duplicate and is never
read. Within a size group, a partial hash (head + tail) filters before any
full read. This is what makes a 5.4 TB run finish in hours rather than days.
"""
import hashlib

from .paths import win_long


def partial_hash(path: str, size: int, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(win_long(path), "rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk * 2:
            fh.seek(-chunk, 2)
            h.update(fh.read(chunk))
    return h.hexdigest()


def full_hash(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(win_long(path), "rb") as fh:
        while True:
            block = fh.read(bufsize)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_candidates(conn) -> dict[str, int]:
    stats = {"partial_hashed": 0, "full_hashed": 0, "errors": 0}

    dup_sizes = [r[0] for r in conn.execute(
        "SELECT size FROM files WHERE kind != 'sidecar' "
        "GROUP BY size HAVING COUNT(*) > 1")]

    for size in dup_sizes:
        rows = conn.execute(
            "SELECT id, path FROM files WHERE size = ? AND kind != 'sidecar' "
            "AND partial_hash IS NULL", (size,)).fetchall()
        for row in rows:
            try:
                ph = partial_hash(row["path"], size)
            except OSError:
                stats["errors"] += 1
                continue
            conn.execute("UPDATE files SET partial_hash = ? WHERE id = ?",
                         (ph, row["id"]))
            stats["partial_hashed"] += 1
        conn.commit()

    dup_parts = [r[0] for r in conn.execute(
        "SELECT partial_hash FROM files WHERE partial_hash IS NOT NULL "
        "GROUP BY partial_hash HAVING COUNT(*) > 1")]

    for ph in dup_parts:
        rows = conn.execute(
            "SELECT id, path FROM files WHERE partial_hash = ? AND sha256 IS NULL",
            (ph,)).fetchall()
        for row in rows:
            try:
                sha = full_hash(row["path"])
            except OSError:
                stats["errors"] += 1
                continue
            conn.execute("UPDATE files SET sha256 = ? WHERE id = ?",
                         (sha, row["id"]))
            stats["full_hashed"] += 1
        conn.commit()

    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_hashing.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/hashing.py tests/unit/test_photo_archive_hashing.py
git commit -m "feat(photo_archive): stage 3 size-bucketed partial then full hashing"
```

---

### Task 6: Stage 4a — tier 1 duplicate plan and keeper rule

**Files:**
- Create: `scripts/photo_archive/dupes.py`
- Test: `tests/unit/test_photo_archive_dupes.py`

**Interfaces:**
- Consumes: `db`, `config`
- Produces: `choose_keeper(rows: list) -> object`, `tier1_groups(conn) -> Iterator[list]`, `plan_tier1(conn, out_csv: str) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_dupes.py
import csv
import pytest
from scripts.photo_archive import db, dupes


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _add(conn, path, top="family_Randoms", rel="", size=100, sha="abc",
         mtime=100.0, dt=None, model=None, ext=".jpg", kind="image"):
    return db.upsert_file(conn, path=path, top_folder=top, rel_dir=rel,
                          filename=path.split("\\")[-1], ext=ext, size=size,
                          mtime=mtime, kind=kind, sha256=sha,
                          exif_dt=dt, camera_model=model)


def _rows(conn):
    return conn.execute("SELECT * FROM files ORDER BY id").fetchall()


def test_keeper_avoids_dupes_named_folder(conn):
    _add(conn, r"P:\0 and 1 star photos originals and dupes\a.jpg",
         top="0 and 1 star photos originals and dupes")
    _add(conn, r"P:\family_Randoms\a.jpg")
    keeper = dupes.choose_keeper(_rows(conn))
    assert keeper["top_folder"] == "family_Randoms"


def test_keeper_avoids_backup_named_folder(conn):
    _add(conn, r"P:\Backup Consolidated Photos\a.jpg",
         top="Backup Consolidated Photos")
    _add(conn, r"P:\family_Randoms\a.jpg")
    assert dupes.choose_keeper(_rows(conn))["top_folder"] == "family_Randoms"


def test_keeper_prefers_richer_exif(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", dt=None)
    _add(conn, r"P:\family_Randoms\b.jpg", dt="2019-01-01T10:00:00")
    assert dupes.choose_keeper(_rows(conn))["exif_dt"] is not None


def test_keeper_prefers_camera_model_when_dates_equal(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", dt="2019-01-01T10:00:00", model=None)
    _add(conn, r"P:\family_Randoms\b.jpg", dt="2019-01-01T10:00:00", model="X-T5")
    assert dupes.choose_keeper(_rows(conn))["camera_model"] == "X-T5"


def test_keeper_prefers_shallower_path(conn):
    _add(conn, r"P:\family_Randoms\deep\deeper\a.jpg", rel="deep\\deeper")
    _add(conn, r"P:\family_Randoms\b.jpg", rel="")
    assert dupes.choose_keeper(_rows(conn))["rel_dir"] == ""


def test_keeper_prefers_older_mtime(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", mtime=500.0)
    _add(conn, r"P:\family_Randoms\b.jpg", mtime=100.0)
    assert dupes.choose_keeper(_rows(conn))["mtime"] == 100.0


def test_keeper_is_deterministic_on_full_tie(conn):
    _add(conn, r"P:\family_Randoms\b.jpg")
    _add(conn, r"P:\family_Randoms\a.jpg")
    first = dupes.choose_keeper(_rows(conn))["path"]
    second = dupes.choose_keeper(list(reversed(_rows(conn))))["path"]
    assert first == second == r"P:\family_Randoms\a.jpg"


def test_raw_and_jpg_never_grouped(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", sha="same", ext=".jpg", kind="image")
    _add(conn, r"P:\family_Randoms\a.raf", sha="same", ext=".raf", kind="raw")
    assert list(dupes.tier1_groups(conn)) == []


def test_tier1_group_requires_two_members(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", sha="solo")
    assert list(dupes.tier1_groups(conn)) == []


def test_plan_tier1_writes_csv_with_keeper_flag(conn, tmp_path):
    _add(conn, r"P:\Backup Consolidated Photos\a.jpg",
         top="Backup Consolidated Photos")
    _add(conn, r"P:\family_Randoms\a.jpg")
    out = tmp_path / "plan.csv"
    stats = dupes.plan_tier1(conn, str(out))
    rows = list(csv.DictReader(out.open()))
    assert stats["groups"] == 1
    assert stats["to_quarantine"] == 1
    keepers = [r for r in rows if r["role"] == "keep"]
    assert len(keepers) == 1
    assert keepers[0]["top_folder"] == "family_Randoms"


def test_sidecars_are_never_grouped_as_duplicates(conn):
    _add(conn, r"P:\family_Randoms\a.xmp", sha="s", ext=".xmp", kind="sidecar")
    _add(conn, r"P:\family_Randoms\b.xmp", sha="s", ext=".xmp", kind="sidecar")
    assert list(dupes.tier1_groups(conn)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_dupes.py -v`
Expected: FAIL with `ImportError: cannot import name 'dupes'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/dupes.py
"""Stage 4 - duplicate grouping and keeper selection."""
import csv

DEMOTE_MARKERS = ("dupes", "backup", "_dupes")


def _demoted(row) -> int:
    haystack = f"{row['top_folder']}/{row['rel_dir']}".lower()
    return 1 if any(m in haystack for m in DEMOTE_MARKERS) else 0


def _sort_key(row):
    """Lower sorts better. Mirrors the spec's keeper rule, in order."""
    return (
        _demoted(row),
        0 if row["exif_dt"] else 1,
        0 if row["camera_model"] else 1,
        len([p for p in str(row["rel_dir"]).split("\\") if p]),
        row["mtime"],
        row["path"],
    )


def choose_keeper(rows):
    return min(rows, key=_sort_key)


def tier1_groups(conn):
    """Byte-identical groups. RAW never groups with non-RAW; sidecars never group."""
    shas = [r[0] for r in conn.execute(
        "SELECT sha256 FROM files WHERE sha256 IS NOT NULL AND kind != 'sidecar' "
        "GROUP BY sha256 HAVING COUNT(*) > 1")]
    for sha in shas:
        rows = conn.execute(
            "SELECT * FROM files WHERE sha256 = ? AND kind != 'sidecar'",
            (sha,)).fetchall()
        for kind_bucket in ("raw", "image", "video", "other"):
            members = [r for r in rows if r["kind"] == kind_bucket]
            if len(members) > 1:
                yield members


FIELDS = ["group_id", "role", "path", "top_folder", "rel_dir", "size",
          "sha256", "exif_dt", "camera_model", "mtime"]


def plan_tier1(conn, out_csv: str) -> dict[str, int]:
    stats = {"groups": 0, "to_quarantine": 0, "bytes_reclaimed": 0}
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for gid, members in enumerate(tier1_groups(conn), start=1):
            keeper = choose_keeper(members)
            stats["groups"] += 1
            for row in sorted(members, key=_sort_key):
                role = "keep" if row["id"] == keeper["id"] else "quarantine"
                if role == "quarantine":
                    stats["to_quarantine"] += 1
                    stats["bytes_reclaimed"] += row["size"]
                writer.writerow({k: row[k] for k in FIELDS if k in row.keys()}
                                | {"group_id": gid, "role": role})
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_dupes.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/dupes.py tests/unit/test_photo_archive_dupes.py
git commit -m "feat(photo_archive): stage 4a tier-1 dupe plan and keeper rule"
```

---

### Task 7: Journal and undo

**Files:**
- Create: `scripts/photo_archive/journal.py`
- Test: `tests/unit/test_photo_archive_journal.py`

**Interfaces:**
- Consumes: `hashing.full_hash`, `win_long`
- Produces: `Journal(path: str)` with `.append(stage, op, src, dst, size, sha256, status) -> None`; `undo(journal_path: str, apply: bool = False) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_journal.py
import csv
import pytest
from scripts.photo_archive import journal, hashing


def test_append_writes_header_once(tmp_path):
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", "a", "b", 1, "s", "ok")
    j.append("quarantine", "move", "c", "d", 1, "s", "ok")
    lines = (tmp_path / "j.csv").read_text().strip().split("\n")
    assert lines[0].startswith("timestamp,")
    assert len(lines) == 3


def test_undo_dry_run_moves_nothing(tmp_path):
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    dst.write_bytes(b"data")
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", str(src), str(dst), 4,
             hashing.full_hash(str(dst)), "ok")
    stats = journal.undo(str(tmp_path / "j.csv"), apply=False)
    assert stats["would_restore"] == 1
    assert dst.exists() and not src.exists()


def test_undo_apply_restores_file(tmp_path):
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    dst.write_bytes(b"data")
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", str(src), str(dst), 4,
             hashing.full_hash(str(dst)), "ok")
    stats = journal.undo(str(tmp_path / "j.csv"), apply=True)
    assert stats["restored"] == 1
    assert src.exists() and not dst.exists()


def test_undo_refuses_when_hash_changed(tmp_path):
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    dst.write_bytes(b"data")
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", str(src), str(dst), 4, "deadbeef", "ok")
    stats = journal.undo(str(tmp_path / "j.csv"), apply=True)
    assert stats["refused"] == 1
    assert dst.exists() and not src.exists()


def test_undo_skips_missing_destination(tmp_path):
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", str(tmp_path / "s"), str(tmp_path / "gone"),
             1, "x", "ok")
    stats = journal.undo(str(tmp_path / "j.csv"), apply=True)
    assert stats["missing"] == 1


def test_undo_replays_in_reverse_order(tmp_path):
    # A then B; undo must process B before A.
    a, b = tmp_path / "a", tmp_path / "b"
    b.write_bytes(b"bb")
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("s", "move", str(a), str(b), 2, hashing.full_hash(str(b)), "ok")
    order = journal.undo(str(tmp_path / "j.csv"), apply=False)["order"]
    assert order == [str(b)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_journal.py -v`
Expected: FAIL with `ImportError: cannot import name 'journal'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/journal.py
"""Append-only operation journal and reverse-replay undo.

The journal lives on C:, so a P: or L: dropout cannot destroy the record of
what was moved.
"""
import csv
import os
import time

from .paths import win_long
from .hashing import full_hash

FIELDS = ["timestamp", "stage", "operation", "src", "dst", "size",
          "sha256", "status"]


class Journal:
    def __init__(self, path: str):
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        if not exists:
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    def append(self, stage, operation, src, dst, size, sha256, status) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "stage": stage, "operation": operation, "src": src,
                "dst": dst, "size": size, "sha256": sha256, "status": status,
            })


def undo(journal_path: str, apply: bool = False) -> dict:
    stats = {"restored": 0, "would_restore": 0, "refused": 0,
             "missing": 0, "order": []}
    with open(journal_path, newline="", encoding="utf-8") as fh:
        entries = [r for r in csv.DictReader(fh)
                   if r["operation"] == "move" and r["status"] == "ok"]

    for entry in reversed(entries):
        src, dst = entry["src"], entry["dst"]
        if not os.path.exists(win_long(dst)):
            stats["missing"] += 1
            continue
        if entry["sha256"]:
            try:
                if full_hash(dst) != entry["sha256"]:
                    stats["refused"] += 1
                    continue
            except OSError:
                stats["refused"] += 1
                continue
        stats["order"].append(dst)
        if not apply:
            stats["would_restore"] += 1
            continue
        os.makedirs(win_long(os.path.dirname(src)), exist_ok=True)
        os.replace(win_long(dst), win_long(src))
        stats["restored"] += 1
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_journal.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/journal.py tests/unit/test_photo_archive_journal.py
git commit -m "feat(photo_archive): append-only journal with hash-verified undo"
```

---

### Task 8: Stage 5 — quarantine executor

**Files:**
- Create: `scripts/photo_archive/execute.py`
- Test: `tests/unit/test_photo_archive_execute.py`

**Interfaces:**
- Consumes: `Journal`, `win_long`, `db.set_state`
- Produces: `quarantine(conn, plan_csv: str, drive_root: str, journal: Journal, apply: bool = False) -> dict[str, int]`, `safe_move(src: str, dst: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_execute.py
import csv
import os
import pytest
from scripts.photo_archive import db, execute, journal


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _plan(tmp_path, rows):
    p = tmp_path / "plan.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group_id", "role", "path",
                                           "top_folder", "rel_dir", "size",
                                           "sha256"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def test_safe_move_suffixes_on_collision(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    result = execute.safe_move(str(a), str(b))
    assert result.endswith("b-2.jpg")
    assert not a.exists()


def test_safe_move_never_overwrites(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    execute.safe_move(str(a), str(b))
    assert b.read_bytes() == b"2"


def test_quarantine_dry_run_moves_nothing(conn, tmp_path):
    src = tmp_path / "family_Randoms" / "a.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "quarantine",
                             "path": str(src), "top_folder": "family_Randoms",
                             "rel_dir": "", "size": 1, "sha256": "s"}])
    j = journal.Journal(str(tmp_path / "j.csv"))
    stats = execute.quarantine(conn, plan, str(tmp_path), j, apply=False)
    assert stats["would_move"] == 1
    assert src.exists()


def test_quarantine_apply_moves_and_preserves_provenance(conn, tmp_path):
    src = tmp_path / "family_Randoms" / "sub" / "a.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "quarantine",
                             "path": str(src), "top_folder": "family_Randoms",
                             "rel_dir": "sub", "size": 1, "sha256": "s"}])
    j = journal.Journal(str(tmp_path / "j.csv"))
    stats = execute.quarantine(conn, plan, str(tmp_path), j, apply=True)
    assert stats["moved"] == 1
    assert not src.exists()
    assert (tmp_path / "_DUPES" / "family_Randoms" / "sub" / "a.jpg").exists()


def test_quarantine_ignores_keeper_rows(conn, tmp_path):
    keeper = tmp_path / "family_Randoms" / "k.jpg"
    keeper.parent.mkdir(parents=True)
    keeper.write_bytes(b"k")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "keep",
                             "path": str(keeper), "top_folder": "family_Randoms",
                             "rel_dir": "", "size": 1, "sha256": "s"}])
    j = journal.Journal(str(tmp_path / "j.csv"))
    stats = execute.quarantine(conn, plan, str(tmp_path), j, apply=True)
    assert stats["moved"] == 0
    assert keeper.exists()


def test_quarantine_journals_every_move(conn, tmp_path):
    src = tmp_path / "family_Randoms" / "a.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "quarantine",
                             "path": str(src), "top_folder": "family_Randoms",
                             "rel_dir": "", "size": 1, "sha256": "s"}])
    jpath = tmp_path / "j.csv"
    execute.quarantine(conn, plan, str(tmp_path), journal.Journal(str(jpath)),
                       apply=True)
    rows = list(csv.DictReader(jpath.open()))
    assert len(rows) == 1 and rows[0]["operation"] == "move"


def test_quarantine_is_resumable(conn, tmp_path):
    src = tmp_path / "family_Randoms" / "a.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "quarantine",
                             "path": str(src), "top_folder": "family_Randoms",
                             "rel_dir": "", "size": 1, "sha256": "s"}])
    j = journal.Journal(str(tmp_path / "j.csv"))
    execute.quarantine(conn, plan, str(tmp_path), j, apply=True)
    stats = execute.quarantine(conn, plan, str(tmp_path), j, apply=True)
    assert stats["missing"] == 1 and stats["moved"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_execute.py -v`
Expected: FAIL with `ImportError: cannot import name 'execute'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/execute.py
"""Stage 5 - quarantine duplicates by same-volume rename.

Nothing is deleted. Provenance is preserved in the target path so the
original location is recoverable without consulting the journal.
"""
import csv
import os

from .paths import win_long

QUARANTINE_DIR = "_DUPES"


def safe_move(src: str, dst: str) -> str:
    """Move src to dst, suffixing -2, -3 ... rather than ever overwriting."""
    os.makedirs(win_long(os.path.dirname(dst)), exist_ok=True)
    stem, ext = os.path.splitext(dst)
    candidate, n = dst, 2
    while os.path.exists(win_long(candidate)):
        candidate = f"{stem}-{n}{ext}"
        n += 1
    os.replace(win_long(src), win_long(candidate))
    return candidate


def quarantine(conn, plan_csv: str, drive_root: str, journal,
               apply: bool = False) -> dict:
    stats = {"moved": 0, "would_move": 0, "missing": 0, "errors": 0}
    with open(plan_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["role"] == "quarantine"]

    for row in rows:
        src = row["path"]
        if not os.path.exists(win_long(src)):
            stats["missing"] += 1
            continue
        dst = os.path.join(drive_root, QUARANTINE_DIR, row["top_folder"],
                           row["rel_dir"], os.path.basename(src))
        if not apply:
            stats["would_move"] += 1
            continue
        try:
            final = safe_move(src, dst)
        except OSError as exc:
            stats["errors"] += 1
            journal.append("quarantine", "move", src, dst, row["size"],
                           row["sha256"], f"error:{exc.errno}")
            continue
        journal.append("quarantine", "move", src, final, row["size"],
                       row["sha256"], "ok")
        conn.execute("UPDATE files SET path = ?, state = 'quarantined' "
                     "WHERE path = ?", (final, src))
        stats["moved"] += 1
    conn.commit()
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_execute.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/execute.py tests/unit/test_photo_archive_execute.py
git commit -m "feat(photo_archive): stage 5 quarantine executor with collision-safe moves"
```

---

### Task 9: Stage 8 — naming and year foldering

**Files:**
- Create: `scripts/photo_archive/organise.py`
- Test: `tests/unit/test_photo_archive_organise.py`

**Interfaces:**
- Consumes: `exif.sanitise_model`, `execute.safe_move`, `Journal`, `config`
- Produces: `target_name(exif_dt: str, model: str | None, ext: str) -> str`, `target_dir(drive_root: str, exif_dt: str) -> str`, `plan_organise(conn, drive_root: str, out_csv: str) -> dict[str, int]`, `apply_organise(conn, plan_csv: str, journal, apply: bool = False) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_organise.py
import csv
import os
import pytest
from scripts.photo_archive import db, organise


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def test_target_name_matches_existing_export_convention():
    assert organise.target_name("2019-01-01T15:41:12", "X-M1", ".jpg") == \
        "2019-01-01 15-41-12-X-M1.jpg"


def test_target_name_drops_trailing_hyphen_without_model():
    # Never produce the dangling "1994-01-01 21-03-33--4.jpg" form.
    assert organise.target_name("1994-01-01T21:03:33", None, ".jpg") == \
        "1994-01-01 21-03-33.jpg"
    assert organise.target_name("1994-01-01T21:03:33", "", ".jpg") == \
        "1994-01-01 21-03-33.jpg"


def test_target_name_preserves_internal_spaces_in_model():
    assert organise.target_name("1994-03-01T14:55:53", "HP pstc5200", ".jpg") == \
        "1994-03-01 14-55-53-HP pstc5200.jpg"


def test_target_name_lowercases_extension():
    assert organise.target_name("2019-01-01T15:41:12", "X-M1", ".JPG").endswith(".jpg")


def test_target_dir_is_year_then_year_month():
    got = organise.target_dir("P:", "2019-03-04T10:00:00")
    assert got == os.path.join("P:", "2019", "2019-03")


def test_undated_rows_route_to_undated_tree(conn, tmp_path):
    db.upsert_file(conn, path=r"P:\family_Randoms\sub\x.jpg",
                   top_folder="family_Randoms", rel_dir="sub", filename="x.jpg",
                   ext=".jpg", size=1, mtime=1.0, kind="image", exif_dt=None)
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    row = next(r for r in csv.DictReader(out.open()))
    assert row["dst"] == os.path.join("P:", "_UNDATED", "family_Randoms",
                                      "sub", "x.jpg")


def test_dated_rows_route_to_year_month(conn, tmp_path):
    db.upsert_file(conn, path=r"P:\family_Randoms\x.jpg",
                   top_folder="family_Randoms", rel_dir="", filename="x.jpg",
                   ext=".jpg", size=1, mtime=1.0, kind="image",
                   exif_dt="2019-03-04T10:00:00", camera_model="X-T5")
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    row = next(r for r in csv.DictReader(out.open()))
    assert row["dst"] == os.path.join("P:", "2019", "2019-03",
                                      "2019-03-04 10-00-00-X-T5.jpg")


def test_sidecar_takes_parent_stem_and_destination(conn, tmp_path):
    parent = db.upsert_file(conn, path=r"P:\family_Randoms\x.raf",
                            top_folder="family_Randoms", rel_dir="",
                            filename="x.raf", ext=".raf", size=1, mtime=1.0,
                            kind="raw", exif_dt="2019-03-04T10:00:00",
                            camera_model="X-T5")
    db.upsert_file(conn, path=r"P:\family_Randoms\x.xmp",
                   top_folder="family_Randoms", rel_dir="", filename="x.xmp",
                   ext=".xmp", size=1, mtime=1.0, kind="sidecar",
                   sidecar_of=parent)
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    rows = {r["src"]: r["dst"] for r in csv.DictReader(out.open())}
    assert rows[r"P:\family_Randoms\x.xmp"] == os.path.join(
        "P:", "2019", "2019-03", "2019-03-04 10-00-00-X-T5.xmp")


def test_quarantined_rows_are_not_organised(conn, tmp_path):
    fid = db.upsert_file(conn, path=r"P:\_DUPES\family_Randoms\x.jpg",
                         top_folder="family_Randoms", rel_dir="",
                         filename="x.jpg", ext=".jpg", size=1, mtime=1.0,
                         kind="image", exif_dt="2019-03-04T10:00:00")
    db.set_state(conn, fid, "quarantined")
    out = tmp_path / "plan.csv"
    stats = organise.plan_organise(conn, "P:", str(out))
    assert stats["planned"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_organise.py -v`
Expected: FAIL with `ImportError: cannot import name 'organise'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/organise.py
"""Stage 8 - rename to the house convention and file under YYYY/YYYY-MM."""
import csv
import os

from .exif import sanitise_model
from .execute import safe_move
from .paths import win_long

UNDATED_DIR = "_UNDATED"
PLAN_FIELDS = ["id", "src", "dst", "kind", "reason"]


def target_name(exif_dt: str, model, ext: str) -> str:
    date, time_part = exif_dt.split("T")
    stamp = f"{date} {time_part.replace(':', '-')}"
    clean = sanitise_model(model)
    suffix = f"-{clean}" if clean else ""
    return f"{stamp}{suffix}{ext.lower()}"


def target_dir(drive_root: str, exif_dt: str) -> str:
    year, month = exif_dt[:4], exif_dt[5:7]
    return os.path.join(drive_root, year, f"{year}-{month}")


def plan_organise(conn, drive_root: str, out_csv: str) -> dict:
    stats = {"planned": 0, "undated": 0}
    rows = conn.execute(
        "SELECT * FROM files WHERE state NOT IN "
        "('quarantined', 'evacuated', 'organised')").fetchall()
    by_id = {r["id"]: r for r in rows}

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            if row["kind"] == "sidecar":
                parent = by_id.get(row["sidecar_of"])
                if parent is None or not parent["exif_dt"]:
                    dst = os.path.join(drive_root, UNDATED_DIR,
                                       row["top_folder"], row["rel_dir"],
                                       row["filename"])
                    reason = "undated"
                    stats["undated"] += 1
                else:
                    name = target_name(parent["exif_dt"],
                                       parent["camera_model"], row["ext"])
                    dst = os.path.join(target_dir(drive_root, parent["exif_dt"]),
                                       name)
                    reason = "sidecar_follows_parent"
                    stats["planned"] += 1
            elif row["exif_dt"]:
                name = target_name(row["exif_dt"], row["camera_model"], row["ext"])
                dst = os.path.join(target_dir(drive_root, row["exif_dt"]), name)
                reason = "dated"
                stats["planned"] += 1
            else:
                dst = os.path.join(drive_root, UNDATED_DIR, row["top_folder"],
                                   row["rel_dir"], row["filename"])
                reason = "undated"
                stats["undated"] += 1
            writer.writerow({"id": row["id"], "src": row["path"], "dst": dst,
                             "kind": row["kind"], "reason": reason})
    return stats


def apply_organise(conn, plan_csv: str, journal, apply: bool = False) -> dict:
    stats = {"moved": 0, "would_move": 0, "missing": 0, "errors": 0}
    with open(plan_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        src, dst = row["src"], row["dst"]
        if not os.path.exists(win_long(src)):
            stats["missing"] += 1
            continue
        if not apply:
            stats["would_move"] += 1
            continue
        try:
            final = safe_move(src, dst)
        except OSError as exc:
            stats["errors"] += 1
            journal.append("organise", "move", src, dst, 0, "",
                           f"error:{exc.errno}")
            continue
        journal.append("organise", "move", src, final, 0, "", "ok")
        conn.execute("UPDATE files SET path = ?, state = 'organised' "
                     "WHERE id = ?", (final, row["id"]))
        stats["moved"] += 1
    conn.commit()
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_organise.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/organise.py tests/unit/test_photo_archive_organise.py
git commit -m "feat(photo_archive): stage 8 naming and YYYY/YYYY-MM foldering"
```

---

### Task 10: CLI and rehearsal runbook

**Files:**
- Create: `scripts/photo_archive/cli.py`
- Create: `scripts/photo_archive/README.md`
- Test: `tests/unit/test_photo_archive_cli.py`

**Interfaces:**
- Consumes: every module above
- Produces: `main(argv: list[str]) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_photo_archive_cli.py
import pytest
from scripts.photo_archive import cli


def test_apply_defaults_to_false():
    args = cli.parse_args(["quarantine", "--db", "x.db", "--plan", "p.csv"])
    assert args.apply is False


def test_apply_flag_is_explicit():
    args = cli.parse_args(["quarantine", "--db", "x.db", "--plan", "p.csv",
                           "--apply"])
    assert args.apply is True


def test_all_stages_are_registered():
    for stage in ("walk", "exif", "hash", "plan-dupes", "quarantine",
                  "plan-organise", "organise", "undo"):
        assert cli.parse_args([stage, "--db", "x.db"]).command == stage


def test_unknown_stage_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.parse_args(["banana", "--db", "x.db"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'cli'` (or `ModuleNotFoundError`) — both are correct; the package exists, the submodule does not yet

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/photo_archive/cli.py
"""Subcommand dispatch for the P: drive pipeline.

Every writing command requires an explicit --apply.
"""
import argparse
import sys

from . import config, db, dupes, execute, hashing, journal, organise, walk

STAGES = ("walk", "exif", "hash", "plan-dupes", "quarantine",
          "plan-organise", "organise", "undo")


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="photo_archive")
    sub = parser.add_subparsers(dest="command", required=True)
    for stage in STAGES:
        p = sub.add_parser(stage)
        p.add_argument("--db", required=True)
        p.add_argument("--drive", default=config.DRIVE)
        p.add_argument("--plan", default=None)
        p.add_argument("--journal", default="photo_archive_journal.csv")
        p.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command not in STAGES:
        parser.error(f"unknown stage: {args.command}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    conn = db.connect(args.db)
    db.init_schema(conn)
    jrn = journal.Journal(args.journal)

    if args.command == "walk":
        print(walk.walk_scope(conn, args.drive, config.SCOPE_ROOTS))
    elif args.command == "hash":
        print(hashing.hash_candidates(conn))
    elif args.command == "plan-dupes":
        print(dupes.plan_tier1(conn, args.plan))
    elif args.command == "quarantine":
        print(execute.quarantine(conn, args.plan, args.drive, jrn, args.apply))
    elif args.command == "plan-organise":
        print(organise.plan_organise(conn, args.drive, args.plan))
    elif args.command == "organise":
        print(organise.apply_organise(conn, args.plan, jrn, args.apply))
    elif args.command == "undo":
        print(journal.undo(args.journal, args.apply))
    else:
        print(f"stage not yet wired: {args.command}", file=sys.stderr)
        return 2
    conn.commit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_cli.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Write the rehearsal runbook**

````markdown
<!-- scripts/photo_archive/README.md -->
# P: drive photo archive pipeline

Runs under Windows Python: `C:\Python311\python.exe -m scripts.photo_archive.cli`

Every writing stage is dry-run unless `--apply` is passed. Nothing is ever
deleted.

## Rehearsal order — do not skip

1. `SMS_Photos` (80 files) — verify by hand.
2. `Google Drive Photos` (1,058 files) — verify by hand.
3. `family_Randoms` (118,975 files) — first at-scale run.
4. Everything else.

Restrict scope for a rehearsal by editing `SCOPE_ROOTS` in `config.py`.

## Full sequence

```bat
set PY=C:\Python311\python.exe
%PY% -m scripts.photo_archive.cli walk          --db C:\pindex.db
%PY% -m scripts.photo_archive.cli exif          --db C:\pindex.db
%PY% -m scripts.photo_archive.cli hash          --db C:\pindex.db
%PY% -m scripts.photo_archive.cli plan-dupes    --db C:\pindex.db --plan dupes.csv
REM  review dupes.csv before the next line
%PY% -m scripts.photo_archive.cli quarantine    --db C:\pindex.db --plan dupes.csv --apply
%PY% -m scripts.photo_archive.cli plan-organise --db C:\pindex.db --plan org.csv
REM  review org.csv before the next line
%PY% -m scripts.photo_archive.cli organise      --db C:\pindex.db --plan org.csv --apply
```

## Undo

```bat
%PY% -m scripts.photo_archive.cli undo --db C:\pindex.db --journal photo_archive_journal.csv
%PY% -m scripts.photo_archive.cli undo --db C:\pindex.db --journal photo_archive_journal.csv --apply
```

Undo refuses any file whose hash changed since it was moved.
````

- [ ] **Step 6: Commit**

```bash
git add scripts/photo_archive/cli.py scripts/photo_archive/README.md \
        tests/unit/test_photo_archive_cli.py
git commit -m "feat(photo_archive): CLI dispatch and rehearsal runbook"
```

---

### Task 11: Collision-aware organise plan

**Files:**
- Modify: `scripts/photo_archive/organise.py` (replace `plan_organise`, add `_unique`)
- Modify: `tests/unit/test_photo_archive_organise.py` (append 4 tests; keep all 9 existing)

**Interfaces:**
- Consumes: `target_name`, `target_dir` (unchanged)
- Produces: `_unique(dst: str, taken: set) -> tuple[str, bool]`; `plan_organise` now also returns `"collisions"` in its stats dict

**Why:** `plan_organise` computes `dst` from EXIF with no collision awareness, so three photos sharing a scan timestamp produce three IDENTICAL destinations in the CSV. `safe_move` disambiguates at apply time so no data is lost, but the CSV is what the operator approves and it currently misrepresents the outcome. Pre-digital scans collide in bulk (1994: 36 exports over 15 timestamps).

**All 9 existing tests must still pass.** The new logic is two-pass: non-sidecars first, then sidecars inheriting their parent's final name — including its collision suffix.

- [ ] **Step 1: Write the failing tests** (append to the existing test file)

```python
def test_colliding_timestamps_get_distinct_destinations(conn, tmp_path):
    for n in ("a", "b", "c"):
        db.upsert_file(conn, path=rf"P:\family_Randoms\{n}.jpg",
                       top_folder="family_Randoms", rel_dir="", filename=f"{n}.jpg",
                       ext=".jpg", size=1, mtime=1.0, kind="image",
                       exif_dt="1994-01-01T21:03:33", camera_model="HP pstc5200")
    out = tmp_path / "plan.csv"
    stats = organise.plan_organise(conn, "P:", str(out))
    dsts = [r["dst"] for r in csv.DictReader(out.open())]
    assert len(dsts) == 3
    assert len(set(dsts)) == 3, "plan must not list the same destination twice"
    assert stats["collisions"] == 2


def test_collision_suffixes_are_deterministic(conn, tmp_path):
    for n in ("a", "b"):
        db.upsert_file(conn, path=rf"P:\family_Randoms\{n}.jpg",
                       top_folder="family_Randoms", rel_dir="", filename=f"{n}.jpg",
                       ext=".jpg", size=1, mtime=1.0, kind="image",
                       exif_dt="1994-01-01T21:03:33", camera_model="X")
    first = tmp_path / "1.csv"
    second = tmp_path / "2.csv"
    organise.plan_organise(conn, "P:", str(first))
    organise.plan_organise(conn, "P:", str(second))
    assert first.read_text() == second.read_text()


def test_sidecar_inherits_parent_collision_suffix(conn, tmp_path):
    # Two RAWs share a timestamp; each has its own sidecar. Each .xmp must
    # follow ITS OWN parent, suffix included - never the other one's.
    for n in ("a", "b"):
        pid = db.upsert_file(conn, path=rf"P:\family_Randoms\{n}.raf",
                             top_folder="family_Randoms", rel_dir="",
                             filename=f"{n}.raf", ext=".raf", size=1, mtime=1.0,
                             kind="raw", exif_dt="2019-03-04T10:00:00",
                             camera_model="X-T5")
        db.upsert_file(conn, path=rf"P:\family_Randoms\{n}.xmp",
                       top_folder="family_Randoms", rel_dir="", filename=f"{n}.xmp",
                       ext=".xmp", size=1, mtime=1.0, kind="sidecar", sidecar_of=pid)
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    plan = {r["src"]: r["dst"] for r in csv.DictReader(out.open())}
    for n in ("a", "b"):
        raw = plan[rf"P:\family_Randoms\{n}.raf"]
        side = plan[rf"P:\family_Randoms\{n}.xmp"]
        assert os.path.splitext(raw)[0] == os.path.splitext(side)[0]
    assert len(set(plan.values())) == 4


def test_unique_returns_collision_flag():
    taken = set()
    assert organise._unique("P:/a/x.jpg", taken) == ("P:/a/x.jpg", False)
    assert organise._unique("P:/a/x.jpg", taken) == ("P:/a/x-2.jpg", True)
    assert organise._unique("P:/a/x.jpg", taken) == ("P:/a/x-3.jpg", True)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_organise.py -v`
Expected: 9 pass, 4 fail (`AttributeError: module ... has no attribute '_unique'`, and KeyError on `collisions`)

- [ ] **Step 3: Replace `plan_organise` and add `_unique`**

```python
def _unique(dst: str, taken: set) -> tuple[str, bool]:
    """Reserve dst, suffixing -2, -3 ... if already taken. Mirrors safe_move."""
    if dst not in taken:
        taken.add(dst)
        return dst, False
    stem, ext = os.path.splitext(dst)
    n = 2
    while f"{stem}-{n}{ext}" in taken:
        n += 1
    final = f"{stem}-{n}{ext}"
    taken.add(final)
    return final, True


def plan_organise(conn, drive_root: str, out_csv: str) -> dict:
    """Two passes: real files claim their names first, then sidecars follow
    their parent's FINAL name so a collision suffix propagates to the sidecar."""
    stats = {"planned": 0, "undated": 0, "collisions": 0}
    rows = conn.execute(
        "SELECT * FROM files WHERE state NOT IN "
        "('quarantined', 'evacuated', 'organised') ORDER BY path").fetchall()
    taken: set = set()
    assigned: dict = {}

    def undated_dst(row):
        return os.path.join(drive_root, UNDATED_DIR, row["top_folder"],
                            row["rel_dir"], row["filename"])

    for row in rows:
        if row["kind"] == "sidecar":
            continue
        if row["exif_dt"]:
            name = target_name(row["exif_dt"], row["camera_model"], row["ext"])
            dst, collided = _unique(
                os.path.join(target_dir(drive_root, row["exif_dt"]), name), taken)
            reason = "dated"
            stats["planned"] += 1
        else:
            dst, collided = _unique(undated_dst(row), taken)
            reason = "undated"
            stats["undated"] += 1
        stats["collisions"] += int(collided)
        assigned[row["id"]] = (dst, reason)

    for row in rows:
        if row["kind"] != "sidecar":
            continue
        parent = assigned.get(row["sidecar_of"])
        if parent is None or parent[1] == "undated":
            dst, collided = _unique(undated_dst(row), taken)
            reason = "undated"
            stats["undated"] += 1
        else:
            dst, collided = _unique(
                os.path.splitext(parent[0])[0] + row["ext"].lower(), taken)
            reason = "sidecar_follows_parent"
            stats["planned"] += 1
        stats["collisions"] += int(collided)
        assigned[row["id"]] = (dst, reason)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            dst, reason = assigned[row["id"]]
            writer.writerow({"id": row["id"], "src": row["path"], "dst": dst,
                             "kind": row["kind"], "reason": reason})
    return stats
```

Add `import os` at the top of the test file if it is not already imported.

- [ ] **Step 4: Run tests to verify all pass**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_organise.py -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_archive/organise.py tests/unit/test_photo_archive_organise.py
git commit -m "fix(photo_archive): collision-aware organise plan so the CSV states the true outcome"
```

---

### Task 12: Wire stage 2 — EXIF into the index

**Files:**
- Modify: `scripts/photo_archive/exif.py` (add `_norm`, `read_exif_into_index`)
- Modify: `scripts/photo_archive/cli.py` (wire the `exif` subcommand)
- Modify: `tests/unit/test_photo_archive_exif.py` (append 5 tests; keep all 7 existing)
- Modify: `tests/unit/test_photo_archive_cli.py` (append 1 test; keep all 4 existing)

**Interfaces:**
- Consumes: `ExifReader`, `pick_datetime`, `sanitise_model`, `paths.strip_long`, the `files` table
- Produces: `_norm(path: str) -> str`, `read_exif_into_index(conn, reader=None, exiftool="exiftool", batch_size=200) -> dict`

**Why:** `exif.py` exists but nothing loops the index through it, so `plan-organise` routes every file to `_UNDATED`. Until this lands, the organise half is built but starved of dates.

**Two traps this must handle:**
1. **exiftool reports `SourceFile` with FORWARD slashes and no `\\?\` prefix**, so a naive dict lookup by our stored Windows path silently misses every record and every photo comes back undated. `_norm` normalises both sides.
2. **Rows must be marked `exif_read` even when no date was found**, otherwise every run re-reads the genuinely undated files forever and the stage is never resumable.

`read_exif_into_index` accepts an injected `reader` so tests never shell out to exiftool.

- [ ] **Step 1: Write the failing tests** (append to `test_photo_archive_exif.py`)

```python
import pytest
from scripts.photo_archive import db


class _FakeReader:
    """Stands in for ExifReader so tests never spawn exiftool."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.closed = False
        self.batches = []

    def read_many(self, paths):
        self.batches.append(list(paths))
        return {p: self.mapping.get(p, {}) for p in paths}

    def close(self):
        self.closed = True


@pytest.fixture
def idx():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _add(conn, path, kind="image", ext=".jpg"):
    return db.upsert_file(conn, path=path, top_folder="t", rel_dir="",
                          filename=path.split("\\")[-1], ext=ext, size=1,
                          mtime=1.0, kind=kind)


def test_norm_matches_exiftool_forward_slash_reporting():
    # exiftool echoes SourceFile with forward slashes; our paths use backslashes.
    assert exif._norm(r"P:\a\b.jpg") == exif._norm("P:/a/b.jpg")
    assert exif._norm("\\\\?\\P:\\a\\b.jpg") == exif._norm("P:/a/b.jpg")


def test_read_exif_writes_date_and_model(idx):
    _add(idx, r"P:\a\b.jpg")
    reader = _FakeReader({r"P:\a\b.jpg": {"DateTimeOriginal": "2019:03:04 10:11:12",
                                          "Model": "X-T5"}})
    stats = exif.read_exif_into_index(idx, reader=reader)
    row = idx.execute("SELECT exif_dt, exif_dt_source, camera_model, state "
                      "FROM files").fetchone()
    assert row["exif_dt"] == "2019-03-04T10:11:12"
    assert row["exif_dt_source"] == "DateTimeOriginal"
    assert row["camera_model"] == "X-T5"
    assert row["state"] == "exif_read"
    assert stats["dated"] == 1


def test_undated_row_is_still_marked_read(idx):
    # Otherwise every run re-reads them forever and the stage never finishes.
    _add(idx, r"P:\a\b.jpg")
    stats = exif.read_exif_into_index(idx, reader=_FakeReader({}))
    row = idx.execute("SELECT exif_dt, state FROM files").fetchone()
    assert row["exif_dt"] is None
    assert row["state"] == "exif_read"
    assert stats["undated"] == 1


def test_second_run_reads_nothing(idx):
    _add(idx, r"P:\a\b.jpg")
    exif.read_exif_into_index(idx, reader=_FakeReader({}))
    stats = exif.read_exif_into_index(idx, reader=_FakeReader({}))
    assert stats["read"] == 0


def test_sidecars_and_other_files_are_not_read(idx):
    _add(idx, r"P:\a\b.xmp", kind="sidecar", ext=".xmp")
    _add(idx, r"P:\a\b.txt", kind="other", ext=".txt")
    stats = exif.read_exif_into_index(idx, reader=_FakeReader({}))
    assert stats["read"] == 0
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_exif.py -v`
Expected: 7 pass, 5 fail (`AttributeError: module ... has no attribute '_norm'`)

- [ ] **Step 3: Add `_norm` and `read_exif_into_index` to `exif.py`**

Add `from .paths import strip_long` to the imports, then append:

```python
def _norm(path: str) -> str:
    """exiftool echoes SourceFile with forward slashes and no \\?\\ prefix.

    Without normalising both sides, the lookup misses every record and every
    photo silently comes back undated.
    """
    return strip_long(path).replace("\\", "/").lower()


def read_exif_into_index(conn, reader=None, exiftool: str = "exiftool",
                         batch_size: int = 200) -> dict:
    """Batch capture date and camera model from disk into the index.

    Resumable: only rows still in state 'walked' are read, and every row
    touched is marked 'exif_read' whether or not a date was found.
    """
    stats = {"read": 0, "dated": 0, "undated": 0}
    rows = conn.execute(
        "SELECT id, path FROM files WHERE state = 'walked' "
        "AND kind IN ('image', 'raw', 'video') ORDER BY id").fetchall()
    if not rows:
        return stats

    owned = reader is None
    reader = reader or ExifReader(exiftool)
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            got = reader.read_many([r["path"] for r in batch])
            lookup = {_norm(k): v for k, v in got.items()}
            for row in batch:
                tags = lookup.get(_norm(row["path"]), {})
                dt, src = pick_datetime(tags)
                model = sanitise_model(tags.get("Model")) or None
                conn.execute(
                    "UPDATE files SET exif_dt = ?, exif_dt_source = ?, "
                    "camera_model = ?, state = 'exif_read' WHERE id = ?",
                    (dt, src, model, row["id"]))
                stats["read"] += 1
                stats["dated" if dt else "undated"] += 1
            conn.commit()
    finally:
        if owned:
            reader.close()
    return stats
```

- [ ] **Step 4: Wire the CLI**

In `cli.py`, replace the line `elif args.command == "hash":` block's preceding branch so that an `exif` branch exists. Specifically, insert immediately after the `walk` branch:

```python
    elif args.command == "exif":
        print(exif.read_exif_into_index(conn))
```

Append this test to `tests/unit/test_photo_archive_cli.py`:

```python
def test_cli_exif_stage_is_wired(tmp_path):
    # Empty index: returns 0 without ever constructing an ExifReader.
    rc = cli.main(["exif", "--db", str(tmp_path / "i.db"),
                   "--journal", str(tmp_path / "j.csv")])
    assert rc == 0
```

- [ ] **Step 5: Run the whole package suite**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_photo_archive_*.py -q`
Expected: PASS, 84 tests

- [ ] **Step 6: Commit**

```bash
git add scripts/photo_archive/exif.py scripts/photo_archive/cli.py \
        tests/unit/test_photo_archive_exif.py tests/unit/test_photo_archive_cli.py
git commit -m "feat(photo_archive): wire stage 2 EXIF read into the index"
```

---

## Known execution wrinkle — Windows Python reading WSL-hosted code

The code lives in the WSL filesystem (`/home/longboardfella/cortex_suite`) but
executes under Windows Python. Windows reaches it via the UNC path
`\\wsl$\Ubuntu\home\longboardfella\cortex_suite`, which works but is slow and
can be flaky under load. Tasks 1–10 are unaffected — they are unit-tested under
WSL. This only matters when actually running a stage against `P:`, at which
point the options are: run from the `\\wsl$` path, or deploy the
`photo_archive/` package to a Windows-local directory (e.g. `C:\photo_archive\`)
and run it there. Decide before the first rehearsal; do not decide it silently
mid-run.

## Deferred to a follow-up plan

These spec sections are deliberately **not** in this plan. Tier 1 dedupe plus organise is a complete, useful, shippable pipeline on its own; these add capability on top and each needs its own test cycle.

- **Stage 2 wiring** — `exif.py` is built and tested here, but the loop that batches paths from the index through `ExifReader` and writes `exif_dt`/`camera_model` back is a follow-up. Until then `plan-organise` routes everything to `_UNDATED`, which is correct-but-useless, so this is the **first** follow-up task.
- **Stage 4b — tiers 2 and 3.** Needs `PIL` and `imagehash` installed into `C:\Python311`. Reviewable reports only, never auto-applied.
- **Stage 6 — evacuation to `L:`.** Hash-verified copy, then source removal, resumable, halting on any mismatch or I/O error.
- **Stage 7 — provenance stamp.** `XMP-dc:Source` plus keywords for `MEANINGFUL_FOLDERS`; must write to the `.xmp` sidecar for `.raf`/`.nef`/`.cr2`, never into the RAW.

---

## Self-Review

**Spec coverage.** Scope and exclusions → Task 1. Index → Task 2. Stage 1 → Task 3. Stage 2 parsing → Task 4 (wiring deferred, noted above). Stage 3 → Task 5. Stage 4a → Task 6. Journal and undo → Task 7. Stage 5 → Task 8. Stage 8 → Task 9. CLI and rehearsal → Task 10. Stages 4b, 6 and 7 are explicitly deferred with reasons.

**Placeholder scan.** No TBD/TODO. Every code step carries real code; every test step carries real assertions.

**Type consistency.** `win_long`/`strip_long` (Task 1) are used unchanged in Tasks 3, 5, 7, 8, 9. `db.upsert_file(conn, **fields) -> int` (Task 2) is called with the same keyword names in Tasks 3, 6, 9. `safe_move(src, dst) -> str` (Task 8) is imported by Task 9. `Journal.append(stage, operation, src, dst, size, sha256, status)` (Task 7) is called with that arity in Tasks 8 and 9. `sanitise_model` (Task 4) is used by Task 9. `choose_keeper`/`tier1_groups` (Task 6) are used only within Task 6 and the CLI.

**Known gap — the organise plan CSV shows COLLIDING destinations** (found
2026-08-21 by testing against real 1994 export filenames). `plan_organise`
computes `dst` from EXIF alone, with no collision awareness; `safe_move`
suffixes `-2`/`-3` only at apply time. Three photos sharing a scan timestamp
therefore appear in `org.csv` as three rows with an IDENTICAL destination,
reading as though two files are about to be lost. **No data is actually at
risk** — `safe_move` never overwrites — but the CSV is the artifact the user
reviews and approves, so it currently misrepresents the outcome.

This is not a rare edge case in this archive: scanned pre-digital photos share
timestamps in bulk (1994: 36 exports over 15 distinct timestamps; 1996: 198 over
35), so it will affect hundreds of rows. **Fix before the first organise run** —
`plan_organise` should track assigned destinations in a set and apply the same
`-2`/`-3` suffixing that `safe_move` uses, so the plan states the true outcome.
Sidecars must receive the suffix that their parent received, not their own.

**Known gap — `undo --dry-run` under-reports on CHAINED moves** (found 2026-08-21
by adversarial test, not by the unit suite). When a file was moved twice
(`a`→`b` during quarantine, then `b`→`c` during evacuation), the dry run reports
`would_restore: 1, missing: 1`, whereas `--apply` correctly restores 2. Cause:
dry-run moves nothing, so when it reaches the earlier entry the intermediate
path does not exist yet and is counted `missing`. **Safety is unaffected** —
`--apply` unwinds the full chain correctly and content was verified intact. But
anyone reading the dry-run counts on a chained history will under-estimate what
undo will do. Fix would be to simulate the moves in memory during the dry run.
Do not "fix" this by making the dry run touch the filesystem.

**Known gap accepted:** `apply_organise` journals `size=0` and `sha256=""` because the organise plan CSV does not carry them, which weakens undo's hash verification for that stage to an existence check. Undo still refuses on a missing destination and still restores correctly. Tightening this means adding `size`/`sha256` to `PLAN_FIELDS`; deferred rather than left silent.
