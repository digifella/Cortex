# Photo Batch Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless CLI (`scripts/photo_batch.py`) that batch-tags 5000+ JPGs with the existing vision pipeline, then reconciles that metadata onto the Lightroom catalog originals — without the Streamlit page's per-batch upload ceiling.

**Architecture:** A single thin script with two subcommands (`tag`, `sync`) that drive the *existing, unchanged* engines `cortex_engine.textifier.DocumentTextifier.keyword_image` (tagging) and `cortex_engine.llm_metadata_sync` (reconciliation). Pure, testable helper functions (description classifier, checkpoint, sync config/scan) are unit-tested; the VLM and exiftool-writing paths are smoke-tested manually. Heavy engine imports are lazy (inside functions) so importing the module for tests stays cheap.

**Tech Stack:** Python 3.11 (cortex_suite venv), argparse, exiftool (`/usr/bin/exiftool`), local Ollama vision (`qwen3-vl:8b`), pytest.

## Global Constraints

- Run everything under the cortex_suite venv: `venv/bin/python`, `venv/bin/pytest`. Working dir = `/home/longboardfella/cortex_suite`.
- **No changes to engine code** — do not modify `cortex_engine/textifier.py` or anything under `cortex_engine/llm_metadata_sync/`. The harness only imports them.
- This is a `scripts/` utility: it does **not** require the page/app version-management workflow (no `version_config.py` change, no `version_manager.py --sync-all`).
- Tag phase processes **top-level** `*.jpg`/`*.JPG` only (matches `run_sync`'s glob). Never pass `clear_location=True` or `clear_keywords=True` to `keyword_image`.
- Default ownership notice (verbatim): `All rights (c) Longboardfella. Contact longboardfella.com for info on use of photos.`
- Default min description length: `40`. Refusal/meta prefixes are matched case-insensitively.
- Sync is **dry-run unless `--apply`** is given; `--apply` is the destructive step (renames originals to `.old`/`.bak`).
- Spec: `docs/superpowers/specs/2026-06-28-photo-batch-harness-design.md`.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Branch: `feature/photo-batch-harness` (already checked out).

---

## File Structure

- **Create** `scripts/photo_batch.py` — the whole harness (module-level constants + pure helpers + orchestrators + `main()`); built up across Tasks 1–5.
- **Create** `tests/unit/test_photo_batch.py` — unit tests for the pure helpers (Tasks 1–3).

Test import preamble (used by `tests/unit/test_photo_batch.py`):

```python
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts import photo_batch as pb
```

`scripts/` is an implicit namespace package (no `__init__.py` needed); `tests/conftest.py` already adds the project root to `sys.path`.

---

### Task 1: Module skeleton + description classifier

**Files:**
- Create: `scripts/photo_batch.py`
- Test: `tests/unit/test_photo_batch.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `description_is_bad(text: str | None, min_len: int = DEFAULT_MIN_DESC_LEN) -> bool`
  - Constants: `DEFAULT_OWNERSHIP: str`, `DEFAULT_MIN_DESC_LEN: int`, `CHECKPOINT_NAME: str`, `REFUSAL_PREFIXES: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_photo_batch.py`:

```python
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts import photo_batch as pb


def test_description_is_bad_empty():
    assert pb.description_is_bad("") is True
    assert pb.description_is_bad(None) is True
    assert pb.description_is_bad("   ") is True


def test_description_is_bad_placeholder():
    assert pb.description_is_bad("[Image: description timed out]") is True


def test_description_is_bad_too_short():
    assert pb.description_is_bad("A dog.") is True            # < 40 chars
    assert pb.description_is_bad("A dog.", min_len=3) is False


def test_description_is_bad_refusal_prefix_even_when_long():
    text = "I must give a thorough and complete description of this scene before I continue"
    assert pb.description_is_bad(text) is True


def test_description_is_good():
    good = ("A wooden sailboat moored at a stone jetty under an overcast sky, "
            "with green hills rising behind the harbour.")
    assert pb.description_is_bad(good) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -p no:cacheprovider`
Expected: collection/import error — `ModuleNotFoundError: No module named 'scripts.photo_batch'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/photo_batch.py`:

```python
#!/usr/bin/env python3
"""photo_batch.py — headless batch photo tagging + Lightroom catalog sync.

Drives the existing cortex_engine tagging (DocumentTextifier.keyword_image) and
reconciliation (cortex_engine.llm_metadata_sync) engines over a directory of
JPGs, so 5000+ photos can be processed without the Streamlit page's per-batch
upload ceiling.

See docs/superpowers/specs/2026-06-28-photo-batch-harness-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_OWNERSHIP = (
    "All rights (c) Longboardfella. Contact longboardfella.com for info on use of photos."
)
DEFAULT_MIN_DESC_LEN = 40
CHECKPOINT_NAME = ".photo_batch_tag.json"

# Lower-cased prefixes that signal a hallucinated / refusal / meta "description"
# rather than a real caption. Matched case-insensitively, independent of length.
REFUSAL_PREFIXES = (
    "i must",
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "sure,",
    "here is a description",
    "here's a description",
    "i will describe",
    "i'd be happy",
)


def description_is_bad(text, min_len: int = DEFAULT_MIN_DESC_LEN) -> bool:
    """Return True when an existing description should be regenerated.

    Bad = empty/whitespace, the engine's "[Image:" placeholder, a refusal/meta
    prefix, or shorter than min_len characters.
    """
    s = (text or "").strip()
    if not s:
        return True
    low = s.lower()
    if low.startswith("[image:"):
        return True
    if low.startswith(REFUSAL_PREFIXES):
        return True
    if len(s) < min_len:
        return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -p no:cacheprovider`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_batch.py tests/unit/test_photo_batch.py
git commit -m "feat(photo_batch): module skeleton + bad-description classifier

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Checkpoint helpers

**Files:**
- Modify: `scripts/photo_batch.py`
- Test: `tests/unit/test_photo_batch.py`

**Interfaces:**
- Consumes: `CHECKPOINT_NAME` (Task 1).
- Produces:
  - `file_key(path: Path) -> str` — `"{name}:{size}:{int(mtime)}"`
  - `checkpoint_path(to_tag_dir: Path) -> Path`
  - `load_checkpoint(to_tag_dir: Path) -> dict`
  - `save_checkpoint(to_tag_dir: Path, data: dict) -> None` (atomic)
  - `is_done(path: Path, checkpoint: dict) -> bool` — True when key present with status `tagged` or `skipped-good`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_photo_batch.py`:

```python
import os
import time


def test_file_key_changes_with_mtime(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    k1 = pb.file_key(f)
    future = time.time() + 100
    os.utime(f, (future, future))
    assert pb.file_key(f) != k1


def test_checkpoint_roundtrip_and_is_done(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"a")
    b = tmp_path / "b.jpg"
    b.write_bytes(b"b")
    cp = {pb.file_key(a): {"status": "tagged"}}
    pb.save_checkpoint(tmp_path, cp)
    loaded = pb.load_checkpoint(tmp_path)
    assert pb.is_done(a, loaded) is True
    assert pb.is_done(b, loaded) is False


def test_is_done_false_after_file_changes(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"a")
    cp = {pb.file_key(a): {"status": "tagged"}}
    a.write_bytes(b"aa")  # size changes -> key changes
    assert pb.is_done(a, cp) is False


def test_load_checkpoint_missing_returns_empty(tmp_path):
    assert pb.load_checkpoint(tmp_path) == {}


def test_skipped_good_counts_as_done(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"a")
    cp = {pb.file_key(a): {"status": "skipped-good"}}
    assert pb.is_done(a, cp) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -k "checkpoint or file_key or is_done or skipped_good" -p no:cacheprovider`
Expected: FAIL — `AttributeError: module 'scripts.photo_batch' has no attribute 'file_key'`.

- [ ] **Step 3: Write minimal implementation**

Add `import json` and `import os` to the imports block of `scripts/photo_batch.py` (top, alongside `import sys`), then append these functions after `description_is_bad`:

```python
def file_key(path: Path) -> str:
    """Identity key for resume: name + size + integer mtime."""
    st = path.stat()
    return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"


def checkpoint_path(to_tag_dir: Path) -> Path:
    return Path(to_tag_dir) / CHECKPOINT_NAME


def load_checkpoint(to_tag_dir: Path) -> dict:
    p = checkpoint_path(to_tag_dir)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_checkpoint(to_tag_dir: Path, data: dict) -> None:
    p = checkpoint_path(to_tag_dir)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, p)


def is_done(path: Path, checkpoint: dict) -> bool:
    entry = checkpoint.get(file_key(path))
    return bool(entry) and entry.get("status") in ("tagged", "skipped-good")
```

The final imports block at the top of the file should read:

```python
import json
import os
import sys
from pathlib import Path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -p no:cacheprovider`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_batch.py tests/unit/test_photo_batch.py
git commit -m "feat(photo_batch): resumable checkpoint helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Sync config, scan, and orchestrator

**Files:**
- Modify: `scripts/photo_batch.py`
- Test: `tests/unit/test_photo_batch.py`

**Interfaces:**
- Consumes: `cortex_engine.llm_metadata_sync.models.SyncConfig`, `.matcher.build_raw_index`, `.matcher.resolve_jpg`, `.sync.run_sync` (all lazy-imported, unchanged).
- Produces:
  - `build_sync_config(raw_root, jpg_dir, *, dry_run, keep_backups=True, filter_keywords=None, timestamp_tolerance=0) -> SyncConfig`
  - `scan_actions(cfg) -> tuple[list, list]` — `(actions, orphaned_jpgs)`
  - `sync_photos(to_tag_dir, raw_root, *, apply, keep_backups=True, filter_keywords=None, timestamp_tolerance=0) -> dict`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_photo_batch.py`:

```python
def test_build_sync_config_dry_run_defaults():
    cfg = pb.build_sync_config("/raw", "/jpg", dry_run=True)
    assert cfg.dry_run is True
    assert cfg.keep_backups is True
    assert cfg.filter_keywords == ["nogps"]
    assert cfg.timestamp_tolerance_seconds == 0
    assert str(cfg.raw_root) == "/raw"
    assert str(cfg.jpg_dir) == "/jpg"


def test_build_sync_config_flags_passthrough():
    cfg = pb.build_sync_config(
        "/raw", "/jpg",
        dry_run=False, keep_backups=False,
        filter_keywords=["x", "y"], timestamp_tolerance=4,
    )
    assert cfg.dry_run is False
    assert cfg.keep_backups is False
    assert cfg.filter_keywords == ["x", "y"]
    assert cfg.timestamp_tolerance_seconds == 4


def test_scan_actions_matches_raw_by_stem(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    jpgdir = tmp_path / "jpg"
    jpgdir.mkdir()
    (raw / "2020-01-01 10-00-00-X-T1.NEF").write_bytes(b"raw")
    (jpgdir / "2020-01-01 10-00-00-X-T1.jpg").write_bytes(b"jpg")

    cfg = pb.build_sync_config(raw, jpgdir, dry_run=True)
    actions, orphaned = pb.scan_actions(cfg)

    assert len(actions) == 1
    assert actions[0].target_path.name == "2020-01-01 10-00-00-X-T1.xmp"
    assert orphaned == []
    # scanning is read-only — no sidecar is created
    assert not (raw / "2020-01-01 10-00-00-X-T1.xmp").exists()


def test_scan_actions_reports_orphan(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    jpgdir = tmp_path / "jpg"
    jpgdir.mkdir()
    (jpgdir / "loner.jpg").write_bytes(b"jpg")

    cfg = pb.build_sync_config(raw, jpgdir, dry_run=True)
    actions, orphaned = pb.scan_actions(cfg)

    assert actions == []
    assert [p.name for p in orphaned] == ["loner.jpg"]


def test_sync_photos_dry_run_writes_nothing(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    jpgdir = tmp_path / "jpg"
    jpgdir.mkdir()
    (raw / "2020-01-01 10-00-00-X-T1.NEF").write_bytes(b"raw")
    (jpgdir / "2020-01-01 10-00-00-X-T1.jpg").write_bytes(b"jpg")

    summary = pb.sync_photos(jpgdir, raw, apply=False)

    assert summary["applied"] is False
    assert summary["actions"] == 1
    assert not (raw / "2020-01-01 10-00-00-X-T1.xmp").exists()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -k "sync_config or scan_actions or sync_photos" -p no:cacheprovider`
Expected: FAIL — `AttributeError: module 'scripts.photo_batch' has no attribute 'build_sync_config'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/photo_batch.py`:

```python
def build_sync_config(
    raw_root,
    jpg_dir,
    *,
    dry_run: bool,
    keep_backups: bool = True,
    filter_keywords=None,
    timestamp_tolerance: int = 0,
):
    """Build a SyncConfig with the same defaults the Streamlit page uses."""
    from cortex_engine.llm_metadata_sync.models import SyncConfig

    return SyncConfig(
        raw_root=Path(raw_root),
        jpg_dir=Path(jpg_dir),
        filter_keywords=list(filter_keywords) if filter_keywords is not None else ["nogps"],
        keep_backups=keep_backups,
        timestamp_tolerance_seconds=timestamp_tolerance,
        dry_run=dry_run,
    )


def scan_actions(cfg):
    """Resolve every top-level JPG in cfg.jpg_dir against cfg.raw_root.

    Returns (actions, orphaned_jpgs). Read-only — builds the index and resolves
    matches, writes nothing.
    """
    from cortex_engine.llm_metadata_sync.matcher import build_raw_index, resolve_jpg

    index = build_raw_index(cfg.raw_root, cfg)
    jpgs = sorted(list(cfg.jpg_dir.glob("*.jpg")) + list(cfg.jpg_dir.glob("*.JPG")))
    actions = []
    orphaned = []
    for jpg in jpgs:
        resolved = resolve_jpg(jpg, index, cfg)
        if resolved:
            actions.extend(resolved)
        else:
            orphaned.append(jpg)
    return actions, orphaned


def sync_photos(
    to_tag_dir,
    raw_root,
    *,
    apply: bool,
    keep_backups: bool = True,
    filter_keywords=None,
    timestamp_tolerance: int = 0,
) -> dict:
    """Dry-run scan (always), then live reconciliation when apply=True."""
    cfg = build_sync_config(
        raw_root,
        to_tag_dir,
        dry_run=not apply,
        keep_backups=keep_backups,
        filter_keywords=filter_keywords,
        timestamp_tolerance=timestamp_tolerance,
    )
    actions, orphaned = scan_actions(cfg)
    matched_jpgs = len({a.jpg_path for a in actions})
    print(f"Scan: {len(actions)} action(s) across {matched_jpgs} matched JPG(s); "
          f"{len(orphaned)} orphaned")
    for a in actions:
        print(f"  {a.jpg_path.name} -> {a.target_path.name} "
              f"[{a.target_type.value}/{a.sidecar_action.value}]")
    if orphaned:
        print(f"Orphaned (no RAW/derivative match): {len(orphaned)}")
        for p in orphaned:
            print(f"  {p.name}")

    if not apply:
        print("DRY RUN — no changes written. Re-run with --apply to perform the sync.")
        return {"actions": len(actions), "orphaned": len(orphaned), "applied": False}

    if not actions:
        print("No actions to apply.")
        return {"actions": 0, "orphaned": len(orphaned), "applied": True,
                "succeeded": 0, "failed": 0}

    from cortex_engine.llm_metadata_sync.sync import run_sync

    ok = fail = kw = desc = loc = 0
    for i, res in enumerate(run_sync(cfg), start=1):
        if res.success:
            ok += 1
            kw += res.keywords_written
            loc += res.location_written
            if res.description_written:
                desc += 1
            print(f"[{i}] OK {res.action.jpg_path.name} -> {res.action.target_path.name}")
        else:
            fail += 1
            print(f"[{i}] FAIL {res.action.jpg_path.name} -> "
                  f"{res.action.target_path.name}: {res.error}")
    print(f"Sync complete: {ok} succeeded, {fail} failed; "
          f"{kw} keywords, {desc} descriptions, {loc} location fields written")
    return {"actions": len(actions), "orphaned": len(orphaned), "applied": True,
            "succeeded": ok, "failed": fail}
```

Note: `run_sync` rebuilds the index internally, so an `--apply` run walks `raw_root` twice (once in `scan_actions`, once in `run_sync`). This mirrors the Streamlit page (Scan then Apply) and keeps the harness a thin wrapper; acceptable for this use.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -p no:cacheprovider`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_batch.py tests/unit/test_photo_batch.py
git commit -m "feat(photo_batch): sync config, scan, and dry-run/apply orchestrator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Tag orchestrator (VLM path)

**Files:**
- Modify: `scripts/photo_batch.py`

**Interfaces:**
- Consumes: `cortex_engine.textifier.DocumentTextifier` (lazy import, unchanged); `description_is_bad`, `file_key`, `load_checkpoint`, `save_checkpoint`, `is_done` (Tasks 1–2).
- Produces:
  - `read_existing_description(path: Path) -> str`
  - `tag_one(path: Path, ownership_notice: str) -> dict`
  - `tag_photos(to_tag_dir, *, min_desc_len=DEFAULT_MIN_DESC_LEN, redescribe_all=False, ownership_notice=DEFAULT_OWNERSHIP, cooldown=0.0) -> dict`

This task's write/VLM paths require Ollama + exiftool and are verified by the manual smoke test in Step 3, not by unit tests (the pure decision logic they use is already covered in Tasks 1–2).

- [ ] **Step 1: Add `import time` and implement the functions**

Add `import time` to the top imports block (so it reads `import json` / `import os` / `import sys` / `import time` / `from pathlib import Path`). Append to `scripts/photo_batch.py`:

```python
def read_existing_description(path: Path) -> str:
    """Read the current caption from a photo, first non-empty of the three
    standard fields. Returns "" if exiftool is unavailable or none is set."""
    import shutil
    import subprocess

    exiftool = shutil.which("exiftool")
    if not exiftool:
        return ""
    try:
        result = subprocess.run(
            [exiftool, "-json",
             "-XMP-dc:Description", "-IPTC:Caption-Abstract", "-EXIF:ImageDescription",
             str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return ""
        rows = json.loads(result.stdout)
        if not rows:
            return ""
        row = rows[0]
        for field in ("Description", "Caption-Abstract", "ImageDescription"):
            val = (row.get(field) or "").strip()
            if val:
                return val
        return ""
    except Exception:
        return ""


def tag_one(path: Path, ownership_notice: str) -> dict:
    """Run the full vision tag on a single photo, in place.

    generate_description=True overwrites the (bad/missing) caption; location is
    fill-missing-only (clear_location stays False); keywords merge (+=).
    """
    from cortex_engine.textifier import DocumentTextifier

    return DocumentTextifier(use_vision=True).keyword_image(
        str(path),
        generate_description=True,
        populate_location=True,
        clear_location=False,
        clear_keywords=False,
        anonymize_keywords=False,
        ownership_notice=ownership_notice,
    )


def tag_photos(
    to_tag_dir,
    *,
    min_desc_len: int = DEFAULT_MIN_DESC_LEN,
    redescribe_all: bool = False,
    ownership_notice: str = DEFAULT_OWNERSHIP,
    cooldown: float = 0.0,
) -> dict:
    """Tag every top-level JPG that needs it, with a resumable checkpoint."""
    to_tag_dir = Path(to_tag_dir)
    photos = sorted(list(to_tag_dir.glob("*.jpg")) + list(to_tag_dir.glob("*.JPG")))
    checkpoint = load_checkpoint(to_tag_dir)
    total = len(photos)
    tagged = skipped = failed = 0

    for i, path in enumerate(photos, start=1):
        if is_done(path, checkpoint):
            skipped += 1
            print(f"[{i}/{total}] SKIP {path.name} (checkpoint)")
            continue

        existing = read_existing_description(path)
        if not redescribe_all and not description_is_bad(existing, min_desc_len):
            checkpoint[file_key(path)] = {
                "status": "skipped-good",
                "description": existing[:120],
            }
            save_checkpoint(to_tag_dir, checkpoint)
            skipped += 1
            print(f"[{i}/{total}] SKIP {path.name} (good description)")
            continue

        try:
            result = tag_one(path, ownership_notice)
            description = (result.get("description") or "")
            # file_key recomputed AFTER the in-place write so the checkpoint key
            # matches the file's new size/mtime (enables fast-skip next run).
            checkpoint[file_key(path)] = {
                "status": "tagged",
                "description": description[:120],
                "keywords": len(result.get("new_keywords") or []),
            }
            tagged += 1
            print(f"[{i}/{total}] TAGGED {path.name}: {description[:120]}")
        except Exception as exc:
            checkpoint[file_key(path)] = {"status": "failed", "error": str(exc)}
            failed += 1
            print(f"[{i}/{total}] FAIL {path.name}: {exc}")

        save_checkpoint(to_tag_dir, checkpoint)
        if cooldown > 0 and i < total:
            time.sleep(cooldown)

    print(f"Tag complete: {tagged} tagged, {skipped} skipped, {failed} failed (of {total})")
    return {"tagged": tagged, "skipped": skipped, "failed": failed, "total": total}
```

- [ ] **Step 2: Verify the module still imports and unit tests still pass**

Run: `venv/bin/python -c "from scripts import photo_batch; print('ok', photo_batch.tag_photos.__name__)"`
Expected: `ok tag_photos` (no import errors; heavy deps are lazy).

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -p no:cacheprovider`
Expected: 15 passed (unchanged — no new unit tests this task).

- [ ] **Step 3: Manual smoke test (requires Ollama running)**

Ensure Ollama is up: `sudo systemctl start ollama`. Pick ~3 real JPGs (at least one with no/short description) into a scratch dir, e.g. `/tmp/claude-1000/-home-longboardfella/d2496cbe-b521-4f9e-a0ad-f4a0d339f29d/scratchpad/smoke_tag/`, then:

Run: `venv/bin/python scripts/photo_batch.py tag <smoke_dir>` *(the CLI lands in Task 5; until then test the function directly:)*
`venv/bin/python -c "from scripts import photo_batch as pb; print(pb.tag_photos('<smoke_dir>'))"`
Expected: per-photo `TAGGED`/`SKIP` lines, a `.photo_batch_tag.json` written in `<smoke_dir>`, and a summary dict. Verify with `exiftool -XMP-dc:Description -XMP-dc:Subject <one.jpg>` that a caption and keywords were written. Re-run the same command and confirm the photos now report `SKIP (checkpoint)` / `SKIP (good description)`.

- [ ] **Step 4: Commit**

```bash
git add scripts/photo_batch.py
git commit -m "feat(photo_batch): resumable tag orchestrator with description gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: CLI (`main`) + executable wiring

**Files:**
- Modify: `scripts/photo_batch.py`

**Interfaces:**
- Consumes: `tag_photos` (Task 4), `sync_photos` (Task 3), constants (Task 1).
- Produces: `main(argv=None) -> None` and the `if __name__ == "__main__"` guard.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_photo_batch.py`:

```python
def test_main_tag_invokes_tag_photos(tmp_path, monkeypatch):
    calls = {}

    def fake_tag_photos(to_tag_dir, **kwargs):
        calls["dir"] = to_tag_dir
        calls["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(pb, "tag_photos", fake_tag_photos)
    pb.main(["tag", str(tmp_path), "--min-desc-len", "10", "--redescribe-all",
             "--cooldown", "1.5", "--no-ownership"])

    assert str(calls["dir"]) == str(tmp_path)
    assert calls["kwargs"]["min_desc_len"] == 10
    assert calls["kwargs"]["redescribe_all"] is True
    assert calls["kwargs"]["cooldown"] == 1.5
    assert calls["kwargs"]["ownership_notice"] == ""


def test_main_sync_defaults_to_dry_run(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    calls = {}

    def fake_sync_photos(to_tag_dir, raw_root, **kwargs):
        calls["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(pb, "sync_photos", fake_sync_photos)
    pb.main(["sync", str(tmp_path), str(raw)])

    assert calls["kwargs"]["apply"] is False
    assert calls["kwargs"]["keep_backups"] is True
    assert calls["kwargs"]["filter_keywords"] == ["nogps"]
    assert calls["kwargs"]["timestamp_tolerance"] == 0


def test_main_sync_apply_and_flags(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    calls = {}
    monkeypatch.setattr(pb, "sync_photos",
                        lambda d, r, **k: calls.update(k) or {})
    pb.main(["sync", str(tmp_path), str(raw), "--apply", "--no-backups",
             "--filter-keywords", "nogps,private", "--timestamp-tolerance", "4"])

    assert calls["apply"] is True
    assert calls["keep_backups"] is False
    assert calls["filter_keywords"] == ["nogps", "private"]
    assert calls["timestamp_tolerance"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -k "main_tag or main_sync" -p no:cacheprovider`
Expected: FAIL — `AttributeError: module 'scripts.photo_batch' has no attribute 'main'`.

- [ ] **Step 3: Implement `main` and the guard**

Add `import argparse` to the top imports block, then append to `scripts/photo_batch.py`:

```python
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="photo_batch",
        description="Headless batch photo tagging + Lightroom catalog sync.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("tag", help="VLM-tag photos in a directory, in place.")
    pt.add_argument("to_tag_dir", type=Path)
    pt.add_argument("--min-desc-len", type=int, default=DEFAULT_MIN_DESC_LEN,
                    help="Existing descriptions shorter than this are regenerated.")
    pt.add_argument("--redescribe-all", action="store_true",
                    help="Regenerate every description regardless of current content.")
    pt.add_argument("--cooldown", type=float, default=0.0,
                    help="Seconds to pause between photos.")
    pt.add_argument("--ownership", default=DEFAULT_OWNERSHIP,
                    help="Ownership/copyright notice to embed.")
    pt.add_argument("--no-ownership", action="store_true",
                    help="Do not write ownership metadata.")

    ps = sub.add_parser("sync", help="Reconcile tagged JPG metadata onto catalog originals.")
    ps.add_argument("to_tag_dir", type=Path)
    ps.add_argument("raw_root", type=Path)
    ps.add_argument("--apply", action="store_true",
                    help="Perform the destructive sync (default is dry-run).")
    ps.add_argument("--no-backups", action="store_true",
                    help="Do not keep .old/.bak backups of modified originals.")
    ps.add_argument("--filter-keywords", default="nogps",
                    help="Comma-separated keywords to drop during sync.")
    ps.add_argument("--timestamp-tolerance", type=int, default=0,
                    help="Allow JPG/RAW capture times to differ by up to N seconds.")

    args = parser.parse_args(argv)

    if args.command == "tag":
        if not args.to_tag_dir.is_dir():
            parser.error(f"Not a directory: {args.to_tag_dir}")
        ownership = "" if args.no_ownership else args.ownership
        tag_photos(
            args.to_tag_dir,
            min_desc_len=args.min_desc_len,
            redescribe_all=args.redescribe_all,
            ownership_notice=ownership,
            cooldown=args.cooldown,
        )
    elif args.command == "sync":
        if not args.to_tag_dir.is_dir():
            parser.error(f"Not a directory: {args.to_tag_dir}")
        if not args.raw_root.is_dir():
            parser.error(f"Not a directory: {args.raw_root}")
        filter_keywords = [k.strip() for k in args.filter_keywords.split(",") if k.strip()]
        sync_photos(
            args.to_tag_dir,
            args.raw_root,
            apply=args.apply,
            keep_backups=not args.no_backups,
            filter_keywords=filter_keywords,
            timestamp_tolerance=args.timestamp_tolerance,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/unit/test_photo_batch.py -p no:cacheprovider`
Expected: 18 passed.

- [ ] **Step 5: Verify the CLI help renders for both subcommands**

Run: `venv/bin/python scripts/photo_batch.py tag --help`
Expected: usage text listing `--min-desc-len`, `--redescribe-all`, `--cooldown`, `--ownership`, `--no-ownership`.

Run: `venv/bin/python scripts/photo_batch.py sync --help`
Expected: usage text listing `--apply`, `--no-backups`, `--filter-keywords`, `--timestamp-tolerance`.

- [ ] **Step 6: End-to-end smoke (dry-run sync, no GPU needed)**

Build a synthetic pair and confirm the CLI dry-run reports a match and writes nothing:

```bash
SMOKE=/tmp/claude-1000/-home-longboardfella/d2496cbe-b521-4f9e-a0ad-f4a0d339f29d/scratchpad/smoke_sync
mkdir -p "$SMOKE/raw" "$SMOKE/jpg"
: > "$SMOKE/raw/2020-01-01 10-00-00-X-T1.NEF"
: > "$SMOKE/jpg/2020-01-01 10-00-00-X-T1.jpg"
venv/bin/python scripts/photo_batch.py sync "$SMOKE/jpg" "$SMOKE/raw"
```
Expected: a scan line showing `1 action(s) across 1 matched JPG(s); 0 orphaned`, a `-> ...xmp [sidecar/create]` line, and `DRY RUN — no changes written.`; confirm no `.xmp` was created under `$SMOKE/raw`.

- [ ] **Step 7: Commit**

```bash
git add scripts/photo_batch.py tests/unit/test_photo_batch.py
git commit -m "feat(photo_batch): argparse CLI with tag and sync subcommands

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Operator runbook (after implementation)

1. `sudo systemctl start ollama`
2. **Tag:** `venv/bin/python scripts/photo_batch.py tag "<to_tag_dir>"` (resumable; re-run after any interruption). Multi-hour for 5000 photos.
3. **Review:** spot-check captions/keywords; inspect `<to_tag_dir>/.photo_batch_tag.json`.
4. **Dry-run sync:** `venv/bin/python scripts/photo_batch.py sync "<to_tag_dir>" "<catalog_raw_root>"` — review matched/orphaned counts.
5. **Apply:** `venv/bin/python scripts/photo_batch.py sync "<to_tag_dir>" "<catalog_raw_root>" --apply` — creates `.old`/`.bak` backups, writes XMP sidecars / embedded / replaces catalog JPGs.
6. Re-import metadata in Lightroom Classic (Metadata → Read Metadata from File).

---

## Self-Review

**Spec coverage:**
- Two-phase tag→review→sync — Tasks 4 (tag), 5 (CLI), 3 (sync); runbook documents the review checkpoint. ✓
- Top-level glob both phases — `tag_photos` and `scan_actions` both glob `*.jpg`/`*.JPG`. ✓
- Tag options description+keywords / location+GPS / ownership, no anonymize/resize — `tag_one` sets exactly those flags. ✓
- Bad-description gate (empty / `[Image:` / too-short / refusal-prefix) + `--min-desc-len` + `--redescribe-all` — Task 1 classifier + Task 4 wiring + Task 5 flags; tested. ✓
- Location fill-missing-only, never overwrite — `clear_location=False`, documented; engine default. ✓
- Resumable checkpoint (name+size+mtime, atomic, statuses) — Task 2 + Task 4. ✓
- Sync dry-run by default, `--apply` destructive, UI-matching SyncConfig defaults, flags `--no-backups`/`--filter-keywords`/`--timestamp-tolerance` — Task 3 + Task 5; tested. ✓
- No engine-code changes — only `scripts/` + `tests/` touched. ✓
- Testing: classifier, checkpoint resume, sync dry-run wiring, arg/config mapping — Tasks 1,2,3,5; manual smoke for VLM path — Task 4/5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has expected output. ✓

**Type consistency:** `description_is_bad`, `file_key`, `is_done`, `load_checkpoint`, `save_checkpoint`, `build_sync_config`, `scan_actions`, `sync_photos`, `read_existing_description`, `tag_one`, `tag_photos`, `main` — names and signatures used in later tasks match their definitions. `keyword_image` returns `description`/`new_keywords` (used in Task 4); `SyncResult` exposes `success`/`keywords_written`/`location_written`/`description_written`/`error`/`action` (used in Task 3) — verified against engine source. ✓
