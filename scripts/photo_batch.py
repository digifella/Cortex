#!/usr/bin/env python3
"""photo_batch.py — headless batch photo tagging + Lightroom catalog sync.

Drives the existing cortex_engine tagging (DocumentTextifier.keyword_image) and
reconciliation (cortex_engine.llm_metadata_sync) engines over a directory of
JPGs, so 5000+ photos can be processed without the Streamlit page's per-batch
upload ceiling.

See docs/superpowers/specs/2026-06-28-photo-batch-harness-design.md
"""
from __future__ import annotations

import json
import os
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
