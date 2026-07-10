#!/usr/bin/env python3
"""photo_batch.py — headless batch photo tagging + Lightroom catalog sync.

Drives the existing cortex_engine tagging (DocumentTextifier.keyword_image) and
reconciliation (cortex_engine.llm_metadata_sync) engines over a directory of
JPGs, so 5000+ photos can be processed without the Streamlit page's per-batch
upload ceiling.

See docs/superpowers/specs/2026-06-28-photo-batch-harness-design.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    # On WSL+OneDrive (9p drvfs) the atomic os.replace() over an existing file
    # intermittently fails with EPERM/EACCES because OneDrive's sync briefly
    # locks the target. Retry, then fall back to a direct in-place write — the
    # checkpoint is only resume state, so a non-atomic write is acceptable.
    for attempt in range(5):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    try:
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
    finally:
        if tmp.exists():
            try:
                os.unlink(tmp)
            except OSError:
                pass


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

    t = DocumentTextifier(use_vision=True)
    # Keyword extraction defaults to the first installed TEXT_MODELS entry, which
    # here is mistral-small3.2 (~23GB VRAM). Loaded per photo alongside LM Studio
    # it saturates the GPU (~44/46GB), stalling the whole machine and adding
    # 15-50s of mistral inference per photo. A small local model derives photo
    # keywords from the caption just as well in ~1-2s with ~10GB. (Vision itself
    # is the Claude Haiku API path, so no local VLM is loaded.)
    t.TEXT_MODELS = ["llama3.2:3b-instruct-q8_0", *t.TEXT_MODELS]
    return t.keyword_image(
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
    limit: int = 0,
) -> dict:
    """Tag every top-level JPG that needs it, with a resumable checkpoint.

    When limit > 0, stop after that many photos have actually been processed
    (tagged or failed) this run — already-good/already-done skips don't count,
    so successive runs march through the backlog in fixed-size batches.
    """
    to_tag_dir = Path(to_tag_dir)
    photos = sorted(
        set(to_tag_dir.glob("*.jpg")) | set(to_tag_dir.glob("*.JPG"))
        | set(to_tag_dir.glob("*.tif")) | set(to_tag_dir.glob("*.TIF"))
        | set(to_tag_dir.glob("*.tiff")) | set(to_tag_dir.glob("*.TIFF"))
    )
    checkpoint = load_checkpoint(to_tag_dir)
    total = len(photos)
    tagged = skipped = failed = processed = 0

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
        processed += 1
        if limit > 0 and processed >= limit:
            print(f"Reached batch limit ({limit} processed) — stopping. "
                  f"Re-run to continue from the checkpoint.")
            break
        if cooldown > 0 and i < total:
            time.sleep(cooldown)

    print(f"Tag complete: {tagged} tagged, {skipped} skipped, {failed} failed (of {total})")
    return {"tagged": tagged, "skipped": skipped, "failed": failed, "total": total}


def load_dotenv_keys() -> None:
    """Load project-root .env into os.environ for any keys not already set.

    The vision tagger prefers the Claude Haiku path when ANTHROPIC_API_KEY is
    present; without it the engine silently falls back to local VLMs that emit
    reasoning-scaffolding instead of captions. Loading .env here makes the
    headless run match the Streamlit app's behaviour. Values may be quoted.
    """
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main(argv=None) -> None:
    load_dotenv_keys()
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
    pt.add_argument("--limit", type=int, default=0,
                    help="Stop after N photos are processed this run (0 = no limit). "
                         "Skips don't count, so re-running marches through the backlog "
                         "in fixed-size batches.")
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
            limit=args.limit,
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
