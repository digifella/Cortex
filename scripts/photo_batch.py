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

# Lightroom Classic "Automatically write changes into XMP". Drives the reminders a
# sync prints. Paul set it OFF (2026-09-27): catalog edits reach the files only on
# Ctrl+S, so Ctrl+S must happen BEFORE a sync and never after it. Update this if
# he turns it back on.
LRC_AUTO_WRITE_XMP = False

# Pixel-verification thresholds for export -> master pairs (a re-export can
# renumber names so the same filename is a different photo; 36/382 were on
# 2026-08-25). Genuine pairs measured: dhash <= 2, corr >= 0.995.
PIXEL_MAX_DHASH = 6
PIXEL_MIN_CORR = 0.9

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


def select_exports(to_tag_dir, dates=None, files=None) -> list[Path]:
    """The export JPGs a surgical sync should touch.

    dates: 'YYYY-MM-DD' prefixes matched against file names in to_tag_dir (export
    names start with the capture date). files: explicit paths, absolute or relative
    to to_tag_dir, which may live in other folders (3 Star / 4+ Star).
    """
    to_tag_dir = Path(to_tag_dir)
    chosen: list[Path] = []
    for d in dates or []:
        chosen += sorted(p for p in to_tag_dir.iterdir()
                         if p.suffix.lower() == ".jpg" and p.name.startswith(d))
    for f in files or []:
        p = Path(f)
        p = p if p.is_absolute() else to_tag_dir / p
        if not p.is_file():
            raise FileNotFoundError(f"--files entry not found: {p}")
        chosen.append(p)
    seen, out = set(), []
    for p in chosen:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def stage_exports(paths, stage_dir) -> Path:
    """Symlink the chosen exports into stage_dir. run_sync always globs a whole
    directory, so staging is how a sync is limited to exactly these photos."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        link = stage_dir / p.name
        if link.exists() or link.is_symlink():
            raise ValueError(f"two selected exports share the name {p.name}; "
                             "sync them in separate runs")
        link.symlink_to(Path(p).resolve())
    return stage_dir


def _load_gray(path: Path, raw_preview: bool = False):
    """Small greyscale PIL image of a photo; for a raw, its embedded JPEG preview."""
    import io
    import subprocess

    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = None
    if raw_preview:
        data = b""
        for tag in ("-JpgFromRaw", "-PreviewImage"):
            data = subprocess.run(["exiftool", "-b", tag, str(path)],
                                  capture_output=True, timeout=120).stdout
            if data:
                break
        if not data:
            raise ValueError("no embedded preview in raw")
        im = Image.open(io.BytesIO(data))
    else:
        im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    im.draft("L", (256, 256))
    return im.convert("L")


def pixel_match(a, b) -> tuple[int, float]:
    """(dhash distance out of 64, 64x64 pixel correlation) for two greyscale images."""
    import numpy as np

    def dh(im):
        x = np.asarray(im.resize((9, 8)), dtype=float)
        return (x[:, 1:] > x[:, :-1]).flatten()

    x = np.asarray(a.resize((64, 64)), dtype=float).ravel()
    y = np.asarray(b.resize((64, 64)), dtype=float).ravel()
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else 0.0
    return int((dh(a) != dh(b)).sum()), corr


def verify_pairs(actions) -> list[str]:
    """Pixel-check every export -> master pair. Returns one problem line per pair
    that fails or cannot be checked (empty list = all verified)."""
    from cortex_engine.llm_metadata_sync.models import TargetType

    problems = []
    for a in actions:
        try:
            if a.target_type == TargetType.SIDECAR:
                if not a.raw_path:
                    raise ValueError("sidecar target without a raw file")
                master = _load_gray(a.raw_path, raw_preview=True)
            else:
                master = _load_gray(a.target_path)
            d, r = pixel_match(_load_gray(a.jpg_path), master)
            if d > PIXEL_MAX_DHASH or r < PIXEL_MIN_CORR:
                problems.append(f"{a.jpg_path.name} -> {a.target_path.name}: "
                                f"DIFFERENT IMAGE? dhash {d}/64, corr {r:.3f}")
        except Exception as exc:  # unreadable = unverified = refused
            problems.append(f"{a.jpg_path.name} -> {a.target_path.name}: "
                            f"could not verify ({exc})")
    return problems


def read_rating_label(paths) -> dict:
    """{path: (Rating, Label)} from XMP; missing files and fields read as None."""
    import subprocess
    import tempfile

    paths = [str(p) for p in paths if Path(p).exists()]
    if not paths:
        return {}
    # -@ argfile: a year-sized run would overflow the command line.
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False) as fh:
        fh.write("\n".join(paths) + "\n")
    try:
        out = subprocess.run(["exiftool", "-j", "-XMP-xmp:Rating", "-XMP-xmp:Label",
                              "-@", fh.name], capture_output=True, text=True,
                             timeout=120 + 2 * len(paths)).stdout
    finally:
        os.unlink(fh.name)
    return {r["SourceFile"]: (r.get("Rating"), r.get("Label"))
            for r in json.loads(out or "[]")}


def lightroom_before_apply_note() -> str:
    if LRC_AUTO_WRITE_XMP:
        return ("Lightroom: auto-write XMP is ON, so Lightroom also writes these masters. "
                "Close Lightroom (or stay out of these folders) until the apply finishes.")
    return ("Lightroom: auto-write XMP is OFF. If you changed metadata on these photos in "
            "Lightroom since your last save, select them and press Ctrl+S (Save Metadata "
            "to File) BEFORE --apply, or those edits are lost when you Read Metadata.")


def lightroom_after_apply_note(actions, raw_root) -> str:
    """Where to Read Metadata in Lightroom, and what not to do now."""
    import collections

    raw_root = Path(raw_root)
    by_folder = collections.defaultdict(list)
    for a in actions:
        master = a.raw_path if a.raw_path else a.target_path
        try:
            folder = str(master.parent.relative_to(raw_root.parent))
        except ValueError:
            folder = str(master.parent)
        by_folder[folder].append(a.jpg_path.name)
    lines = ["", "── Lightroom next step " + "─" * 40]
    for folder, names in sorted(by_folder.items()):
        days = collections.Counter(n[:10] for n in names)
        times = sorted(n[11:16].replace("-", ":") for n in names)
        span = f", {times[0]}–{times[-1]}" if len(days) == 1 else ""
        lines.append(f"  {folder}: {len(names)} photo(s) — "
                     + ", ".join(f"{d} ×{c}" for d, c in sorted(days.items())) + span)
    lines.append("  Select them → Metadata › Read Metadata from Files.")
    if LRC_AUTO_WRITE_XMP:
        lines.append("  Auto-write XMP is ON: Lightroom picks up the files' metadata when "
                     "you read it; keep out of these folders until then.")
    else:
        lines.append("  Auto-write XMP is OFF: do NOT Save Metadata to File (Ctrl+S) on these "
                     "before reading — that writes the catalog's old metadata over this sync.")
    return "\n".join(lines)


def sync_photos(
    to_tag_dir,
    raw_root,
    *,
    apply: bool,
    keep_backups: bool = True,
    filter_keywords=None,
    timestamp_tolerance: int = 0,
    dates=None,
    files=None,
    verify_pixels=None,
    allow_rating_change: bool = False,
) -> dict:
    """Dry-run scan (always), then live reconciliation when apply=True.

    dates/files make it SURGICAL: only those exports are synced (staged as symlinks,
    since run_sync globs a whole directory), and pixel verification defaults on.
    """
    import tempfile

    surgical = bool(dates or files)
    if verify_pixels is None:
        verify_pixels = surgical
    if surgical:
        chosen = select_exports(to_tag_dir, dates=dates, files=files)
        if not chosen:
            print("No exports match --date/--files; nothing to do.")
            return {"actions": 0, "orphaned": 0, "applied": False}
        to_tag_dir = stage_exports(chosen, tempfile.mkdtemp(prefix="photo_sync_stage_"))
        print(f"Surgical sync: {len(chosen)} export(s) staged in {to_tag_dir}")
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

    from cortex_engine.llm_metadata_sync.models import TargetType

    blocked = False
    if verify_pixels and actions:
        problems = verify_pairs(actions)
        print(f"Pixel check: {len(actions) - len(problems)}/{len(actions)} pair(s) verified")
        for line in problems:
            print(f"  ✗ {line}")
        blocked |= bool(problems)

    # The sync copies the export's Rating onto the master (a JPG replace brings its
    # Label too), so predict every master's post-sync Rating/Label and surface any
    # rating it would change before anything is written.
    exp_rl = read_rating_label([a.jpg_path.resolve() for a in actions])
    before = read_rating_label([a.target_path for a in actions])
    expected, rating_changes = {}, []
    for a in actions:
        er, el = exp_rl.get(str(a.jpg_path.resolve()), (None, None))
        br, bl = before.get(str(a.target_path), (None, None))
        if a.target_type == TargetType.JPG_REPLACE:
            expected[str(a.target_path)] = (er, el)
        else:
            expected[str(a.target_path)] = (er if er is not None else br, bl)
        if str(a.target_path) in before and expected[str(a.target_path)][0] != br:
            rating_changes.append(f"{a.target_path.name}: rating {br} -> "
                                  f"{expected[str(a.target_path)][0]}")
    if rating_changes:
        print(f"Rating changes this sync would make: {len(rating_changes)}")
        for line in rating_changes:
            print(f"  {line}")
        # Only surgical syncs block: a whole-year /photo-year run carries export
        # ratings to the masters on purpose.
        if surgical and not allow_rating_change:
            print("  (refused on --apply unless --allow-rating-change)")
            blocked = True

    if not apply:
        print(lightroom_before_apply_note())
        print("DRY RUN — no changes written. Re-run with --apply to perform the sync.")
        return {"actions": len(actions), "orphaned": len(orphaned), "applied": False,
                "blocked": blocked}

    if not actions:
        print("No actions to apply.")
        return {"actions": 0, "orphaned": len(orphaned), "applied": True,
                "succeeded": 0, "failed": 0}

    if blocked:
        print("REFUSED — fix the pairs/ratings listed above (or drop those photos from "
              "--files) and re-run. Nothing was written.")
        return {"actions": len(actions), "orphaned": len(orphaned), "applied": False,
                "blocked": True}

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
    after = read_rating_label([a.target_path for a in actions])
    drift = [f"{Path(p).name}: expected {expected[p]}, now {after.get(p)}"
             for p in expected if p in after and after[p] != expected[p]]
    if drift:
        print(f"⚠ RATING/LABEL DRIFT on {len(drift)} master(s) — check these in Lightroom:")
        for line in drift:
            print(f"  {line}")
    else:
        print(f"Rating/label check: {len(after)} master(s) exactly as expected")
    print(lightroom_after_apply_note(actions, cfg.raw_root))
    return {"actions": len(actions), "orphaned": len(orphaned), "applied": True,
            "succeeded": ok, "failed": fail, "drift": len(drift)}


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


def tag_one(path: Path, ownership_notice: str, local_vision: bool = False) -> dict:
    """Run the full vision tag on a single photo, in place.

    generate_description=True overwrites the (bad/missing) caption; location is
    fill-missing-only (clear_location stays False); keywords merge (+=).

    local_vision=True skips Claude and captions with the VLM already resident in
    LM Studio (see --local-vision).
    """
    from cortex_engine.textifier import DocumentTextifier

    t = DocumentTextifier(use_vision=True, prefer_local_vision=local_vision)
    # Keyword extraction defaults to the first installed TEXT_MODELS entry, which
    # here is mistral-small3.2 (~23GB VRAM). Loaded per photo alongside LM Studio
    # it saturates the GPU (~44/46GB), stalling the whole machine and adding
    # 15-50s of mistral inference per photo. A small local model derives photo
    # keywords from the caption just as well in ~1-2s with ~10GB.
    #
    # This is now only the *fallback*: extract_keywords prefers the model already
    # loaded in LM Studio (~2s, no extra VRAM), dropping to this Ollama list only
    # when LM Studio is unreachable or has nothing resident.
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
    local_vision: bool = False,
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
            result = tag_one(path, ownership_notice, local_vision=local_vision)
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
    present. Without it — and without --local-vision — the engine falls through
    to Ollama VLMs, which emit reasoning-scaffolding instead of captions. (The
    LM Studio route used by --local-vision does not: it sends
    reasoning_effort="none".) .env also carries CORTEX_LMSTUDIO_BASE_URL, so
    loading it here makes the headless run match the Streamlit app's behaviour.
    Values may be quoted.
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
    pt.add_argument("--local-vision", action="store_true",
                    help="Caption with the VLM already loaded in LM Studio instead of "
                         "the Claude Haiku API (no API cost, no extra VRAM — it reuses "
                         "the resident model). Falls back to Ollama if LM Studio is "
                         "unreachable, which does leak reasoning text into captions.")

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
    ps.add_argument("--date", action="append", metavar="YYYY-MM-DD",
                    help="SURGICAL: sync only exports in to_tag_dir whose name starts "
                         "with this date. Repeatable.")
    ps.add_argument("--files", type=Path, metavar="LIST",
                    help="SURGICAL: sync only the exports listed in this file (one path "
                         "per line, absolute or relative to to_tag_dir).")
    ps.add_argument("--verify-pixels", dest="verify_pixels", action="store_true",
                    default=None,
                    help="Pixel-check every export/master pair (default ON for "
                         "--date/--files, OFF for a whole directory).")
    ps.add_argument("--no-verify-pixels", dest="verify_pixels", action="store_false")
    ps.add_argument("--allow-rating-change", action="store_true",
                    help="Let a surgical sync change a master's star rating to the "
                         "export's (refused by default).")

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
            local_vision=args.local_vision,
            limit=args.limit,
        )
    elif args.command == "sync":
        if not args.to_tag_dir.is_dir():
            parser.error(f"Not a directory: {args.to_tag_dir}")
        if not args.raw_root.is_dir():
            parser.error(f"Not a directory: {args.raw_root}")
        filter_keywords = [k.strip() for k in args.filter_keywords.split(",") if k.strip()]
        files = None
        if args.files:
            files = [ln.strip() for ln in args.files.read_text().splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#")]
        result = sync_photos(
            args.to_tag_dir,
            args.raw_root,
            apply=args.apply,
            keep_backups=not args.no_backups,
            filter_keywords=filter_keywords,
            timestamp_tolerance=args.timestamp_tolerance,
            dates=args.date,
            files=files,
            verify_pixels=args.verify_pixels,
            allow_rating_change=args.allow_rating_change,
        )
        if result.get("blocked") and args.apply:
            sys.exit(2)
        if result.get("drift"):
            sys.exit(3)


if __name__ == "__main__":
    main()
