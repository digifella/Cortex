"""Two-pass photo enrichment: fast model first, stronger model for the failures.

Pass 1 uses the VRAM-appropriate model (fast, handles most photos). Any photo it
cannot describe gets a "[Image: ...]" placeholder rather than a caption; those
are collected and retried in pass 2 with the highest-quality installed model,
which is slower but succeeds where the small one gave up.

This is deliberately two passes rather than inline escalation: escalating mid-run
makes batch duration unpredictable, and the failures are usually few enough that
a separate slow pass costs little.

    photo_enrich_batch.py <folder> [--after "YYYY-MM-DD HH:MM:SS"] [--only-empty]
                          [--pass1 MODEL] [--pass2 MODEL] [--no-retry] [--dry-run]

Resumable: completed files are recorded, so an interrupted run picks up where it
stopped without re-describing (or re-charging for) work already done.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cortex_engine.textifier import DocumentTextifier
from cortex_engine.vision_model_selector import (
    VISION_MODEL_PROFILES,
    installed_models,
    select_vision_model,
)
from cortex_engine.utils import convert_windows_to_wsl_path

PHOTO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".gif", ".bmp"}
OWNERSHIP = ("All rights (c) Longboardfella. Contact longboardfella.com "
             "for info on use of photos.")


def collect(folder: str) -> list:
    root = Path(convert_windows_to_wsl_path(folder.strip()))
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    return sorted(
        str(p) for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS
        and not p.name.endswith("_original")
    )


def exif(path: str, tag: str) -> str:
    return subprocess.run(["exiftool", "-q", "-m", "-s3", tag, path],
                          capture_output=True, text=True).stdout.strip()


def strongest_installed(exclude: str = "") -> str:
    """Highest-quality installed model, ignoring VRAM fit.

    Pass 2 accepts CPU spill: slow but correct beats fast but empty.
    """
    inst = set(installed_models())
    ranked = [p.name for p in sorted(VISION_MODEL_PROFILES, key=lambda x: -x.quality)
              if p.name in inst and p.name != exclude]
    return ranked[0] if ranked else ""


def enrich(paths, model, label, state_path, dry_run=False):
    """Describe each photo. Returns (ok, failed, placeholders)."""
    done = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    todo = [p for p in paths if p not in done]
    print(f"\n=== {label}: model={model} | {len(todo)} of {len(paths)} to process ===",
          flush=True)
    if dry_run or not todo:
        return 0, 0, []

    os.environ.pop("ANTHROPIC_API_KEY", None)
    t = DocumentTextifier(geocode_mode="auto", prefer_local_vision=True,
                          auto_select_vision=False)
    t.VISION_MODELS = [model]
    t.VISION_FALLBACK_MODELS = []

    ok = failed = 0
    placeholders = []
    for i, p in enumerate(todo, 1):
        name = Path(p).name
        try:
            r = t.keyword_image(p, generate_description=True, populate_location=True,
                                clear_keywords=False, clear_location=False,
                                ownership_notice=OWNERSHIP)
            desc = (r.get("description") or "").strip()
            if r.get("error"):
                failed += 1
                print(f"[{i}/{len(todo)}] FAIL {name}: {r['error']}", flush=True)
            elif DocumentTextifier.is_placeholder_description(desc):
                placeholders.append(p)
                print(f"[{i}/{len(todo)}] PLACEHOLDER {name}: {desc}", flush=True)
            else:
                ok += 1
                done.add(p)
                state_path.write_text(json.dumps(sorted(done)))
                print(f"[{i}/{len(todo)}] OK {name}\n      {desc[:130]}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(todo)}] ERROR {name}: {exc}", flush=True)
        time.sleep(0.3)
    return ok, failed, placeholders


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--after", default=None, help='only photos captured after "YYYY-MM-DD HH:MM:SS"')
    ap.add_argument("--only-empty", action="store_true",
                    help="skip photos that already have a real (non-placeholder) caption")
    ap.add_argument("--pass1", default=None, help="fast model; default = VRAM-appropriate pick")
    ap.add_argument("--pass2", default=None, help="retry model; default = strongest installed")
    ap.add_argument("--no-retry", action="store_true", help="skip the slow second pass")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default=None, help="resume file path")
    args = ap.parse_args()

    paths = collect(args.folder)
    if args.after:
        cutoff = args.after.replace("-", ":", 2)
        paths = [p for p in paths if (exif(p, "-DateTimeOriginal") or "") > cutoff]
    if args.only_empty:
        kept = []
        for p in paths:
            cap = exif(p, "-IPTC:Caption-Abstract")
            if not cap or DocumentTextifier.is_placeholder_description(cap):
                kept.append(p)
        paths = kept

    pass1 = args.pass1 or (select_vision_model()[0] or "")
    pass2 = args.pass2 or strongest_installed(exclude=pass1)
    print(f"{len(paths)} photo(s) selected | pass1={pass1} | pass2={pass2 or '(none)'}",
          flush=True)
    if args.dry_run:
        for p in paths[:10]:
            print("   ", Path(p).name)
        return 0
    if not paths:
        return 0
    if not pass1:
        print("FATAL: no vision model installed", flush=True)
        return 1

    state = Path(args.state) if args.state else Path("/tmp/photo_enrich_state.json")
    ok1, failed1, placeholders = enrich(paths, pass1, "PASS 1 (fast)", state)

    ok2 = failed2 = 0
    still = []
    if placeholders and pass2 and not args.no_retry:
        print(f"\n{len(placeholders)} photo(s) the fast model could not describe — "
              f"retrying with {pass2} (slower)", flush=True)
        ok2, failed2, still = enrich(placeholders, pass2, "PASS 2 (strong)", state)
    elif placeholders:
        still = placeholders

    print(f"\nCOMPLETE pass1={pass1} ok={ok1} failed={failed1} | "
          f"pass2={pass2 or '-'} ok={ok2} failed={failed2} | "
          f"still undescribed={len(still)}", flush=True)
    for p in still:
        print(f"  UNDESCRIBED {Path(p).name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
