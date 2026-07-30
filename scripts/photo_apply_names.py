"""Re-apply person names to captions already written to disk.

Name substitution normally runs during enrichment, using the keywords present at
that moment. Photos tagged *after* they were captioned keep the generic phrasing
("A woman is walking...") because there was no name to substitute at the time.

This pass fixes those without re-running the vision model: it reads each photo's
current caption and keywords and rewrites the caption if a person-tag now
matches. No inference, so it takes seconds rather than hours.

    photo_apply_names.py <folder> [--tags "Paul_C=Paul, Jacqui_C=Jacqui"] [--apply]

Defaults to a dry run — pass --apply to write.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cortex_engine.photo_name_tags import apply_names, parse_name_tags
from cortex_engine.textifier import DocumentTextifier
from cortex_engine.utils import convert_windows_to_wsl_path

PHOTO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".gif", ".bmp"}


def read_fields(path: str):
    out = subprocess.run(
        ["exiftool", "-q", "-m", "-s3", "-IPTC:Caption-Abstract", "-IPTC:Keywords", path],
        capture_output=True, text=True).stdout.split("\n")
    caption = out[0].strip() if out else ""
    keywords = [k.strip() for k in (out[1] if len(out) > 1 else "").split(",") if k.strip()]
    return caption, keywords


def write_caption(path: str, caption: str) -> bool:
    r = subprocess.run(
        ["exiftool", "-overwrite_original",
         f"-IPTC:Caption-Abstract={caption}",
         f"-XMP-dc:Description={caption}", path],
        capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tags", default="", help='"Paul_C=Paul, Jacqui_C=Jacqui"')
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    root = Path(convert_windows_to_wsl_path(args.folder.strip()))
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1
    name_tags = parse_name_tags(args.tags)
    print(f"tags: {name_tags} | mode: {'APPLY' if args.apply else 'dry run'}\n", flush=True)

    paths = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS
                   and not p.name.endswith("_original"))

    changed = skipped = written = 0
    for p in paths:
        caption, keywords = read_fields(str(p))
        if not caption or DocumentTextifier.is_placeholder_description(caption):
            continue
        updated = apply_names(caption, keywords, name_tags)
        if updated == caption:
            skipped += 1
            continue
        changed += 1
        print(f"{p.name}\n  before: {caption[:110]}\n  after : {updated[:110]}", flush=True)
        if args.apply and write_caption(str(p), updated):
            written += 1

    print(f"\n{len(paths)} photos | {changed} would change | {written} written | "
          f"{skipped} unchanged", flush=True)
    if changed and not args.apply:
        print("re-run with --apply to write", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
