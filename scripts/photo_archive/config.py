"""Scope, exclusions and file classification for the P: drive pipeline."""

DRIVE = "P:"

# NOTE: "Catalogued Photos_Predig and Preraw backups_19 Aug 2026" (22,453 files
# / 284 GB) was in scope at design time but Paul removed it from P: on
# 2026-08-21, mid-project. Confirmed deliberate. Not present on any mounted
# volume as of that date. Removed from scope rather than left to fail the
# walk - restore this line if it ever comes back.
SCOPE_ROOTS = [
    "0 and 1 star photos originals and dupes",
    "Backup Consolidated Photos",
    "3+ Star_RAW_Backups",
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
