from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TargetType(Enum):
    SIDECAR = "sidecar"      # write to .xmp sidecar
    EMBEDDED = "embedded"    # write into TIF/PSD/DNG/PSB
    JPG_REPLACE = "jpg_replace"  # back up original catalog JPG (.old), copy described JPG in its place


class SidecarAction(Enum):
    CREATE = "create"        # XMP does not exist yet
    MERGE = "merge"          # XMP exists; clear then repopulate
    NONE = "none"            # embedded targets have no sidecar action


@dataclass
class SyncAction:
    jpg_path: Path
    target_path: Path        # XMP sidecar path OR embedded file path
    target_type: TargetType
    sidecar_action: SidecarAction
    raw_path: Path | None    # original RAW for sidecar mode (informational)


@dataclass
class SyncResult:
    action: SyncAction
    success: bool
    keywords_written: int
    description_written: bool
    location_written: int = 0   # number of location fields copied (city/state/country/gps)
    error: str | None = None


@dataclass
class SyncReport:
    actions: list[SyncAction]
    results: list[SyncResult]
    orphaned_jpgs: list[Path]


@dataclass
class SyncConfig:
    raw_root: Path
    jpg_dir: Path
    filter_keywords: list[str] = field(default_factory=lambda: ["nogps"])
    keep_backups: bool = True
    dry_run: bool = False
    rating_suffix_range: tuple[int, int] = (1, 5)
    raw_extensions: tuple[str, ...] = (
        "RAF", "NEF", "CR2", "CR3", "ARW", "DNG",
        "RW2", "ORF", "PEF", "SRW", "IIQ", "3FR",
        "PNG",   # PNG files → XMP sidecar alongside the file
    )
    embed_extensions: tuple[str, ...] = ("tif", "tiff", "psd", "psb", "dng")
    jpg_extensions: tuple[str, ...] = ("jpg", "jpeg")  # catalog JPGs → JPG_REPLACE action
    timestamp_tolerance_seconds: int = 0
    deriv_patterns: tuple[str, ...] = (
        # Compound patterns must come first so re.search matches at the leftmost
        # position (e.g. -Enhanced-NR-Edit-Edit) rather than just the trailing -Edit-Edit.
        r"-Enhanced-NR(?:-Edit)+-\d+", r"-Enhanced-NR(?:-Edit)+",
        r"-Enhanced(?:-Edit)+-\d+",    r"-Enhanced(?:-Edit)+",
        r"-Pano(?:-Edit)+-\d+",        r"-Pano(?:-Edit)+",
        r"-HDR(?:-Edit)+-\d+",         r"-HDR(?:-Edit)+",
        r"(?:-Edit)+-\d+",             r"(?:-Edit)+",
        r"-Enhanced-NR", r"-Enhanced",
        r"-HDR-\d+",     r"-HDR",
        r"-Pano-\d+",    r"-Pano",
    )
