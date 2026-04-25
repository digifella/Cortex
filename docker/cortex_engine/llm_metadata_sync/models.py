from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TargetType(Enum):
    SIDECAR = "sidecar"      # write to .xmp sidecar
    EMBEDDED = "embedded"    # write into TIF/PSD/DNG/PSB


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
    deriv_patterns: tuple[str, ...] = (
        r"-Edit", r"-Edit-\d+",
        r"-Enhanced", r"-Enhanced-NR",
        r"-HDR", r"-HDR-\d+",
        r"-Pano", r"-Pano-\d+",
    )
