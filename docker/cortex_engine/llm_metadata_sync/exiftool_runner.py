from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import TargetType

_BENIGN_WARNINGS = ("IPTCDigest is not current",)


class ExifToolNotFoundError(RuntimeError):
    pass


def exiftool_path() -> str:
    """Return path to exiftool binary, raising ExifToolNotFoundError if absent."""
    path = shutil.which("exiftool")
    if not path:
        raise ExifToolNotFoundError(
            "exiftool not found on PATH. "
            "Install with: sudo apt install libimage-exiftool-perl"
        )
    return path


def is_available() -> bool:
    """Return True if exiftool is on PATH."""
    return shutil.which("exiftool") is not None


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def filtered_stderr(self) -> str:
        lines = [
            line
            for line in self.stderr.splitlines()
            if not any(w in line for w in _BENIGN_WARNINGS)
        ]
        return "\n".join(lines)


def _run(args: list[str]) -> RunResult:
    result = subprocess.run(args, capture_output=True, text=True)
    return RunResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=args,
    )


def _backup_flag(keep_backups: bool) -> str:
    return "-overwrite_original" if keep_backups else "-overwrite_original_in_place"


def clear_keyword_lists(
    target: Path, target_type: TargetType, keep_backups: bool
) -> RunResult:
    """Step 1 of two-step write: clear keyword lists on target.

    SIDECAR: clears xmp-dc:subject only.
    EMBEDDED: clears xmp-dc:subject AND iptc:Keywords.
    """
    et = exiftool_path()
    backup = _backup_flag(keep_backups)

    if target_type == TargetType.SIDECAR:
        args = [et, backup, "-xmp-dc:subject=", str(target)]
    else:
        args = [et, backup, "-xmp-dc:subject=", "-iptc:Keywords=", str(target)]

    return _run(args)


def write_metadata(
    jpg: Path,
    target: Path,
    target_type: TargetType,
    keywords: list[str],
    description: str,
    keep_backups: bool,
    location_fields: set[str] = frozenset(),
) -> RunResult:
    """Step 2 of two-step write: populate keywords, description, and/or location.

    SIDECAR: writes to xmp-dc / xmp-photoshop namespaces.
    EMBEDDED: writes to both xmp and iptc namespaces (kept in sync).
    Description and location copied from JPG via -tagsfromfile.
    location_fields: subset of {"city", "state", "country", "gps"} to copy.
    """
    et = exiftool_path()
    backup = _backup_flag(keep_backups)
    args = [et, backup]

    # Direct keyword writes
    if target_type == TargetType.SIDECAR:
        for kw in keywords:
            args.append(f"-xmp-dc:subject+={kw}")
    else:
        for kw in keywords:
            args.append(f"-xmp-dc:subject+={kw}")
            args.append(f"-iptc:Keywords+={kw}")

    # Build -tagsfromfile copy tags for description and location
    copy_tags: list[str] = []

    if description:
        copy_tags.append("-xmp-dc:description<iptc:Caption-Abstract")
        if target_type == TargetType.EMBEDDED:
            copy_tags.append("-iptc:Caption-Abstract<iptc:Caption-Abstract")

    if "city" in location_fields:
        copy_tags.append("-XMP-photoshop:City<XMP-photoshop:City")
        if target_type == TargetType.EMBEDDED:
            copy_tags.append("-IPTC:City<XMP-photoshop:City")

    if "state" in location_fields:
        copy_tags.append("-XMP-photoshop:State<XMP-photoshop:State")
        if target_type == TargetType.EMBEDDED:
            copy_tags.append("-IPTC:Province-State<XMP-photoshop:State")

    if "country" in location_fields:
        copy_tags.append("-XMP-photoshop:Country<XMP-photoshop:Country")
        if target_type == TargetType.EMBEDDED:
            copy_tags.append("-IPTC:Country-PrimaryLocationName<XMP-photoshop:Country")

    if "gps" in location_fields:
        copy_tags += [
            "-GPSLatitude<GPSLatitude",
            "-GPSLongitude<GPSLongitude",
            "-GPSLatitudeRef<GPSLatitudeRef",
            "-GPSLongitudeRef<GPSLongitudeRef",
        ]

    if copy_tags:
        args += ["-tagsfromfile", str(jpg)] + copy_tags

    args.append(str(target))
    return _run(args)
