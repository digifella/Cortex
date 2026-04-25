from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def build_keyword_union(
    existing: list[str],
    new: list[str],
    filter_list: list[str],
) -> list[str]:
    """Return union of existing + new keywords, deduplicated, filtered.

    - existing keywords appear before new ones
    - dedup is case-sensitive, first-seen wins
    - filter_list matching is case-insensitive
    """
    filter_set = {f.lower() for f in filter_list}
    seen: dict[str, None] = {}
    for kw in existing + new:
        if kw.lower() not in filter_set and kw not in seen:
            seen[kw] = None
    return list(seen.keys())


def read_jpg_metadata(jpg: Path) -> tuple[list[str], str]:
    """Read IPTC keywords and caption from a JPG via exiftool.

    Returns (keywords, description). Both empty if exiftool fails or file has none.
    """
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return [], ""
    result = subprocess.run(
        [exiftool, "-json", "-s", "-IPTC:Keywords", "-IPTC:Caption-Abstract", str(jpg)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return [], ""
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [], ""
    if not payload:
        return [], ""
    row = payload[0]
    kws = row.get("Keywords", [])
    if isinstance(kws, str):
        kws = [kws]
    keywords = [k.strip() for k in kws if k.strip()]
    description = (row.get("Caption-Abstract") or "").strip()
    return keywords, description


def read_existing_keywords(target: Path) -> list[str]:
    """Read XMP-dc:Subject keywords from a target file via exiftool.

    Returns empty list if file doesn't exist or exiftool fails.
    """
    if not target.exists():
        return []
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return []
    result = subprocess.run(
        [exiftool, "-json", "-s", "-XMP-dc:Subject", str(target)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not payload:
        return []
    val = payload[0].get("Subject", [])
    if isinstance(val, str):
        val = [val]
    return [k.strip() for k in val if k.strip()]
