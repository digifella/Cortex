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


def read_location(path: Path) -> dict[str, str]:
    """Read location fields from any file via exiftool.

    Returns dict with keys city, state, country, gps_lat, gps_lon (empty string if absent).
    Works on JPGs, XMP sidecars, and embedded TIF/PSD/DNG files.
    """
    if not path.exists():
        return {"city": "", "state": "", "country": "", "gps_lat": "", "gps_lon": ""}
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return {"city": "", "state": "", "country": "", "gps_lat": "", "gps_lon": ""}
    result = subprocess.run(
        [
            exiftool, "-json", "-s",
            "-XMP-photoshop:City", "-IPTC:City",
            "-XMP-photoshop:State", "-IPTC:Province-State",
            "-XMP-photoshop:Country", "-IPTC:Country-PrimaryLocationName",
            "-GPSLatitude", "-GPSLongitude",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"city": "", "state": "", "country": "", "gps_lat": "", "gps_lon": ""}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"city": "", "state": "", "country": "", "gps_lat": "", "gps_lon": ""}
    if not payload:
        return {"city": "", "state": "", "country": "", "gps_lat": "", "gps_lon": ""}
    row = payload[0]
    city = (row.get("City") or "").strip()
    state = (row.get("State") or row.get("Province-State") or "").strip()
    country = (row.get("Country") or row.get("Country-PrimaryLocationName") or "").strip()
    gps_lat = str(row.get("GPSLatitude") or "").strip()
    gps_lon = str(row.get("GPSLongitude") or "").strip()
    return {"city": city, "state": state, "country": country, "gps_lat": gps_lat, "gps_lon": gps_lon}


def build_location_update(jpg_location: dict[str, str], existing_location: dict[str, str]) -> set[str]:
    """Return set of location field names present in jpg but absent in target.

    Field names returned: "city", "state", "country", "gps".
    A field is only included when the JPG has it AND the target is missing it.
    """
    update: set[str] = set()
    for field in ("city", "state", "country"):
        if jpg_location.get(field) and not existing_location.get(field):
            update.add(field)
    jpg_has_gps = bool(jpg_location.get("gps_lat") and jpg_location.get("gps_lon"))
    target_has_gps = bool(existing_location.get("gps_lat") and existing_location.get("gps_lon"))
    if jpg_has_gps and not target_has_gps:
        update.add("gps")
    return update


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
