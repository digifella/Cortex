from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from .models import SidecarAction, SyncAction, SyncConfig, TargetType

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})(-.+)?$")


def strip_rating_suffix(stem: str, suffix_range: tuple[int, int]) -> str:
    """Remove trailing -N rating suffix if N is within suffix_range."""
    lo, hi = suffix_range
    for n in range(lo, hi + 1):
        suffix = f"-{n}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _deriv_regex(patterns: tuple[str, ...]) -> re.Pattern:
    alternation = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(f"({alternation})$", re.IGNORECASE)


def build_raw_index(raw_root: Path, config: SyncConfig) -> dict[str, list[Path]]:
    """Walk raw_root once and return {case-folded stem: [target paths]}.

    Target paths:
    - .xmp sidecar path (may not exist) for raw originals
    - embedded file path for TIF/PSD/DNG/PSB derivatives
    - the JPG file itself for catalog JPGs (JPG_REPLACE)
    ACR files are always skipped.
    """
    raw_exts = {e.lower() for e in config.raw_extensions}
    embed_exts = {e.lower() for e in config.embed_extensions}
    jpg_exts = {e.lower() for e in config.jpg_extensions}
    deriv_re = _deriv_regex(config.deriv_patterns)

    index: dict[str, list[Path]] = {}

    for dirpath, _dirs, filenames in os.walk(raw_root):
        dir_path = Path(dirpath)
        for filename in filenames:
            path = dir_path / filename
            ext = path.suffix.lstrip(".").lower()
            stem = path.stem

            if ext == "acr":
                continue

            if ext in embed_exts:
                m = deriv_re.search(stem)
                if m:
                    # Derivative embed (e.g. shot-Edit.tif, shot-Pano.dng) — includes
                    # DNG panoramas/HDR merges created by Lightroom.  Check embed_exts
                    # before raw_exts so derivative-suffix DNGs land here rather than
                    # being indexed as raw originals under their full (un-stripped) stem.
                    base_stem = stem[: m.start()]
                    key = base_stem.lower()
                    index.setdefault(key, []).append(path)
                else:
                    # Standalone embed-format file or original DNG capture (no suffix)
                    # → write to an XMP sidecar alongside it
                    key = stem.lower()
                    sidecar = dir_path / f"{stem}.xmp"
                    index.setdefault(key, []).append(sidecar)

            elif ext in raw_exts:
                key = stem.lower()
                sidecar = dir_path / f"{stem}.xmp"
                index.setdefault(key, []).append(sidecar)

            elif ext in jpg_exts:
                # Catalog / mobile JPG (no raw original) → JPG_REPLACE target.
                # Only index files without a derivative suffix; rated/described JPGs
                # (e.g. shot-5.jpg, shot-Edit.tif equivalents) live in jpg_dir and
                # are the *source* of metadata, not the target.
                if not deriv_re.search(stem):
                    key = stem.lower()
                    index.setdefault(key, []).append(path)

    return index


def _parse_key_ts(key: str):
    """Return (datetime, camera_suffix) from a key, or None if not parseable."""
    m = _TS_RE.match(key)
    if not m:
        return None
    ts_str = m.group(1)  # "YYYY-MM-DD HH-MM-SS"
    cam = (m.group(2) or "").lower()
    try:
        ts = datetime(
            int(ts_str[0:4]), int(ts_str[5:7]), int(ts_str[8:10]),
            int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
        )
        return ts, cam
    except ValueError:
        return None


def _fuzzy_targets(key: str, index: dict[str, list[Path]], tolerance_s: int) -> list[Path]:
    """Find index targets whose key matches key's camera suffix with timestamp ≤ tolerance_s away."""
    parsed = _parse_key_ts(key)
    if parsed is None:
        return []
    ts, cam = parsed
    results: list[Path] = []
    for idx_key, idx_targets in index.items():
        if idx_key == key:
            continue
        parsed2 = _parse_key_ts(idx_key)
        if parsed2 is None:
            continue
        idx_ts, idx_cam = parsed2
        if idx_cam != cam:
            continue
        if 0 < abs((idx_ts - ts).total_seconds()) <= tolerance_s:
            results.extend(idx_targets)
    return results


def resolve_jpg(
    jpg_path: Path, index: dict[str, list[Path]], config: SyncConfig
) -> list[SyncAction]:
    """Resolve a JPG to a list of SyncActions against the pre-built index."""
    stem = strip_rating_suffix(jpg_path.stem, config.rating_suffix_range)
    key = stem.lower()
    targets = index.get(key, [])

    # Fuzzy fallback: panorama DNGs are often timestamped 2–4 s before the
    # exported JPG.  When tolerance_seconds > 0 and exact match fails, scan
    # the index for the same camera suffix within the allowed window.
    if not targets and config.timestamp_tolerance_seconds > 0:
        targets = _fuzzy_targets(key, index, config.timestamp_tolerance_seconds)

    embed_exts = {e.lower() for e in config.embed_extensions}
    jpg_exts = {e.lower() for e in config.jpg_extensions}
    actions: list[SyncAction] = []

    for target in targets:
        ext = target.suffix.lstrip(".").lower()

        if ext == "xmp":
            sidecar_action = SidecarAction.MERGE if target.exists() else SidecarAction.CREATE
            raw_path: Path | None = None
            for raw_ext in config.raw_extensions:
                candidate = target.parent / f"{target.stem}.{raw_ext}"
                if candidate.exists():
                    raw_path = candidate
                    break
            actions.append(
                SyncAction(
                    jpg_path=jpg_path,
                    target_path=target,
                    target_type=TargetType.SIDECAR,
                    sidecar_action=sidecar_action,
                    raw_path=raw_path,
                )
            )

        elif ext in embed_exts:
            actions.append(
                SyncAction(
                    jpg_path=jpg_path,
                    target_path=target,
                    target_type=TargetType.EMBEDDED,
                    sidecar_action=SidecarAction.NONE,
                    raw_path=None,
                )
            )

        elif ext in jpg_exts:
            # Don't create a self-replace action (happens when jpg_dir == raw_root)
            if target.resolve() == jpg_path.resolve():
                continue
            actions.append(
                SyncAction(
                    jpg_path=jpg_path,
                    target_path=target,
                    target_type=TargetType.JPG_REPLACE,
                    sidecar_action=SidecarAction.NONE,
                    raw_path=None,
                )
            )

    return actions
