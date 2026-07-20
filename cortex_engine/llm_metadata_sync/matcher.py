from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from .models import SidecarAction, SyncAction, SyncConfig, TargetType

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})(-.+)?$")
_TS_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})(.*)$")


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


def _normalized_key(stem: str) -> str:
    return stem.lower().rstrip("- ")


def _candidate_keys(stem: str, config: SyncConfig) -> list[str]:
    """Return one or more lookup keys for a stem.

    Primary key is the normalized stem itself. For timestamp-based filenames that
    have editor-added underscore blocks in the middle, also add a collapsed key
    using only the leading timestamp and the trailing underscore-delimited chunk.
    Example:
    2020-02-13 11-17-31_Sri Lanka_4896 x 3264_X-T1-Enhanced-NR-Edit
    -> 2020-02-13 11-17-31-x-t1
    """
    deriv_re = _deriv_regex(config.deriv_patterns)
    m_deriv = deriv_re.search(stem)
    if m_deriv:
        stem = stem[: m_deriv.start()]

    keys: list[str] = []

    primary = _normalized_key(stem)
    if primary:
        keys.append(primary)

    m_ts = _TS_PREFIX_RE.match(stem)
    if not m_ts:
        return keys

    ts_prefix, remainder = m_ts.groups()
    remainder = remainder.strip()
    if not remainder or "_" not in remainder:
        return keys

    tail = remainder.split("_")[-1].strip(" _-")
    if not tail:
        return keys

    collapsed = _normalized_key(f"{ts_prefix}-{tail}")
    if collapsed and collapsed not in keys:
        keys.append(collapsed)

    return keys


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
                    for key in _candidate_keys(base_stem, config):
                        index.setdefault(key, []).append(path)
                elif ext == "png":
                    # PNG sources should be updated in place rather than routed
                    # through a synthetic XMP sidecar path.
                    for key in _candidate_keys(stem, config):
                        index.setdefault(key, []).append(path)
                elif ext in ("tif", "tiff", "psd", "psb"):
                    # Edited raster master (e.g. a Photoshop TIF) with no derivative
                    # suffix — embed metadata into the file itself (that's where
                    # Lightroom reads TIF/PSD metadata from, not a sidecar). These
                    # often keep the export rating suffix (-N) in their filename
                    # while the source JPG has it stripped, so key off the stripped
                    # stem to align the two.
                    base_stem = strip_rating_suffix(stem, config.rating_suffix_range)
                    for key in _candidate_keys(base_stem, config):
                        index.setdefault(key, []).append(path)
                else:
                    # Standalone original DNG capture (no suffix) → XMP sidecar
                    sidecar = dir_path / f"{stem}.xmp"
                    for key in _candidate_keys(stem, config):
                        index.setdefault(key, []).append(sidecar)

            elif ext in raw_exts:
                # rstrip("- ") handles empty camera-model stems like "2025-10-10 10-15-24-"
                sidecar = dir_path / f"{stem}.xmp"
                for key in _candidate_keys(stem, config):
                    index.setdefault(key, []).append(sidecar)

            elif ext in jpg_exts:
                # Catalog / mobile JPG (no raw original) → JPG_REPLACE target.
                # Only index files without a derivative suffix; rated/described JPGs
                # (e.g. shot-5.jpg, shot-Edit.tif equivalents) live in jpg_dir and
                # are the *source* of metadata, not the target. Some catalogs (e.g.
                # pre-RAW JPG-only years) keep the export rating suffix (-N) baked
                # into the catalog filename itself, identical to the jpg_dir source —
                # strip it the same way as the TIF/PSD embed path above so those still
                # match instead of being spuriously orphaned.
                if not deriv_re.search(stem):
                    base_stem = strip_rating_suffix(stem, config.rating_suffix_range)
                    for key in _candidate_keys(base_stem, config):
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


def _camera_suffix_targets(
    jpg_stem: str, index: dict[str, list[Path]], config: SyncConfig
) -> list[Path]:
    """Match injected-token exports by timestamp + camera-token-as-suffix.

    Some Lightroom export templates inject hyphen-delimited '-<location>-<W x H>-'
    (or '-<W x H>-<tag>-') blocks between the timestamp and the camera model, and
    the token order is inconsistent across batches. Rather than parse that variable
    junk, key off what is stable: the leading timestamp, and the RAW-side camera
    token appearing as a suffix of the (deriv-stripped) JPG stem. Among index keys
    at the exact same timestamp, the longest matching camera token wins, so a
    second body that fired the same second never gets the metadata.
    """
    deriv_re = _deriv_regex(config.deriv_patterns)
    m_deriv = deriv_re.search(jpg_stem)
    if m_deriv:
        jpg_stem = jpg_stem[: m_deriv.start()]
    parsed = _parse_key_ts(_normalized_key(jpg_stem))
    if parsed is None:
        return []
    ts, _ = parsed
    jl = jpg_stem.lower()

    # (abs_seconds_from_ts, camera_token, targets) for every same-camera candidate.
    matches: list[tuple[float, str, list[Path]]] = []
    tol = config.timestamp_tolerance_seconds
    for idx_key, idx_targets in index.items():
        parsed2 = _parse_key_ts(idx_key)
        if parsed2 is None:
            continue
        idx_ts, idx_cam = parsed2
        delta = abs((idx_ts - ts).total_seconds())
        if delta > tol:
            continue
        cam = idx_cam.lstrip("-")
        # Empty-camera targets only qualify at the exact timestamp; matching them
        # across a tolerance window would be too loose (they match any stem).
        if delta > 0 and cam == "":
            continue
        if cam == "" or jl.endswith(cam):
            matches.append((delta, cam, idx_targets))
    if not matches:
        return []

    # Prefer the nearest timestamp, then the longest (most specific) camera token,
    # so an exact hit always beats a within-tolerance neighbour.
    nearest = min(delta for delta, _, _ in matches)
    at_nearest = [(cam, tgts) for delta, cam, tgts in matches if delta == nearest]
    longest = max(len(cam) for cam, _ in at_nearest)
    results: list[Path] = []
    for cam, idx_targets in at_nearest:
        if len(cam) == longest:
            results.extend(idx_targets)
    return results


def resolve_jpg(
    jpg_path: Path, index: dict[str, list[Path]], config: SyncConfig
) -> list[SyncAction]:
    """Resolve a JPG to a list of SyncActions against the pre-built index."""
    stem = strip_rating_suffix(jpg_path.stem, config.rating_suffix_range)

    keys = _candidate_keys(stem, config)
    targets: list[Path] = []
    for key in keys:
        targets.extend(index.get(key, []))

    # Fallback for exports that inject hyphen-delimited location/dimension tokens
    # between the timestamp and camera (see _camera_suffix_targets). Only runs when
    # exact key matching found nothing, so existing matches are never altered.
    if not targets:
        targets = _camera_suffix_targets(stem, index, config)

    if targets:
        seen: set[Path] = set()
        deduped_targets: list[Path] = []
        for target in targets:
            if target in seen:
                continue
            seen.add(target)
            deduped_targets.append(target)
        targets = deduped_targets

    # Fuzzy fallback: panorama DNGs are often timestamped 2–4 s before the
    # exported JPG.  When tolerance_seconds > 0 and exact match fails, scan
    # the index for the same camera suffix within the allowed window.
    if not targets and config.timestamp_tolerance_seconds > 0:
        for key in keys:
            fuzzy_targets = _fuzzy_targets(key, index, config.timestamp_tolerance_seconds)
            if fuzzy_targets:
                targets = fuzzy_targets
                break

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
