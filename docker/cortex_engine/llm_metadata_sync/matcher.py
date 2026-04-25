from __future__ import annotations

import os
import re
from pathlib import Path

from .models import SidecarAction, SyncAction, SyncConfig, TargetType


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
    ACR files are always skipped.
    """
    raw_exts = {e.lower() for e in config.raw_extensions}
    embed_exts = {e.lower() for e in config.embed_extensions}
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

            if ext in raw_exts:
                key = stem.lower()
                sidecar = dir_path / f"{stem}.xmp"
                index.setdefault(key, []).append(sidecar)

            elif ext in embed_exts:
                m = deriv_re.search(stem)
                if m:
                    base_stem = stem[: m.start()]
                    key = base_stem.lower()
                    index.setdefault(key, []).append(path)

    return index


def resolve_jpg(
    jpg_path: Path, index: dict[str, list[Path]], config: SyncConfig
) -> list[SyncAction]:
    """Resolve a JPG to a list of SyncActions against the pre-built index."""
    stem = strip_rating_suffix(jpg_path.stem, config.rating_suffix_range)
    key = stem.lower()
    targets = index.get(key, [])

    embed_exts = {e.lower() for e in config.embed_extensions}
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

    return actions
