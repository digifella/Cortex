from __future__ import annotations

import shutil
from pathlib import Path
from typing import Generator

from . import exiftool_runner
from .matcher import build_raw_index, resolve_jpg
from .merger import (
    build_keyword_union,
    build_location_update,
    read_existing_keywords,
    read_jpg_metadata,
    read_location,
)
from .models import SyncAction, SyncConfig, SyncResult, TargetType


def run_sync(config: SyncConfig) -> Generator[SyncResult, None, None]:
    """Generator orchestrator. Yields one SyncResult per matched action.

    Builds the RAW index once. Orphaned JPGs (no match) are silently skipped.
    """
    index = build_raw_index(config.raw_root, config)

    jpgs = sorted(
        list(config.jpg_dir.glob("*.jpg")) + list(config.jpg_dir.glob("*.JPG"))
    )

    for jpg in jpgs:
        actions = resolve_jpg(jpg, index, config)
        for action in actions:
            yield _process_action(action, config)


def _process_jpg_replace(action: SyncAction, config: SyncConfig) -> SyncResult:
    """Back up the catalog JPG as .old and copy the described JPG into its place.

    The described JPG already carries all metadata (keywords, description, location),
    so no exiftool calls are needed.
    """
    jpg_keywords, description = read_jpg_metadata(action.jpg_path)
    jpg_location = read_location(action.jpg_path)
    location_written = len([v for v in jpg_location.values() if v]) if jpg_location else 0

    if config.dry_run:
        return SyncResult(
            action=action,
            success=True,
            keywords_written=len(jpg_keywords),
            description_written=bool(description),
            location_written=location_written,
            error=None,
        )

    old_path = action.target_path.with_suffix(".old")
    try:
        action.target_path.rename(old_path)
    except Exception as exc:
        return SyncResult(
            action=action, success=False,
            keywords_written=0, description_written=False, location_written=0,
            error=f"Could not rename original to .old: {exc}",
        )
    try:
        shutil.copy2(action.jpg_path, action.target_path)
    except Exception as exc:
        # Restore original so the catalog isn't left without the file
        try:
            old_path.rename(action.target_path)
        except Exception:
            pass
        return SyncResult(
            action=action, success=False,
            keywords_written=0, description_written=False, location_written=0,
            error=f"Could not copy described JPG into place: {exc}",
        )

    return SyncResult(
        action=action,
        success=True,
        keywords_written=len(jpg_keywords),
        description_written=bool(description),
        location_written=location_written,
        error=None,
    )


def _process_action(action: SyncAction, config: SyncConfig) -> SyncResult:
    if action.target_type == TargetType.JPG_REPLACE:
        return _process_jpg_replace(action, config)

    try:
        jpg_keywords, description = read_jpg_metadata(action.jpg_path)
        jpg_location = read_location(action.jpg_path)

        # Determine location fields to copy before deciding whether to skip
        existing_location = read_location(action.target_path)
        location_fields = build_location_update(jpg_location, existing_location)

        if not jpg_keywords and not description and not location_fields:
            return SyncResult(
                action=action,
                success=True,
                keywords_written=0,
                description_written=False,
                location_written=0,
                error=None,
            )

        existing_keywords = read_existing_keywords(action.target_path)
        merged_keywords = build_keyword_union(
            existing_keywords, jpg_keywords, config.filter_keywords
        )

        if config.dry_run:
            return SyncResult(
                action=action,
                success=True,
                keywords_written=len(merged_keywords),
                description_written=bool(description),
                location_written=len(location_fields),
                error=None,
            )

        # Step 1: clear keyword lists (only if target already exists)
        if action.target_path.exists():
            clear_result = exiftool_runner.clear_keyword_lists(
                action.target_path, action.target_type, config.keep_backups
            )
            if not clear_result.ok:
                return SyncResult(
                    action=action,
                    success=False,
                    keywords_written=0,
                    description_written=False,
                    location_written=0,
                    error=clear_result.filtered_stderr or "exiftool clear failed",
                )

        # Step 2: write keywords, description, and any missing location fields
        write_result = exiftool_runner.write_metadata(
            action.jpg_path,
            action.target_path,
            action.target_type,
            merged_keywords,
            description,
            config.keep_backups,
            location_fields,
        )
        if not write_result.ok:
            return SyncResult(
                action=action,
                success=False,
                keywords_written=0,
                description_written=False,
                location_written=0,
                error=write_result.filtered_stderr or "exiftool write failed",
            )

        return SyncResult(
            action=action,
            success=True,
            keywords_written=len(merged_keywords),
            description_written=bool(description),
            location_written=len(location_fields),
            error=None,
        )

    except Exception as exc:
        return SyncResult(
            action=action,
            success=False,
            keywords_written=0,
            description_written=False,
            location_written=0,
            error=str(exc),
        )
