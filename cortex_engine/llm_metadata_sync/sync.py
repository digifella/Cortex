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
    read_rating,
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
            # Never let one bad photo (unexpected metadata shape, transient IO)
            # abort the whole run — surface it as a failed result and continue.
            try:
                yield _process_action(action, config)
            except Exception as exc:
                yield SyncResult(
                    action=action, success=False,
                    keywords_written=0, description_written=False,
                    location_written=0,
                    error=f"Unhandled error: {exc}",
                )


def _process_jpg_replace(action: SyncAction, config: SyncConfig) -> SyncResult:
    """Back up the catalog JPG as .old and copy the described JPG into its place.

    The described JPG already carries all metadata (keywords, description, location),
    so no exiftool calls are needed.
    """
    jpg_keywords, description = read_jpg_metadata(action.jpg_path)
    jpg_location = read_location(action.jpg_path)
    jpg_rating = read_rating(action.jpg_path)
    location_written = len([v for v in jpg_location.values() if v]) if jpg_location else 0

    if config.dry_run:
        return SyncResult(
            action=action,
            success=True,
            keywords_written=len(jpg_keywords),
            description_written=bool(description or jpg_rating is not None),
            location_written=location_written,
            error=None,
        )

    old_path = action.target_path.with_suffix(".old")
    if old_path.exists():
        # A .old backup already exists — either this target was replaced in a
        # prior run, or the user has a pre-existing backup. Skip so we never
        # clobber the original backup with an already-described copy. This also
        # makes the run safely resumable after an interruption.
        return SyncResult(
            action=action, success=True,
            keywords_written=len(jpg_keywords), description_written=bool(description),
            location_written=location_written, error=None,
        )
    try:
        action.target_path.rename(old_path)
    except Exception as exc:
        return SyncResult(
            action=action, success=False,
            keywords_written=0, description_written=False, location_written=0,
            error=f"Could not rename original to .old: {exc}",
        )
    try:
        # copyfile (data only), not copy2: copy2's copystat() calls os.chmod,
        # which returns EPERM on WSL drvfs mounts (e.g. the L: catalog), failing
        # every JPG replace. The described JPG already carries all metadata in
        # its bytes, so file permissions/timestamps don't need copying.
        shutil.copyfile(action.jpg_path, action.target_path)
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
        jpg_rating = read_rating(action.jpg_path)

        # Determine location fields to copy before deciding whether to skip
        existing_location = read_location(action.target_path)
        location_fields = build_location_update(jpg_location, existing_location)

        if not jpg_keywords and not description and not location_fields and jpg_rating is None:
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
                description_written=bool(description or jpg_rating is not None),
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
            rating=jpg_rating,
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
            description_written=bool(description or jpg_rating is not None),
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
