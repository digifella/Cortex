from __future__ import annotations

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


def _process_action(action: SyncAction, config: SyncConfig) -> SyncResult:
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
