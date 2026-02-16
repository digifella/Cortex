from __future__ import annotations

import json
import logging
import os
import getpass
from argparse import Namespace
from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_cortex_sync_input
from cortex_engine.utils.path_utils import convert_windows_to_wsl_path

logger = logging.getLogger(__name__)


def _list_local_topic_dirs() -> list[str]:
    site_root = os.environ.get(
        "CORTEX_SYNC_SITE_ROOT",
        str(Path.home() / "longboardfella_website" / "site"),
    ).strip()
    knowledge_root = Path(site_root) / "chatbot" / "knowledge"
    if not knowledge_root.exists():
        return []
    return sorted([p.name for p in knowledge_root.iterdir() if p.is_dir()])


def _normalize_input_path(raw_path: str) -> str:
    """Normalize path text to a usable local path in WSL/Linux."""
    text = str(raw_path or "").strip()
    if not text:
        return text
    try:
        converted = convert_windows_to_wsl_path(text)
        return str(Path(converted).expanduser())
    except Exception:
        return text


def _resolve_existing_path(raw_path: str) -> str:
    """
    Resolve a submitted path to an existing local path.
    Tries direct path, home-user remap, then public_html -> local site root remap.
    """
    normalized = _normalize_input_path(raw_path)
    candidates: list[str] = []
    if normalized:
        candidates.append(normalized)

    # Remap /home/<submitted_user>/... -> /home/<local_user>/...
    parts = Path(normalized).parts if normalized else ()
    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "home":
        tail = parts[3:]  # after /home/<user>/
        local_user = getpass.getuser()
        remapped = str(Path("/home") / local_user / Path(*tail))
        if remapped not in candidates:
            candidates.append(remapped)

    # Remap cPanel style .../public_html/... into local website mirror root.
    public_html_marker = "/public_html/"
    if normalized and public_html_marker in normalized:
        rel = normalized.split(public_html_marker, 1)[1].lstrip("/")
        site_root = os.environ.get(
            "CORTEX_SYNC_SITE_ROOT",
            str(Path.home() / "longboardfella_website" / "site"),
        ).strip()
        remapped_site = str(Path(site_root) / rel)
        if remapped_site not in candidates:
            candidates.append(remapped_site)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            if candidate != normalized:
                logger.info("cortex_sync remapped path: %s -> %s", raw_path, candidate)
            return candidate
    return normalized


def _load_doc_id_map(processed_log_path: str) -> dict[str, str]:
    """
    Read ingested_files.log.
    Supports current JSON format and legacy line format fallback.
    """
    path = Path(processed_log_path)
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return {}

    # Current format: JSON object {doc_posix_path: doc_id}
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            result: dict[str, str] = {}
            for k, v in payload.items():
                key = str(k).strip()
                if not key:
                    continue
                value = ""
                if isinstance(v, list) and v:
                    value = str(v[0]).strip()
                else:
                    value = str(v).strip()
                result[os.path.normpath(key)] = value
            return result
    except Exception:
        pass

    # Fallback legacy format: "path | doc_id" lines
    result: dict[str, str] = {}
    for line in raw.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = [p.strip() for p in row.split("|", 1)]
        key = os.path.normpath(parts[0])
        value = parts[1] if len(parts) > 1 else ""
        result[key] = value
    return result


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Sync website knowledge documents into Cortex ChromaDB.

    input_data contract:
    {
      "file_paths": ["/absolute/path/to/file.md", ...],
      "collection_name": "Website - Wellbeing",
      "topic": "wellbeing",
      "fresh": false
    }
    """
    payload = validate_cortex_sync_input(input_data or {})
    file_paths = list(payload.get("file_paths") or [])
    collection_name = str(payload.get("collection_name") or "Website - All Topics").strip()
    topic = str(payload.get("topic") or "").strip()
    fresh = bool(payload.get("fresh", False))

    # Validate files exist on disk.
    valid_paths: list[str] = []
    errors: list[dict[str, str]] = []
    missing_paths: list[str] = []
    for fp in file_paths:
        normalized = _resolve_existing_path(fp)
        if normalized and os.path.exists(normalized):
            valid_paths.append(normalized)
        else:
            errors.append({"file": str(fp), "error": "File not found on disk"})
            missing_paths.append(str(fp))
            logger.warning("cortex_sync file not found: %s", fp)

    if not valid_paths:
        site_root = os.environ.get(
            "CORTEX_SYNC_SITE_ROOT",
            str(Path.home() / "longboardfella_website" / "site"),
        ).strip()
        topics = _list_local_topic_dirs()
        hint = (
            f"None of the {len(file_paths)} files exist on disk. "
            f"Configured CORTEX_SYNC_SITE_ROOT='{site_root}'. "
            f"Local topic dirs: {topics if topics else 'none found'}; "
            f"missing sample: {missing_paths[:3]}"
        )
        raise ValueError(hint)

    if progress_cb:
        progress_cb(5, f"Validated {len(valid_paths)} files", "validate")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before initialization")

    # Resolve db_path from ConfigManager (avoid path bugs from remote payload).
    from cortex_engine.config_manager import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.get_config()
    db_path = str(config.get("ai_database_path") or "").strip()
    if not db_path:
        raise ValueError("ai_database_path not configured in Cortex ConfigManager")

    logger.info(
        "cortex_sync: %d files -> collection '%s' (fresh=%s, db_path=%s)",
        len(valid_paths), collection_name, fresh, db_path,
    )

    # Build args namespace matching ingest_cortex expectations.
    args = Namespace(
        db_path=db_path,
        skip_image_processing=True,
        llm_timeout=90.0,
        throttle_delay=1.0,
        cpu_threshold=70.0,
        gpu_threshold=60.0,
        max_throttle_delay=8.0,
        cooldown_every=25,
        cooldown_seconds=20.0,
        image_workers=1,
        index_batch_cooldown=1.0,
        gpu_intensity=75,
        ingest_backend="default",
        migration_mode=None,
        target_collection=collection_name,
    )

    from cortex_engine.ingest_cortex import (
        INGESTED_FILES_LOG,
        analyze_documents,
        finalize_ingestion,
        initialize_script,
    )

    if progress_cb:
        progress_cb(10, "Initializing Cortex ingestion", "init")
    initialize_script()

    if progress_cb:
        progress_cb(20, f"Analyzing {len(valid_paths)} documents", "analyze")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before analysis")
    analyze_documents(
        include_paths=valid_paths,
        fresh_start=fresh,
        args=args,
        target_collection=collection_name,
    )

    if progress_cb:
        progress_cb(65, "Finalizing ingestion and embeddings", "finalize")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before finalization")
    finalize_ingestion(db_path=db_path, args=args)

    if progress_cb:
        progress_cb(85, "Collecting sync results", "results")

    processed_log_path = os.path.join(db_path, "knowledge_hub_db", INGESTED_FILES_LOG)
    doc_map = _load_doc_id_map(processed_log_path)
    valid_norm_set = {os.path.normpath(vp) for vp in valid_paths}

    doc_ids: list[str] = []
    ingested_paths: set[str] = set()
    for p, doc_id in doc_map.items():
        if p in valid_norm_set:
            ingested_paths.add(p)
            if doc_id:
                doc_ids.append(doc_id)

    # Any valid paths not seen in ingestion log are reported as errors.
    for vp in valid_paths:
        nvp = os.path.normpath(vp)
        if nvp not in ingested_paths:
            errors.append({"file": vp, "error": "Not found in ingestion log after processing"})

    # Update collection references.
    if progress_cb:
        progress_cb(95, "Updating working collection", "collection")
    if doc_ids:
        try:
            from cortex_engine.collection_manager import WorkingCollectionManager

            wcm = WorkingCollectionManager()
            if collection_name not in wcm.get_collection_names():
                wcm.create_collection(collection_name)
                logger.info("Created collection: %s", collection_name)
            wcm.add_docs_by_id_to_collection(collection_name, doc_ids)
            logger.info("Added %d doc_ids to collection '%s'", len(doc_ids), collection_name)
        except Exception as e:
            logger.exception("Collection update failed for %s", collection_name)
            errors.append({"file": "", "error": f"Collection update failed: {e}"})

    success_count = len(doc_ids)
    error_count = len(errors)
    if progress_cb:
        progress_cb(100, f"Done: {success_count} synced, {error_count} errors", "done")

    return {
        "output_data": {
            "success_count": success_count,
            "error_count": error_count,
            "doc_ids": doc_ids,
            "collection_name": collection_name,
            "topic": topic,
            "errors": errors,
            "submitted_files": len(file_paths),
            "validated_files": len(valid_paths),
        }
    }
