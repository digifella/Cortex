from __future__ import annotations

import json
import logging
import os
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_cortex_sync_input

logger = logging.getLogger(__name__)


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
    result2: dict[str, str] = {}
    for line in raw.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = [p.strip() for p in row.split("|", 1)]
        key = os.path.normpath(parts[0])
        value = parts[1] if len(parts) > 1 else ""
        result2[key] = value
    return result2


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Sync website knowledge documents into Cortex ChromaDB.

    The website ZIPs knowledge files (category/filename structure) and attaches
    them as the job's input file. input_data carries a manifest mapping zip
    paths to document metadata.

    input_data contract (new — ZIP-based):
    {
        "manifest": [
            {"zip_path": "digital-health/file.md", "doc_id": 123, "source": "file", "category": "digital-health", "title": "..."},
            ...
        ],
        "collection_name": "Website - Digital Health",
        "topic": "digital-health",
        "fresh": false
    }
    """
    payload = validate_cortex_sync_input(input_data or {})
    manifest = payload.get("manifest") or []
    collection_name = str(payload.get("collection_name") or "Website - All Topics").strip()
    topic = str(payload.get("topic") or "").strip()
    fresh = bool(payload.get("fresh", False))

    # Extract ZIP from downloaded input file
    if not input_path or not input_path.exists():
        raise ValueError("cortex_sync requires an input ZIP file (downloaded from queue)")

    extract_dir = input_path.parent / f"{input_path.stem}_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting ZIP %s to %s", input_path, extract_dir)
    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(extract_dir)

    extracted_supported_files = [
        str(p.relative_to(extract_dir))
        for p in sorted(extract_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf", ".docx", ".pptx"}
    ]

    # Build file list from extracted ZIP, keyed by zip_path for error reporting
    # zip_path format: category/filename.ext
    valid_paths: list[str] = []
    zip_path_to_local: dict[str, str] = {}  # zip_path -> local extracted path
    errors: list[dict[str, str]] = []
    manifest_count = len(manifest)
    matched_count = 0

    if manifest:
        # Use manifest to find expected files
        for entry in manifest:
            zip_path = entry.get("zip_path", "")
            if os.path.isabs(zip_path) or ".." in Path(zip_path).parts:
                errors.append({"file": zip_path, "error": "Invalid zip_path in manifest"})
                logger.warning("Invalid manifest zip_path: %s", zip_path)
                continue
            local_path = str(extract_dir / zip_path)
            if os.path.exists(local_path):
                valid_paths.append(local_path)
                zip_path_to_local[zip_path] = local_path
                matched_count += 1
            else:
                errors.append({"file": zip_path, "error": "File not found in extracted ZIP"})
                logger.warning("Expected file not in ZIP: %s", zip_path)
    else:
        # No manifest — discover files from ZIP
        for rel in extracted_supported_files:
            path_obj = extract_dir / rel
            valid_paths.append(str(path_obj))
            zip_path_to_local[rel] = str(path_obj)
        matched_count = len(valid_paths)

    if not valid_paths:
        raise ValueError(f"No valid files found in extracted ZIP ({len(errors)} errors)")

    if progress_cb:
        progress_cb(5, f"Extracted {len(valid_paths)} files from ZIP", "validate")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before initialization")

    # Resolve db_path from ConfigManager (avoid path bugs from remote payload)
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

    # Build args namespace matching ingest_cortex expectations
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

    # Stage 2: Analyze documents
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

    # Stage 3: Finalize (embed into ChromaDB + build graph)
    if progress_cb:
        progress_cb(65, "Finalizing ingestion and embeddings", "finalize")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before finalization")
    finalize_ingestion(db_path=db_path, args=args)

    # Stage 4: Read ingested doc_ids
    if progress_cb:
        progress_cb(85, "Collecting sync results", "results")

    processed_log_path = os.path.join(db_path, "knowledge_hub_db", INGESTED_FILES_LOG)
    doc_map = _load_doc_id_map(processed_log_path)

    # Build reverse map: local extracted path -> zip_path
    local_to_zip: dict[str, str] = {v: k for k, v in zip_path_to_local.items()}

    doc_ids: list[str] = []
    ingested_local_paths: set[str] = set()
    for log_path, doc_id in doc_map.items():
        # Check if this log entry matches any of our extracted files
        for vp in valid_paths:
            if os.path.normpath(log_path) == os.path.normpath(vp):
                ingested_local_paths.add(vp)
                if doc_id:
                    doc_ids.append(doc_id)
                break

    # Report files not found in ingestion log — use zip_path for error key
    for vp in valid_paths:
        if vp not in ingested_local_paths:
            zip_path = local_to_zip.get(vp, os.path.basename(vp))
            errors.append({"file": zip_path, "error": "Not found in ingestion log after processing"})

    # Stage 5: Update working collection
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

    logger.info(
        "cortex_sync complete: %d synced, %d errors, collection='%s'",
        success_count, error_count, collection_name,
    )

    return {
        "output_data": {
            "success_count": success_count,
            "error_count": error_count,
            "doc_ids": doc_ids,
            "collection_name": collection_name,
            "topic": topic,
            "errors": errors,
            "manifest_count": manifest_count,
            "extracted_count": len(extracted_supported_files),
            "matched_count": matched_count,
            "missing_manifest_entries": max(0, manifest_count - matched_count) if manifest_count else 0,
        }
    }
