# ## File: pages/6_Maintenance.py
# Version: v5.6.0
# Date: 2026-01-17
# Purpose: Consolidated maintenance and administrative functions for the Cortex Suite.
#          Combines database maintenance, system terminal, and other administrative functions
#          from various pages into a single, organized maintenance interface.
#          - FEATURE (v5.1.0): Added Database Embedding Inspector with Qwen3-VL compatibility
#            matrix. Shows stored dimensions, identifies original model, and provides
#            migration guidance for switching to Qwen3-VL multimodal embeddings.
#          - FEATURE (v4.11.0): Added Embedding Model Status display with compatibility checking
#            and actionable solutions for embedding model mismatches
#          - BUGFIX (v1.0.1): Fixed import error by using ConfigManager instead of load_config
#          - ENHANCEMENT (v4.4.0): Added Clean Start function for complete system reset
#            Specifically addresses ChromaDB schema conflicts and provides fresh installation state

import streamlit as st
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import time
import contextlib
from pages.components._Maintenance_ResetRecovery import (
    render_reset_recovery_section as shared_render_reset_recovery_section,
)
from pages.components._Maintenance_HealthCheck import (
    render_database_health_check_section as shared_render_database_health_check_section,
)
from pages.components._Maintenance_EmbeddingStatus import (
    render_embedding_model_status_panel as shared_render_embedding_model_status_panel,
)
from pages.components._Maintenance_EmbeddingInspector import (
    render_database_embedding_inspector_panel as shared_render_database_embedding_inspector_panel,
)
from pages.components._Maintenance_PathTools import (
    render_database_path_tools as shared_render_database_path_tools,
)
from pages.components._Maintenance_Dedup import (
    render_database_dedup_section as shared_render_database_dedup_section,
)

# Configure page
st.set_page_config(
    page_title="Maintenance", 
    page_icon="🔧",
    layout="wide"
)

# Page configuration
PAGE_VERSION = None

# Import Cortex modules
try:
    from cortex_engine.config import INGESTED_FILES_LOG
    from cortex_engine.version_config import VERSION_STRING
    from cortex_engine.config_manager import ConfigManager
    from cortex_engine.utils import (
        get_logger,
        convert_to_docker_mount_path,
        convert_windows_to_wsl_path,
        ensure_directory,
        resolve_db_root_path,
    )
    from cortex_engine.utils.command_executor import display_command_executor_widget, SafeCommandExecutor
    from cortex_engine.utils.performance_monitor import get_performance_monitor, get_all_stats, get_session_summary
    from cortex_engine.utils.gpu_monitor import get_gpu_memory_info, get_device_recommendations
    from cortex_engine.ingestion_recovery import IngestionRecoveryManager
    from cortex_engine.collection_manager import WorkingCollectionManager
    from cortex_engine.setup_manager import SetupManager
    from cortex_engine.backup_manager import BackupManager
    from cortex_engine.sync_backup_manager import SyncBackupManager
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from cortex_engine.config import COLLECTION_NAME
except ImportError as e:
    st.error(f"Failed to import required Cortex modules: {e}")
    st.stop()

PAGE_VERSION = VERSION_STRING

# Set up logging
logger = get_logger(__name__)

# Initialize session state
if 'maintenance_config' not in st.session_state:
    st.session_state.maintenance_config = None

def clear_ingestion_session_state(reason: str = "maintenance") -> None:
    """Terminate any active ingestion subprocess and clear related Streamlit state."""
    proc = st.session_state.get("ingestion_process")
    if proc:
        try:
            proc.terminate()
            logger.info(f"{reason}: terminated active ingestion process during maintenance")
        except Exception as e:
            logger.warning(f"{reason}: failed to terminate ingestion process ({e})")
        finally:
            st.session_state.pop("ingestion_process", None)

    stop_event = st.session_state.get("ingestion_reader_stop")
    if stop_event:
        with contextlib.suppress(Exception):
            stop_event.set()
    reader_thread = st.session_state.get("ingestion_reader_thread")
    if reader_thread:
        with contextlib.suppress(Exception):
            reader_thread.join(timeout=0.25)

    # Known ingestion-related keys that keep the UI in a running state
    transient_keys = {
        "files_to_review",
        "staged_files",
        "file_selections",
        "edited_staged_files",
        "review_page",
        "batch_ingest_mode",
        "batch_mode_active",
        "batch_auto_processed",
        "log_messages",
        "resume_mode_enabled",
        "current_scan_config",
        "force_batch_mode",
        "last_ingested_doc_ids",
        "target_collection_name",
        "current_doc_number",
        "total_docs_in_batch",
        "show_logs",
        "processing_metrics",
        "ingestion_start_time",
    }

    for key in list(st.session_state.keys()):
        if key in transient_keys or key.startswith("ingestion_"):
            st.session_state.pop(key, None)

def purge_ingestion_state_files(
    db_root: Path, deleted_items: list, debug_log_lines: list | None = None, errors: list | None = None
) -> None:
    """Remove staging, batch, and progress artifacts so failed ingests don't stick around."""
    project_root = Path(__file__).parent.parent

    def _record(message: str) -> None:
        if debug_log_lines is not None:
            debug_log_lines.append(message)
        logger.info(message)

    def _error(message: str, exc: Exception) -> None:
        if errors is not None:
            errors.append(f"{message}: {exc}")
        logger.warning(f"{message}: {exc}")

    staging_patterns = [
        "staging_ingestion.json",
        "staging_*.json",
        "staging_test.json",
        "batch_progress.json",
        "batch_state.json",
        "ingestion_progress.json",
        "failed_ingestion.json",
        "scan_config.json",
    ]

    for base in (project_root, db_root):
        for pattern in staging_patterns:
            for staging_file in base.glob(pattern):
                if staging_file.is_file():
                    try:
                        staging_file.unlink()
                        deleted_items.append(f"State file: {staging_file}")
                        _record(f"Removed state file: {staging_file}")
                    except Exception as e:
                        _error(f"Failed to remove state file {staging_file}", e)
    # Ingestion progress directory (auto-created by tracker)
    progress_dir = db_root / "ingestion_progress"
    if progress_dir.exists():
        try:
            shutil.rmtree(progress_dir)
            deleted_items.append(f"Ingestion progress dir: {progress_dir}")
            _record(f"Removed ingestion progress directory: {progress_dir}")
        except Exception as e:
            _error(f"Failed to remove ingestion progress directory {progress_dir}", e)

    # Recovery metadata/state
    recovery_metadata_dir = db_root / "recovery_metadata"
    if recovery_metadata_dir.exists():
        try:
            shutil.rmtree(recovery_metadata_dir)
            deleted_items.append(f"Recovery metadata: {recovery_metadata_dir}")
            _record(f"Removed recovery metadata directory: {recovery_metadata_dir}")
        except Exception as e:
            _error(f"Failed to remove recovery metadata {recovery_metadata_dir}", e)

    recovery_state_file = db_root / "knowledge_hub_db" / "recovery_state.json"
    if recovery_state_file.exists():
        try:
            recovery_state_file.unlink()
            deleted_items.append(f"Recovery state file: {recovery_state_file}")
            _record(f"Removed recovery state file: {recovery_state_file}")
        except Exception as e:
            _error(f"Failed to remove recovery state file {recovery_state_file}", e)

    # Local logs (ingestion/query) to clear stale "processing" status
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            try:
                log_file.unlink()
                deleted_items.append(f"Log file: {log_file}")
                _record(f"Cleared log file: {log_file}")
            except Exception as e:
                _error(f"Failed to clear log file {log_file}", e)


def _remove_path(target: Path, deleted_items: list, errors: list, label: str) -> None:
    """Delete a file or directory and record results."""
    try:
        if not target.exists():
            return
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        deleted_items.append(f"{label}: {target}")
    except Exception as e:
        errors.append(f"Failed to delete {label} ({target}): {e}")


def _terminate_ingest_processes_for_db(db_root: Path) -> list:
    """
    Stop ingest subprocesses for the given database path.
    This prevents another Streamlit session from recreating data during reset.
    """
    stopped = []
    try:
        import psutil  # type: ignore
    except Exception:
        logger.warning("psutil not available; cannot terminate external ingest processes")
        return stopped

    db_resolved = str(db_root.resolve())
    for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue
            cmd_text = " ".join(cmdline)
            if "ingest_cortex.py" not in cmd_text and "cortex_engine.ingest_cortex" not in cmd_text:
                continue

            should_stop = False
            if "--db-path" in cmdline:
                idx = cmdline.index("--db-path")
                if idx + 1 < len(cmdline):
                    proc_db = cmdline[idx + 1]
                    proc_db_root = resolve_db_root_path(proc_db)
                    if proc_db_root and str(proc_db_root.resolve()) == db_resolved:
                        should_stop = True
            else:
                # Conservative fallback: only stop if DB path text appears in command
                if db_resolved in cmd_text:
                    should_stop = True

            if not should_stop:
                continue

            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            stopped.append(proc.pid)
        except Exception:
            continue
    return stopped


def _reset_database_artifacts(db_path: str, include_extended_state: bool = False) -> tuple:
    """Core reset routine shared by delete/reset actions."""
    db_root = resolve_db_root_path(db_path)
    if not db_root:
        raise ValueError("Database path is empty or invalid.")

    deleted_items = []
    errors = []

    # Stop ingestion from this and other sessions to avoid instant re-creation.
    clear_ingestion_session_state("maintenance.reset.start")
    stopped_pids = _terminate_ingest_processes_for_db(db_root)
    if stopped_pids:
        deleted_items.append(f"Stopped ingest processes: {stopped_pids}")

    project_root = Path(__file__).parent.parent

    # Core KB artifacts
    core_targets = [
        ("ChromaDB directory", db_root / "knowledge_hub_db"),
        ("Knowledge graph", db_root / "knowledge_cortex.gpickle"),
        ("Collections file", db_root / "working_collections.json"),
        ("Legacy collections file", project_root / "working_collections.json"),
    ]
    for label, target in core_targets:
        _remove_path(target, deleted_items, errors, label)

    # Remove ingestion state files and progress artifacts.
    purge_ingestion_state_files(db_root, deleted_items, errors=errors)

    # Extended clean-start artifacts from DB root only.
    if include_extended_state:
        extended_targets = [
            ("Entity profiles", db_root / "entity_profiles"),
            ("Workspaces", db_root / "workspaces"),
            ("Structured data", db_root / "structured_data"),
            ("Structured knowledge", db_root / "structured_knowledge.json"),
            ("Entities cache", db_root / "entities.json"),
        ]
        for label, target in extended_targets:
            _remove_path(target, deleted_items, errors, label)

        # Delete any residual staging files directly under DB root.
        for state_file in db_root.glob("staging_*.json"):
            _remove_path(state_file, deleted_items, errors, "Staging file")

    clear_ingestion_session_state("maintenance.reset.complete")
    return deleted_items, errors

def delete_ingested_document_database(db_path: str):
    """Delete the ingested document database with proper error handling and logging."""
    try:
        deleted_items, errors = _reset_database_artifacts(db_path, include_extended_state=False)
        if deleted_items:
            st.success("✅ Database artifacts deleted:\n" + "\n".join(f"- {item}" for item in deleted_items))
        if errors:
            st.error("❌ Some items could not be deleted:\n" + "\n".join(f"- {err}" for err in errors))
        if not deleted_items and not errors:
            st.warning("⚠️ No ingested database artifacts were found to delete.")
    except Exception as e:
        logger.error(f"Ingested document database deletion failed: {e}")
        st.error(f"❌ Failed to delete ingested document database: {e}")

def perform_clean_start(db_path: str):
    """Perform complete KB reset with a concise, deterministic cleanup flow."""
    try:
        resolved = resolve_db_root_path(db_path)
        if not resolved:
            st.error("❌ Database path is empty. Set a valid knowledge base path before running Clean Start.")
            return

        st.info(f"🧹 Running Clean Start for `{resolved}`")
        deleted_items, errors = _reset_database_artifacts(str(resolved), include_extended_state=True)

        st.success(f"✅ Clean Start completed. Removed {len(deleted_items)} item(s).")
        with st.expander("Show removed items", expanded=False):
            if deleted_items:
                st.text("\n".join(f"- {item}" for item in deleted_items))
            else:
                st.text("No matching artifacts were found.")

        if errors:
            st.warning(f"⚠️ {len(errors)} cleanup issue(s) were reported.")
            with st.expander("Show cleanup warnings", expanded=False):
                st.text("\n".join(f"- {err}" for err in errors))

        # Verify critical files are gone.
        verify_targets = [
            resolved / "knowledge_hub_db",
            resolved / "working_collections.json",
            resolved / "knowledge_cortex.gpickle",
            resolved / "staging_ingestion.json",
            resolved / "batch_state.json",
        ]
        remaining = [str(p) for p in verify_targets if p.exists()]
        if remaining:
            st.error("❌ Clean Start incomplete. Remaining critical artifacts:\n" + "\n".join(f"- {p}" for p in remaining))
        else:
            st.success("✅ Verification passed: no critical KB artifacts remain.")

        st.info(
            "Next: go to Knowledge Ingest, keep the same DB path, and start a fresh ingest. "
            "Collection assignment now happens only during finalization."
        )
    except Exception as e:
        logger.error(f"Clean Start failed: {e}")
        st.error(f"❌ Clean Start failed: {e}")

@st.cache_resource
def init_chroma_client(db_path):
    """Initialize ChromaDB client for maintenance operations"""
    if not db_path:
        return None
        
    wsl_db_path = convert_windows_to_wsl_path(db_path)
    chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
    
    if not os.path.isdir(chroma_db_path):
        return None
        
    try:
        db_settings = ChromaSettings(anonymized_telemetry=False)
        return chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return None

def load_maintenance_config():
    """Load configuration for maintenance operations"""
    if st.session_state.maintenance_config is None:
        try:
            config_manager = ConfigManager()
            config = config_manager.get_config()
            
            # Map ConfigManager keys to expected keys - use proper default path detection
            from cortex_engine.utils.default_paths import get_default_ai_database_path
            default_db_path = get_default_ai_database_path()
            maintenance_config = {
                'db_path': config.get('ai_database_path', default_db_path),
                'knowledge_source_path': config.get('knowledge_source_path', ''),
            }
            
            st.session_state.maintenance_config = maintenance_config
            return maintenance_config
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            return None
    return st.session_state.maintenance_config

def clear_ingestion_log_file():
    """Clear the ingestion log file to allow re-ingestion of all files"""
    try:
        config = load_maintenance_config()
        if not config:
            return
        
        from cortex_engine.utils.default_paths import get_default_ai_database_path
        db_path = config.get('db_path', get_default_ai_database_path())
        db_root = resolve_db_root_path(db_path)
        if not db_root:
            st.error("❌ Database path is empty. Set a database path before clearing logs.")
            return

        log_path = db_root / "knowledge_hub_db" / INGESTED_FILES_LOG
        
        if log_path.exists():
            os.remove(log_path)
            st.success(f"✅ Ingestion log cleared successfully: {log_path}")
            logger.info(f"Ingestion log cleared: {log_path}")
        else:
            st.warning(f"Log file not found: {log_path}")
    except Exception as e:
        st.error(f"❌ Failed to clear ingestion log: {e}")
        logger.error(f"Failed to clear ingestion log: {e}")

def delete_ingested_document_database_simple(db_path):
    """Deprecated wrapper for backward compatibility."""
    logger.info("delete_ingested_document_database_simple() is deprecated; delegating to canonical delete function")
    delete_ingested_document_database(db_path)

def display_header():
    """Display page header with navigation and information"""
    st.title("🔧 7. Maintenance & Administration")
    st.caption(f"Version: {PAGE_VERSION} • Consolidated System Maintenance Interface")
    
    # Quick access to System Terminal
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💻 Open System Terminal", use_container_width=True, help="Access the secure system terminal for command execution"):
            st.switch_page("pages/_System_Terminal.py")
    
    st.markdown("""
    **⚠️ Important:** This page contains powerful system maintenance functions that can modify or delete data.  
    Please use these functions with caution and ensure you have appropriate backups before proceeding.
    """)
    
    st.divider()


@st.cache_data(ttl=120)
def _get_detected_db_dimension(db_path: str):
    """Cache embedding-dimension detection for maintenance diagnostics."""
    if not db_path:
        return None
    try:
        from cortex_engine.utils.embedding_validator import get_database_embedding_dimension
        return get_database_embedding_dimension(db_path)
    except Exception as e:
        logger.debug(f"Could not detect embedding dimension for maintenance diagnostics: {e}")
        return None


_CREDIBILITY_BY_VALUE = {
    5: ("peer-reviewed", "Peer-Reviewed"),
    4: ("institutional", "Institutional"),
    3: ("pre-print", "Pre-Print"),
    2: ("editorial", "Editorial"),
    1: ("commentary", "Commentary"),
    0: ("unclassified", "Unclassified"),
}
_CREDIBILITY_BY_KEY = {k: (v, label) for v, (k, label) in _CREDIBILITY_BY_VALUE.items()}
_CREDIBILITY_BY_LABEL = {label.lower(): (v, key) for v, (key, label) in _CREDIBILITY_BY_VALUE.items()}


def _normalize_credibility_tier(metadata_json: dict) -> None:
    if not isinstance(metadata_json, dict):
        return

    default_value = 0
    default_key, default_label = _CREDIBILITY_BY_VALUE[default_value]

    raw_value = metadata_json.get("credibility_tier_value")
    raw_key = str(metadata_json.get("credibility_tier_key", "")).strip().lower()
    raw_label = str(metadata_json.get("credibility_tier_label", "")).strip().lower()

    if raw_value is not None:
        try:
            value = int(raw_value)
            if value in _CREDIBILITY_BY_VALUE:
                key, label = _CREDIBILITY_BY_VALUE[value]
                metadata_json["credibility_tier_value"] = value
                metadata_json["credibility_tier_key"] = key
                metadata_json["credibility_tier_label"] = label
                return
        except Exception:
            pass

    if raw_key in _CREDIBILITY_BY_KEY:
        value, label = _CREDIBILITY_BY_KEY[raw_key]
        metadata_json["credibility_tier_value"] = value
        metadata_json["credibility_tier_key"] = raw_key
        metadata_json["credibility_tier_label"] = label
        return

    if raw_label in _CREDIBILITY_BY_LABEL:
        value, key = _CREDIBILITY_BY_LABEL[raw_label]
        label = _CREDIBILITY_BY_VALUE[value][1]
        metadata_json["credibility_tier_value"] = value
        metadata_json["credibility_tier_key"] = key
        metadata_json["credibility_tier_label"] = label
        return

    metadata_json["credibility_tier_value"] = default_value
    metadata_json["credibility_tier_key"] = default_key
    metadata_json["credibility_tier_label"] = default_label


def _detect_document_stage(meta: dict, doc_text: str) -> str:
    file_name = str(meta.get("file_name", "") or "")
    source_type = str(meta.get("source_type", "") or "")
    combined = f"{file_name}\n{source_type}\n{doc_text}".lower()
    return "Draft" if "draft" in combined else "Final"


def _detect_marker_credibility_value(meta: dict, doc_text: str) -> int:
    source_type = str(meta.get("source_type", "") or "")
    file_name = str(meta.get("file_name", "") or "")
    publisher = str(meta.get("publisher", "") or "")
    available_at = str(meta.get("available_at", "") or "")
    summary = str(meta.get("summary", "") or "")
    combined = f"{source_type}\n{file_name}\n{publisher}\n{available_at}\n{summary}\n{doc_text}".lower()

    marker_map = {
        5: ["pubmed", "nlm", "nature", "lancet", "jama", "bmj", "peer-reviewed", "peer reviewed"],
        4: ["who", "un ", "ipcc", "oecd", "world bank", "government", "department", "ministry", "university", "institute", "centre", "center"],
        3: ["arxiv", "ssrn", "biorxiv", "researchgate", "preprint", "pre-print"],
        2: ["scientific american", "the conversation", "hbr", "harvard business review", "editorial"],
        1: ["blog", "newsletter", "opinion", "consulting report", "whitepaper", "white paper"],
    }
    for tier in (5, 4, 3, 2, 1):
        if any(marker in combined for marker in marker_map[tier]):
            return tier
    return 0


def _enforce_credibility_policy_inplace(meta: dict, doc_text: str) -> bool:
    if not isinstance(meta, dict):
        return False

    before = dict(meta)

    if "credibility_tier_value" not in meta and "credibility_value" in meta:
        meta["credibility_tier_value"] = meta.get("credibility_value")
    if "credibility_tier_key" not in meta and "credibility_source" in meta:
        meta["credibility_tier_key"] = str(meta.get("credibility_source", "")).strip().lower()

    _normalize_credibility_tier(meta)

    normalized_value = int(meta.get("credibility_tier_value", 0) or 0)
    marker_value = _detect_marker_credibility_value(meta, doc_text)
    source_type = str(meta.get("source_type", "") or "")
    combined_lower = f"{meta.get('summary', '')}\n{doc_text}".lower()
    ai_markers = ["ai generated report", "generated by ai", "chatgpt", "openai", "claude", "gemini", "perplexity"]
    is_ai_generated = source_type.lower() == "ai generated report" or any(m in combined_lower for m in ai_markers)

    if is_ai_generated:
        final_value = 1
    elif marker_value > 0:
        final_value = marker_value
    else:
        final_value = normalized_value

    key, label = _CREDIBILITY_BY_VALUE.get(final_value, _CREDIBILITY_BY_VALUE[0])
    stage = _detect_document_stage(meta, doc_text)
    meta["credibility_tier_value"] = final_value
    meta["credibility_tier_key"] = key
    meta["credibility_tier_label"] = label
    meta["credibility"] = f"{stage} {label} Report"
    return meta != before


def _extract_candidate_doc_ids(meta: dict) -> set[str]:
    candidates = set()
    if not isinstance(meta, dict):
        return candidates

    for key in ("doc_id", "document_id", "ref_doc_id"):
        value = meta.get(key)
        if value:
            candidates.add(str(value))

    node_content = meta.get("_node_content")
    if isinstance(node_content, str) and node_content.strip():
        try:
            parsed = json.loads(node_content)
            md = parsed.get("metadata", {}) if isinstance(parsed, dict) else {}
            nested_doc_id = md.get("doc_id") if isinstance(md, dict) else None
            if nested_doc_id:
                candidates.add(str(nested_doc_id))
        except Exception:
            pass

    return candidates


def _extract_primary_doc_key(meta: dict, chunk_id: str) -> str:
    candidates = _extract_candidate_doc_ids(meta)
    if candidates:
        # Prefer deterministic lexical choice for stable grouping.
        return sorted(candidates)[0]
    doc_path = str(meta.get("doc_posix_path", "") or "").strip()
    if doc_path:
        return f"path:{doc_path}"
    file_name = str(meta.get("file_name", "") or "").strip()
    if file_name:
        return f"file:{file_name}"
    return f"chunk:{chunk_id}"


def _extract_stage_from_credibility(credibility_text: str) -> str:
    text = str(credibility_text or "").strip().lower()
    return "Draft" if text.startswith("draft ") else "Final"


def _run_credibility_retrofit(db_path: str, selected_collections: list, dry_run: bool, include_docs: bool) -> dict:
    client = init_chroma_client(db_path)
    if not client:
        raise RuntimeError("Could not connect to ChromaDB client.")

    collection = client.get_collection(COLLECTION_NAME)
    working_mgr = WorkingCollectionManager()
    requested_ids = set()
    for name in selected_collections:
        requested_ids.update(working_mgr.get_doc_ids_by_name(name))

    # Collections can contain either canonical doc_id values or chunk IDs.
    # Resolve both so retrofit scope behaves as users expect.
    allowed_doc_ids = set(requested_ids)
    allowed_chunk_ids = set()
    if requested_ids:
        ids_list = list(requested_ids)
        for i in range(0, len(ids_list), 200):
            probe_ids = ids_list[i:i + 200]
            try:
                resolved = collection.get(ids=probe_ids, include=["metadatas"])
                for rid, rmeta in zip(resolved.get("ids", []), resolved.get("metadatas", [])):
                    allowed_chunk_ids.add(rid)
                    if isinstance(rmeta, dict) and rmeta.get("doc_id"):
                        allowed_doc_ids.add(str(rmeta.get("doc_id")))
            except Exception:
                # Ignore probe failures and keep best-effort scope matching.
                continue

    total = collection.count()
    if total <= 0:
        return {"processed": 0, "updated": 0, "skipped": 0, "errors": []}

    processed = 0
    updated = 0
    skipped = 0
    errors = []

    include_fields = ["metadatas", "documents"] if include_docs else ["metadatas"]
    batch_size = 300
    scoped_records = []
    for offset in range(0, total, batch_size):
        batch = collection.get(limit=batch_size, offset=offset, include=include_fields)
        ids = batch.get("ids", [])
        metadatas = batch.get("metadatas", [])
        documents = batch.get("documents", []) if include_docs else [""] * len(ids)

        for cid, meta, doc in zip(ids, metadatas, documents):
            if not isinstance(meta, dict):
                skipped += 1
                continue
            if requested_ids:
                doc_candidates = _extract_candidate_doc_ids(meta)
                in_scope = (
                    bool(doc_candidates.intersection(allowed_doc_ids))
                    or (cid in allowed_chunk_ids)
                    or (cid in requested_ids)
                )
                if not in_scope:
                    skipped += 1
                    continue

            processed += 1
            scoped_records.append((cid, dict(meta), str(doc or "")))

    # First pass: per-chunk classification
    classified = []
    groups = {}
    for cid, safe_meta, doc_text in scoped_records:
        try:
            _enforce_credibility_policy_inplace(safe_meta, doc_text)
            doc_key = _extract_primary_doc_key(safe_meta, cid)
            tier_value = int(safe_meta.get("credibility_tier_value", 0) or 0)
            stage = _extract_stage_from_credibility(safe_meta.get("credibility", ""))
            classified.append((cid, safe_meta, doc_key, tier_value, stage))
            groups.setdefault(doc_key, {"tiers": {}, "stages": {"Final": 0, "Draft": 0}})
            groups[doc_key]["tiers"][tier_value] = groups[doc_key]["tiers"].get(tier_value, 0) + 1
            groups[doc_key]["stages"][stage] = groups[doc_key]["stages"].get(stage, 0) + 1
        except Exception as e:
            errors.append(f"{cid}: {e}")

    # Choose one credibility tier per document (majority vote; tie -> higher tier).
    group_choice = {}
    for doc_key, payload in groups.items():
        tier_counts = payload["tiers"]
        chosen_tier = sorted(tier_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
        chosen_stage = "Draft" if payload["stages"].get("Draft", 0) > 0 else "Final"
        group_choice[doc_key] = (chosen_tier, chosen_stage)

    update_ids = []
    update_metas = []
    for cid, safe_meta, doc_key, _, _ in classified:
        chosen_tier, chosen_stage = group_choice[doc_key]
        chosen_key, chosen_label = _CREDIBILITY_BY_VALUE.get(chosen_tier, _CREDIBILITY_BY_VALUE[0])
        final_cred = f"{chosen_stage} {chosen_label} Report"

        before_tuple = (
            safe_meta.get("credibility_tier_value"),
            safe_meta.get("credibility_tier_key"),
            safe_meta.get("credibility_tier_label"),
            safe_meta.get("credibility"),
        )
        after_tuple = (chosen_tier, chosen_key, chosen_label, final_cred)

        safe_meta["credibility_tier_value"] = chosen_tier
        safe_meta["credibility_tier_key"] = chosen_key
        safe_meta["credibility_tier_label"] = chosen_label
        safe_meta["credibility"] = final_cred

        if before_tuple != after_tuple:
            updated += 1
            if not dry_run:
                update_ids.append(cid)
                update_metas.append(safe_meta)

    if update_ids and not dry_run:
        for i in range(0, len(update_ids), 300):
            collection.update(ids=update_ids[i:i + 300], metadatas=update_metas[i:i + 300])

    return {"processed": processed, "updated": updated, "skipped": skipped, "errors": errors}

def display_database_maintenance():
    """Display database maintenance and recovery functions"""
    st.header("🗄️ Database Maintenance")
    
    config = load_maintenance_config()
    if not config:
        st.error("Cannot load configuration for database operations")
        return
    
    # Use proper default path detection
    from cortex_engine.utils.default_paths import get_default_ai_database_path
    default_db_path = get_default_ai_database_path()
    db_path = config.get('ai_database_path', config.get('db_path', default_db_path))

    # Database Path Configuration block
    with st.container(border=True):
        st.subheader("📁 Database Path Configuration")
        docker_mode = os.path.exists('/.dockerenv')
        st.caption(f"Environment: {'🐳 Docker' if docker_mode else '💻 Host'}")

        # Current value and normalized preview
        current_input = st.text_input(
            "AI Database Path",
            value=db_path,
            help="Enter the folder that contains your knowledge base (e.g., C:/ai_databases)."
        )
        # Keep the most recent user input in session state for downstream actions
        st.session_state["maintenance_current_db_input"] = current_input
        try:
            preview = convert_windows_to_wsl_path(current_input)
            st.code(f"Resolved path: {preview}")
        except Exception:
            pass

        runtime_root = resolve_db_root_path(current_input)
        runtime_base = str(runtime_root) if runtime_root else (current_input or "")
        runtime_path = convert_to_docker_mount_path(runtime_base) if runtime_base else ""
        detected_dim = _get_detected_db_dimension(runtime_path) if runtime_path else None
        dim_text = f"{detected_dim}D" if detected_dim else "unknown/new database"
        path_text = runtime_path or "(not resolved)"
        st.caption(f"🧪 Runtime DB path: `{path_text}` • detected embedding dimension: `{dim_text}`")

    # Embedding Model Status Display
    shared_render_embedding_model_status_panel()

    # Database Embedding Inspector
    shared_render_database_embedding_inspector_panel(
        config_manager_cls=ConfigManager,
        convert_windows_to_wsl_path_fn=convert_windows_to_wsl_path,
        chromadb_module=chromadb,
        chroma_settings_cls=ChromaSettings,
        collection_name=COLLECTION_NAME,
        logger=logger,
    )

    shared_render_database_path_tools(
        current_input=current_input,
        docker_mode=docker_mode,
        convert_windows_to_wsl_path_fn=convert_windows_to_wsl_path,
        config_manager_cls=ConfigManager,
        discovered_paths_key="discovered_db_paths",
    )
    
    # Database Operations section header
    st.markdown("---")

    shared_render_database_health_check_section(
        db_path=db_path,
        recovery_manager_cls=IngestionRecoveryManager,
    )

    shared_render_reset_recovery_section(
        db_path=db_path,
        ingested_files_log=INGESTED_FILES_LOG,
        clear_ingestion_log_file_fn=clear_ingestion_log_file,
        perform_clean_start_fn=perform_clean_start,
        delete_ingested_document_database_fn=delete_ingested_document_database,
        expander_title="🧹 Reset & Recovery",
    )

    # Add database deduplication section
    shared_render_database_dedup_section(
        db_path=db_path,
        init_chroma_client_fn=init_chroma_client,
        collection_name=COLLECTION_NAME,
        working_collection_manager_cls=WorkingCollectionManager,
        logger=logger,
    )

    with st.expander("🏷️ Credibility Metadata Retrofit", expanded=False):
        st.caption("Batch-apply canonical credibility metadata to existing Chroma documents.")
        try:
            mgr = WorkingCollectionManager()
            collections = mgr.list_collections()
        except Exception as e:
            collections = []
            st.warning(f"Could not load working collections: {e}")

        selected_collections = st.multiselect(
            "Collections to retrofit",
            options=collections,
            default=collections,
            key="maintenance_credibility_collections",
            help="Only documents belonging to selected collections will be updated.",
        )
        include_docs = st.checkbox(
            "Use document text during classification",
            value=False,
            key="maintenance_credibility_include_docs",
            help="Enables marker checks against chunk text. Slower on large databases.",
        )
        dry_run = st.checkbox(
            "Dry run (show counts, do not write)",
            value=True,
            key="maintenance_credibility_dry_run",
        )

        if st.button("Run Credibility Retrofit", key="maintenance_run_credibility_retrofit", type="primary"):
            if not selected_collections:
                st.error("Select at least one collection.")
            else:
                with st.spinner("Applying credibility policy to collection metadata..."):
                    try:
                        result = _run_credibility_retrofit(
                            db_path=db_path,
                            selected_collections=selected_collections,
                            dry_run=dry_run,
                            include_docs=include_docs,
                        )
                        mode_text = "Dry run complete" if dry_run else "Retrofit complete"
                        st.success(f"{mode_text}: processed={result['processed']} updated={result['updated']} skipped={result['skipped']}")
                        if result["errors"]:
                            st.warning(f"Completed with {len(result['errors'])} item errors. Showing first 10:")
                            for err in result["errors"][:10]:
                                st.code(err)
                    except Exception as e:
                        st.error(f"Retrofit failed: {e}")

    st.info("💡 Tip: Run **Database Health Check** first; use **Reset & Recovery** only when repair is insufficient.")

def display_system_terminal():
    """Display system terminal and command execution interface"""
    st.header("💻 System Terminal")
    
    st.markdown("""
    This secure terminal interface allows you to execute system commands safely within the Cortex Suite environment.
    Only whitelisted commands are permitted to ensure system security.
    """)
    
    # Quick action buttons for common tasks
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📦 Check Models", use_container_width=True):
            st.session_state.quick_command = "ollama list"
            st.rerun()
    
    with col2:
        if st.button("🔍 System Status", use_container_width=True):
            st.session_state.quick_command = "docker ps"
            st.rerun()
    
    with col3:
        if st.button("📊 Disk Usage", use_container_width=True):
            st.session_state.quick_command = "df -h"
            st.rerun()
    
    with col4:
        if st.button("🔄 Clear Terminal", use_container_width=True):
            st.session_state.quick_command = None
            if 'terminal_output' in st.session_state:
                del st.session_state.terminal_output
            st.rerun()
    
    # Display command executor widget
    try:
        display_command_executor_widget()
    except Exception as e:
        st.error(f"Failed to load command executor: {e}")

def display_setup_maintenance():
    """Display setup and installation maintenance functions"""
    st.header("⚙️ Setup & Installation")
    
    with st.expander("🔄 Reset System Setup", expanded=False):
        st.markdown("""
        Reset the system setup state if installation gets stuck or needs to be rerun.
        This will clear setup progress and allow you to run the setup wizard again.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset Setup State", use_container_width=True):
                try:
                    setup_manager = SetupManager()
                    setup_manager.reset_setup()
                    
                    # Clear related session state
                    keys_to_clear = [k for k in st.session_state.keys() if 'setup' in k.lower() or 'installation' in k.lower()]
                    for key in keys_to_clear:
                        del st.session_state[key]
                    
                    st.success("✅ Setup state reset successfully!")
                    logger.info("Setup state reset via maintenance page")
                    
                except Exception as e:
                    st.error(f"❌ Failed to reset setup: {e}")
        
        with col2:
            st.info("After resetting, you can navigate to the Setup Wizard to reconfigure the system.")

def display_backup_management():
    """Display database portability functions for transfer between machines"""
    st.header("🔄 Database Transfer")

    config = load_maintenance_config()
    if not config:
        st.error("Cannot load configuration for transfer operations")
        return

    try:
        from cortex_engine.utils.default_paths import get_default_ai_database_path
        db_path = config.get('db_path', get_default_ai_database_path())

        st.caption("Transfer your knowledge base between machines with automatic embedding model configuration.")

        # ----- EXPORT DATABASE -----
        with st.expander("📤 Export Database", expanded=False):
            st.markdown("""
            Create a portable database package that can be transferred to another machine.
            The export includes embedding model configuration for automatic setup on import.
            """)

            export_col1, export_col2 = st.columns([2, 1])

            with export_col1:
                export_dest = st.text_input(
                    "Destination Folder:",
                    value=str(Path.home() / "cortex_exports"),
                    key="export_destination_folder",
                    help="Folder where the export zip will be saved"
                )

            with export_col2:
                export_name = st.text_input(
                    "Export Name (optional):",
                    value="",
                    key="export_custom_name",
                    help="Custom name for the export (timestamp added automatically)"
                )

            if st.button("📦 Create Portable Export", type="primary", use_container_width=True, key="btn_create_export"):
                try:
                    from cortex_engine.database_exporter import DatabaseExporter, get_export_summary

                    exporter = DatabaseExporter(db_path)

                    # Progress display
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(stage: str, percent: int):
                        progress_bar.progress(percent)
                        status_text.text(f"{stage}...")

                    success, message, manifest = exporter.create_export(
                        destination_folder=export_dest,
                        export_name=export_name if export_name else None,
                        progress_callback=update_progress
                    )

                    if success:
                        st.success(f"✅ {message}")

                        # Show export summary
                        if manifest:
                            summary = get_export_summary(manifest)
                            with st.container(border=True):
                                st.markdown("**Export Summary:**")
                                sum_col1, sum_col2, sum_col3 = st.columns(3)
                                with sum_col1:
                                    st.metric("Documents", summary["documents"])
                                    st.caption(f"Model: {summary['model_size']}")
                                with sum_col2:
                                    st.metric("Collections", summary["collections"])
                                    st.caption(f"Dims: {summary['dimensions']}")
                                with sum_col3:
                                    st.metric("VRAM Required", summary["vram_required"])
                                    st.caption(f"Version: {summary['cortex_version']}")
                    else:
                        st.error(f"❌ {message}")

                except Exception as e:
                    st.error(f"❌ Export failed: {e}")

        # ----- IMPORT DATABASE -----
        with st.expander("📥 Import Database", expanded=False):
            st.markdown("""
            Import a portable database package from another machine.
            The embedding model will be automatically configured based on the export settings.
            """)

            # File uploader for zip files
            uploaded_zip = st.file_uploader(
                "Choose a database export package:",
                type=['zip'],
                key="import_zip_uploader",
                help="Select a cortex_export_*.zip file"
            )

            import_path = None
            if uploaded_zip:
                # Save uploaded file to temp location
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / "cortex_imports"
                temp_dir.mkdir(exist_ok=True)

                temp_zip_path = temp_dir / uploaded_zip.name
                with open(temp_zip_path, 'wb') as f:
                    f.write(uploaded_zip.getvalue())

                import_path = str(temp_zip_path)
                st.success(f"📦 File loaded: {uploaded_zip.name} ({uploaded_zip.size / (1024*1024):.1f} MB)")

            if import_path:
                # Validate button
                if st.button("🔍 Validate Package", use_container_width=True, key="btn_validate_import"):
                    try:
                        from cortex_engine.database_exporter import DatabaseImporter, get_export_summary

                        importer = DatabaseImporter(import_path)
                        valid, message, manifest = importer.validate()

                        if valid and manifest:
                            st.success(f"✅ {message}")

                            # Show package info
                            summary = get_export_summary(manifest)
                            with st.container(border=True):
                                st.markdown("**Package Contents:**")
                                info_col1, info_col2 = st.columns(2)
                                with info_col1:
                                    st.write(f"**Source:** {summary['source']}")
                                    st.write(f"**Date:** {summary['date'][:10]}")
                                    st.write(f"**Documents:** {summary['documents']}")
                                    st.write(f"**Collections:** {summary['collections']}")
                                with info_col2:
                                    st.write(f"**Model:** {summary['embedding_model'].split('/')[-1]}")
                                    st.write(f"**Size:** {summary['model_size']}")
                                    st.write(f"**Dimensions:** {summary['dimensions']}")
                                    st.write(f"**VRAM Required:** {summary['vram_required']}")

                            # Check hardware compatibility
                            compatible, compat_msg, details = importer.check_hardware_compatibility()

                            if compatible:
                                st.success(f"✅ {compat_msg}")
                            else:
                                st.warning(f"⚠️ {compat_msg}")

                            st.caption(f"Your GPU: {details.get('gpu_name', 'Unknown')} ({details.get('available_vram_gb', 0):.1f} GB)")

                            # Store validation state
                            st.session_state.import_validated = True
                            st.session_state.import_manifest = manifest
                            st.session_state.import_compatible = compatible
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.import_validated = False

                    except Exception as e:
                        st.error(f"❌ Validation failed: {e}")
                        st.session_state.import_validated = False

                # Show import options if validated
                if st.session_state.get('import_validated', False):
                    st.markdown("---")

                    overwrite_confirm = st.checkbox(
                        "I understand this will replace my current database",
                        key="import_overwrite_confirm",
                        help="Your existing database will be backed up before replacement"
                    )

                    auto_scan_fix = st.checkbox(
                        "Auto-scan and fix database after import",
                        value=True,
                        key="import_auto_scan_fix",
                        help="Recommended: Removes orphaned entries from the ingestion log so you start with a clean database"
                    )

                    import_disabled = not overwrite_confirm

                    if st.button(
                        "📥 Import Now",
                        type="primary",
                        use_container_width=True,
                        disabled=import_disabled,
                        key="btn_import_now"
                    ):
                        try:
                            from cortex_engine.database_exporter import DatabaseImporter

                            importer = DatabaseImporter(import_path)

                            # Progress display
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            def update_progress(stage: str, percent: int):
                                progress_bar.progress(percent)
                                status_text.text(f"{stage}...")

                            success, message = importer.import_database(
                                destination_path=db_path,
                                overwrite=True,
                                progress_callback=update_progress
                            )

                            if success:
                                st.success(f"✅ {message}")

                                # CRITICAL: Clear stale ingestion state from previous ingest attempts
                                # This prevents the UI from showing batch progress from a different database
                                try:
                                    from cortex_engine.batch_manager import clear_all_ingestion_state
                                    cleanup_result = clear_all_ingestion_state(db_path)
                                    if cleanup_result["batch_state_cleared"] or cleanup_result["staging_cleared"]:
                                        st.info("🧹 Cleared stale ingestion state from previous database")
                                    if cleanup_result["errors"]:
                                        for err in cleanup_result["errors"]:
                                            st.warning(f"⚠️ Cleanup warning: {err}")
                                except Exception as cleanup_e:
                                    st.warning(f"⚠️ Could not clear ingestion state: {cleanup_e}")

                                # Run auto-scan and fix if enabled
                                if auto_scan_fix:
                                    status_text.text("🔍 Scanning database for inconsistencies...")
                                    progress_bar.progress(90)

                                    try:
                                        from cortex_engine.ingestion_recovery import IngestionRecoveryManager

                                        recovery = IngestionRecoveryManager(db_path)
                                        analysis = recovery.analyze_ingestion_state()

                                        orphaned_count = analysis['statistics'].get('orphaned_count', 0)
                                        if orphaned_count > 0:
                                            status_text.text(f"🔧 Found {orphaned_count} orphaned log entries, cleaning up...")
                                            cleanup_result = recovery.cleanup_orphaned_log_entries()

                                            if cleanup_result.get('status') == 'success':
                                                st.success(f"✅ Cleaned up {cleanup_result['entries_removed']} orphaned log entries")
                                                st.caption("These files were in the ingestion log but not in the database. They can now be re-ingested if needed.")
                                            else:
                                                st.warning(f"⚠️ Cleanup encountered issues: {cleanup_result.get('error', 'Unknown')}")
                                        else:
                                            st.success("✅ Database scan complete - no issues found")

                                        progress_bar.progress(100)
                                        status_text.text("✅ Import and cleanup complete!")

                                    except Exception as scan_e:
                                        st.warning(f"⚠️ Auto-scan encountered an error: {scan_e}")
                                        st.caption("Import was successful, but you may want to run maintenance manually.")

                                st.info("🔄 Embedding model has been auto-configured. Refresh the page to use the imported database.")

                                # Clear all relevant session state for clean slate
                                st.session_state.import_validated = False
                                # Clear model caches
                                for key in ['model_info_cache', 'available_models_cache']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                # Clear ingestion-related session state
                                ingest_keys_to_clear = [
                                    'ingestion_process', 'ingestion_running', 'batch_status',
                                    'current_throttle_delay', 'last_progress_update',
                                    'ingestion_output', 'process_output_queue'
                                ]
                                for key in ingest_keys_to_clear:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.info("🧹 Session state cleared - Knowledge Ingest will show fresh state")
                            else:
                                st.error(f"❌ {message}")

                        except Exception as e:
                            st.error(f"❌ Import failed: {e}")

    except Exception as e:
        st.error(f"Failed to initialize backup manager: {e}")

def display_performance_dashboard():
    """Display performance monitoring dashboard with metrics and analytics"""
    st.markdown("## 📊 Performance Monitoring Dashboard")
    st.markdown("Real-time performance metrics for critical operations (v4.9.0+)")

    # Get performance monitor
    monitor = get_performance_monitor()

    # Create columns for high-level metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        session_summary = get_session_summary()
        st.metric(
            "Total Operations",
            session_summary.get("total_operations", 0),
            help="Total number of monitored operations this session"
        )

    with col2:
        cache_stats = monitor.get_cache_stats()
        hit_rate = cache_stats.get("hit_rate", 0.0) * 100
        st.metric(
            "Cache Hit Rate",
            f"{hit_rate:.1f}%",
            help="Percentage of queries served from cache"
        )

    with col3:
        gpu_info = get_gpu_memory_info()
        device_name = gpu_info.device_name
        if len(device_name) > 20:
            device_name = device_name[:17] + "..."
        st.metric(
            "Device",
            device_name,
            help=f"Full name: {gpu_info.device_name}"
        )

    with col4:
        session_duration = session_summary.get("session_duration_formatted", "N/A")
        st.metric(
            "Session Duration",
            session_duration,
            help="Performance monitoring session duration"
        )

    st.markdown("---")

    # Overview Charts
    all_stats = get_all_stats()
    if all_stats:
        st.markdown("### 📊 Performance Overview")

        # Create two columns for overview charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Operation distribution pie chart
            import pandas as pd
            import plotly.express as px

            op_data = []
            for op_type, stats in all_stats.items():
                op_data.append({
                    'Operation': op_type.replace('_', ' ').title(),
                    'Count': stats.total_operations,
                    'Time (s)': stats.total_duration
                })

            if op_data:
                df_ops = pd.DataFrame(op_data)

                fig_pie = px.pie(
                    df_ops,
                    values='Count',
                    names='Operation',
                    title='Operation Distribution',
                    hole=0.4  # Donut chart
                )
                fig_pie.update_layout(height=350)
                st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            # Total time by operation (bar chart)
            if op_data:
                fig_bar = px.bar(
                    df_ops,
                    x='Operation',
                    y='Time (s)',
                    title='Total Time by Operation Type',
                    color='Time (s)',
                    color_continuous_scale='Blues'
                )
                fig_bar.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

    # GPU/Device Information
    with st.expander("🖥️ Device & GPU Information", expanded=True):
        gpu_info = get_gpu_memory_info()
        device_recs = get_device_recommendations()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Device Information")
            st.markdown(f"**Device:** {gpu_info.device_name}")
            st.markdown(f"**Type:** {'CUDA GPU' if gpu_info.is_cuda else ('Apple Silicon (MPS)' if gpu_info.is_mps else 'CPU')}")

            if gpu_info.is_cuda:
                st.markdown(f"**Total Memory:** {gpu_info.total_memory_gb:.2f} GB")
                st.markdown(f"**Free Memory:** {gpu_info.free_memory_gb:.2f} GB")
                st.markdown(f"**Utilization:** {gpu_info.utilization_percent:.1f}%")

                # Progress bar for memory utilization
                st.progress(min(gpu_info.utilization_percent / 100.0, 1.0))

        with col_b:
            st.markdown("### Batch Size Recommendations")
            batch_recs = device_recs.get("batch_recommendations", {})
            perf_tier = device_recs.get("performance_tier", "Unknown")

            st.markdown(f"**Recommended Batch Size:** {batch_recs.get('recommended', 32)}")
            st.markdown(f"**Conservative:** {batch_recs.get('conservative', 32)}")
            st.markdown(f"**Aggressive:** {batch_recs.get('aggressive', 32)}")
            st.markdown(f"**Performance Tier:** {perf_tier}")

    # Operation Statistics
    with st.expander("⏱️ Operation Performance Statistics", expanded=True):
        all_stats = get_all_stats()

        if not all_stats:
            st.info("No performance data collected yet. Run some operations (ingestion, search, etc.) to see metrics.")
        else:
            # Create tabs for different operation types
            op_types = list(all_stats.keys())
            if op_types:
                tabs = st.tabs([f"{'📸' if op == 'image_processing' else '🔢' if op == 'embedding_batch' else '🔍' if op == 'query' else '⚡'} {op.replace('_', ' ').title()}" for op in op_types])

                for tab, op_type in zip(tabs, op_types):
                    with tab:
                        stats = all_stats[op_type]

                        # Metrics row
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Total Ops", stats.total_operations)
                        with m2:
                            st.metric("Successful", stats.successful_operations)
                        with m3:
                            st.metric("Failed", stats.failed_operations)
                        with m4:
                            success_rate = (stats.successful_operations / stats.total_operations * 100) if stats.total_operations > 0 else 0
                            st.metric("Success Rate", f"{success_rate:.1f}%")

                        # Timing statistics
                        st.markdown("### ⏱️ Timing Statistics")

                        t1, t2, t3 = st.columns(3)
                        with t1:
                            st.markdown(f"**Average Duration:** {stats.avg_duration:.3f}s")
                            st.markdown(f"**Min Duration:** {stats.min_duration:.3f}s")
                            st.markdown(f"**Max Duration:** {stats.max_duration:.3f}s")

                        with t2:
                            st.markdown(f"**Median (P50):** {stats.p50_duration:.3f}s")
                            st.markdown(f"**95th Percentile:** {stats.p95_duration:.3f}s")
                            st.markdown(f"**99th Percentile:** {stats.p99_duration:.3f}s")

                        with t3:
                            st.markdown(f"**Total Time:** {stats.total_duration:.2f}s")
                            st.markdown(f"**First Seen:** {stats.first_seen.split('T')[1][:8]}")
                            st.markdown(f"**Last Seen:** {stats.last_seen.split('T')[1][:8]}")

                        # Performance Charts
                        recent = monitor.get_recent_metrics(op_type, limit=50)
                        if recent and len(recent) >= 2:
                            import pandas as pd
                            import plotly.express as px
                            import plotly.graph_objects as go

                            st.markdown("### 📈 Performance Charts")

                            # Create DataFrame for plotting
                            df = pd.DataFrame([
                                {
                                    'timestamp': metric.timestamp,
                                    'duration': metric.duration,
                                    'success': metric.success,
                                    'operation': op_type
                                }
                                for metric in recent
                            ])

                            # Chart 1: Duration over time (line chart)
                            fig1 = px.line(
                                df,
                                x='timestamp',
                                y='duration',
                                title=f'{op_type.replace("_", " ").title()} - Duration Over Time',
                                labels={'duration': 'Duration (seconds)', 'timestamp': 'Time'},
                                markers=True
                            )
                            fig1.add_hline(y=stats.avg_duration, line_dash="dash", line_color="red",
                                          annotation_text=f"Avg: {stats.avg_duration:.3f}s")
                            fig1.update_layout(height=400)
                            st.plotly_chart(fig1, use_container_width=True)

                            # Chart 2: Duration distribution (histogram)
                            fig2 = px.histogram(
                                df,
                                x='duration',
                                title=f'{op_type.replace("_", " ").title()} - Duration Distribution',
                                labels={'duration': 'Duration (seconds)', 'count': 'Frequency'},
                                nbins=20
                            )
                            fig2.add_vline(x=stats.p50_duration, line_dash="dash", line_color="green",
                                          annotation_text=f"P50: {stats.p50_duration:.3f}s")
                            fig2.add_vline(x=stats.p95_duration, line_dash="dash", line_color="orange",
                                          annotation_text=f"P95: {stats.p95_duration:.3f}s")
                            fig2.update_layout(height=350)
                            st.plotly_chart(fig2, use_container_width=True)

                        # Recent operations (text summary)
                        if recent and len(recent) >= 1:
                            st.markdown("### 📋 Recent Operations (Last 5)")
                            for i, metric in enumerate(list(recent)[-5:], 1):
                                status = "✅" if metric.success else "❌"
                                metadata_str = ", ".join(f"{k}={v}" for k, v in metric.metadata.items())
                                st.markdown(f"{i}. {status} **{metric.duration:.3f}s** - {metadata_str}")

    # Cache Statistics
    with st.expander("💾 Query Cache Statistics", expanded=False):
        cache_stats = monitor.get_cache_stats()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Queries", cache_stats.get("total_queries", 0))
        with c2:
            st.metric("Cache Hits", cache_stats.get("cache_hits", 0))
        with c3:
            st.metric("Cache Misses", cache_stats.get("cache_misses", 0))
        with c4:
            hit_rate = cache_stats.get("hit_rate", 0.0) * 100
            st.metric("Hit Rate", f"{hit_rate:.1f}%")

        if cache_stats.get("total_queries", 0) > 0:
            st.markdown(f"**Cache Efficiency:** {cache_stats.get('cache_hits', 0)} instant responses out of {cache_stats.get('total_queries', 0)} queries")

            # Calculate time saved
            query_stats = all_stats.get("query")
            if query_stats and cache_stats.get("cache_hits", 0) > 0:
                avg_search_time = query_stats.avg_duration
                time_saved = cache_stats.get("cache_hits", 0) * avg_search_time
                st.success(f"⚡ Estimated time saved by caching: **{time_saved:.2f} seconds**")

    # Actions
    with st.expander("⚙️ Performance Monitoring Actions", expanded=False):
        col_x, col_y = st.columns(2)

        with col_x:
            if st.button("💾 Save Metrics to File", help="Export current metrics to JSON file"):
                try:
                    file_path = monitor.save_to_file()
                    st.success(f"✅ Metrics saved to: `{file_path}`")
                except Exception as e:
                    st.error(f"❌ Failed to save metrics: {e}")

            if st.button("🔄 Refresh Display", help="Reload performance data"):
                st.rerun()

        with col_y:
            if st.button("🧹 Clear Metrics", help="Reset all performance metrics (new session)", type="primary"):
                monitor.clear()
                st.success("✅ Performance metrics cleared - new session started")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.info("💡 **Tip:** Performance metrics are collected automatically during ingestion, search, and other operations. The data resets when you clear metrics or restart the application.")


def display_changelog_viewer():
    """Display the project changelog viewer"""
    st.markdown("## 📋 Project Changelog")
    st.markdown("View the complete development history and version changes for the Cortex Suite.")
    
    # Get project root path
    project_root = Path(__file__).parent.parent
    changelog_path = project_root / "CHANGELOG.md"
    
    if not changelog_path.exists():
        st.error("❌ CHANGELOG.md not found in project root")
        return
    
    try:
        # Read the changelog
        with open(changelog_path, 'r', encoding='utf-8') as f:
            changelog_content = f.read()
        
        # Display changelog info
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info(f"📁 **File Location:** `{changelog_path.relative_to(project_root)}`")
        
        with col2:
            # Get file stats
            stat = changelog_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            st.info(f"🕒 **Last Updated:** {last_modified}")
        
        with col3:
            # File size
            size_kb = stat.st_size / 1024
            st.info(f"📏 **Size:** {size_kb:.1f} KB")
        
        st.divider()
        
        # Display options
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            show_full = st.checkbox("📖 Show Full Changelog", 
                                  value=st.session_state.get("changelog_show_full", False),
                                  key="changelog_show_full_checkbox")
        
        with col2:
            if st.button("🔄 Refresh", key="changelog_refresh"):
                st.rerun()
        
        with col3:
            # Download button
            st.download_button(
                label="💾 Download Changelog",
                data=changelog_content,
                file_name="CHANGELOG.md",
                mime="text/markdown",
                key="changelog_download"
            )
        
        st.divider()
        
        # Parse and display changelog sections
        if show_full:
            # Show entire changelog
            st.markdown("### 📚 Complete Changelog")
            st.markdown(changelog_content)
        else:
            # Show recent versions (first few entries)
            lines = changelog_content.split('\n')
            
            # Find recent version entries (lines starting with ##)
            version_lines = []
            current_section = []
            version_count = 0
            
            for line in lines:
                if line.startswith('## v') and version_count < 5:  # Show last 5 versions
                    if current_section:
                        version_lines.append('\n'.join(current_section))
                        current_section = []
                        version_count += 1
                    current_section = [line]
                elif line.startswith('## ') and not line.startswith('## ['):
                    # Stop at non-version headers
                    break
                elif current_section:
                    current_section.append(line)
            
            # Add the last section
            if current_section and version_count < 5:
                version_lines.append('\n'.join(current_section))
            
            if version_lines:
                st.markdown("### 🆕 Recent Updates (Last 5 Versions)")
                for section in version_lines:
                    st.markdown(section)
                    st.divider()
                
                st.info("💡 **Tip:** Check 'Show Full Changelog' above to see complete version history.")
            else:
                st.warning("⚠️ Could not parse changelog sections. Showing raw content:")
                st.text(changelog_content[:2000] + "..." if len(changelog_content) > 2000 else changelog_content)
    
    except Exception as e:
        st.error(f"❌ Failed to read changelog: {e}")
        logger.error(f"Changelog viewer error: {e}")

def main():
    """Main function to orchestrate the maintenance interface"""
    display_header()
    
    # Create tabs for different maintenance categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗄️ Database",
        "💻 Terminal",
        "⚙️ Setup",
        "🔄 Transfer",
        "📊 Performance",
        "📋 Changelog",
        "ℹ️ Info"
    ])

    with tab1:
        display_database_maintenance()

    with tab2:
        display_system_terminal()

    with tab3:
        display_setup_maintenance()

    with tab4:
        display_backup_management()

    with tab5:
        display_performance_dashboard()

    with tab6:
        display_changelog_viewer()

    with tab7:
        st.markdown("""
        ## 📋 Maintenance Information

        This maintenance interface consolidates system administration functions:

        **🗄️ Database Tab:**
        - Configure database path and embedding model settings
        - **Health Check** - Scan for and fix orphaned entries, collection issues
        - Clear ingestion logs, delete/rebuild knowledge base
        - Database deduplication and optimization
        - System reset (Clean Start) for severe issues

        **💻 Terminal Tab:**
        - Execute safe system commands
        - Check model availability and system status
        - Monitor disk usage and resources

        **⚙️ Setup Tab:**
        - Reset installation state if setup gets stuck

        **🔄 Transfer Tab:**
        - Export portable database packages with embedding model config
        - Import databases with automatic model configuration
        - Transfer knowledge bases between machines

        **📊 Performance Tab:**
        - Monitor operation performance
        - View query cache statistics
        - GPU and device information

        **⚠️ Important Notes:**
        - Always backup data before destructive operations
        - Use Health Check before Export to ensure clean database
        - Check logs for detailed error information
        """)
        
        st.markdown(f"**Page Version:** {PAGE_VERSION} | **Date:** 2025-08-27")

if __name__ == "__main__":
    main()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
