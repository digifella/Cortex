# ## File: pages/2_Knowledge_Ingest.py [MAIN VERSION]
# Version: v5.7.0
# Date: 2026-01-27
# Purpose: GUI for knowledge base ingestion.
#          - FEATURE (v5.7.0): When navigating to a directory with no subdirectories,
#            now shows individual files with checkboxes for selection. Users can
#            select specific files and click "Proceed with X Selected File(s)" to
#            ingest them directly without needing to select parent directories.
#          - FEATURE (v5.1.0): Added Qwen3-VL multimodal embedding status and
#            configuration display in sidebar. Shows model size, reranker status,
#            and setup instructions when disabled.
#          - UX FIX (v4.11.0): Fixed confusing status display during finalization.
#            Now clearly shows "Finalization in Progress" instead of "Analysis Complete!"
#            while embedding documents. Added FINALIZE_START event handler.
#          - REFACTOR (v39.3.0): Moved maintenance functions to dedicated Maintenance page
#            for better organization and UI cleanup. Functions moved: clear_ingestion_log_file,
#            delete_knowledge_base, and all database recovery tools.
#          - REFACTOR (v39.0.0): Updated to use centralized utilities for path handling,
#            logging, and error handling. Removed code duplication.

import streamlit as st
import os
import json
import subprocess
import sys
import shutil
import re
import shlex
from pathlib import Path
from fnmatch import fnmatch
from collections import defaultdict
from datetime import datetime
from typing import List, Optional
import time
import threading
import queue

import fitz

# Import version from centralized config
from cortex_engine.version_config import VERSION_STRING
import docx

# --- Project Setup ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import centralized utilities
from cortex_engine.utils import (
    convert_windows_to_wsl_path,
    get_logger,
    validate_path_exists,
    convert_to_docker_mount_path,
    convert_source_path_to_docker_mount,
    ensure_directory_writable,
)
from cortex_engine.utils.model_checker import model_checker
from cortex_engine.config import STAGING_INGESTION_FILE, INGESTED_FILES_LOG, DEFAULT_EXCLUSION_PATTERNS_STR
from cortex_engine.config_manager import ConfigManager
from cortex_engine.ingest_cortex import RichMetadata
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.document_type_manager import get_document_type_manager
from cortex_engine.help_system import help_system
from cortex_engine.batch_manager import BatchState
from cortex_engine.ingestion_recovery import IngestionRecoveryManager
from cortex_engine.pre_ingest_organizer import PreIngestScanCancelled, run_pre_ingest_organizer_scan
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.model_manager import (
    get_available_embedding_models,
    get_recommended_embedding_model,
    check_model_cached,
    validate_model_available,
    download_model,
    get_model_info_summary,
    get_pytorch_cuda_install_command
)
from pages.components._Ingest_Maintenance import (
    render_maintenance_link,
    check_recovery_needed as shared_check_recovery_needed,
    render_recovery_section as shared_render_recovery_section,
    render_recovery_quick_actions,
    recover_collection_from_ingest_log,
)
from pages.components._Ingest_Workflow import (
    detect_orphaned_session_from_log,
    render_orphaned_session_notice,
    render_stage as render_ingest_stage,
)
from pages.components._Ingest_Batch import (
    render_active_batch_management as shared_render_active_batch_management,
    render_batch_processing_ui as shared_render_batch_processing_ui,
)
from pages.components._Ingest_Processing import (
    render_log_and_review_ui as shared_render_log_and_review_ui,
)
from pages.components._Ingest_Review import (
    render_metadata_review_ui as shared_render_metadata_review_ui,
)
from pages.components._Ingest_DocTypes import (
    render_document_type_management as shared_render_document_type_management,
)
from pages.components._Ingest_Recovery import (
    render_recovery_panels as shared_render_recovery_panels,
)
from pages.components._Ingest_ServiceStatus import (
    render_ollama_status_panel as shared_render_ollama_status_panel,
)
from pages.components._Ingest_Shell import (
    render_ingest_page_shell as shared_render_ingest_page_shell,
)

# Set up logging
logger = get_logger(__name__)


def install_missing_models(missing_models: list) -> bool:
    """Attempt to install missing models using the model_checker helper commands.

    Returns True on successful installs so the caller can trigger a re-check/rerun.
    """
    if not missing_models:
        st.info("No missing models to install.")
        return False

    service_ok, service_msg = model_checker.check_ollama_service()
    if not service_ok:
        st.error(f"Cannot install models because Ollama is not reachable: {service_msg}")
        return False

    try:
        with st.status("Installing models...", expanded=True) as status:
            for model_name in missing_models:
                cmd = ["ollama", "pull", model_name]
                status.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}")
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    if result.stdout:
                        status.write(result.stdout)
                    if result.stderr:
                        status.write(f"stderr: {result.stderr}")
                except subprocess.CalledProcessError as e:
                    status.write(f"❌ Failed: {e}")
                    if e.stdout:
                        status.write(e.stdout)
                    if e.stderr:
                        status.write(e.stderr)
                    status.update(label="Model install failed", state="error")
                    return False
            status.update(label="Models installed", state="complete")
        return True
    except Exception as e:
        st.error(f"Failed to install models automatically: {e}")
        return False

st.set_page_config(layout="wide", page_title="Knowledge Ingest")

# Apply refined editorial theme
apply_theme()

# Add global CSS for left-aligned directory buttons and light-mode code blocks
st.markdown("""
<style>
/* Left-align folder directory navigation buttons */
.stButton > button:has-text("📁") {
    text-align: left !important;
    justify-content: flex-start !important;
}
/* Alternative selector for buttons containing folder emoji */
button[data-testid="baseButton-secondary"]:contains("📁") {
    text-align: left !important;
    justify-content: flex-start !important;
}
/* Fallback: target all secondary buttons in directory areas */
div[data-testid="column"] .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
}

/* Ensure readable high-contrast buttons regardless of app theme */
div[data-testid="stButton"] > button,
.stButton > button {
    background-color: #F8F9FA !important;
    color: #111111 !important;
    border: 1px solid #C5CCD3 !important;
}
div[data-testid="stButton"] > button:hover,
.stButton > button:hover {
    background-color: #EEF2F6 !important;
    color: #111111 !important;
    border-color: #AEB8C3 !important;
}
div[data-testid="stButton"] > button:focus,
.stButton > button:focus {
    color: #111111 !important;
    border-color: #7A8794 !important;
    box-shadow: 0 0 0 2px rgba(122, 135, 148, 0.25) !important;
}
div[data-testid="stButton"] > button:disabled,
.stButton > button:disabled {
    background-color: #ECEFF2 !important;
    color: #6B7280 !important;
    border-color: #D4DAE0 !important;
}

/* Streamlit button variants (primary/secondary) - enforce readable contrast */
button[data-testid="baseButton-primary"] {
    background-color: #2A4362 !important;
    color: #FFFFFF !important;
    border: 1px solid #1F3550 !important;
}
button[data-testid="baseButton-primary"]:hover {
    background-color: #324f73 !important;
    color: #FFFFFF !important;
}
button[data-testid="baseButton-primary"]:disabled {
    background-color: #E2E8F0 !important;
    color: #374151 !important;
    border-color: #CBD5E1 !important;
    opacity: 1 !important;
}
button[data-testid="baseButton-secondary"] {
    background-color: #F8F9FA !important;
    color: #111111 !important;
    border: 1px solid #C5CCD3 !important;
}
button[data-testid="baseButton-secondary"]:disabled {
    background-color: #ECEFF2 !important;
    color: #4B5563 !important;
    border-color: #D4DAE0 !important;
    opacity: 1 !important;
}
/* Force light mode for code blocks (processing log) */
pre, code, .stCodeBlock, [data-testid="stCodeBlock"] {
    background-color: #F8F9FA !important;
    color: #1A1A1A !important;
}
[data-testid="stCodeBlock"] code {
    background-color: #F8F9FA !important;
    color: #1A1A1A !important;
}
/* Ensure code text is readable */
.stCodeBlock pre {
    background-color: #F8F9FA !important;
    color: #1A1A1A !important;
}
</style>
""", unsafe_allow_html=True)

# --- Path Helpers ---
def _resolve_db_path(raw_path: str) -> str:
    """Convert the user-provided DB path into a runtime-safe path."""
    return convert_to_docker_mount_path(raw_path) if raw_path else ""


def get_runtime_db_path() -> str:
    """
    Get the active runtime database path.
    Falls back to the current text-input value if no runtime override is stored.
    """
    runtime_path = st.session_state.get("db_path_runtime")
    if runtime_path:
        return runtime_path
    raw = st.session_state.get("db_path", "")
    resolved = _resolve_db_path(raw)
    if resolved:
        st.session_state.db_path_runtime = resolved
    return resolved


def set_runtime_db_path(resolved_path: Optional[str] = None) -> str:
    """Persist the runtime DB path for the current ingestion session."""
    if resolved_path is None:
        resolved_path = _resolve_db_path(st.session_state.get("db_path", ""))
    if resolved_path:
        st.session_state.db_path_runtime = resolved_path
    else:
        st.session_state.pop("db_path_runtime", None)
    return resolved_path


def filter_existing_doc_ids_for_collection(db_root_path: str, candidate_doc_ids: List[str]) -> List[str]:
    """Return only doc_ids that currently exist in the vector store metadata."""
    if not db_root_path or not candidate_doc_ids:
        return []
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        chroma_db_path = Path(db_root_path) / "knowledge_hub_db"
        if not chroma_db_path.exists():
            return []

        client = chromadb.PersistentClient(
            path=str(chroma_db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.get(where={"doc_id": {"$in": candidate_doc_ids}}, include=["metadatas"])

        valid_ids = []
        seen = set()
        for meta in results.get("metadatas", []):
            if not isinstance(meta, dict):
                continue
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in seen:
                valid_ids.append(doc_id)
                seen.add(doc_id)
        return valid_ids
    except Exception as e:
        logger.warning(f"Could not validate candidate doc IDs against vector store: {e}")
        return []

# --- Constants & State ---
REVIEW_PAGE_SIZE = 10
MAX_AUTO_FINALIZE_RETRIES = 12  # Give staging writes up to ~10 seconds to settle
# SPRINT 21: Removed image files from the unsupported list. They are now processed by the backend.
UNSUPPORTED_EXTENSIONS = {
    # Multimedia (Video)
    '.mp4', '.mov', '.avi', '.wmv', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg',
    # Multimedia (Audio)
    '.mp3', '.wav', '.aac', '.flac', '.ogg', '.wma', '.m4a', '.opus',
    # Archives & Compressed
    '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.cab', '.dmg', '.iso',
    # Interactive media
    '.swf',
    # Email files (Outlook message files are typically binary)
    '.msg', '.eml', '.pst', '.ost',
    # Office binary formats (use newer XML formats instead)
    '.xls', '.xlsx', '.xlsm', '.pptm', '.potm', '.ppsm',
    # Keynote and other Apple formats
    '.key', '.numbers', '.pages',
    # Executables & System
    '.exe', '.msi', '.app', '.deb', '.rpm', '.pkg', '.run', '.bin', '.com', '.bat', '.cmd', '.sh',
    # Temporary & Cache files
    '.tmp', '.temp', '.cache', '.bak', '.old', '.orig', '.swp', '.~', '.lock',
    # System & Hidden
    '.lnk', '.ini', '.cfg', '.reg', '.sys', '.dll', '.so', '.dylib',
    # Database files (binary)
    '.db', '.sqlite', '.mdb', '.accdb',
    # Font files
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    # 3D & CAD files
    '.obj', '.fbx', '.dae', '.3ds', '.blend', '.dwg', '.dxf',
    # Virtual machine & disk images
    '.vmdk', '.vdi', '.qcow2', '.img'
}


# Get dynamic document type options from the document type manager
def get_document_type_options():
    doc_type_manager = get_document_type_manager()
    return ["Any"] + doc_type_manager.get_all_document_types()

DOC_TYPE_OPTIONS = get_document_type_options()
PROPOSAL_OUTCOME_OPTIONS = RichMetadata.model_fields['proposal_outcome'].annotation.__args__

def _is_wsl_env() -> bool:
    try:
        import os, platform
        if os.environ.get("WSL_DISTRO_NAME"):
            return True
        rel = platform.release().lower()
        ver = platform.version().lower()
        return ("microsoft" in rel) or ("microsoft" in ver)
    except Exception:
        return False


def build_ingestion_command(container_db_path, files_to_process, target_collection=None, resume=False):
    """Build ingestion command with collection assignment support"""
    # Use direct script path to avoid module resolution confusion
    script_path = project_root / "cortex_engine" / "ingest_cortex.py"
    command = [
        sys.executable, "-u", str(script_path),  # -u flag for unbuffered stdout
        "--analyze-only", "--db-path", container_db_path,
        "--include", *files_to_process
    ]

    if resume:
        command.append("--resume")

    if target_collection:
        command.extend(["--target-collection", target_collection])

    ingest_backend = st.session_state.get("ingest_backend", "default")
    command.extend(["--ingest-backend", ingest_backend])

    # Add skip image processing flag if enabled
    if st.session_state.get("skip_image_processing", False):
        command.append("--skip-image-processing")

    # Add throttle delay - choose a safer default on WSL for NVIDIA 8GB class
    default_throttle = 2.0 if _is_wsl_env() else 0.5
    throttle_delay = st.session_state.get("throttle_delay", default_throttle)
    command.extend(["--throttle-delay", str(throttle_delay)])

    # Add GPU intensity control (affects batch size and inter-batch delays)
    gpu_intensity = st.session_state.get("gpu_intensity", 75)
    command.extend(["--gpu-intensity", str(gpu_intensity)])

    # Apply conservative runtime stability defaults automatically on WSL
    if _is_wsl_env():
        # Cooler cadence to avoid GPU stalls and UI freezes
        command.extend(["--cooldown-every", "20", "--cooldown-seconds", "15"])  # defaults are 25/20; tighten for WSL
        # Lower thresholds to start throttling earlier on laptops
        command.extend(["--gpu-threshold", "50", "--cpu-threshold", "60"]) 
        # Slow down indexing slightly between batches to reduce I/O churn
        command.extend(["--index-batch-cooldown", "2.0"]) 
        # Enforce hard LLM timeout to avoid indefinite stalls
        command.extend(["--llm-timeout", "120"])  # seconds

    return command

# ---- Subprocess helpers for robust streaming ----
def spawn_ingest(command: list) -> subprocess.Popen:
    """Launch an ingestion subprocess with unbuffered stdout and merged stderr."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure python runs unbuffered even when using -m mode
    if len(command) >= 2 and command[0] == sys.executable and command[1] != "-u":
        # Insert -u right after the interpreter if missing
        command = [command[0], "-u", *command[1:]]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

def _reader_to_queue(pipe, q: queue.Queue, stop_event: threading.Event):
    try:
        while not stop_event.is_set():
            line = pipe.readline()
            if not line:
                # EOF reached
                break
            q.put(line)
    except Exception as e:
        logger.debug(f"Reader thread ended: {e}")

def start_ingest_reader(proc: subprocess.Popen) -> None:
    """Start a background thread that enqueues stdout lines for non-blocking UI reads."""
    q = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(target=_reader_to_queue, args=(proc.stdout, q, stop_event), daemon=True)
    t.start()
    st.session_state.ingestion_output_queue = q
    st.session_state.ingestion_reader_stop = stop_event
    st.session_state.ingestion_reader_thread = t

def get_ingest_lines(max_lines: int = 50) -> list:
    """Drain up to max_lines from the ingestion output queue without blocking."""
    q = st.session_state.get("ingestion_output_queue")
    if not q:
        return []
    lines = []
    for _ in range(max_lines):
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line:
            s = line.strip()
            if s:
                lines.append(s)
    return lines

def _read_staging_payload_with_retries(container_db_path: str, retries: int = 5, delay_seconds: float = 0.4):
    """
    Attempt to read the staging_ingestion.json file with a short retry window.
    This smooths over network-drive latency where the file can lag behind the
    ingestion subprocess completing.
    """
    if not container_db_path:
        return None, None

    staging_path = Path(container_db_path) / "staging_ingestion.json"
    last_error = None
    for attempt in range(retries):
        if staging_path.exists():
            try:
                with open(staging_path, 'r') as f:
                    return json.load(f), staging_path
            except json.JSONDecodeError as exc:
                last_error = exc
                logger.debug(
                    "Staging file not parseable yet (attempt %s/%s): %s",
                    attempt + 1,
                    retries,
                    exc,
                )
        else:
            logger.debug(
                "Staging file not found yet (attempt %s/%s): %s",
                attempt + 1,
                retries,
                staging_path,
            )
        time.sleep(delay_seconds)

    if last_error:
        logger.warning(f"Staging file never stabilized at {staging_path}: {last_error}")
    else:
        logger.debug(f"Staging file still missing after retries: {staging_path}")
    return None, staging_path

def should_auto_finalize():
    """Check if automatic finalization should proceed"""
    try:
        # Check if we have a database path
        if not st.session_state.get('db_path'):
            logger.debug("Auto-finalize check: No db_path in session state")
            return False

        container_db_path = get_runtime_db_path()
        staging_data, staging_path = _read_staging_payload_with_retries(
            container_db_path,
            retries=6,
            delay_seconds=0.5,
        )

        if not staging_data:
            logger.debug(f"Auto-finalize check: staging file not ready yet ({staging_path})")
            return False

        # Handle both old and new staging formats
        if isinstance(staging_data, list):
            doc_count = len(staging_data)
            logger.debug(f"Auto-finalize check: {doc_count} docs detected in staging (list) @ {staging_path}")
            return doc_count > 0
        else:
            doc_count = len(staging_data.get('documents', []))
            logger.debug(f"Auto-finalize check: {doc_count} docs detected in staging (dict) @ {staging_path}")
            return doc_count > 0

    except Exception as e:
        logger.error(f"Error checking auto-finalization: {e}", exc_info=True)
        return False

def start_automatic_finalization():
    """Start automatic finalization subprocess"""
    try:
        from cortex_engine.utils import convert_windows_to_wsl_path
        
        container_db_path = get_runtime_db_path()
        
        # Build finalization command
        command = [
            sys.executable, "-m", "cortex_engine.ingest_cortex", 
            "--finalize-from-staging", "--db-path", container_db_path
        ]
        
        # Add skip image processing flag if enabled
        if st.session_state.get("skip_image_processing", False):
            command.append("--skip-image-processing")
        
        # Start finalization subprocess
        st.session_state.log_messages = ["Starting automatic finalization..."]
        st.session_state.ingestion_stage = "finalizing"
        st.session_state.auto_finalize_retry_attempts = 0
        st.session_state.auto_finalize_triggered = True
        st.session_state.ingestion_process = spawn_ingest(command)
        start_ingest_reader(st.session_state.ingestion_process)
        
        logger.info(f"Started automatic finalization with command: {' '.join(command[:4])}...")
        
    except Exception as e:
        logger.error(f"Failed to start automatic finalization: {e}")
        st.error(f"❌ Failed to start automatic finalization: {e}")
        # Fall back to manual mode
        st.session_state.ingestion_stage = "metadata_review"
        st.session_state.batch_auto_finalize_started = False

def show_collection_migration_healthcheck():
    """Warn if a project-root collections file exists and offer migration to external DB path."""
    try:
        project_collections = (Path(__file__).parent.parent / "working_collections.json")
        # Cache the manager instance to avoid repeated file I/O
        if 'collection_mgr_cache' not in st.session_state:
            st.session_state.collection_mgr_cache = WorkingCollectionManager()
        mgr = st.session_state.collection_mgr_cache
        external_path = Path(mgr.collections_file)
        # If a project-root collections file exists and is different from external path
        if project_collections.exists() and project_collections.resolve() != external_path.resolve():
            st.warning("Detected collections file in project root. Migrate to external database path to avoid inconsistencies.")
            col_a, col_b = st.columns([1,2])
            with col_a:
                if st.button("🔄 Migrate Collections", key="migrate_collections"):
                    try:
                        # Load both sets
                        import json
                        with open(project_collections, 'r') as f:
                            project_data = json.load(f)
                        external_data = mgr.collections or {}
                        # Merge doc_ids per collection
                        for name, data in project_data.items():
                            ids = data.get('doc_ids', data if isinstance(data, list) else [])
                            mgr.add_docs_by_id_to_collection(name, ids)
                        st.success("Collections migrated to external DB path.")
                    except Exception as me:
                        st.error(f"Failed to migrate collections: {me}")
            with col_b:
                st.caption(f"Project file: {project_collections}\nExternal file: {external_path}")
    except Exception as e:
        # Non-fatal; just skip if anything goes wrong
        logger.debug(f"Collection migration healthcheck skipped: {e}")

def initialize_state(force_reset: bool = False):
    # On force_reset, clear all session state first, then re-read config
    if force_reset:
        keys_to_reset = list(st.session_state.keys())
        for key in keys_to_reset:
            del st.session_state[key]

    # Cache config to avoid file reads on every interaction
    if "cached_config" not in st.session_state:
        config_manager = ConfigManager()
        st.session_state.cached_config = config_manager.get_config()
    config = st.session_state.cached_config

    # Always sync with config - update session state if config has different values
    config_knowledge_path = config.get("knowledge_source_path", "")
    config_db_path = config.get("ai_database_path", "")

    # Update session state from config (this fixes stale session values)
    # Initialize with config values if not set OR if currently empty (handles Docker first-run case)
    if "knowledge_source_path" not in st.session_state or not st.session_state.get("knowledge_source_path"):
        if config_knowledge_path:  # Only override if config has a value
            st.session_state.knowledge_source_path = config_knowledge_path
    if "db_path" not in st.session_state or not st.session_state.get("db_path"):
        if config_db_path:  # Only override if config has a value
            st.session_state.db_path = config_db_path
    if "db_path_runtime" not in st.session_state and st.session_state.get("db_path"):
        st.session_state.db_path_runtime = _resolve_db_path(st.session_state.db_path)

    # Pick safer defaults automatically on WSL
    try:
        import os, platform
        _is_wsl_default = bool(os.environ.get("WSL_DISTRO_NAME") or "microsoft" in platform.release().lower())
    except Exception:
        _is_wsl_default = False

    defaults = {
        "ingestion_stage": "config", "dir_selections": {},
        "files_to_review": [], "staged_files": [], "file_selections": {},
        "edited_staged_files": [], "staged_metadata": {}, "review_page": 0, "ingestion_process": None,
        "skip_image_processing": False,  # Option to skip VLM image processing
        "ingest_backend": "docling",  # default|docling|auto
        # Delay between documents; on WSL default to 1.5s for stability
        "throttle_delay": 1.5 if _is_wsl_default else 0.5,
        "batch_ingest_mode": False,  # Option to bypass preview check for large ingests
        "batch_mode_active": False,  # Persistent flag set when batch processing starts
        "batch_auto_processed": False,  # Flag to prevent re-processing in batch mode
        "batch_auto_finalize_started": False,
        "auto_finalize_retry_attempts": 0,
        "auto_finalize_triggered": False,
        "log_messages": [], "filter_exclude_common": True, "filter_prefer_docx": True,
        "filter_deduplicate": True, "enable_pattern_exclusion": False,
        "exclude_patterns_input": "", "show_confirm_clear_log": False,
        "show_confirm_delete_kb": False, "last_ingested_doc_ids": [],
        "target_collection_name": "", "collection_assignment_mode": "default",
        # Throttle status tracking
        "current_throttle_delay": 0.0,
        "current_gpu_util": None,
        "current_cpu_util": None,
        "throttle_active": False
    }
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val

    # Initialize directory_scan_path to match knowledge_source_path (don't use stale values)
    # Always sync with current knowledge_source_path to prevent showing non-existent directories
    if "directory_scan_path" not in st.session_state or not st.session_state.directory_scan_path:
        st.session_state.directory_scan_path = st.session_state.get("knowledge_source_path", "")

# Path handling now handled by centralized utilities

# Note: delete_knowledge_base function moved to pages/6_Maintenance.py

@st.cache_data
def get_full_file_content(file_path_str: str) -> str:
    file_path = Path(file_path_str)
    MAX_PREVIEW_SIZE = 50 * 1024 * 1024  # 50MB limit
    MAX_PREVIEW_CHARS = 10000  # 10k character limit for preview
    
    try:
        # Check file size first
        file_size = file_path.stat().st_size
        if file_size > MAX_PREVIEW_SIZE:
            return f"[File too large for preview: {file_size / (1024*1024):.1f}MB. Preview limited to large files.]"
        
        if file_path.suffix.lower() == '.pdf':
            with fitz.open(file_path) as doc:
                # Limit to first 10 pages for preview to prevent freezing
                max_pages = min(10, doc.page_count)
                pages_text = []
                for i in range(max_pages):
                    page = doc[i]
                    pages_text.append(page.get_text())
                text = "\n".join(pages_text)
                if doc.page_count > max_pages:
                    text += f"\n\n[Preview truncated - showing first {max_pages} pages of {doc.page_count} total pages]"
                    
        elif file_path.suffix.lower() == '.docx':
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs]
            text = "\n".join(paragraphs)
            
        elif file_path.suffix.lower() == '.pptx':
            return "[PowerPoint file preview not available in text format. The content for ingestion will be extracted by the document reader during processing.]"
            
        # SPRINT 21: Add a preview handler for images (show a message, since text preview is not applicable)
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return "[Image file preview not available. The content for ingestion is generated by the Vision AI model.]"
        else:
            # Only try to read as text for actual text files
            if file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            else:
                return f"[Preview not available for {file_path.suffix.upper()} files. Content will be processed during ingestion.]"
        
        # Truncate if too long
        if len(text) > MAX_PREVIEW_CHARS:
            text = text[:MAX_PREVIEW_CHARS] + f"\n\n[Preview truncated - showing first {MAX_PREVIEW_CHARS:,} characters of {len(text):,} total characters]"
            
        return text.strip() if text.strip() else "[No preview available or empty file]"
        
    except Exception as e: 
        return f"[Error generating preview: {e}]"

def load_staged_files():
    st.session_state.staged_files = []
    st.session_state.staged_metadata = {}
    # Use database-specific staging path instead of hardcoded project path
    db_path = st.session_state.get('db_path')
    if not db_path:
        return

    container_db_path = get_runtime_db_path()
    raw_data, staging_path = _read_staging_payload_with_retries(
        container_db_path,
        retries=6,
        delay_seconds=0.5,
    )

    if not raw_data:
        logger.debug(f"No staging payload available yet at {staging_path}")
        return

    try:
        if isinstance(raw_data, dict):
            st.session_state.staged_metadata = raw_data
            st.session_state.staged_files = raw_data.get('documents', [])
            target = raw_data.get('target_collection')
            if target and not st.session_state.get('target_collection_name'):
                st.session_state.target_collection_name = target
        else:
            st.session_state.staged_metadata = {
                "documents": raw_data,
                "target_collection": st.session_state.get('target_collection_name', ''),
            }
            st.session_state.staged_files = raw_data
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Error reading staging file: {e}")
        st.session_state.staged_files = []

def serialize_staging_payload(documents: List[dict], target_collection: Optional[str] = None) -> dict:
    """Build staging payload preserving metadata when user edits documents."""
    base_metadata = st.session_state.get('staged_metadata')
    payload = dict(base_metadata) if isinstance(base_metadata, dict) else {}
    payload['documents'] = documents
    collection = target_collection or payload.get('target_collection') or st.session_state.get('target_collection_name', '')
    payload['target_collection'] = collection or ""
    payload.setdefault('version', '2.0')
    payload.setdefault('created_at', datetime.now().isoformat())
    return payload

def scan_for_files(selected_dirs: List[str]):
    container_db_path = get_runtime_db_path()
    chroma_db_dir = Path(container_db_path) / "knowledge_hub_db"
    ingested_log_path = chroma_db_dir / INGESTED_FILES_LOG

    ingested_files = {}
    if ingested_log_path.exists():
        with open(ingested_log_path, 'r') as f:
            try: ingested_files = json.load(f)
            except json.JSONDecodeError: pass

    # Enhanced file scanning with progress monitoring
    all_files = []
    
    # Create progress containers
    scan_progress_container = st.empty()
    file_count_container = st.empty()
    
    total_dirs_to_scan = len(selected_dirs)
    total_scan_start = time.time()
    
    for dir_idx, dir_path in enumerate(selected_dirs, 1):
        wsl_dir_path = convert_windows_to_wsl_path(dir_path)
        dir_name = Path(dir_path).name
        
        # Update progress display
        scan_progress_container.info(f"🔍 **Scanning directory {dir_idx}/{total_dirs_to_scan}**: {dir_name}")
        
        try:
            path_obj = Path(wsl_dir_path)
            if not path_obj.exists():
                st.warning(f"⚠️ Directory not found: {dir_path}")
                continue
                
            # Count files in directory first (for better progress indication)
            dir_files = []
            file_count = 0
            
            # Use iterative approach with progress updates
            start_time = time.time()
            for file_path in path_obj.rglob('*'):
                if file_path.is_file():
                    dir_files.append(file_path)
                    file_count += 1
                    
                    # Update file count every 50 files (more frequent updates)
                    if file_count % 50 == 0:
                        elapsed = time.time() - start_time
                        file_count_container.text(f"📁 **{dir_name}**: {file_count} files found ({elapsed:.1f}s elapsed)...")
            
            all_files.extend(dir_files)
            file_count_container.success(f"✅ **{dir_name}**: {file_count} files found")
            
        except Exception as e:
            st.error(f"❌ Error scanning {dir_path}: {str(e)}")
            continue
    
    total_scan_time = time.time() - total_scan_start
    scan_progress_container.success(f"🎉 **Scan complete!** Found {len(all_files)} total files across {total_dirs_to_scan} directories in {total_scan_time:.1f}s")

    # Apply initial filtering with progress updates
    filter_progress_container = st.empty()
    filter_progress_container.info("🔍 **Applying filters and exclusions...**")
    
    # Normalize ingested_files keys to POSIX format for consistent comparison
    ingested_posix = {Path(k).as_posix() for k in ingested_files}
    candidate_files = [f.as_posix() for f in all_files if f.as_posix() not in ingested_posix and f.suffix.lower() not in UNSUPPORTED_EXTENSIONS]
    filter_progress_container.text(f"📋 After excluding unsupported formats: {len(candidate_files)} files")

    if st.session_state.filter_exclude_common:
        exclude_keywords = [
            "working", "temp", "archive", "ignore", "backup", "node_modules",
            ".git", "exclude", "draft", "invoice", "timesheet", "contract", "receipt",
            "prezi.app"
        ]
        # Remove "data" from exclusions as it conflicts with Docker mount paths like /data/
        candidate_files = [f for f in candidate_files if not any(k in part.lower() for k in exclude_keywords for part in Path(f).parts)]
        filter_progress_container.text(f"📋 After excluding common folders: {len(candidate_files)} files")

    if st.session_state.enable_pattern_exclusion:
        patterns = [p.strip() for p in st.session_state.exclude_patterns_input.split('\n') if p.strip()]
        candidate_files = [f for f in candidate_files if not any(fnmatch(Path(f).name, p) for p in patterns)]
        filter_progress_container.text(f"📋 After pattern exclusions: {len(candidate_files)} files")

    if st.session_state.filter_prefer_docx:
        docx_stems = {Path(f).stem for f in candidate_files if f.lower().endswith('.docx')}
        candidate_files = [f for f in candidate_files if not (f.lower().endswith('.pdf') and Path(f).stem in docx_stems)]
        filter_progress_container.text(f"📋 After preferring .docx over .pdf: {len(candidate_files)} files")

    if st.session_state.filter_deduplicate:
        grouped_files = defaultdict(list)
        for f_path in candidate_files:
            pattern = r'[\s_-]*(v\d+(\.\d+)*|draft|\d{8}|final|revised)[\s_-]*'
            normalized = re.sub(pattern, "", Path(f_path).stem, flags=re.IGNORECASE).strip()
            grouped_files[normalized].append(f_path)
        candidate_files = [max(file_list, key=lambda f: Path(f).stat().st_mtime) if len(file_list) > 1 else file_list[0] for _, file_list in grouped_files.items()]
        filter_progress_container.text(f"📋 After deduplication: {len(candidate_files)} files")
    
    # Final summary
    filter_progress_container.success(f"✅ **Filtering complete!** {len(candidate_files)} files ready for processing")

    st.session_state.files_to_review = sorted(candidate_files)
    st.session_state.file_selections = {fp: True for fp in st.session_state.files_to_review}
    st.session_state.review_page = 0
    
    # Handle resume mode - create batch state with proper filtering
    if st.session_state.get("resume_mode_enabled"):
        container_db_path = get_runtime_db_path()
        batch_manager = BatchState(container_db_path)
        set_runtime_db_path(str(batch_manager.db_path))
        set_runtime_db_path(str(batch_manager.db_path))
        
        # Create batch with resume logic, passing the scan configuration
        scan_config = st.session_state.get("current_scan_config", {})
        
        # Check if chunked processing should be used
        chunk_size = None
        if st.session_state.get("use_chunked_processing", False):
            chunk_size = st.session_state.get("chunk_size", 250)
        
        batch_id, files_to_process, completed_count = batch_manager.resume_or_create_batch(candidate_files, scan_config, chunk_size)
        
        if files_to_process:
            st.session_state.files_to_review = files_to_process
            st.session_state.file_selections = {fp: True for fp in files_to_process}
            
            # Show resume info
            st.success(f"✅ Resume batch created: {completed_count} files already processed, {len(files_to_process)} remaining")
            st.session_state.force_batch_mode = True  # Use different key to avoid conflict
        else:
            st.info("✅ All files from the original batch have been processed!")
        
        # Clear resume mode flag
        st.session_state.resume_mode_enabled = False

def log_failed_documents(failed_docs, db_path):
    """Log documents that failed during batch processing to a separate failure log."""
    container_db_path = convert_to_docker_mount_path(db_path)
    chroma_db_dir = Path(container_db_path) / "knowledge_hub_db"
    chroma_db_dir.mkdir(parents=True, exist_ok=True)
    
    failure_log_path = chroma_db_dir / "ingest_failures.log"
    
    try:
        # Load existing failure log
        existing_failures = {}
        if failure_log_path.exists():
            with open(failure_log_path, 'r') as f:
                try:
                    existing_failures = json.load(f)
                except json.JSONDecodeError:
                    existing_failures = {}
        
        # Add new failures with timestamp
        timestamp = datetime.now().isoformat()
        for doc_path, error_msg in failed_docs.items():
            existing_failures[doc_path] = {
                "error": error_msg,
                "timestamp": timestamp,
                "batch_mode": True
            }
        
        # Write updated failure log
        with open(failure_log_path, 'w') as f:
            json.dump(existing_failures, f, indent=2)
        
        logger.info(f"Logged {len(failed_docs)} failed documents to {failure_log_path}")
        return str(failure_log_path)
        
    except Exception as e:
        logger.error(f"Failed to write failure log: {e}")
        return None

def render_batch_processing_ui():
    """Handle batch processing that bypasses preview and logs errors separately."""
    shared_render_batch_processing_ui(
        get_runtime_db_path=get_runtime_db_path,
        set_runtime_db_path=set_runtime_db_path,
        batch_state_cls=BatchState,
        build_ingestion_command=build_ingestion_command,
        spawn_ingest=spawn_ingest,
        start_ingest_reader=start_ingest_reader,
        auto_resume_from_batch_config=auto_resume_from_batch_config,
        initialize_state=initialize_state,
        logger=logger,
    )

def render_active_batch_management(batch_manager: BatchState, batch_status: dict):
    """Render the active batch management section with consolidated controls."""
    shared_render_active_batch_management(
        batch_manager=batch_manager,
        batch_status=batch_status,
        auto_resume_from_batch_config=auto_resume_from_batch_config,
        start_ingest_reader=start_ingest_reader,
        get_ingest_lines=get_ingest_lines,
        should_auto_finalize=should_auto_finalize,
        start_automatic_finalization=start_automatic_finalization,
        initialize_state=initialize_state,
        logger=logger,
    )

def auto_resume_from_batch_config(batch_manager: BatchState) -> bool:
    """Automatically restore scan configuration and resume batch processing"""
    try:
        scan_config = batch_manager.get_scan_config()
        if not scan_config:
            # Fallback: attempt to resume directly from files_remaining in batch state
            state = batch_manager.load_state() or {}
            files_remaining = state.get('files_remaining', [])
            if files_remaining:
                container_db_path = str(batch_manager.db_path)
                st.session_state.log_messages = []
                st.session_state.ingestion_stage = "analysis_running"
                st.session_state.batch_mode_active = True
                target_collection = st.session_state.get('target_collection_name', '')
                command = build_ingestion_command(container_db_path, files_remaining, target_collection, resume=True)
                try:
                    st.session_state.ingestion_process = spawn_ingest(command)
                    start_ingest_reader(st.session_state.ingestion_process)
                    return True
                except Exception as e:
                    st.error(f"❌ Failed to start ingestion from batch state: {e}")
                    logger.error(f"Fallback resume failed: {e}")
            # If no remaining files or start failed, try staging file path
            return try_resume_from_staging_file(batch_manager)
        
        return resume_from_scan_config(batch_manager, scan_config)
    except Exception as e:
        st.error(f"❌ Failed to resume batch automatically: {e}")
        st.info("💡 **Tip:** Try clearing the batch state and starting fresh if this persists.")
        return False

def try_resume_from_staging_file(batch_manager: BatchState) -> bool:
    """Try to resume processing using existing staging file"""
    try:
        # Look for staging file in the database directory, not project root
        staging_file = batch_manager.db_path / "staging_ingestion.json"
        if not staging_file.exists():
            st.error("❌ No staging file found to resume from")
            st.info("🔄 **No recovery options available.** Please start a new scan with your original paths.")
            return False
        
        st.info("🔄 Found existing staging file - attempting to resume from metadata review stage")
        
        # Switch to metadata review stage to process the staging file
        st.session_state.ingestion_stage = "metadata_review"
        st.success("✅ Resumed from staging file - please review and approve the metadata below")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to resume from staging file: {e}")
        st.info("💡 **Recovery suggestion:** Clear the batch state and start a new scan.")
        return False

def resume_from_scan_config(batch_manager: BatchState, scan_config: dict) -> bool:
    """Resume using stored scan configuration"""
    try:
        
        # Restore session state from batch configuration
        st.session_state.knowledge_source_path = scan_config.get("knowledge_source_path", "")
        st.session_state.db_path = scan_config.get("db_path", "")
        runtime_path = scan_config.get("db_path_runtime")
        set_runtime_db_path(runtime_path if runtime_path else _resolve_db_path(st.session_state.db_path))
        st.session_state.filter_exclude_common = scan_config.get("filter_exclude_common", False)
        st.session_state.enable_pattern_exclusion = scan_config.get("enable_pattern_exclusion", False)
        st.session_state.exclude_patterns_input = scan_config.get("exclude_patterns_input", "")
        st.session_state.filter_prefer_docx = scan_config.get("filter_prefer_docx", False)
        # Don't restore batch_ingest_mode directly to avoid widget conflicts
        # st.session_state.batch_ingest_mode = scan_config.get("batch_ingest_mode", False)
        
        # Get the original directories and perform scan
        selected_dirs = scan_config.get("selected_directories", [])
        if not selected_dirs:
            st.error("❌ No directories found in original scan configuration")
            return False
            
        # Perform the scan with original configuration and enhanced progress monitoring
        st.info(f"🔄 Restoring original scan configuration and processing {len(selected_dirs)} directories...")
        scan_for_files(selected_dirs)
        
        # Get the files that were found and start processing immediately
        files_to_process = st.session_state.get("files_to_review", [])
        if files_to_process:
            # Start the actual ingestion process
            container_db_path = get_runtime_db_path()
            batch_manager_instance = BatchState(container_db_path)
            set_runtime_db_path(str(batch_manager_instance.db_path))
            # Ensure we clear any paused flag from a prior session
            try:
                batch_manager_instance.start_new_session()
            except Exception:
                pass
            
            # For chunked processing, get current chunk files
            if batch_manager_instance.is_chunked_processing():
                chunk_files = batch_manager_instance.get_current_chunk_files()
                if chunk_files:
                    files_to_process = chunk_files
                    st.info(f"🔄 Processing chunk {batch_manager_instance.get_status()['current_chunk']}/{batch_manager_instance.get_status()['total_chunks']} ({len(chunk_files)} files)")
            
            st.session_state.log_messages = []
            st.session_state.ingestion_stage = "analysis_running"
            st.session_state.batch_mode_active = True
            
            # Build and start the ingestion command
            target_collection = st.session_state.get('target_collection_name', '')
            command = build_ingestion_command(container_db_path, files_to_process, target_collection, resume=True)
            
            try:
                st.session_state.ingestion_process = spawn_ingest(command)
                start_ingest_reader(st.session_state.ingestion_process)
                st.success(f"✅ Auto-resume started! Processing {len(files_to_process)} files...")
                return True
            except Exception as start_error:
                st.error(f"❌ Failed to start ingestion process: {start_error}")
                logger.error(f"Auto-resume subprocess failed: {start_error}")
                return False
        else:
            st.warning("⚠️ No files found to process after scan")
            return False
        
    except Exception as e:
        st.error(f"❌ Failed to restore scan configuration: {e}")
        logger.error(f"Auto-resume failed: {e}")
        return False

# Note: clear_ingestion_log_file function moved to pages/6_Maintenance.py

def render_model_status_bar():
    """Render persistent model configuration status bar at top of page"""
    # Use cached model info to avoid repeated system checks
    # Invalidate cache if it's missing the new embedding_strategy field (old cache format)
    if 'model_info_cache' not in st.session_state or 'embedding_strategy' not in st.session_state.model_info_cache:
        st.session_state.model_info_cache = get_model_info_summary()
    model_info = st.session_state.model_info_cache
    gpu_info = model_info.get('gpu_info', {})
    embed_strategy = model_info.get('embedding_strategy', {})

    # Build embedding model display name
    embed_approach = embed_strategy.get('approach', '')
    if embed_approach == 'qwen3vl':
        # Show clean Qwen3-VL name with size
        # PRIORITY: Use effective size from session state (set by sidebar after compatibility check)
        # This ensures the display matches the sidebar selection, not stale config
        from cortex_engine.config import QWEN3_VL_MODEL_SIZE
        effective_size = st.session_state.get('effective_qwen3vl_size', QWEN3_VL_MODEL_SIZE)
        size_to_dims = {"2B": 2048, "8B": 4096}
        if effective_size in size_to_dims:
            embed_dims = size_to_dims[effective_size]
            size_display = effective_size.upper()
        else:
            # "auto" - derive from what the GPU would select
            vram_gb = gpu_info.get('memory_total_gb', 0)
            embed_dims = 4096 if vram_gb >= 16 else 2048
            size_display = "8B" if vram_gb >= 16 else "2B"
        embed_display = f"Qwen3-VL ({size_display}, {embed_dims}D)"
    elif embed_approach == 'nv-embed':
        embed_display = "NV-Embed-v2"
    elif embed_approach == 'bge':
        embed_display = "BGE-base"
    else:
        embed_display = model_info['embedding_model'].split('/')[-1][:25]

    # Custom styled status bar
    status_html = f"""
    <div style="
        background: linear-gradient(135deg, #2A4362 0%, #3A5575 100%);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div style="flex: 1; min-width: 200px;">
                <div style="color: #FFFFFF; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                    System Configuration
                </div>
                <div style="color: #FFFFFF; font-size: 1.35rem; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.4);">
                    {'🎮 ' + gpu_info.get('device_name', 'No GPU') if model_info['has_nvidia_gpu'] else '💻 CPU Mode'}
                </div>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="color: #FFFFFF; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                    Embedding Model
                </div>
                <div style="color: #FFFFFF; font-size: 1.35rem; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.4);">
                    {embed_display}
                </div>
            </div>
            <div style="flex: 0; min-width: 140px; text-align: right;">
                <div style="color: #FFFFFF; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                    Quick Access
                </div>
                <div style="color: #FFD54F; font-size: 1.05rem; font-weight: 700; cursor: pointer; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                    → Configure in Sidebar
                </div>
            </div>
        </div>
    """

    # Show setup warning if GPU detected but not accessible
    if model_info['has_nvidia_gpu'] and gpu_info.get('method') == 'wsl-windows-nvidia-smi':
        status_html += """<div style="margin-top: 14px; padding: 18px 24px; background: #FFA726; border-left: 5px solid #FF6F00; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"><div style="color: #1A1A1A; font-size: 1.15rem; font-weight: 800; letter-spacing: 0.02em;">⚠️ GPU detected but not accessible - See sidebar for setup instructions</div></div>"""

    status_html += "</div>"

    st.markdown(status_html, unsafe_allow_html=True)


def render_sidebar_model_config():
    """Render comprehensive model configuration in sidebar"""

    # Note: Removed dark mode CSS that was forcing white text on light backgrounds
    # Let Streamlit handle theme colors naturally

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🤖 Model Configuration")

    # Cache model info to avoid repeated system checks on every rerun
    # Invalidate cache if it's missing the new embedding_strategy field (old cache format)
    if 'model_info_cache' not in st.session_state or 'embedding_strategy' not in st.session_state.model_info_cache:
        st.session_state.model_info_cache = get_model_info_summary()
    if 'available_models_cache' not in st.session_state:
        st.session_state.available_models_cache = get_available_embedding_models()

    model_info = st.session_state.model_info_cache
    available_models = st.session_state.available_models_cache
    gpu_info = model_info.get('gpu_info', {})

    # GPU Status Section
    st.sidebar.markdown("### GPU Status")

    if model_info['has_nvidia_gpu']:
        gpu_name = gpu_info.get('device_name', 'Unknown')
        detection_method = gpu_info.get('method', 'unknown')

        st.sidebar.success(f"🎮 **{gpu_name}**")

        if 'memory_info' in gpu_info:
            st.sidebar.caption(f"Memory: {gpu_info['memory_info']}")
        elif 'memory_total_gb' in gpu_info and gpu_info['memory_total_gb'] > 0:
            st.sidebar.caption(f"Memory: {gpu_info['memory_total_gb']:.1f}GB")

        # WSL Setup Instructions
        if detection_method == 'wsl-windows-nvidia-smi':
            st.sidebar.warning("⚠️ **GPU Not Accessible**")
            st.sidebar.markdown("**Enable GPU Acceleration:**")
            st.sidebar.markdown("Copy and run this **complete command** in WSL terminal:")
            st.sidebar.code(get_pytorch_cuda_install_command(), language="bash")
            st.sidebar.caption("⚠️ Make sure to copy the ENTIRE command including 'pip3'")
            st.sidebar.caption("Then restart Streamlit to activate GPU acceleration")
    else:
        st.sidebar.info("💻 **CPU Mode**")
        if gpu_info.get('issues'):
            with st.sidebar.expander("Detection Issues"):
                for issue in gpu_info['issues']:
                    st.sidebar.caption(f"• {issue}")

    # Check Qwen3-VL status FIRST to determine which UI to show
    embed_strategy = model_info.get('embedding_strategy', {})
    qwen3vl_active = embed_strategy.get('approach') == 'qwen3vl'

    # Import config values
    from cortex_engine.config import (
        QWEN3_VL_MODEL_SIZE,
        QWEN3_VL_RERANKER_ENABLED,
        QWEN3_VL_RERANKER_TOP_K,
    )

    # =========================================================================
    # UNIFIED EMBEDDING MODEL SECTION
    # =========================================================================
    st.sidebar.markdown("### Embedding Model")

    # CRITICAL: Check database embedding dimension to prevent corruption
    from cortex_engine.utils.embedding_validator import get_compatible_qwen3vl_sizes
    db_path = st.session_state.get('db_path', '')
    db_compat = get_compatible_qwen3vl_sizes(db_path) if db_path else {
        "is_new_database": True,
        "compatible_sizes": ["2B", "8B"],
        "incompatible_sizes": [],
        "database_dimension": None,
        "warning_message": None,
    }

    if qwen3vl_active:
        # ----- QWEN3-VL ACTIVE: Show unified Qwen3-VL controls -----
        st.sidebar.success("✅ **Qwen3-VL Multimodal**")

        # Show database dimension status
        if not db_compat["is_new_database"]:
            st.sidebar.info(f"📊 Database: {db_compat['database_dimension']}D embeddings")

        # Filter model sizes based on database compatibility
        all_sizes = ["2B", "8B"]
        size_to_dims = {"2B": 2048, "8B": 4096}

        if db_compat["is_new_database"]:
            # New database - all sizes available, include auto
            size_options = ["auto", "2B", "8B"]
            size_labels = {
                "auto": "Auto (based on VRAM)",
                "2B": "2B (5GB VRAM, 2048 dims)",
                "8B": "8B (16GB VRAM, 4096 dims)",
            }
        else:
            # Existing database - only show compatible sizes
            size_options = db_compat["compatible_sizes"]
            size_labels = {}
            for size in all_sizes:
                if size in db_compat["compatible_sizes"]:
                    size_labels[size] = f"{size} ({size_to_dims[size]} dims) ✓ Compatible"
                else:
                    size_labels[size] = f"{size} ({size_to_dims[size]} dims) ❌ Incompatible"

            # Show warning about restricted options
            if db_compat["warning_message"]:
                st.sidebar.warning(f"⚠️ {db_compat['warning_message']}")

        # Handle case where no compatible sizes exist
        if not size_options:
            st.sidebar.error("❌ No compatible Qwen3-VL models for this database!")
            st.sidebar.caption(f"Database dimension: {db_compat['database_dimension']}")
            st.sidebar.caption("You may need to rebuild the database or use a different embedding model.")
            selected_size = None
        else:
            current_size = QWEN3_VL_MODEL_SIZE

            # Ensure current size is in options, or default to first compatible
            if current_size not in size_options:
                if "auto" in size_options:
                    current_size = "auto"
                else:
                    current_size = size_options[0]
                    st.sidebar.warning(f"⚠️ Switched to {current_size} (previous selection incompatible)")

            # Store effective model size in session state for display consistency
            st.session_state['effective_qwen3vl_size'] = current_size

            try:
                current_index = size_options.index(current_size)
            except ValueError:
                current_index = 0

            selected_size = st.sidebar.selectbox(
                "Model Size:",
                options=size_options,
                index=current_index,
                format_func=lambda x: size_labels.get(x, x),
                key="qwen3_vl_size_selector",
                help="Only models matching the database dimension are shown to prevent corruption."
            )

            # Show dimensions for the SELECTED size (what user will get after Apply)
            if selected_size in size_to_dims:
                display_dims = size_to_dims[selected_size]
            elif selected_size == "auto":
                # For "auto", show what the GPU would select
                vram_gb = gpu_info.get('memory_total_gb', 0)
                display_dims = 4096 if vram_gb >= 16 else 2048
            else:
                display_dims = 2048  # fallback
            st.sidebar.caption(f"**Dimensions**: {display_dims}")

            # Validate selected size against database before allowing apply
            selected_dims = size_to_dims.get(selected_size)
            if selected_size == "auto":
                vram_gb = gpu_info.get('memory_total_gb', 0)
                selected_dims = 4096 if vram_gb >= 16 else 2048

            # Check if "auto" would select an incompatible model
            if selected_size == "auto" and not db_compat["is_new_database"]:
                if selected_dims != db_compat["database_dimension"]:
                    st.sidebar.error(
                        f"⚠️ 'Auto' would select {selected_dims}D model but database uses "
                        f"{db_compat['database_dimension']}D. Please select a specific compatible model."
                    )
                    selected_size = None  # Block the selection

        # Show apply button if size changed and selection is valid
        if selected_size and selected_size != current_size:
            if st.sidebar.button("🔄 Apply Size Change", type="primary", use_container_width=True):
                import os
                os.environ["QWEN3_VL_MODEL_SIZE"] = selected_size

                try:
                    from cortex_engine.qwen3_vl_embedding_service import reset_service
                    reset_service()
                except Exception as e:
                    st.sidebar.warning(f"Service reset warning: {e}")

                from cortex_engine.config import invalidate_embedding_cache
                invalidate_embedding_cache()

                if 'model_info_cache' in st.session_state:
                    del st.session_state.model_info_cache

                # Update session state for display consistency
                st.session_state['effective_qwen3vl_size'] = selected_size

                st.sidebar.success(f"✅ Changed to {size_labels[selected_size]}")
                st.rerun()

        # Reranker status
        if QWEN3_VL_RERANKER_ENABLED:
            st.sidebar.caption(f"🔄 **Reranker**: Enabled (top-{QWEN3_VL_RERANKER_TOP_K})")
        else:
            st.sidebar.caption("🔄 **Reranker**: Disabled")

        # Capabilities expander
        with st.sidebar.expander("ℹ️ Qwen3-VL Capabilities"):
            st.markdown("""
            **Multimodal features:**
            - Image embedding (charts, diagrams)
            - Visual document search
            - Cross-modal search (text ↔ image)
            - Neural reranking for precision
            """)
    else:
        # ----- QWEN3-VL NOT ACTIVE: Show legacy model selection -----
        current_model = model_info['embedding_model']
        is_cached, _ = check_model_cached(current_model)

        # Show current approach info
        current_approach = embed_strategy.get('approach', 'unknown')
        reason = embed_strategy.get('reason', '')

        if current_approach == 'nv-embed':
            st.sidebar.info("**NV-Embed-v2** (GPU-optimized)")
        elif current_approach == 'bge':
            st.sidebar.info("**BGE** (CPU-friendly)")
        else:
            st.sidebar.info(f"**{current_model.split('/')[-1]}**")

        st.sidebar.caption(f"Reason: {reason[:60]}..." if len(reason) > 60 else f"Reason: {reason}")

        # Model dropdown for manual selection
        with st.sidebar.expander("🔧 Manual Model Selection"):
            model_options = list(available_models.keys())
            model_labels = {}

            for model_id in model_options:
                model_data = available_models[model_id]
                label = model_data['name'][:35]
                if model_id == model_info['recommended_model']:
                    label += " ⭐"
                if model_id == current_model:
                    label += " ✓"
                model_labels[model_id] = label

            try:
                current_index = model_options.index(current_model)
            except ValueError:
                current_index = 0

            selected_model = st.selectbox(
                "Select Model:",
                options=model_options,
                index=current_index,
                format_func=lambda x: model_labels[x],
                key="sidebar_model_selection"
            )

            if selected_model:
                model_data = available_models[selected_model]
                st.caption(f"**Best for**: {model_data['recommended_for']}")

                if selected_model != current_model:
                    if st.button("🔄 Apply Change", type="primary", use_container_width=True):
                        st.session_state.selected_embedding_model = selected_model
                        from cortex_engine import config
                        config.EMBED_MODEL = selected_model
                        st.success(f"✅ Changed to {selected_model.split('/')[-1]}")
                        st.rerun()
                elif not is_cached:
                    is_available, status_msg = validate_model_available(selected_model)
                    st.info(status_msg)
                    if not is_available:
                        if st.button("⬇️ Download Now", use_container_width=True):
                            with st.spinner("Downloading..."):
                                success, message = download_model(selected_model)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)

    # LLM Models Info
    st.sidebar.markdown("### LLM Models")
    st.sidebar.caption(f"📝 **Text**: {model_info['llm_model']}")
    st.sidebar.caption(f"👁️ **Vision**: {model_info['vlm_model']}")
    st.sidebar.caption("Managed via Ollama")

    # Add refresh button for cache
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Model Info", use_container_width=True, help="Refresh GPU and model status if you've made changes"):
        # Clear all model-related caches (session state AND module-level)
        from cortex_engine.config import invalidate_embedding_cache
        invalidate_embedding_cache()  # Clear module-level config cache

        if 'model_info_cache' in st.session_state:
            del st.session_state.model_info_cache
        if 'available_models_cache' in st.session_state:
            del st.session_state.available_models_cache
        st.rerun()


def init_effective_model_size():
    """
    Initialize the effective Qwen3-VL model size based on database compatibility.
    This must be called BEFORE render_model_status_bar() to ensure display consistency.
    Recomputes when the database path changes.
    """
    # Get current config
    from cortex_engine.config import QWEN3_VL_MODEL_SIZE
    from cortex_engine.utils.embedding_validator import get_compatible_qwen3vl_sizes

    db_path = st.session_state.get('db_path', '')

    # Recompute if db_path changed since last calculation
    last_db_path = st.session_state.get('_effective_size_db_path', None)
    if 'effective_qwen3vl_size' in st.session_state and last_db_path == db_path:
        return

    # Track which db_path we computed for
    st.session_state['_effective_size_db_path'] = db_path

    if not db_path:
        # No database path yet, use config value
        st.session_state['effective_qwen3vl_size'] = QWEN3_VL_MODEL_SIZE
        return

    # Check database compatibility
    db_compat = get_compatible_qwen3vl_sizes(db_path)

    if db_compat["is_new_database"]:
        # New database - use config value
        st.session_state['effective_qwen3vl_size'] = QWEN3_VL_MODEL_SIZE
    else:
        # Existing database - check if config size is compatible
        current_size = QWEN3_VL_MODEL_SIZE
        if current_size not in db_compat["compatible_sizes"] and current_size != "auto":
            # Config size is incompatible - switch to first compatible
            if db_compat["compatible_sizes"]:
                st.session_state['effective_qwen3vl_size'] = db_compat["compatible_sizes"][0]
            else:
                st.session_state['effective_qwen3vl_size'] = QWEN3_VL_MODEL_SIZE
        else:
            st.session_state['effective_qwen3vl_size'] = QWEN3_VL_MODEL_SIZE


def render_config_and_scan_ui():
    # Initialize effective model size before rendering status bar
    init_effective_model_size()

    # Render model status bar at top
    render_model_status_bar()

    st.header("Ingest New Documents")
    st.info("Set your paths, navigate folders, and select directories to scan.")

    def reset_scan_path():
        st.session_state.directory_scan_path = st.session_state.knowledge_source_path
        st.session_state.dir_selections = {}

    st.text_input("1. Root Source Documents Path",
                  key="knowledge_source_path",
                  on_change=reset_scan_path,
                  help="📁 Path to your source documents folder. This is where Cortex will scan for files to ingest. You can use Windows paths (C:\\Documents) or Linux paths (/home/user/docs).")
    st.text_input("2. Database Storage Path (Destination)",
                  key="db_path",
                  help="💾 Path where your knowledge base will be stored. This directory will contain the processed documents, embeddings, and knowledge graph. Needs sufficient space for your document collection.")
    resolved_preview = _resolve_db_path(st.session_state.get("db_path", ""))
    if resolved_preview:
        st.caption(f"Resolved runtime path: `{resolved_preview}`")
    st.markdown("---")
    st.markdown("**3. Select Directories to Scan**")

    root_display_path = st.session_state.knowledge_source_path
    # Use the appropriate path converter that handles Docker/WSL environments
    root_wsl_path = convert_source_path_to_docker_mount(root_display_path)
    # Use Docker-aware path validation that checks both normal and Docker mount paths
    is_knowledge_path_valid = validate_path_exists(root_display_path, must_be_dir=True)

    if is_knowledge_path_valid:
        # Initialize directory_scan_path if not set or if it's from a different root
        if 'directory_scan_path' not in st.session_state or not st.session_state.directory_scan_path:
            st.session_state.directory_scan_path = root_display_path
        current_display_path = st.session_state.directory_scan_path
        st.text_input("Current Directory:", current_display_path, disabled=True)
        # Use the appropriate path converter that handles Docker/WSL environments
        current_scan_path_converted = convert_source_path_to_docker_mount(current_display_path)
        current_scan_path_wsl = Path(current_scan_path_converted)
        try:
            # Cache directory scan results to avoid repeated I/O
            cache_key = f"dir_scan_{current_display_path}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = sorted([d.name for d in os.scandir(current_scan_path_wsl) if d.is_dir()], key=str.lower)
            subdirs = st.session_state[cache_key]
            c1, c2, c3 = st.columns(3)
            if c1.button("Select All Visible", use_container_width=True):
                for d in subdirs: st.session_state.dir_selections[str(Path(current_display_path) / d)] = True
                st.rerun()
            if c2.button("Deselect All Visible", use_container_width=True):
                for d in subdirs: st.session_state.dir_selections[str(Path(current_display_path) / d)] = False
                st.rerun()
            if current_scan_path_wsl != Path(root_wsl_path):
                if c3.button("⬆️ Go Up One Level", use_container_width=True):
                    # Clear cache for old directory before navigating
                    old_cache_key = f"dir_scan_{current_display_path}"
                    if old_cache_key in st.session_state:
                        del st.session_state[old_cache_key]
                    st.session_state.directory_scan_path = str(Path(current_display_path).parent)
                    st.rerun()
            st.markdown("---")
            if subdirs:
                cols = st.columns(3)
                for i, dirname in enumerate(subdirs):
                    col = cols[i % 3]
                    with col:
                        full_display_path = str(Path(current_display_path) / dirname)
                        is_selected = st.session_state.dir_selections.get(full_display_path, False)
                        with st.container(border=True):
                            # Top row: Directory name button (left-aligned)
                            if st.button(f"📁 {dirname}", key=f"nav_{full_display_path}", help=f"Navigate into {dirname}", use_container_width=True):
                                # Clear cache for old directory before navigating
                                old_cache_key = f"dir_scan_{current_display_path}"
                                if old_cache_key in st.session_state:
                                    del st.session_state[old_cache_key]
                                st.session_state.directory_scan_path = full_display_path
                                st.rerun()
                            
                            # Bottom row: Selection checkbox
                            st.session_state.dir_selections[full_display_path] = st.checkbox(
                                f"Include in scan", 
                                value=is_selected, 
                                key=f"cb_{full_display_path}",
                                help=f"Include {dirname} in the file scan"
                            )
            else:
                # No subdirectories found - offer to select files directly
                st.info("📂 No subdirectories found. Select individual files below for ingestion.")

                # Get files in current directory (non-recursive, direct files only)
                try:
                    direct_files = [f for f in os.scandir(current_scan_path_wsl) if f.is_file()]
                    supported_files = sorted(
                        [f for f in direct_files if Path(f.name).suffix.lower() not in UNSUPPORTED_EXTENSIONS],
                        key=lambda x: x.name.lower()
                    )
                    file_count = len(supported_files)

                    if file_count > 0:
                        # Initialize direct file selections if needed
                        if 'direct_file_selections' not in st.session_state:
                            st.session_state.direct_file_selections = {}

                        st.write(f"**{file_count} supported files** found:")

                        # Select All / Deselect All buttons
                        sel_col1, sel_col2 = st.columns(2)
                        if sel_col1.button("✅ Select All Files", key="select_all_direct", use_container_width=True):
                            for f in supported_files:
                                full_path = str(Path(current_display_path) / f.name)
                                st.session_state.direct_file_selections[full_path] = True
                            st.rerun()
                        if sel_col2.button("❌ Deselect All Files", key="deselect_all_direct", use_container_width=True):
                            for f in supported_files:
                                full_path = str(Path(current_display_path) / f.name)
                                st.session_state.direct_file_selections[full_path] = False
                            st.rerun()

                        # Show files with checkboxes (paginated if many files)
                        files_per_page = 20
                        if 'direct_files_page' not in st.session_state:
                            st.session_state.direct_files_page = 0

                        total_pages = max(1, -(-file_count // files_per_page))
                        current_page = st.session_state.direct_files_page
                        start_idx = current_page * files_per_page
                        end_idx = min(start_idx + files_per_page, file_count)

                        # File list with checkboxes
                        with st.container(border=True, height=400):
                            for f in supported_files[start_idx:end_idx]:
                                full_path = str(Path(current_display_path) / f.name)
                                wsl_path = str(Path(current_scan_path_wsl) / f.name)

                                # Get file info
                                try:
                                    mod_time = datetime.fromtimestamp(Path(wsl_path).stat().st_mtime)
                                    mod_str = mod_time.strftime('%Y-%m-%d')
                                except:
                                    mod_str = "?"

                                is_selected = st.session_state.direct_file_selections.get(full_path, False)
                                new_val = st.checkbox(
                                    f"**{f.name}** ({mod_str})",
                                    value=is_selected,
                                    key=f"df_{full_path}"
                                )
                                st.session_state.direct_file_selections[full_path] = new_val

                        # Pagination controls
                        if total_pages > 1:
                            pag_col1, pag_col2, pag_col3 = st.columns([1, 2, 1])
                            if current_page > 0:
                                if pag_col1.button("⬅️ Prev", key="df_prev"):
                                    st.session_state.direct_files_page -= 1
                                    st.rerun()
                            pag_col2.write(f"Page {current_page + 1} of {total_pages} ({file_count} files)")
                            if current_page < total_pages - 1:
                                if pag_col3.button("Next ➡️", key="df_next"):
                                    st.session_state.direct_files_page += 1
                                    st.rerun()

                        # Count selected files
                        selected_count = sum(1 for v in st.session_state.direct_file_selections.values() if v)
                        if selected_count > 0:
                            st.success(f"✅ **{selected_count} files selected** for ingestion")
                        else:
                            st.warning("No files selected. Check the boxes above to select files.")
                    else:
                        st.warning("No supported files found in this directory.")
                except Exception as e:
                    st.warning(f"Could not list files: {e}")
        except Exception as e:
            st.warning(f"Could not read directory: {e}")
    else:
        st.warning("Please provide a valid root source path to enable navigation.")

    st.markdown("---")
    st.markdown("**4. Processing Mode**")
    st.checkbox("🚀 **Batch ingest mode** - Skip file preview and process all automatically", 
                key="batch_ingest_mode",
                help="🚀 **Recommended for large collections (100+ files)**: Automatically processes all files without manual review. Failed documents are logged to 'ingest_failures.log' for later inspection. Perfect for initial bulk imports or when you trust your document quality.")
    
    st.markdown("---")
    st.markdown("**5. Collection Assignment**")

    # Collection assignment section - use cached manager
    try:
        if 'collection_mgr_cache' not in st.session_state:
            st.session_state.collection_mgr_cache = WorkingCollectionManager()
        collection_mgr = st.session_state.collection_mgr_cache
        existing_collections = collection_mgr.get_collection_names()
    except Exception as e:
        existing_collections = []
        logger.warning(f"Could not load existing collections: {e}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Collection assignment options
        assignment_mode = st.radio(
            "Where should the ingested documents be assigned?",
            options=["default", "existing", "new"],
            format_func=lambda x: {
                "default": "📂 Default Collection (no organization)",
                "existing": "📁 Add to Existing Collection", 
                "new": "✨ Create New Collection"
            }[x],
            key="collection_assignment_mode",
            help="Choose how to organize your documents during ingestion"
        )
        
        # Initialize collection assignment in session state
        if "target_collection_name" not in st.session_state:
            st.session_state.target_collection_name = ""
            
        if assignment_mode == "existing":
            if existing_collections:
                selected_collection = st.selectbox(
                    "Select existing collection:",
                    options=existing_collections,
                    help="Documents will be added to this existing collection"
                )
                st.session_state.target_collection_name = selected_collection
            else:
                st.warning("⚠️ No existing collections found. Create a new collection or ingest without assignment.")
                st.session_state.target_collection_name = ""
                
        elif assignment_mode == "new":
            new_collection_name = st.text_input(
                "New collection name:",
                placeholder="e.g., Project Alpha Documents",
                help="Enter a name for the new collection to organize these documents"
            )
            if new_collection_name:
                if new_collection_name in existing_collections:
                    st.error(f"❌ Collection '{new_collection_name}' already exists. Choose a different name or select the existing collection.")
                    st.session_state.target_collection_name = ""
                else:
                    st.session_state.target_collection_name = new_collection_name
            else:
                st.session_state.target_collection_name = ""
        else:  # default
            st.session_state.target_collection_name = ""
    
    with col2:
        st.markdown("**📋 Assignment Summary**")
        if assignment_mode == "default":
            st.info("Documents will be available in the general knowledge base without specific collection organization.")
        elif assignment_mode == "existing" and st.session_state.target_collection_name:
            st.success(f"✅ **Target:** {st.session_state.target_collection_name}")
            st.caption("Documents will be added to this existing collection.")
        elif assignment_mode == "new" and st.session_state.target_collection_name:
            st.success(f"✨ **New Collection:** {st.session_state.target_collection_name}")
            st.caption("This collection will be created and populated with the ingested documents.")
        else:
            st.warning("⚠️ Collection assignment incomplete")

    st.markdown("---")
    st.markdown("**6. Advanced Options**")
    with st.expander("Filtering & Pattern-Based Exclusion"):
        st.info("The 'Smart Filtering' options below provide robust, default exclusions. Use 'Pattern-Based' for more specific needs.")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Smart Filtering**")
            st.checkbox("Exclude common temp/archive/business folders", key="filter_exclude_common",
                       help="🗂️ Automatically excludes folders with names like 'temp', 'archive', 'backup', 'working', 'node_modules', etc. Highly recommended for cleaner ingestion.")
            st.checkbox("Prefer .docx over .pdf", key="filter_prefer_docx",
                       help="📄 When both PDF and DOCX versions of the same document exist, only ingest the DOCX version (usually has better text extraction).")
            st.checkbox("Deduplicate by latest version", key="filter_deduplicate", 
                       help="🔄 Automatically detects files with version numbers or dates (e.g., 'report_v2.pdf', 'proposal_final.docx') and only ingests the latest version.")
            st.checkbox("⚡ Skip image processing (faster, but loses visual content)", key="skip_image_processing",
                       value=False,
                       help="🖼️ Skip AI vision analysis of JPG/PNG files. Image processing is now optimized with parallel execution (30s timeout). Only skip if you don't need OCR, charts, or diagram analysis.")
            st.selectbox(
                "📚 Ingestion backend",
                options=["docling", "auto", "default"],
                key="ingest_backend",
                help=(
                    "docling = force Docling processing (recommended); "
                    "auto = gradual migration with fallback; "
                    "default = safest profile (legacy in Docker, gradual elsewhere)."
                ),
            )
            # Use session default (which is WSL-aware) for initial value
            st.number_input(
                "⏱️ Throttle delay (seconds between documents)",
                key="throttle_delay",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("throttle_delay", 1.5 if _is_wsl_env() else 0.5)),
                step=0.5,
                help=(
                    "🎛️ Smart adaptive throttling with GPU/CPU monitoring. Set baseline delay (0-10s) — "
                    "system automatically increases delay when GPU/CPU load is high to prevent freezing. "
                    "0 = no baseline (auto-throttle only), 1–2s recommended on laptops/WSL."
                ),
            )

            # GPU Intensity Control - allows other GPU tasks to run during ingest
            st.slider(
                "🎮 GPU Intensity (%)",
                min_value=25,
                max_value=100,
                value=int(st.session_state.get("gpu_intensity", 75)),
                step=5,
                key="gpu_intensity",
                help=(
                    "Controls GPU usage during embedding. Lower = smaller batches + longer pauses between batches. "
                    "Use 50-75% if running other GPU tasks (photo editing, etc). "
                    "100% = maximum speed, 25% = minimal GPU load."
                ),
            )
            gpu_intensity = st.session_state.get("gpu_intensity", 75)
            if gpu_intensity < 50:
                st.caption(f"⚠️ Low intensity ({gpu_intensity}%) - embedding will be slower but GPU stays available for other tasks")
            elif gpu_intensity < 75:
                st.caption(f"🎮 Balanced ({gpu_intensity}%) - good for multitasking")
            else:
                st.caption(f"⚡ High intensity ({gpu_intensity}%) - faster embedding, higher GPU load")

            if _is_wsl_env():
                st.caption("WSL profile active: baseline default 1.5s; auto-applies cooldown every 20 docs for 15s and LLM timeout 120s.")
        with col2:
            st.write("**Pattern-Based Exclusion**")
            st.checkbox("Enable pattern-based exclusion", key="enable_pattern_exclusion", 
                       on_change=lambda: setattr(st.session_state, 'exclude_patterns_input', DEFAULT_EXCLUSION_PATTERNS_STR if st.session_state.enable_pattern_exclusion else ""))
            if st.session_state.enable_pattern_exclusion: st.text_area("File Patterns (one per line)", key="exclude_patterns_input", height=150)

    st.markdown("---")
    converted_db_path = _resolve_db_path(st.session_state.get("db_path", ""))
    if converted_db_path and "db_path_runtime" not in st.session_state and not st.session_state.get("ingestion_process"):
        st.session_state.db_path_runtime = converted_db_path
    db_parent = Path(converted_db_path).parent if converted_db_path else None
    is_db_path_valid = bool(converted_db_path) and db_parent.exists()
    db_path_writable = False
    db_path_error = None
    if is_db_path_valid:
        db_path_writable, db_path_error = ensure_directory_writable(converted_db_path)
    elif st.session_state.db_path:
        parent_display = db_parent if db_parent else Path(st.session_state.db_path).parent
        db_path_error = f"Parent directory '{parent_display}' is not accessible."

    if db_path_error and not db_path_writable:
        st.error(f"Database storage path issue: {db_path_error}")

    selected_to_scan = [path for path, selected in st.session_state.dir_selections.items() if selected]

    # Check for directly selected files (when no subdirectories exist)
    direct_files_selected = [
        path for path, selected in st.session_state.get('direct_file_selections', {}).items() if selected
    ]

    # Show appropriate button based on selection mode
    has_direct_files = len(direct_files_selected) > 0 and len(selected_to_scan) == 0
    has_dir_selection = len(selected_to_scan) > 0

    # Optional pre-ingest organizer pass (scan-only manifest generation)
    if has_dir_selection:
        with st.expander("🧭 Pre-Ingest Organizer (Recommended for messy repositories)", expanded=False):
            st.caption(
                "Runs a fast pre-ingestion triage scan and writes "
                "`<db_path>/pre_ingest/pre_ingest_manifest.json` "
                "with include/exclude/review recommendations."
            )
            worker_state = st.session_state.setdefault(
                "pre_ingest_worker",
                {
                    "status": "idle",
                    "progress_pct": 0.0,
                    "progress_text": "Not started",
                    "log_lines": [],
                    "event_queue": None,
                    "thread": None,
                    "pause_event": None,
                    "stop_event": None,
                    "error": "",
                },
            )

            def _append_pre_ingest_log(line: str) -> None:
                lines = worker_state.get("log_lines", [])
                lines.append(line)
                worker_state["log_lines"] = lines[-500:]

            def _format_pre_ingest_progress(event, data):
                if event == "scan_started":
                    return f"[scan] Starting scan across {data.get('source_dir_count', 0)} directories"
                if event == "scan_dir_start":
                    return (
                        f"[scan] Directory {data.get('directory_index', 0)}/{data.get('directory_count', 0)}: "
                        f"{data.get('source_dir', '')}"
                    )
                if event == "scan_dir_progress":
                    return (
                        f"[scan] {data.get('source_dir', '')} | scanned={data.get('scanned_in_dir', 0)} "
                        f"discovered={data.get('discovered_total', 0)} | {data.get('current_path', '')}"
                    )
                if event == "scan_dir_done":
                    return (
                        f"[scan] Completed {data.get('source_dir', '')} | scanned={data.get('scanned_in_dir', 0)} "
                        f"discovered={data.get('discovered_total', 0)}"
                    )
                if event == "scan_complete":
                    return f"[scan] Candidate discovery complete: {data.get('discovered_total', 0)} files"
                if event == "analyze_progress":
                    return (
                        f"[analyze] {data.get('processed', 0)}/{data.get('total', 0)} | "
                        f"{data.get('current_path', '')}"
                    )
                if event == "scan_truncated":
                    return (
                        f"[scan] Max file limit reached ({data.get('max_file_count', 0)}), "
                        f"truncated at {data.get('discovered_total', 0)} files"
                    )
                if event == "manifest_written":
                    return f"[done] Manifest written: {data.get('manifest_path', '')}"
                if event == "scan_dir_missing":
                    return (
                        f"[warn] Missing source directory: {data.get('source_dir', '')} "
                        f"-> {data.get('resolved_path', '')}"
                    )
                return f"[{event}] {json.dumps(data, default=str)}"

            def _drain_pre_ingest_events() -> None:
                event_queue = worker_state.get("event_queue")
                if not event_queue:
                    return
                while True:
                    try:
                        event = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    kind = event.get("kind")
                    if kind == "progress":
                        evt = event.get("event", "")
                        data = event.get("data", {})
                        _append_pre_ingest_log(_format_pre_ingest_progress(evt, data))
                        if evt == "scan_complete":
                            worker_state["progress_pct"] = 0.5
                            worker_state["progress_text"] = (
                                f"Discovery complete: {data.get('discovered_total', 0)} files. Analyzing..."
                            )
                        elif evt == "analyze_progress":
                            total = max(1, int(data.get("total", 1)))
                            processed = max(0, min(int(data.get("processed", 0)), total))
                            worker_state["progress_pct"] = 0.5 + (processed / total) * 0.5
                            worker_state["progress_text"] = f"Analyzing files: {processed}/{total}"
                    elif kind == "done":
                        result = event.get("result", {})
                        worker_state["status"] = "completed"
                        worker_state["progress_pct"] = 1.0
                        worker_state["progress_text"] = "Pre-ingest organizer complete"
                        st.session_state.pre_ingest_manifest_path = result.get("manifest_path")
                        st.session_state.pre_ingest_manifest_paths = list(
                            result.get("manifest_paths") or ([result.get("manifest_path")] if result.get("manifest_path") else [])
                        )
                        if st.session_state.get("pre_ingest_manifest_path"):
                            st.session_state.pre_ingest_manifest_selected = st.session_state.get("pre_ingest_manifest_path")
                        st.session_state.pre_ingest_summary = result.get("summary", {})
                        _append_pre_ingest_log(f"[done] Completed scan ({result.get('total_files', 0)} files)")
                    elif kind == "stopped":
                        worker_state["status"] = "stopped"
                        worker_state["progress_text"] = "Stopped by operator"
                        _append_pre_ingest_log(f"[stop] {event.get('message', 'Scan stopped')}")
                    elif kind == "error":
                        worker_state["status"] = "failed"
                        worker_state["error"] = str(event.get("error", "Unknown error"))
                        worker_state["progress_text"] = "Failed"
                        _append_pre_ingest_log(f"[error] {worker_state['error']}")

                thread_obj = worker_state.get("thread")
                if thread_obj and (not thread_obj.is_alive()) and worker_state.get("status") in {"running", "paused", "stopping"}:
                    worker_state["status"] = "stopped"
                    worker_state["progress_text"] = "Stopped"

            _drain_pre_ingest_events()

            status = worker_state.get("status", "idle")
            status_col, prog_col = st.columns([1, 3])
            status_col.metric("Status", status)
            prog_col.progress(float(worker_state.get("progress_pct", 0.0)), text=str(worker_state.get("progress_text", "")))

            force_rerun = st.checkbox(
                "Force fresh re-run (clear previous manifest summary first)",
                key="pre_ingest_force_rerun",
                value=False,
                help="Use this if a previous run's results are still shown or you want a clean re-scan cycle.",
            )

            c1, c2, c3, c4 = st.columns(4)
            can_start = status in {"idle", "completed", "failed", "stopped"}
            if c1.button(
                f"Run/Re-run Pre-Ingest Organizer ({len(selected_to_scan)} dirs)",
                key="run_pre_ingest_organizer",
                use_container_width=True,
                type="secondary",
                disabled=not can_start,
            ):
                if not (is_knowledge_path_valid and is_db_path_valid and db_path_writable):
                    st.error("Valid source path and writable DB path are required.")
                else:
                    if force_rerun:
                        st.session_state.pre_ingest_manifest_path = ""
                        st.session_state.pre_ingest_manifest_paths = []
                        st.session_state.pre_ingest_manifest_selected = ""
                        st.session_state.pre_ingest_summary = {}
                        st.session_state.pre_ingest_manifest_preview = []
                    resolved_runtime = set_runtime_db_path(converted_db_path)
                    event_queue = queue.Queue()
                    pause_event = threading.Event()
                    pause_event.set()
                    stop_event = threading.Event()
                    worker_state["status"] = "running"
                    worker_state["progress_pct"] = 0.0
                    worker_state["progress_text"] = "Pre-ingest organizer starting..."
                    worker_state["log_lines"] = []
                    worker_state["event_queue"] = event_queue
                    worker_state["pause_event"] = pause_event
                    worker_state["stop_event"] = stop_event
                    worker_state["error"] = ""

                    def _progress_callback(event, data):
                        event_queue.put({"kind": "progress", "event": event, "data": data})

                    def _control_callback():
                        while not pause_event.is_set():
                            if stop_event.is_set():
                                raise PreIngestScanCancelled("Scan stopped by operator")
                            time.sleep(0.2)
                        if stop_event.is_set():
                            raise PreIngestScanCancelled("Scan stopped by operator")

                    def _worker_target():
                        try:
                            result = run_pre_ingest_organizer_scan(
                                source_dirs=selected_to_scan,
                                db_path=resolved_runtime,
                                progress_callback=_progress_callback,
                                control_callback=_control_callback,
                            )
                            event_queue.put({"kind": "done", "result": result})
                        except PreIngestScanCancelled as exc:
                            event_queue.put({"kind": "stopped", "message": str(exc)})
                        except Exception as exc:
                            event_queue.put({"kind": "error", "error": str(exc)})

                    thread_obj = threading.Thread(target=_worker_target, daemon=True, name="pre-ingest-organizer")
                    worker_state["thread"] = thread_obj
                    thread_obj.start()
                    st.rerun()

            if c2.button(
                "Pause",
                key="pause_pre_ingest_organizer",
                use_container_width=True,
                disabled=status != "running",
            ):
                pause_event = worker_state.get("pause_event")
                if pause_event:
                    pause_event.clear()
                    worker_state["status"] = "paused"
                    _append_pre_ingest_log("[control] Pause requested")
                    st.rerun()

            if c3.button(
                "Resume",
                key="resume_pre_ingest_organizer",
                use_container_width=True,
                disabled=status != "paused",
            ):
                pause_event = worker_state.get("pause_event")
                if pause_event:
                    pause_event.set()
                    worker_state["status"] = "running"
                    _append_pre_ingest_log("[control] Resume requested")
                    st.rerun()

            if c4.button(
                "Stop",
                key="stop_pre_ingest_organizer",
                use_container_width=True,
                disabled=status not in {"running", "paused"},
            ):
                stop_event = worker_state.get("stop_event")
                pause_event = worker_state.get("pause_event")
                if stop_event:
                    stop_event.set()
                if pause_event:
                    pause_event.set()
                worker_state["status"] = "stopping"
                worker_state["progress_text"] = "Stopping..."
                _append_pre_ingest_log("[control] Stop requested")
                st.rerun()

            lines = worker_state.get("log_lines", [])
            if lines:
                st.code("\n".join(lines[-120:]), language="text")
            if worker_state.get("status") == "failed" and worker_state.get("error"):
                st.error(f"Pre-ingest organizer failed: {worker_state.get('error')}")
            if worker_state.get("status") == "completed":
                st.success(f"Pre-ingest manifest created at: `{st.session_state.get('pre_ingest_manifest_path', '')}`")

            if worker_state.get("status") in {"running", "paused", "stopping"}:
                time.sleep(1)
                st.rerun()

            summary = st.session_state.get("pre_ingest_summary")
            if summary:
                pol = summary.get("policy_counts", {})
                own = summary.get("ownership_counts", {})
                st.markdown(
                    f"- Policy: include={pol.get('include', 0)}, exclude={pol.get('exclude', 0)}, "
                    f"review_required={pol.get('review_required', 0)}, do_not_ingest={pol.get('do_not_ingest', 0)}"
                )
                st.markdown(
                    f"- Ownership: first_party={own.get('first_party', 0)}, "
                    f"client_owned={own.get('client_owned', 0)}, external_ip={own.get('external_ip', 0)}"
                )
            runtime_db_for_pre_ingest = set_runtime_db_path(converted_db_path) if converted_db_path else ""
            pre_ingest_dir = Path(runtime_db_for_pre_ingest) / "pre_ingest" if runtime_db_for_pre_ingest else None
            disk_manifests = []
            if pre_ingest_dir and pre_ingest_dir.exists():
                disk_manifests = sorted(
                    [p.as_posix() for p in pre_ingest_dir.glob("pre_ingest_manifest*.json") if p.is_file()],
                    key=lambda p: Path(p).stat().st_mtime,
                    reverse=True,
                )
            known_manifests = list(st.session_state.get("pre_ingest_manifest_paths", []))
            manifest_options = []
            seen = set()
            for p in (disk_manifests + known_manifests):
                if p and p not in seen:
                    seen.add(p)
                    manifest_options.append(p)

            if manifest_options:
                default_selected = st.session_state.get("pre_ingest_manifest_selected") or manifest_options[0]
                if default_selected not in manifest_options:
                    default_selected = manifest_options[0]
                selected_manifest = st.selectbox(
                    "Manifest File",
                    options=manifest_options,
                    index=manifest_options.index(default_selected),
                    key="pre_ingest_manifest_selected",
                    help="Choose a saved manifest to review/edit and prepare for ingest.",
                )
                st.caption(f"Selected manifest: `{selected_manifest}`")

                preview_col1, preview_col2 = st.columns([1, 1])
                if preview_col1.button(
                    "Load Selected Manifest",
                    key="load_pre_ingest_manifest_preview",
                    use_container_width=True,
                ):
                    try:
                        with open(selected_manifest, "r", encoding="utf-8") as handle:
                            payload = json.load(handle)
                        records = list(payload.get("records", []))
                        st.session_state.pre_ingest_manifest_preview = records
                        st.session_state.pre_ingest_manifest_path = selected_manifest
                        st.session_state.pre_ingest_summary = payload.get("summary", {})
                        st.success(f"Loaded {len(records)} manifest records.")
                    except Exception as e:
                        st.error(f"Failed to load manifest preview: {e}")

                if preview_col2.button(
                    "Clear Loaded Manifest",
                    key="clear_pre_ingest_manifest_preview",
                    use_container_width=True,
                ):
                    st.session_state.pre_ingest_manifest_preview = []
                    st.rerun()

                preview_records = list(st.session_state.get("pre_ingest_manifest_preview", []))
                if preview_records:
                    st.markdown("**Review + Edit Manifest Decisions**")
                    st.caption("Editable fields: policy, sensitivity, ownership, and operator notes.")

                    table_rows = []
                    for rec in preview_records:
                        table_rows.append(
                            {
                                "file_name": rec.get("file_name", ""),
                                "doc_class": rec.get("doc_class", ""),
                                "ingest_policy_class": rec.get("ingest_policy_class", "review_required"),
                                "sensitivity_level": rec.get("sensitivity_level", "public"),
                                "source_ownership": rec.get("source_ownership", "first_party"),
                                "is_canonical_version": bool(rec.get("is_canonical_version", False)),
                                "operator_note": rec.get("operator_note", ""),
                                "file_path": rec.get("file_path", ""),
                            }
                        )

                    edited_rows = st.data_editor(
                        table_rows,
                        key="pre_ingest_manifest_editor",
                        use_container_width=True,
                        hide_index=True,
                        height=360,
                        disabled=["file_name", "is_canonical_version", "file_path"],
                        column_config={
                            "doc_class": st.column_config.SelectboxColumn(
                                "doc_class",
                                options=["work_knowledge", "admin_finance", "legal_contract", "draft", "unknown"],
                                required=True,
                            ),
                            "ingest_policy_class": st.column_config.SelectboxColumn(
                                "ingest_policy_class",
                                options=["include", "exclude", "review_required", "do_not_ingest"],
                                required=True,
                            ),
                            "sensitivity_level": st.column_config.SelectboxColumn(
                                "sensitivity_level",
                                options=["public", "internal", "confidential", "restricted"],
                                required=True,
                            ),
                            "source_ownership": st.column_config.SelectboxColumn(
                                "source_ownership",
                                options=["first_party", "client_owned", "external_ip"],
                                required=True,
                            ),
                            "operator_note": st.column_config.TextColumn(
                                "operator_note",
                                help="Optional override/context note for this document decision.",
                            ),
                        },
                    )

                    if hasattr(edited_rows, "to_dict"):
                        edited_records = edited_rows.to_dict(orient="records")
                    elif isinstance(edited_rows, list):
                        edited_records = edited_rows
                    else:
                        edited_records = table_rows

                    edit_col1, edit_col2 = st.columns([1, 1])
                    if edit_col1.button(
                        "Save Decisions to Manifest",
                        key="save_pre_ingest_manifest_edits",
                        use_container_width=True,
                        type="primary",
                    ):
                        try:
                            with open(selected_manifest, "r", encoding="utf-8") as handle:
                                payload = json.load(handle)
                            records = list(payload.get("records", []))
                            by_path = {str(r.get("file_path", "")): r for r in records}

                            for row in edited_records:
                                fp = str(row.get("file_path", ""))
                                target = by_path.get(fp)
                                if not target:
                                    continue
                                new_doc_class = str(row.get("doc_class", target.get("doc_class", "unknown")))
                                old_doc_class = str(target.get("doc_class", "unknown"))
                                target["doc_class"] = new_doc_class
                                target["doc_class_overridden"] = bool(new_doc_class != old_doc_class)
                                target["ingest_policy_class"] = str(row.get("ingest_policy_class", target.get("ingest_policy_class", "review_required")))
                                target["sensitivity_level"] = str(row.get("sensitivity_level", target.get("sensitivity_level", "public")))
                                target["source_ownership"] = str(row.get("source_ownership", target.get("source_ownership", "first_party")))
                                target["operator_note"] = str(row.get("operator_note", target.get("operator_note", ""))).strip()

                            payload["summary"] = {
                                "total_files": len(records),
                                "policy_counts": {
                                    "include": sum(1 for r in records if r.get("ingest_policy_class") == "include"),
                                    "exclude": sum(1 for r in records if r.get("ingest_policy_class") == "exclude"),
                                    "review_required": sum(1 for r in records if r.get("ingest_policy_class") == "review_required"),
                                    "do_not_ingest": sum(1 for r in records if r.get("ingest_policy_class") == "do_not_ingest"),
                                },
                                "ownership_counts": {
                                    "first_party": sum(1 for r in records if r.get("source_ownership") == "first_party"),
                                    "client_owned": sum(1 for r in records if r.get("source_ownership") == "client_owned"),
                                    "external_ip": sum(1 for r in records if r.get("source_ownership") == "external_ip"),
                                },
                            }

                            with open(selected_manifest, "w", encoding="utf-8") as handle:
                                json.dump(payload, handle, indent=2)

                            st.session_state.pre_ingest_manifest_preview = records
                            st.session_state.pre_ingest_manifest_path = selected_manifest
                            st.session_state.pre_ingest_summary = payload["summary"]
                            st.success("Saved manifest decisions.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save manifest decisions: {e}")

                    include_review = edit_col2.toggle(
                        "Include review_required in approved set",
                        key="pre_ingest_include_review_required",
                        value=False,
                    )

                    if st.button(
                        "Prepare Ingest From Approved Policies",
                        key="prepare_ingest_from_manifest_policies",
                        use_container_width=True,
                    ):
                        allowed = {"include"}
                        if include_review:
                            allowed.add("review_required")

                        selected_files = sorted(
                            {
                                str(r.get("file_path", "")).strip()
                                for r in st.session_state.get("pre_ingest_manifest_preview", [])
                                if str(r.get("ingest_policy_class", "")).strip() in allowed
                                and str(r.get("file_path", "")).strip()
                            }
                        )

                        if not selected_files:
                            st.warning("No files match approved policies. Adjust decisions and save first.")
                        else:
                            st.session_state.files_to_review = selected_files
                            st.session_state.file_selections = {fp: True for fp in selected_files}
                            st.session_state.review_page = 0
                            scan_config = {
                                "selected_directories": selected_to_scan,
                                "knowledge_source_path": st.session_state.knowledge_source_path,
                                "db_path": st.session_state.db_path,
                                "db_path_runtime": set_runtime_db_path(converted_db_path),
                                "batch_ingest_mode": st.session_state.get("batch_ingest_mode", False),
                                "manifest_path": selected_manifest,
                                "scan_timestamp": datetime.now().isoformat(),
                            }
                            if "current_scan_config" not in st.session_state:
                                st.session_state.current_scan_config = {}
                            st.session_state.current_scan_config.update(scan_config)
                            st.session_state.ingestion_stage = "pre_analysis"
                            st.success(f"Prepared {len(selected_files)} approved files for ingest review.")
                            st.rerun()

                    if st.toggle(
                        "Show Raw Manifest JSON (first 50 records)",
                        key="show_pre_ingest_raw_json",
                        value=False,
                    ):
                        st.json(st.session_state.get("pre_ingest_manifest_preview", [])[:50])

    if has_direct_files:
        # Direct file selection mode - button to proceed with selected files
        if st.button(f"📄 Proceed with {len(direct_files_selected)} Selected File(s)", type="primary", use_container_width=True):
            if is_knowledge_path_valid and is_db_path_valid and db_path_writable:
                resolved_runtime = set_runtime_db_path(converted_db_path)
                config_manager = ConfigManager()
                config_manager.update_config({"knowledge_source_path": st.session_state.knowledge_source_path, "ai_database_path": st.session_state.db_path})

                # Check model availability before proceeding
                include_images = not st.session_state.get("skip_image_processing", False)
                model_check = model_checker.check_ingestion_requirements(include_images=include_images)

                if not model_check["can_proceed"]:
                    st.error("❌ **Cannot proceed with ingestion - Missing required models**")
                    st.markdown(model_checker.format_status_message(model_check))
                    return

                if model_check["warnings"]:
                    st.warning(model_checker.format_status_message(model_check))
                else:
                    st.success("✅ All required models available - proceeding with ingestion")

                # Convert Windows paths to WSL/container paths for the selected files
                converted_files = []
                for file_path in direct_files_selected:
                    wsl_path = convert_windows_to_wsl_path(file_path)
                    converted_files.append(wsl_path)

                # Populate files_to_review directly (bypass directory scanning)
                st.session_state.files_to_review = sorted(converted_files)
                st.session_state.file_selections = {fp: True for fp in st.session_state.files_to_review}
                st.session_state.review_page = 0

                # Save scan configuration
                scan_config = {
                    "selected_directories": [],
                    "direct_files": direct_files_selected,
                    "knowledge_source_path": st.session_state.knowledge_source_path,
                    "db_path": st.session_state.db_path,
                    "db_path_runtime": resolved_runtime,
                    "batch_ingest_mode": st.session_state.get("batch_ingest_mode", False),
                    "scan_timestamp": datetime.now().isoformat()
                }
                if "current_scan_config" not in st.session_state:
                    st.session_state.current_scan_config = {}
                st.session_state.current_scan_config.update(scan_config)

                # Determine next stage based on batch mode
                if st.session_state.get("batch_ingest_mode", False):
                    files_to_process = st.session_state.files_to_review
                    if files_to_process:
                        container_db_path = get_runtime_db_path()
                        batch_manager = BatchState(container_db_path)
                        batch_manager.create_batch(files_to_process, scan_config)

                        st.session_state.log_messages = []
                        st.session_state.ingestion_stage = "analysis_running"
                        st.session_state.batch_mode_active = True

                        target_collection = st.session_state.get('target_collection_name', '')
                        command = build_ingestion_command(container_db_path, files_to_process, target_collection)

                        try:
                            st.session_state.ingestion_process = spawn_ingest(command)
                            start_ingest_reader(st.session_state.ingestion_process)
                            logger.info(f"Direct file mode: Auto-started processing {len(files_to_process)} files")
                        except Exception as e:
                            st.error(f"❌ Failed to start processing: {e}")
                            logger.error(f"Auto-start failed: {e}")
                            st.session_state.ingestion_stage = "pre_analysis"
                    else:
                        st.session_state.ingestion_stage = "pre_analysis"
                else:
                    st.session_state.ingestion_stage = "pre_analysis"

                st.rerun()
            else:
                if not is_knowledge_path_valid:
                    st.error("Root Source Path is not valid.")
                if not is_db_path_valid:
                    st.error(f"DB path's parent directory is not accessible: {db_parent}")
                elif not db_path_writable:
                    st.error(db_path_error or "Database path is not writable.")

    elif has_dir_selection:
        # Directory selection mode - original scan button
        if st.button(f"🔎 Scan {len(selected_to_scan)} Selected Director(y/ies) for New Files", type="primary", use_container_width=True):
            if is_knowledge_path_valid and is_db_path_valid and db_path_writable:
                resolved_runtime = set_runtime_db_path(converted_db_path)
                config_manager = ConfigManager(); config_manager.update_config({"knowledge_source_path": st.session_state.knowledge_source_path, "ai_database_path": st.session_state.db_path})

                # Capture scan configuration for batch resume (avoid modifying existing widget keys)
                scan_config = {
                    "selected_directories": selected_to_scan,
                    "knowledge_source_path": st.session_state.knowledge_source_path,
                    "db_path": st.session_state.db_path,
                    "db_path_runtime": resolved_runtime,
                    "filter_exclude_common": st.session_state.get("filter_exclude_common", False),
                    "enable_pattern_exclusion": st.session_state.get("enable_pattern_exclusion", False),
                    "exclude_patterns_input": st.session_state.get("exclude_patterns_input", ""),
                    "filter_prefer_docx": st.session_state.get("filter_prefer_docx", False),
                    "batch_ingest_mode": st.session_state.get("batch_ingest_mode", False),
                    "auto_finalize_enabled": st.session_state.get("batch_ingest_mode", False),
                    "scan_timestamp": datetime.now().isoformat()
                }
                # Store in a way that won't conflict with widgets
                if "current_scan_config" not in st.session_state:
                    st.session_state.current_scan_config = {}
                st.session_state.current_scan_config.update(scan_config)

                # Check model availability before scanning
                include_images = not st.session_state.get("skip_image_processing", False)
                model_check = model_checker.check_ingestion_requirements(include_images=include_images)

                if not model_check["can_proceed"]:
                    st.error("❌ **Cannot proceed with ingestion - Missing required models**")
                    st.markdown(model_checker.format_status_message(model_check))

                    # Show quick fix options
                    if model_check["missing_models"]:
                        st.markdown("### Quick Fix Options:")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Option 1: Install missing models**")
                            install_cmds = model_checker.get_model_installation_commands(model_check["missing_models"])
                            for cmd in install_cmds:
                                st.code(cmd, language="bash")
                            if st.button("Install missing models now", key="install_missing_models", use_container_width=True):
                                if install_missing_models(model_check["missing_models"]):
                                    st.success("Models installed. Re-checking availability...")
                                    st.rerun()

                        with col2:
                            if "llava" in model_check["missing_models"]:
                                st.markdown("**Option 2: Skip image processing**")
                                st.info("📋 **To proceed without image analysis:**\n\n"
                                       "1. Expand **'Advanced Options'** section below\n"
                                       "2. Under **'Smart Filtering'**, check the box:\n"
                                       "   ⚡ **Skip image processing (faster ingestion)**\n"
                                       "3. Click **'Scan 1 Selected Directory(ies)'** again")
                                st.markdown("↓ *Scroll down to Advanced Options* ↓")
                    return

                # Show successful model check
                if model_check["warnings"]:
                    st.warning(model_checker.format_status_message(model_check))
                else:
                    st.success("✅ All required models available - proceeding with ingestion")

                # Start file scanning with enhanced progress monitoring (no spinner to allow progress updates to show)
                scan_for_files(selected_to_scan)

                # After scanning, determine next stage based on batch mode
                if st.session_state.get("batch_ingest_mode", False):
                    # In batch mode, skip the batch management screen and start processing immediately
                    files_to_process = st.session_state.get("files_to_review", [])
                    if files_to_process:
                        # Initialize batch state
                        container_db_path = get_runtime_db_path()
                        batch_manager = BatchState(container_db_path)
                        batch_manager.create_batch(files_to_process, scan_config)

                        # Start processing immediately - go directly to analysis_running
                        st.session_state.log_messages = []
                        st.session_state.ingestion_stage = "analysis_running"
                        st.session_state.batch_mode_active = True

                        # Build and start ingestion command
                        target_collection = st.session_state.get('target_collection_name', '')
                        command = build_ingestion_command(container_db_path, files_to_process, target_collection)

                        try:
                            st.session_state.ingestion_process = spawn_ingest(command)
                            start_ingest_reader(st.session_state.ingestion_process)
                            logger.info(f"Batch mode: Auto-started processing {len(files_to_process)} files")
                        except Exception as e:
                            st.error(f"❌ Failed to start processing: {e}")
                            logger.error(f"Auto-start failed: {e}")
                            st.session_state.ingestion_stage = "batch_processing"  # Fallback to manual start
                    else:
                        st.session_state.ingestion_stage = "batch_processing"  # No files, show batch screen
                else:
                    st.session_state.ingestion_stage = "pre_analysis"

                st.rerun()
            else:
                if not is_knowledge_path_valid:
                    st.error("Root Source Path is not valid.")
                if not is_db_path_valid:
                    st.error(f"DB path's parent directory is not accessible: {db_parent}")
                elif not db_path_writable:
                    st.error(db_path_error or "Database path is not writable. Please choose a directory on a mounted drive that allows write access.")

    else:
        # No files or directories selected - show disabled button
        st.button("🔎 Select files or directories above to enable scanning", type="primary", use_container_width=True, disabled=True)

    render_maintenance_link(
        "pages/6_Maintenance.py",
        button_key="ingest_open_maintenance_top",
    )

def render_pre_analysis_ui():
    st.header("Pre-Analysis Review")
    files_to_review = st.session_state.get("files_to_review", [])
    total_files = len(files_to_review)

    if not files_to_review:
        st.success("Scan complete. No new documents found based on your criteria.")
        if st.button("⬅️ Back to Configuration", key="back_config_no_new_docs"): initialize_state(force_reset=True); st.rerun()
        return

    def update_selection(f_path, is_selected):
        st.session_state.file_selections[f_path] = is_selected

    with st.container(border=True):
        sc1, sc2 = st.columns(2)
        def sort_by_name(): st.session_state.files_to_review.sort(key=lambda f: Path(f).name)
        def _safe_mtime(f):
            try:
                return Path(f).stat().st_mtime
            except OSError:
                return 0.0
        def sort_by_date(): st.session_state.files_to_review.sort(key=_safe_mtime, reverse=True)
        sc1.button("Sort by Name (A-Z)", on_click=sort_by_name, use_container_width=True)
        sc2.button("Sort by Date (Newest First)", on_click=sort_by_date, use_container_width=True)

    page = st.session_state.review_page
    start_idx, end_idx = page * REVIEW_PAGE_SIZE, (page + 1) * REVIEW_PAGE_SIZE
    paginated_files = files_to_review[start_idx:end_idx]
    total_pages = -(-total_files // REVIEW_PAGE_SIZE) or 1

    num_selected = sum(1 for f in st.session_state.file_selections.values() if f)
    st.info(f"Found **{total_files}** documents and images. Currently selecting **{num_selected}** for processing. Displaying page {page + 1} of {total_pages}.")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Select All on Page", use_container_width=True): st.session_state.file_selections.update({f: True for f in paginated_files}); st.rerun()
    if c2.button("Deselect All on Page", use_container_width=True): st.session_state.file_selections.update({f: False for f in paginated_files}); st.rerun()
    if c3.button("Select All (All Pages)", use_container_width=True): st.session_state.file_selections.update({f: True for f in files_to_review}); st.rerun()
    if c4.button("Deselect All (All Pages)", use_container_width=True): st.session_state.file_selections.update({f: False for f in files_to_review}); st.rerun()

    st.markdown("---")

    for fp in paginated_files:
        try:
            mod_time = datetime.fromtimestamp(Path(fp).stat().st_mtime)
        except OSError:
            mod_time = datetime.min
        label = f"**{Path(fp).name}** (`{mod_time.strftime('%Y-%m-%d %H:%M:%S')}`)"
        is_selected = st.session_state.file_selections.get(fp, False)
        new_selection = st.checkbox(label, value=is_selected, key=f"cb_{fp}")
        if new_selection != is_selected:
            update_selection(fp, new_selection)
            st.rerun()
        with st.expander("Show Preview"):
             st.text_area("Preview", get_full_file_content(fp), height=200, disabled=True, key=f"preview_{fp}")

    st.divider()
    nav_cols = st.columns([1, 1, 5])
    if page > 0: nav_cols[0].button("⬅️ Previous", on_click=lambda: st.session_state.update(review_page=page - 1), use_container_width=True)
    if end_idx < total_files: nav_cols[1].button("Next ➡️", on_click=lambda: st.session_state.update(review_page=page + 1), use_container_width=True)
    nav_cols[2].write(f"Page {page + 1} of {total_pages}")

    st.divider()
    globally_selected = [f for f, s in st.session_state.file_selections.items() if s]
    globally_ignored = [f for f, s in st.session_state.file_selections.items() if not s]
    proc_c1, proc_c2 = st.columns(2)
    if proc_c1.button("⬅️ Back to Configuration", use_container_width=True): initialize_state(force_reset=True); st.rerun()
    if proc_c2.button(f"Process {len(globally_selected)} files & Ignore {len(globally_ignored)}", type="primary", use_container_width=True, disabled=not globally_selected):
        container_db_path = get_runtime_db_path()
        if globally_ignored:
            chroma_db_dir = Path(container_db_path) / "knowledge_hub_db"
            chroma_db_dir.mkdir(parents=True, exist_ok=True)
            ingested_log_path = chroma_db_dir / INGESTED_FILES_LOG
            ingested_log = {}
            if ingested_log_path.exists():
                try:
                    with open(ingested_log_path, 'r') as f: ingested_log = json.load(f)
                except json.JSONDecodeError: pass
            for fp in globally_ignored: ingested_log[fp] = "user_excluded"
            with open(ingested_log_path, 'w') as f: json.dump(ingested_log, f, indent=4)

        # Save scan configuration for batch resume capability
        scan_config = {
            "root_path": st.session_state.get("directory_scan_path", ""),
            "db_path": st.session_state.get("db_path", ""),
            "selected_dirs": st.session_state.get("selected_dirs", []),
            "filter_exclude_common": st.session_state.get("filter_exclude_common", True),
            "filter_prefer_docx": st.session_state.get("filter_prefer_docx", True),
            "filter_deduplicate": st.session_state.get("filter_deduplicate", True),
            "enable_pattern_exclusion": st.session_state.get("enable_pattern_exclusion", False),
            "exclude_patterns_input": st.session_state.get("exclude_patterns_input", ""),
            "target_collection_name": st.session_state.get("target_collection_name", "")
        }
        scan_config_path = Path(container_db_path) / "scan_config.json"
        with open(scan_config_path, 'w') as f:
            json.dump(scan_config, f, indent=2)

        st.session_state.log_messages = []; st.session_state.ingestion_stage = "analysis_running"
        target_collection = st.session_state.get('target_collection_name', '')
        command = build_ingestion_command(container_db_path, globally_selected, target_collection)
        st.session_state.ingestion_process = spawn_ingest(command)
        start_ingest_reader(st.session_state.ingestion_process)
        st.rerun()

def render_log_and_review_ui(stage_title: str, on_complete_stage: str):
    shared_render_log_and_review_ui(
        stage_title=stage_title,
        on_complete_stage=on_complete_stage,
        initialize_state=initialize_state,
        get_runtime_db_path=get_runtime_db_path,
        set_runtime_db_path=set_runtime_db_path,
        auto_resume_from_batch_config=auto_resume_from_batch_config,
        start_ingest_reader=start_ingest_reader,
        get_ingest_lines=get_ingest_lines,
        load_staged_files=load_staged_files,
        should_auto_finalize=should_auto_finalize,
        start_automatic_finalization=start_automatic_finalization,
        max_auto_finalize_retries=MAX_AUTO_FINALIZE_RETRIES,
        logger=logger,
    )

def render_completion_screen():
    st.success("✅ Finalization complete! Your knowledge base is up to date.")

    # Success toast with collection and count info
    # Get actual document count from database instead of relying on session state
    try:
        from cortex_engine.utils.path_utils import get_database_path
        import chromadb

        db_path = get_database_path()
        chroma_path = Path(db_path) / "knowledge_hub_db"

        if chroma_path.exists():
            client = chromadb.PersistentClient(path=str(chroma_path))
            collection = client.get_or_create_collection(name="knowledge_base")
            total_docs = collection.count()

            target_collection = st.session_state.get('target_collection_name', '') or 'default'

            # Try to get ingested IDs from session state, but fallback to showing total
            ingested_ids = st.session_state.get('last_ingested_doc_ids', []) or []

            if ingested_ids:
                st.info(f"📚 Collection: {target_collection} • 📄 Documents added this session: {len(ingested_ids)} • Total documents: {total_docs}")
            else:
                st.info(f"📚 Total documents in knowledge base: {total_docs}")
    except Exception as e:
        logger.warning(f"Could not retrieve document count: {e}")
        pass
    
    st.info(
        "Collection assignment is finalized inside the ingestion engine from staged metadata. "
        "Post-finalization UI reassignment is disabled to prevent orphan references."
    )
    st.session_state.last_ingested_doc_ids = []
    st.session_state.target_collection_name = ""
    st.markdown("---")
    # Collections file quick access and preview
    try:
        mgr = WorkingCollectionManager()
        collections_path = mgr.collections_file
        st.caption("Collections file location (for troubleshooting):")
        st.code(collections_path, language="text")
        with st.expander("🔎 Preview collections JSON", expanded=False):
            import json
            if os.path.exists(collections_path):
                with open(collections_path, 'r') as f:
                    data = json.load(f)
                st.json(data)
            else:
                st.warning("Collections file not found at this path.")
    except Exception:
        pass
    if st.button("⬅️ Start a New Ingestion"):
        initialize_state(force_reset=True)
        st.rerun()


def render_metadata_review_ui():
    shared_render_metadata_review_ui(
        should_auto_finalize=should_auto_finalize,
        start_automatic_finalization=start_automatic_finalization,
        log_failed_documents=log_failed_documents,
        get_runtime_db_path=get_runtime_db_path,
        serialize_staging_payload=serialize_staging_payload,
        project_root=project_root,
        spawn_ingest=spawn_ingest,
        start_ingest_reader=start_ingest_reader,
        initialize_state=initialize_state,
        get_full_file_content=get_full_file_content,
        review_page_size=REVIEW_PAGE_SIZE,
        doc_type_options=DOC_TYPE_OPTIONS,
        proposal_outcome_options=PROPOSAL_OUTCOME_OPTIONS,
        get_document_type_manager=get_document_type_manager,
        convert_windows_to_wsl_path=convert_windows_to_wsl_path,
    )



def render_document_type_management():
    shared_render_document_type_management(
        get_document_type_manager=get_document_type_manager,
    )

# --- Main App Logic ---
shared_render_ingest_page_shell(
    initialize_state=initialize_state,
    version_string=VERSION_STRING,
    help_system=help_system,
    render_ollama_status_panel=lambda: shared_render_ollama_status_panel(cache_ttl_seconds=60),
    render_recovery_panels=lambda: shared_render_recovery_panels(
        config_manager_cls=ConfigManager,
        recovery_manager_cls=IngestionRecoveryManager,
        logger=logger,
        shared_check_recovery_needed_fn=shared_check_recovery_needed,
        shared_render_recovery_section_fn=shared_render_recovery_section,
        render_recovery_quick_actions_fn=render_recovery_quick_actions,
        recover_collection_from_ingest_log_fn=recover_collection_from_ingest_log,
        filter_valid_doc_ids_fn=filter_existing_doc_ids_for_collection,
        collection_manager_cls=WorkingCollectionManager,
    ),
    get_runtime_db_path=get_runtime_db_path,
    set_runtime_db_path=set_runtime_db_path,
    batch_state_cls=BatchState,
    render_active_batch_management=render_active_batch_management,
    detect_orphaned_session_from_log=detect_orphaned_session_from_log,
    render_orphaned_session_notice=render_orphaned_session_notice,
    render_document_type_management=render_document_type_management,
    show_collection_migration_healthcheck=show_collection_migration_healthcheck,
    render_sidebar_model_config=render_sidebar_model_config,
    render_ingest_stage=render_ingest_stage,
    stage_handlers={
        "config": render_config_and_scan_ui,
        "pre_analysis": render_pre_analysis_ui,
        "batch_processing": render_batch_processing_ui,
        "analysis_running": lambda: render_log_and_review_ui("Live Analysis Log", "metadata_review"),
        "metadata_review": render_metadata_review_ui,
        "finalizing": lambda: render_log_and_review_ui("Live Finalization Log", "config_done"),
        "config_done": render_completion_screen,
    },
    project_root=project_root,
)
