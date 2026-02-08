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
    read_ingested_log,
    recover_collection_from_ingest_log,
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
        commands = model_checker.get_model_installation_commands(missing_models)
        with st.status("Installing models...", expanded=True) as status:
            for cmd in commands:
                status.write(f"$ {cmd}")
                try:
                    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
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
    except Exception:
        pass

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
    except Exception:
        # Non-fatal; just skip if anything goes wrong
        pass

def initialize_state(force_reset: bool = False):
    # Cache config to avoid file reads on every interaction
    if "cached_config" not in st.session_state or force_reset:
        config_manager = ConfigManager()
        st.session_state.cached_config = config_manager.get_config()
    config = st.session_state.cached_config

    if force_reset:
        keys_to_reset = list(st.session_state.keys())
        for key in keys_to_reset:
            del st.session_state[key]

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
    
    candidate_files = [f.as_posix() for f in all_files if f.as_posix() not in ingested_files and f.suffix.lower() not in UNSUPPORTED_EXTENSIONS]
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
    st.header("Batch Processing Mode")
    files_to_process = st.session_state.get("files_to_review", [])
    
    # Check for existing batch state
    container_db_path = get_runtime_db_path()
    batch_manager = BatchState(container_db_path)
    set_runtime_db_path(str(batch_manager.db_path))
    batch_status = batch_manager.get_status()
    
    # If resuming from main screen, use batch remaining files
    if batch_status["active"] and not files_to_process:
        batch_state = batch_manager.load_state()
        if batch_state:
            files_to_process = batch_state.get('files_remaining', [])
            st.session_state.files_to_review = files_to_process
    
    # Handle orphaned session resume - create proper batch state
    elif not batch_status["active"] and not files_to_process and st.session_state.get("resume_mode_enabled"):
        st.warning("⚠️ Resume mode is enabled but no files selected. Please go back and select your original directories.")
        if st.button("⬅️ Back to Configuration", key="back_config_orphaned", use_container_width=True):
            st.session_state.ingestion_stage = "config"
            st.rerun()
        return
    
    total_files = len(files_to_process)
    
    # Show batch status if there's an active batch
    if batch_status["active"]:
        st.subheader("📊 Batch Status")
        
        # Manual refresh button for updating progress
        if st.button("🔄 Refresh Progress", key="refresh_batch_progress"):
            st.rerun()
        
        if batch_status.get('is_chunked', False):
            if batch_status.get('auto_pause_after_chunks'):
                # Show session-based metrics for auto-pause batches with document-level progress
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Overall Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
                with col2:
                    st.metric("Current Chunk", f"{batch_status['current_chunk']}/{batch_status['total_chunks']}")
                with col3:
                    # Show document progress within current chunk
                    chunk_progress = f"{batch_status.get('current_chunk_progress', 0)}/{batch_status.get('current_chunk_total', batch_status.get('chunk_size', 0))}"
                    st.metric("Chunk Documents", chunk_progress)
                with col4:
                    session_progress = f"{batch_status['chunks_processed_in_session']}/{batch_status['auto_pause_after_chunks']}"
                    st.metric("Session Chunks", session_progress)
                with col5:
                    st.metric("Remaining Files", batch_status['remaining'])
                with col6:
                    st.metric("Errors", batch_status['error_count'])
            else:
                # Regular chunked display with document progress
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Overall Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
                with col2:
                    st.metric("Current Chunk", f"{batch_status['current_chunk']}/{batch_status['total_chunks']}")
                with col3:
                    # Show document progress within current chunk
                    chunk_progress = f"{batch_status.get('current_chunk_progress', 0)}/{batch_status.get('current_chunk_total', batch_status.get('chunk_size', 0))}"
                    st.metric("Chunk Documents", chunk_progress)
                with col4:
                    st.metric("Chunk Size", batch_status['chunk_size'])
                with col5:
                    st.metric("Remaining Files", batch_status['remaining'])
                with col6:
                    st.metric("Errors", batch_status['error_count'])
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
            with col2:
                st.metric("Completed", f"{batch_status['progress_percent']}%")
            with col3:
                st.metric("Remaining", batch_status['remaining'])
            with col4:
                st.metric("Errors", batch_status['error_count'])
        
        # Progress bars
        if batch_status.get('is_chunked', False):
            # Show dual progress: overall and current chunk
            st.markdown("**Overall Progress**")
            overall_progress = batch_status['progress_percent'] / 100.0
            st.progress(overall_progress, text=f"Total: {batch_status['completed']}/{batch_status.get('total_files', 0)} files ({batch_status['progress_percent']}%)")
            
            st.markdown("**Current Chunk Progress**")
            chunk_progress_percent = batch_status.get('chunk_progress_percent', 0) / 100.0
            chunk_docs = batch_status.get('current_chunk_progress', 0)
            chunk_total = batch_status.get('current_chunk_total', batch_status.get('chunk_size', 0))
            st.progress(chunk_progress_percent, text=f"Chunk {batch_status['current_chunk']}: {chunk_docs}/{chunk_total} documents ({batch_status.get('chunk_progress_percent', 0)}%)")
        else:
            # Single progress bar for non-chunked processing
            progress = batch_status['progress_percent'] / 100.0
            st.progress(progress, text=f"Processing files... {batch_status['completed']}/{batch_status.get('total_files', 0)}")
        
        # Show pause/resume status
        if batch_status['paused']:
            if batch_status.get('auto_pause_after_chunks') and batch_status.get('chunks_processed_in_session', 0) >= batch_status.get('auto_pause_after_chunks', 0):
                st.success(f"🎯 **Session Complete!** Processed {batch_status['chunks_processed_in_session']} chunks ({batch_status['chunks_processed_in_session'] * batch_status.get('chunk_size', 0)} files)")
                st.info("Ready to start next session during your off-peak hours")
            else:
                st.warning("⏸️ Batch processing is **PAUSED**")
        else:
            st.info(f"🔄 Batch processing active (Started: {batch_status['started_at'][:19]})")
        
        st.markdown("---")
    
    if not files_to_process and not batch_status["active"]:
        st.success("✅ Batch processing complete!")
        
        # Check if there are successfully processed documents for collection creation
        if not st.session_state.get('last_ingested_doc_ids'):
            # Note: Document IDs are now tracked directly during ingestion
            # The completion screen queries ChromaDB for accurate counts
            pass
        
        # Show collection creation option if we have successfully processed documents
        if st.session_state.get('last_ingested_doc_ids'):
            # Check for pre-configured target collection
            target_collection = st.session_state.get('target_collection_name', '')
            
            if target_collection:
                # Auto-assign to target collection
                try:
                    collection_mgr = WorkingCollectionManager()
                    existing_collections = collection_mgr.get_collection_names()
                    
                    if target_collection not in existing_collections:
                        # Create new collection
                        if collection_mgr.create_collection(target_collection):
                            st.success(f"✅ Created new collection: **{target_collection}**")
                        else:
                            st.error(f"❌ Failed to create collection: {target_collection}")
                            target_collection = ""
                    
                    if target_collection:
                        # Assign documents to target collection
                        collection_mgr.add_docs_by_id_to_collection(target_collection, st.session_state.last_ingested_doc_ids)
                        success_count = len(st.session_state.last_ingested_doc_ids)
                        st.success(f"🎯 **{success_count} documents automatically assigned** to collection: **{target_collection}**")
                        st.session_state.last_ingested_doc_ids = []
                        st.session_state.target_collection_name = ""
                        
                except Exception as e:
                    st.error(f"❌ Failed to assign to collection '{target_collection}': {e}")
                    target_collection = ""  # Fall back to manual
            
            # Manual collection creation (if no target or auto-assignment failed)
            if not target_collection and st.session_state.get('last_ingested_doc_ids'):
                with st.form("batch_completion_collection"):
                    st.subheader("📂 Create Collection from Successfully Processed Documents")
                    success_count = len(st.session_state.last_ingested_doc_ids)
                    st.info(f"You have {success_count} successfully processed documents available for collection creation.")
                    
                    collection_name = st.text_input(
                        "Collection Name", 
                        placeholder="e.g., Recent Import - Documents",
                        help="Name for the new collection containing successfully processed documents"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("📂 Create Collection", type="primary", use_container_width=True):
                            if collection_name:
                                collection_mgr = WorkingCollectionManager()
                                if collection_name in collection_mgr.get_collection_names():
                                    st.error(f"Collection '{collection_name}' already exists. Please choose a different name.")
                                else:
                                    try:
                                        collection_mgr.add_docs_by_id_to_collection(collection_name, st.session_state.last_ingested_doc_ids)
                                        st.success(f"✅ Successfully created collection '{collection_name}' with {success_count} documents!")
                                        st.session_state.last_ingested_doc_ids = []  # Clear after successful creation
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to create collection: {e}")
                            else:
                                st.warning("Please provide a name for the collection.")
                    
                    with col2:
                        if st.form_submit_button("Skip", use_container_width=True):
                            st.session_state.last_ingested_doc_ids = []  # Clear the list
                            st.info("Skipped collection creation.")
                            st.rerun()
        else:
            st.info("No successfully processed documents found for collection creation.")
        
        st.markdown("---")
        if st.button("⬅️ Back to Configuration", key="back_config_no_files", use_container_width=True): 
            initialize_state(force_reset=True)
            st.rerun()
        return
    
    # Show different UI based on batch state
    if batch_status["active"]:
        st.info(f"🚀 **Batch Mode Active:** Resume processing or manage the current batch.")
    else:
        st.info(f"🚀 **Batch Mode Enabled:** Processing all {total_files} files automatically. Files with errors will be logged separately for later review.")
    
    # Automatic chunked processing for large batches
    if total_files > 500:
        # Automatically enable chunked processing for large batches
        # Determine optimal chunk size based on batch size
        if total_files > 2000:
            auto_chunk_size = 250  # Smaller chunks for very large batches
        elif total_files > 1000:
            auto_chunk_size = 500  # Medium chunks for large batches
        else:
            auto_chunk_size = 500  # Standard chunk size

        estimated_chunks = (total_files + auto_chunk_size - 1) // auto_chunk_size

        # Auto-enable chunked processing
        st.session_state.use_chunked_processing = True
        st.session_state.chunk_size = auto_chunk_size

        # Show informational message about automatic chunking
        st.info(f"🔧 **Automatic Chunked Processing Enabled**: {total_files} files detected. "
                f"Processing in {estimated_chunks} chunks of {auto_chunk_size} files each for optimal performance.")

        # Allow user to customize if desired
        with st.expander("⚙️ **Customize Chunked Processing** (Optional)", expanded=False):
            st.markdown("Chunked processing is automatically enabled for large batches. You can adjust the settings below if needed.")

            col1, col2 = st.columns(2)
            with col1:
                custom_chunk_size = st.selectbox(
                    "Chunk Size:",
                    options=[100, 250, 500, 1000],
                    index=[100, 250, 500, 1000].index(auto_chunk_size) if auto_chunk_size in [100, 250, 500, 1000] else 1,
                    key="custom_chunk_size",
                    help="Number of files to process in each chunk"
                )

            with col2:
                custom_estimated_chunks = (total_files + custom_chunk_size - 1) // custom_chunk_size
                st.metric("Estimated Chunks", custom_estimated_chunks)

            if st.button("✅ Apply Custom Chunk Size", key="apply_custom_chunk"):
                st.session_state.chunk_size = custom_chunk_size
                st.success(f"✅ Custom chunk size applied: {custom_estimated_chunks} chunks of {custom_chunk_size} files each")
                st.rerun()
    
    # Show a summary of what will be processed
    with st.expander("📋 Files to Process", expanded=False):
        for i, fp in enumerate(files_to_process[:10], 1):  # Show first 10
            st.write(f"{i}. {Path(fp).name}")
        if total_files > 10:
            st.write(f"... and {total_files - 10} more files")
    
    st.markdown("---")
    
    # Batch control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    # Check if we should start processing or if it's already running
    process_running = st.session_state.get("ingestion_process") is not None
    
    with col1:
        # Start/Resume button
        if not process_running:
            if batch_status["active"] and batch_status["remaining"] > 0:
                # For active batches with stored config, use automatic resume
                if batch_status.get("has_scan_config", False):
                    if st.button("▶️ Auto Resume", type="primary", use_container_width=True, key="auto_resume_batch"):
                        if auto_resume_from_batch_config(batch_manager):
                            st.rerun()
                        else:
                            st.error("❌ Auto-resume failed. Check logs.")
                else:
                    # Fallback for batches without stored config
                    if st.button("▶️ Resume Processing", type="primary", use_container_width=True, key="resume_processing_fallback"):
                        # Validate files_to_process before starting
                        if not files_to_process:
                            st.error("❌ No files to process. Please check your file selection or batch state.")
                            return
                        
                        # Clear any paused flag before resuming
                        try:
                            batch_manager.start_new_session()
                        except Exception:
                            pass

                        st.session_state.log_messages = []
                        st.session_state.ingestion_stage = "analysis_running"
                        st.session_state.batch_mode_active = True
                        # Clear batch_ingest_mode to allow transition to analysis_running stage
                        st.session_state.batch_ingest_mode = False

                        # Build command with resume flag and collection assignment
                        target_collection = st.session_state.get('target_collection_name', '')
                        command = build_ingestion_command(container_db_path, files_to_process, target_collection, resume=True)
                        
                        # Debug logging
                        logger.info(f"Starting batch processing with {len(files_to_process)} files")
                        logger.info(f"Command: {' '.join(command[:6])}... (truncated)")
                        
                        try:
                            st.session_state.ingestion_process = spawn_ingest(command)
                            start_ingest_reader(st.session_state.ingestion_process)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to start batch processing: {e}")
                            logger.error(f"Failed to start subprocess: {e}")
            else:
                # New batch
                if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True, key="start_batch_processing"):
                    # Validate files_to_process before starting
                    if not files_to_process:
                        st.error("❌ No files to process. Please check your file selection or batch state.")
                        return
                    
                    st.session_state.log_messages = []
                    st.session_state.ingestion_stage = "analysis_running"
                    st.session_state.batch_mode_active = True
                    # Clear batch_ingest_mode to allow transition to analysis_running stage
                    st.session_state.batch_ingest_mode = False

                    # Build command with collection assignment
                    target_collection = st.session_state.get('target_collection_name', '')
                    command = build_ingestion_command(container_db_path, files_to_process, target_collection)
                    
                    # Debug logging
                    logger.info(f"Starting batch processing with {len(files_to_process)} files")
                    logger.info(f"Command: {' '.join(command[:6])}... (truncated)")
                    
                    try:
                        st.session_state.ingestion_process = spawn_ingest(command)
                        start_ingest_reader(st.session_state.ingestion_process)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to start batch processing: {e}")
                        logger.error(f"Failed to start subprocess: {e}")
        else:
            st.info("Processing is running...")
    
    with col2:
        # Pause button
        if process_running and not batch_status.get("paused", False):
            if st.button("⏸️ Pause", use_container_width=True):
                batch_manager.pause_batch()
                st.success("Pause request sent")
                st.rerun()
        elif batch_status.get("paused", False):
            st.info("⏸️ Paused")
    
    with col3:
        # Clear batch button
        if batch_status["active"]:
            if st.button("🗑️ Clear Batch", key="clear_batch_processing", use_container_width=True):
                batch_manager.clear_batch()
                st.success("Batch cleared")
                st.rerun()
        else:
            st.empty()
    
    with col4:
        # Back to config button
        if st.button("⬅️ Back to Config", key="back_config_batch_processing", use_container_width=True):
            initialize_state(force_reset=True)
            st.rerun()

def render_active_batch_management(batch_manager: BatchState, batch_status: dict):
    """Render the active batch management section with consolidated controls"""
    st.subheader("📊 Active Batch Management")

    # CRITICAL: Always show Stop/Pause controls at the top
    st.markdown("### 🎛️ Batch Controls")
    st.warning("⚠️ **Batch processing is active.** Use the controls below to manage it.")

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)

    with control_col1:
        if st.button("⏹️ **STOP BATCH**", key="stop_active_batch", type="secondary", use_container_width=True):
            # Force stop the batch and subprocess
            if st.session_state.get('ingestion_process'):
                try:
                    st.session_state.ingestion_process.terminate()
                except Exception as e:
                    logger.warning(f"Process termination failed: {e}")
                st.session_state.ingestion_process = None

            # Signal reader thread to stop
            stop_ev = st.session_state.get("ingestion_reader_stop")
            if stop_ev:
                stop_ev.set()

            # Clear the batch
            batch_manager.clear_batch()

            # Reset to config
            st.session_state.ingestion_stage = "config"
            st.success("✅ Batch stopped and cleared!")
            st.rerun()

    with control_col2:
        if not batch_status.get("paused", False):
            if st.button("⏸️ Pause", key="pause_active_batch", use_container_width=True):
                batch_manager.pause_batch()
                st.success("Pause request sent")
                st.rerun()
        else:
            st.info("⏸️ Paused")

    with control_col3:
        if st.button("🗑️ Clear Batch", key="clear_active_batch", use_container_width=True):
            batch_manager.clear_batch()
            st.success("Batch cleared")
            st.rerun()

    with control_col4:
        if st.button("⬅️ Back", key="back_from_active_batch", use_container_width=True):
            initialize_state(force_reset=True)
            st.rerun()

    st.markdown("---")

    # Manual/auto refresh controls
    refresher_col1, refresher_col2 = st.columns([1,1])
    with refresher_col1:
        if st.button("🔄 Refresh Progress", key="refresh_active_batch_progress"):
            st.rerun()
    with refresher_col2:
        auto_refresh = st.checkbox("Auto refresh (3s)", key="auto_refresh_active_batch")
        if auto_refresh:
            import time as _t
            _t.sleep(3)
            st.rerun()

    # If batch is marked processing but no subprocess is attached, try to auto-start and attach
    process_obj = st.session_state.get("ingestion_process")
    if (
        batch_status.get('active', False)
        and not batch_status.get('paused', False)
        and batch_status.get('remaining', 0) > 0
        and not process_obj
    ):
        try:
            st.info("🔌 Reattaching to batch and starting processing…")
            if auto_resume_from_batch_config(batch_manager):
                st.rerun()
                return
        except Exception as _auto_err:
            logger.warning(f"Auto-start from active state failed: {_auto_err}")

    # If a subprocess exists but no reader is attached (e.g., after refresh), reattach it
    if process_obj and st.session_state.get("ingestion_output_queue") is None:
        try:
            start_ingest_reader(process_obj)
            st.info("🔌 Reattached log reader to running process…")
        except Exception as _reattach_err:
            logger.warning(f"Failed to reattach reader in active mgmt: {_reattach_err}")
    
    process_running = False
    if process_obj:
        try:
            process_running = process_obj.poll() is None
        except Exception:
            process_running = False
    
    # Show batch status metrics
    if batch_status.get('is_chunked', False):
        if batch_status.get('auto_pause_after_chunks'):
            # Show session-based metrics for auto-pause batches with document-level progress
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Overall Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
            with col2:
                st.metric("Current Chunk", f"{batch_status['current_chunk']}/{batch_status['total_chunks']}")
            with col3:
                # Show document progress within current chunk
                chunk_progress = f"{batch_status.get('current_chunk_progress', 0)}/{batch_status.get('current_chunk_total', batch_status.get('chunk_size', 0))}"
                st.metric("Chunk Documents", chunk_progress)
            with col4:
                session_progress = f"{batch_status['chunks_processed_in_session']}/{batch_status['auto_pause_after_chunks']}"
                st.metric("Session Chunks", session_progress)
            with col5:
                st.metric("Remaining Files", batch_status['remaining'])
            with col6:
                st.metric("Errors", batch_status['error_count'])
        else:
            # Regular chunked display with document progress
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Overall Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
            with col2:
                st.metric("Current Chunk", f"{batch_status['current_chunk']}/{batch_status['total_chunks']}")
            with col3:
                # Show document progress within current chunk
                chunk_progress = f"{batch_status.get('current_chunk_progress', 0)}/{batch_status.get('current_chunk_total', batch_status.get('chunk_size', 0))}"
                st.metric("Chunk Documents", chunk_progress)
            with col4:
                st.metric("Chunk Size", batch_status['chunk_size'])
            with col5:
                st.metric("Remaining Files", batch_status['remaining'])
            with col6:
                st.metric("Errors", batch_status['error_count'])
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
        with col2:
            st.metric("Completed", f"{batch_status['progress_percent']}%")
        with col3:
            st.metric("Remaining", batch_status['remaining'])
        with col4:
            st.metric("Errors", batch_status['error_count'])

    # Progress bars  
    if batch_status.get('is_chunked', False):
        # Show dual progress: overall and current chunk
        st.markdown("**Overall Progress**")
        overall_progress = batch_status['progress_percent'] / 100.0
        st.progress(overall_progress, text=f"Total: {batch_status['completed']}/{batch_status.get('total_files', 0)} files ({batch_status['progress_percent']}%)")
        
        st.markdown("**Current Chunk Progress**")
        chunk_progress_percent = batch_status.get('chunk_progress_percent', 0) / 100.0
        chunk_docs = batch_status.get('current_chunk_progress', 0)
        chunk_total = batch_status.get('current_chunk_total', batch_status.get('chunk_size', 0))
        st.progress(chunk_progress_percent, text=f"Chunk {batch_status['current_chunk']}: {chunk_docs}/{chunk_total} documents ({batch_status.get('chunk_progress_percent', 0)}%)")
    else:
        # Single progress bar for non-chunked processing
        progress = batch_status['progress_percent'] / 100.0
        st.progress(progress, text=f"Processing files... {batch_status['completed']}/{batch_status.get('total_files', 0)}")

    # Drain any available process output to keep metrics/log fresh while in active batch view
    proc = st.session_state.get('ingestion_process')
    if proc and proc.poll() is None:
        for line in get_ingest_lines(max_lines=50):
            if line.startswith("CORTEX_THROTTLE::"):
                try:
                    _, delay_str, gpu_str, cpu_str = line.split("::", 3)
                    st.session_state.current_throttle_delay = float(delay_str)
                    st.session_state.current_gpu_util = None if gpu_str == "N/A" else float(gpu_str)
                    st.session_state.current_cpu_util = None if cpu_str == "N/A" else float(cpu_str)
                    st.session_state.last_heartbeat_ts = time.time()
                    st.session_state.log_messages.append(
                        f"THROTTLE delay={float(delay_str):.1f}s GPU={gpu_str}% CPU={cpu_str}%"
                    )
                except Exception:
                    pass
            elif line.startswith("CORTEX_HEARTBEAT::"):
                try:
                    _, gpu_str, cpu_str = line.split("::", 2)
                    st.session_state.current_gpu_util = None if gpu_str == "N/A" else float(gpu_str)
                    st.session_state.current_cpu_util = None if cpu_str == "N/A" else float(cpu_str)
                    st.session_state.last_heartbeat_ts = time.time()
                    ts = time.strftime('%H:%M:%S')
                    st.session_state.log_messages.append(f"[{ts}] HEARTBEAT GPU={gpu_str}% CPU={cpu_str}%")
                except Exception:
                    pass

    # Performance Tuning Display - Always show when batch is active
    st.markdown("---")
    st.markdown("**⚡ Performance Tuning**")
    throttle_col1, throttle_col2, throttle_col3 = st.columns([1, 1, 1])
    with throttle_col1:
        delay_val = st.session_state.get('current_throttle_delay', 1.0)
        st.metric("⏱️ Throttle Delay", f"{delay_val:.1f}s", help="Current delay between documents (1.0s baseline for responsiveness, auto-adjusts if system load increases)")
    with throttle_col2:
        gpu_val = st.session_state.get('current_gpu_util', None)
        if gpu_val is not None:
            gpu_delta = f"+{gpu_val - 80:.0f}%" if gpu_val > 80 else None
            st.metric("🎮 GPU Load", f"{gpu_val:.0f}%", delta=gpu_delta, delta_color="inverse", help="GPU utilization (auto-throttles at >80%)")
        else:
            st.metric("🎮 GPU Load", "N/A", help="GPU monitoring unavailable (nvidia-smi not found)")
    with throttle_col3:
        cpu_val = st.session_state.get('current_cpu_util', None)
        if cpu_val is not None:
            cpu_delta = f"+{cpu_val - 85:.0f}%" if cpu_val > 85 else None
            st.metric("💻 CPU Load", f"{cpu_val:.0f}%", delta=cpu_delta, delta_color="inverse", help="CPU utilization (auto-throttles at >85%)")
        else:
            st.metric("💻 CPU Load", "N/A", help="CPU monitoring unavailable (psutil not found)")

    st.markdown("---")

    # Show pause/resume status
    if batch_status['paused']:
        if batch_status.get('auto_pause_after_chunks') and batch_status.get('chunks_processed_in_session', 0) >= batch_status.get('auto_pause_after_chunks', 0):
            st.success(f"🎯 **Session Complete!** Processed {batch_status['chunks_processed_in_session']} chunks ({batch_status['chunks_processed_in_session'] * batch_status.get('chunk_size', 0)} files)")
            st.info("Ready to start next session during your off-peak hours")
        else:
            st.warning("⏸️ Batch processing is **PAUSED**")
    else:
        st.info(f"🔄 Batch processing active (Started: {batch_status['started_at'][:19]})")
    
    # Show error information if there are errors
    if batch_status.get('error_count', 0) > 0:
        error_count = batch_status['error_count']
        completed = batch_status['completed']
        total = batch_status.get('total_files', 0)
        success_rate = round((completed / total) * 100, 1) if total > 0 else 0
        
        # Use warning instead of error for normal processing issues
        if success_rate >= 95:
            st.warning(f"📝 **{error_count} files skipped** during processing " +
                      f"({success_rate}% success rate: {completed}/{total} files completed). " +
                      "Common causes: corrupted files, unsupported formats, or metadata extraction issues. " +
                      "Check logs for details if needed.")
        else:
            st.error(f"⚠️ **{error_count} errors encountered** " + 
                    f"({success_rate}% success rate: {completed}/{total} files completed). " +
                    "Check logs for details or clear batch to restart with fresh setup.")

    # Check if batch is effectively complete (no remaining files) but still marked as active
    if batch_status.get('remaining', 0) == 0 and batch_status.get('completed', 0) > 0:
        st.success("🎉 **Batch Processing Complete!** All files have been processed.")
        
        # Populate document IDs if not already done
        # Note: Document IDs are now tracked directly during ingestion
        # The completion screen queries ChromaDB for accurate counts
        if not st.session_state.get('last_ingested_doc_ids'):
            pass
        
        # Show collection creation option
        if st.session_state.get('last_ingested_doc_ids'):
            # Check for pre-configured target collection
            target_collection = st.session_state.get('target_collection_name', '')
            
            if target_collection:
                # Auto-assign to target collection
                try:
                    collection_mgr = WorkingCollectionManager()
                    existing_collections = collection_mgr.get_collection_names()
                    
                    if target_collection not in existing_collections:
                        # Create new collection
                        if collection_mgr.create_collection(target_collection):
                            st.success(f"✅ Created new collection: **{target_collection}**")
                        else:
                            st.error(f"❌ Failed to create collection: {target_collection}")
                            target_collection = ""
                    
                    if target_collection:
                        # Assign documents to target collection
                        collection_mgr.add_docs_by_id_to_collection(target_collection, st.session_state.last_ingested_doc_ids)
                        success_count = len(st.session_state.last_ingested_doc_ids)
                        error_count = batch_status.get('error_count', 0)
                        
                        if error_count > 0:
                            st.success(f"🎯 **{success_count} documents automatically assigned** to collection: **{target_collection}** (with {error_count} errors)")
                        else:
                            st.success(f"🎯 **{success_count} documents automatically assigned** to collection: **{target_collection}**")
                            
                        st.session_state.last_ingested_doc_ids = []
                        st.session_state.target_collection_name = ""
                        batch_manager.clear_batch()  # Clear the completed batch
                        
                except Exception as e:
                    st.error(f"❌ Failed to assign to collection '{target_collection}': {e}")
                    target_collection = ""  # Fall back to manual
            
            # Manual collection creation (if no target or auto-assignment failed)
            if not target_collection and st.session_state.get('last_ingested_doc_ids'):
                with st.form("active_batch_completion_collection"):
                    st.subheader("📂 Create Collection from Processed Documents")
                    success_count = len(st.session_state.last_ingested_doc_ids)
                    error_count = batch_status.get('error_count', 0)
                    
                    if error_count > 0:
                        st.info(f"✅ **{success_count} documents successfully processed** (with {error_count} errors)")
                    else:
                        st.info(f"✅ **{success_count} documents successfully processed**")
                    
                    collection_name = st.text_input(
                        "Collection Name", 
                        placeholder="e.g., Batch Import - Documents",
                        help="Name for the new collection containing successfully processed documents"
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.form_submit_button("📂 Create Collection", type="primary", use_container_width=True):
                            if collection_name:
                                collection_mgr = WorkingCollectionManager()
                                if collection_name in collection_mgr.get_collection_names():
                                    st.error(f"Collection '{collection_name}' already exists. Please choose a different name.")
                                else:
                                    try:
                                        collection_mgr.add_docs_by_id_to_collection(collection_name, st.session_state.last_ingested_doc_ids)
                                        st.success(f"✅ Successfully created collection '{collection_name}' with {success_count} documents!")
                                        st.session_state.last_ingested_doc_ids = []
                                        batch_manager.clear_batch()  # Clear the completed batch
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to create collection: {e}")
                            else:
                                st.warning("Please provide a name for the collection.")
                    
                    with col2:
                        if st.form_submit_button("Skip Collection", use_container_width=True):
                            st.session_state.last_ingested_doc_ids = []
                            batch_manager.clear_batch()  # Clear the completed batch
                            st.info("Skipped collection creation and cleared batch.")
                            st.rerun()
                            
                    with col3:
                        if st.form_submit_button("Clear Batch", use_container_width=True):
                            st.session_state.last_ingested_doc_ids = []
                            batch_manager.clear_batch()
                            st.success("Batch cleared.")
                            st.rerun()

    # Check if there are staged documents ready for finalization
    auto_finalize_ready = should_auto_finalize() and batch_status.get('completed', 0) > 0 and batch_status.get('remaining', 0) == 0
    auto_finalize_enabled = bool(
        st.session_state.get("batch_mode_active", False) or
        batch_status.get("auto_finalize_enabled", False)
    )
    if auto_finalize_ready:
        st.markdown("---")
        st.info("📦 **Analysis Complete!** Your documents are ready to be added to the knowledge base.")

        finalize_col1, finalize_col2 = st.columns([3, 1])

        if auto_finalize_enabled and not st.session_state.get("batch_auto_finalize_started"):
            st.success("🚀 Starting automatic finalization for batch mode…")
            st.session_state.batch_auto_finalize_started = True
            st.session_state.auto_finalize_triggered = True
            start_automatic_finalization()
            st.rerun()
            return

        with finalize_col1:
            if st.button("✅ Complete Ingestion (Finalize to Database)", type="primary", use_container_width=True, key="manual_finalize_batch"):
                with st.spinner("Finalizing documents to database..."):
                    try:
                        container_db_path = convert_to_docker_mount_path(st.session_state.db_path)
                        command = [
                            sys.executable, "-m", "cortex_engine.ingest_cortex",
                            "--finalize-from-staging", "--db-path", container_db_path
                        ]

                        result = subprocess.run(command, capture_output=True, text=True, timeout=300)

                        if result.returncode == 0:
                            st.success("✅ Documents successfully added to knowledge base!")
                            batch_manager.clear_batch()
                            # Set flag to indicate finalization completed successfully
                            st.session_state.manual_finalize_success = True
                            st.session_state.auto_finalize_triggered = False
                            st.session_state.ingestion_stage = "config_done"
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Finalization failed: {result.stderr}")
                            logger.error(f"Finalization error: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Failed to finalize: {e}")
                        logger.error(f"Manual finalization error: {e}", exc_info=True)

        with finalize_col2:
            st.caption("This will add your analyzed documents to the searchable database.")

    # Consolidated action buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Check if batch is actively processing (use batch state, not subprocess state)
        # A batch is "processing" if it's active, not paused, and has remaining files
        batch_is_processing = (
            batch_status.get('active', False) and
            not batch_status.get('paused', False) and
            batch_status.get('remaining', 0) > 0
        )

        is_paused = batch_status.get('paused', False)

        # Dynamic button based on batch state (not subprocess state)
        if batch_is_processing:
            # Batch is actively processing - show Pause button
            if st.button("⏸️ Pause Processing", type="secondary", use_container_width=True, key="pause_processing_main"):
                batch_manager.pause_batch()
                st.success("⏸️ Processing paused")
                st.rerun()
        else:
            # Batch is paused or stopped - show Resume/Start button
            if batch_status.get('auto_pause_after_chunks') and is_paused:
                if batch_status.get('chunks_processed_in_session', 0) >= batch_status.get('auto_pause_after_chunks', 0):
                    button_text = "🔄 Start Next Session"
                else:
                    button_text = "▶️ Resume Processing"
            else:
                button_text = "▶️ Resume Processing"

            if st.button(button_text, type="primary", use_container_width=True, key="resume_processing_main"):
                # If paused for any reason, clear paused flag and session counters
                try:
                    if is_paused:
                        batch_manager.start_new_session()
                except Exception:
                    pass

                # Resume processing
                if auto_resume_from_batch_config(batch_manager):
                    st.rerun()
                else:
                    st.error("❌ Failed to resume batch automatically. Please check the logs.")
    
    with col2:
        if st.button("🗑️ Clear This Batch", key="clear_active_batch", use_container_width=True):
            batch_manager.clear_batch()
            st.success("Batch cleared")
            st.rerun()
            
    with col3:
        if st.button("📋 View Ingestion Logs", key="view_logs_batch", use_container_width=True):
            st.session_state.show_logs = True
            st.rerun()
            
    with col4:
        if st.button("⬅️ Back to Config", key="back_to_config_batch", use_container_width=True):
            batch_manager.clear_batch()
            initialize_state(force_reset=True)
            st.rerun()
    
    # Show ingestion logs if requested
    if st.session_state.get("show_logs", False):
        st.markdown("---")
        st.subheader("📋 Recent Ingestion Logs")
        
        ingestion_log_path = Path(__file__).parent.parent / "logs" / "ingestion.log"
        if ingestion_log_path.exists():
            try:
                with open(ingestion_log_path, 'r') as log_file:
                    log_content = log_file.read()
                    if log_content.strip():
                        log_lines = log_content.strip().split('\n')
                        
                        # Enhanced log display controls
                        log_col1, log_col2, log_col3 = st.columns([2, 1, 1])
                        with log_col1:
                            # Line count selector
                            line_options = [25, 50, 100, 200, len(log_lines)]
                            line_labels = ["Last 25 lines", "Last 50 lines", "Last 100 lines", "Last 200 lines", f"All {len(log_lines)} lines"]
                            selected_lines = st.selectbox("Log Display:", 
                                                         options=line_options,
                                                         format_func=lambda x: line_labels[line_options.index(x)],
                                                         index=1,  # Default to 50 lines
                                                         key="log_display_lines")
                        with log_col2:
                            if st.button("🔄 Refresh Logs", key="refresh_logs"):
                                st.rerun()
                        with log_col3:
                            if st.button("❌ Hide Logs", key="hide_logs"):
                                st.session_state.show_logs = False
                                st.rerun()
                        
                        # Select lines to display
                        if selected_lines >= len(log_lines):
                            display_lines = log_lines
                            st.caption(f"Showing all {len(log_lines)} log entries")
                        else:
                            display_lines = log_lines[-selected_lines:]
                            st.caption(f"Showing last {selected_lines} of {len(log_lines)} log entries")
                        
                        # Use text_area for better scrolling with scroll bar
                        log_text = '\n'.join(f"{i+1:4d}: {line}" for i, line in enumerate(display_lines))
                        st.text_area("📝 Ingestion Log Output:", 
                                   value=log_text,
                                   height=400,
                                   disabled=True,
                                   key="log_display_area",
                                   help="Use scroll bar or arrow keys to navigate through log entries")
                    else:
                        st.info("No log entries found.")
            except Exception as e:
                st.error(f"Error reading logs: {e}")
        else:
            st.warning("Ingestion log file not found.")
    
    # Show processing configuration options - only when NOT actively processing
    # Use batch state to determine if actively processing (not subprocess state)
    batch_is_processing = (
        batch_status.get('active', False) and
        not batch_status.get('paused', False) and
        batch_status.get('remaining', 0) > 0
    )

    if not batch_is_processing:
        st.markdown("---")
        st.subheader("⚙️ Processing Configuration")

        if batch_status.get('is_chunked', False):
            # Already chunked - show current settings and allow modification when paused
            current_chunk_size = batch_status.get('chunk_size', 250)
            current_auto_pause = batch_status.get('auto_pause_after_chunks')

            st.info(f"✅ **Chunked Processing Active**: {current_chunk_size} files per chunk" +
                    (f", auto-pause after {current_auto_pause} chunks" if current_auto_pause else ", no auto-pause"))

            with st.expander("🔧 **Modify Processing Settings**", expanded=False):
                st.info("Adjust chunk size and session length for your processing needs. Changes will apply after current pause.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    new_chunk_size = st.selectbox(
                        "New Chunk Size:",
                        options=[100, 250, 500, 1000],
                        index=[100, 250, 500, 1000].index(current_chunk_size) if current_chunk_size in [100, 250, 500, 1000] else 1,
                        key="modify_chunk_size",
                        help="Number of files to process in each chunk"
                    )

                with col2:
                    new_estimated_chunks = (batch_status.get('total_files', 0) + new_chunk_size - 1) // new_chunk_size
                    st.metric("New Total Chunks", new_estimated_chunks)

                with col3:
                    new_auto_pause = st.selectbox(
                        "Auto-pause after:",
                        options=[1, 2, 3, 5, 10, "No auto-pause"],
                        index=([1, 2, 3, 5, 10].index(current_auto_pause) if current_auto_pause and current_auto_pause in [1, 2, 3, 5, 10] else 5) if current_auto_pause else 2,
                        key="modify_auto_pause",
                        help="Automatically pause after processing this many chunks"
                    )

                # Show timing estimates
                if new_auto_pause != "No auto-pause":
                    files_per_session = new_chunk_size * new_auto_pause
                    sessions_needed = (batch_status.get('total_files', 0) + files_per_session - 1) // files_per_session
                    st.info(f"💡 **New Processing Plan**: {files_per_session} files per session, ~{sessions_needed} sessions needed")

                if st.button("🔄 Apply New Settings", type="secondary", use_container_width=True):
                    # Modify existing batch with new settings
                            batch_state = batch_manager.load_state()
                            if batch_state:
                                remaining_files = batch_state.get('files_remaining', [])
                                if remaining_files:
                                    # Update batch with new settings
                                    batch_manager.clear_batch()
                                    scan_config = batch_state.get('scan_config', {})
                                    auto_finalize_enabled = batch_state.get(
                                        'auto_finalize_enabled',
                                        scan_config.get('auto_finalize_enabled', False)
                                    )
                                    batch_manager.create_batch(
                                        remaining_files,
                                        scan_config,
                                        new_chunk_size,
                                        new_auto_pause if new_auto_pause != "No auto-pause" else None,
                                        auto_finalize_enabled=auto_finalize_enabled
                                    )

                            if new_auto_pause != "No auto-pause":
                                st.success(f"✅ Settings updated! New session mode: {new_auto_pause} chunks ({new_auto_pause * new_chunk_size} files) then auto-pause")
                            else:
                                st.success(f"✅ Settings updated! New chunk size: {new_chunk_size} files per chunk, continuous processing")

                            st.rerun()
        else:
            # Not chunked yet - show conversion options only if large batch and paused
            if batch_status.get('total_files', 0) > 500:
                st.warning(f"⚠️ **Large Batch**: {batch_status.get('total_files', 0)} files may cause memory issues")

                with st.expander("🔧 **Convert to Chunked Processing** (Recommended)", expanded=True):
                    st.info("Convert this large batch to chunked processing for better memory management and session control.")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        chunk_size_main = st.selectbox(
                            "Chunk Size:",
                            options=[100, 250, 500, 1000],
                            index=1,  # Default to 250
                            key="convert_chunk_size",
                            help="Number of files to process in each chunk"
                        )

                    with col2:
                        estimated_chunks = (batch_status.get('total_files', 0) + chunk_size_main - 1) // chunk_size_main
                        st.metric("Estimated Chunks", estimated_chunks)

                    with col3:
                        auto_pause_chunks = st.selectbox(
                            "Auto-pause after:",
                            options=[1, 2, 3, 5, 10, "No auto-pause"],
                            index=2,  # Default to 3 chunks
                            key="convert_auto_pause",
                            help="Automatically pause after processing this many chunks"
                        )

                    # Show timing estimates
                    if auto_pause_chunks != "No auto-pause":
                        files_per_session = chunk_size_main * auto_pause_chunks
                        sessions_needed = (batch_status.get('total_files', 0) + files_per_session - 1) // files_per_session
                        st.info(f"💡 **Processing Plan**: {files_per_session} files per session, ~{sessions_needed} sessions needed")

                    if st.button("🔄 Convert to Chunked Processing", type="secondary", use_container_width=True):
                        # Convert existing batch to chunked
                        batch_state = batch_manager.load_state()
                        if batch_state:
                            remaining_files = batch_state.get('files_remaining', [])
                            if remaining_files:
                                # Clear old batch and create new chunked one
                                batch_manager.clear_batch()
                                scan_config = batch_state.get('scan_config', {})
                                auto_finalize_enabled = batch_state.get(
                                    'auto_finalize_enabled',
                                    scan_config.get('auto_finalize_enabled', False)
                                )
                                batch_manager.create_batch(
                                    remaining_files,
                                    scan_config,
                                    chunk_size_main,
                                    auto_pause_chunks if auto_pause_chunks != "No auto-pause" else None,
                                    auto_finalize_enabled=auto_finalize_enabled
                                )

                                if auto_pause_chunks != "No auto-pause":
                                    st.success(f"✅ Session mode enabled! Will process {auto_pause_chunks} chunks then auto-pause")
                                else:
                                    st.success(f"✅ Chunked processing enabled! {estimated_chunks} chunks of {chunk_size_main} files each")

                                st.rerun()
            else:
                st.info("💡 **Small Batch**: No chunking needed for this batch size.")

    if process_running and not batch_status.get('paused', False):
        time.sleep(1)
        st.rerun()

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
        # Show clean Qwen3-VL name with size - derive dims from config, not stale cache
        from cortex_engine.config import QWEN3_VL_MODEL_SIZE
        size_to_dims = {"2B": 2048, "8B": 4096}
        if QWEN3_VL_MODEL_SIZE in size_to_dims:
            embed_dims = size_to_dims[QWEN3_VL_MODEL_SIZE]
            size_display = QWEN3_VL_MODEL_SIZE.upper()
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


def render_config_and_scan_ui():
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
                        container_db_path = convert_to_docker_mount_path(st.session_state.db_path)
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
                        container_db_path = convert_to_docker_mount_path(st.session_state.db_path)
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
        def sort_by_date(): st.session_state.files_to_review.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
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
        mod_time = datetime.fromtimestamp(Path(fp).stat().st_mtime)
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
    st.header(stage_title)
    
    # If the UI is in processing mode but no subprocess exists (e.g., app reload),
    # attempt an automatic resume from stored batch configuration.
    try:
        if st.session_state.get('ingestion_stage') == 'analysis_running' and not st.session_state.get('ingestion_process'):
            container_db_path = convert_to_docker_mount_path(st.session_state.get('db_path', ''))
            if container_db_path:
                batch_manager = BatchState(container_db_path)
                set_runtime_db_path(str(batch_manager.db_path))
                batch_state = batch_manager.load_state()
                if batch_state and batch_state.get('files_remaining'):
                    if auto_resume_from_batch_config(batch_manager):
                        st.info("🔄 Auto-resume started after reload; continuing processing…")
                        st.rerun()
                        return
    except Exception as _e:
        logger.warning(f"Auto-resume on reload skipped: {_e}")
    
    # Add control buttons for pause/stop (always visible during processing)
    st.markdown("### 🎛️ Ingestion Controls")
    st.info("💡 **Tip:** You can stop the ingestion at any time using the Stop button below.")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("⏸️ Pause", key="pause_processing", use_container_width=True):
            # Get batch manager and pause the batch
            container_db_path = get_runtime_db_path()
            batch_manager = BatchState(container_db_path)
            set_runtime_db_path(str(batch_manager.db_path))
            batch_manager.pause_batch()
            st.success("Pause requested")

    with col2:
        # Make Stop button more prominent and always functional
        if st.button("⏹️ **STOP INGESTION**", key="stop_processing", use_container_width=True, type="secondary"):
            # Always allow stopping, even if process doesn't exist
            if st.session_state.get('ingestion_process'):
                try:
                    st.session_state.ingestion_process.terminate()
                except Exception as e:
                    logger.warning(f"Process termination failed: {e}")
                st.session_state.ingestion_process = None

            # Signal reader thread to stop
            stop_ev = st.session_state.get("ingestion_reader_stop")
            if stop_ev:
                stop_ev.set()

            # Force reset the ingestion stage to allow fresh start
            st.session_state.ingestion_stage = "config"
            st.warning("⚠️ Ingestion stopped by user. Returning to configuration...")
            st.rerun()
    
    with col3:
        if st.button("⬅️ Back", key="back_from_processing", use_container_width=True):
            initialize_state(force_reset=True)
            st.rerun()
    
    # Initialize current document tracking if not exists
    if "current_doc_number" not in st.session_state:
        st.session_state.current_doc_number = 0
    if "total_docs_in_batch" not in st.session_state:
        st.session_state.total_docs_in_batch = 0
    # Embedding progress tracking
    if "embedding_current" not in st.session_state:
        st.session_state.embedding_current = 0
    if "embedding_total" not in st.session_state:
        st.session_state.embedding_total = 0
    
    # Progress counter display (prominent)
    if st.session_state.total_docs_in_batch > 0:
        # Distinguish between analysis and finalization phases
        if st.session_state.get('finalize_started_detected', False):
            progress_text = f"🔄 Indexing batch {st.session_state.current_doc_number} of {st.session_state.total_docs_in_batch}"
            status_header = "**Finalization in Progress** (Embedding & Indexing)"
        else:
            progress_text = f"📄 Analyzing document {st.session_state.current_doc_number} of {st.session_state.total_docs_in_batch}"
            status_header = "**Analysis in Progress** (Scanning & Metadata Extraction)"

        st.markdown(f"### {status_header}")
        st.markdown(f"{progress_text}")
        progress_percent = st.session_state.current_doc_number / st.session_state.total_docs_in_batch
        progress_bar = st.progress(progress_percent, text=progress_text)

        # Show embedding progress during finalization
        if st.session_state.get('finalize_started_detected', False) and st.session_state.embedding_total > 0:
            embed_current = st.session_state.embedding_current
            embed_total = st.session_state.embedding_total
            embed_pct = (embed_current / embed_total * 100) if embed_total > 0 else 0
            embed_text = f"🔢 Embedding: {embed_current}/{embed_total} vectors ({embed_pct:.0f}%)"
            st.progress(embed_current / embed_total if embed_total > 0 else 0, text=embed_text)
    else:
        progress_bar = st.progress(0, text="Starting process...")

    # Throttle status indicator (always visible; values update as lines arrive)
    throttle_col1, throttle_col2, throttle_col3 = st.columns([1, 1, 1])
    with throttle_col1:
        delay_val = st.session_state.get('current_throttle_delay', 0.0)
        st.metric("⏱️ Throttle Delay", f"{delay_val:.1f}s", help="Current delay between documents to reduce system load")
    with throttle_col2:
        gpu_val = st.session_state.get('current_gpu_util', None)
        if gpu_val is not None:
            gpu_delta = f"+{gpu_val - 80:.0f}%" if gpu_val > 80 else None
            st.metric("🎮 GPU Load", f"{gpu_val:.0f}%", delta=gpu_delta, delta_color="inverse", help="GPU utilization (throttle increases when high)")
        else:
            st.metric("🎮 GPU Load", "N/A", help="GPU monitoring unavailable or not yet sampled")
    with throttle_col3:
        cpu_val = st.session_state.get('current_cpu_util', None)
        if cpu_val is not None:
            cpu_delta = f"+{cpu_val - 85:.0f}%" if cpu_val > 85 else None
            st.metric("💻 CPU Load", f"{cpu_val:.0f}%", delta=cpu_delta, delta_color="inverse", help="CPU utilization (throttle increases when high)")
        else:
            st.metric("💻 CPU Load", "N/A", help="CPU monitoring unavailable or not yet sampled")

    # Visual indicator when throttling is active
    if (gpu_val and gpu_val > 80) or (cpu_val and cpu_val > 85):
        st.warning("🎛️ **Adaptive throttling active** - System load detected, automatically slowing down to prevent freezing.")

    # Heartbeat freshness indicator
    hb_ts = st.session_state.get('last_heartbeat_ts')
    if hb_ts:
        try:
            age = max(0, int(time.time() - hb_ts))
            st.caption(f"Last update {age}s ago")
        except Exception:
            pass

    # Check if process is still running BEFORE the expander (so rerun logic works)
    process_still_running = False
    if st.session_state.ingestion_process:
        poll_result = st.session_state.ingestion_process.poll()
        process_still_running = (poll_result is None)

        # If reader lost (e.g., browser refresh/new session), reattach to running process
        if process_still_running and not st.session_state.get("ingestion_output_queue"):
            try:
                start_ingest_reader(st.session_state.ingestion_process)
                st.info("🔌 Reattached to running process; resuming live logs…")
            except Exception as _reattach_err:
                logger.warning(f"Failed to reattach reader: {_reattach_err}")

        # Drain available lines from background reader
        for line in get_ingest_lines(max_lines=50):
            # Parse progress updates
            if line.startswith("CORTEX_PROGRESS::"):
                try:
                    _, progress_part, filename_part = line.split("::", 2)
                    current, total = map(int, progress_part.split('/'))
                    st.session_state.current_doc_number = current
                    st.session_state.total_docs_in_batch = total
                    st.session_state.log_messages.append(f"Processing {current}/{total}: {filename_part}")
                except (ValueError, IndexError):
                    st.session_state.log_messages.append(line)
            elif line.startswith("CORTEX_EMBEDDING::"):
                # Parse embedding progress: CORTEX_EMBEDDING::current/total::batch_info
                try:
                    _, progress_part, batch_info = line.split("::", 2)
                    current, total = map(int, progress_part.split('/'))
                    st.session_state.embedding_current = current
                    st.session_state.embedding_total = total
                    if current > 0:
                        pct = (current / total * 100) if total > 0 else 0
                        st.session_state.log_messages.append(f"🔢 Embedding: {current}/{total} ({pct:.0f}%)")
                except (ValueError, IndexError):
                    st.session_state.log_messages.append(line)
            elif line.startswith("CORTEX_THROTTLE::"):
                try:
                    _, delay_str, gpu_str, cpu_str = line.split("::", 3)
                    st.session_state.current_throttle_delay = float(delay_str)
                    gpu_val = None if gpu_str == "N/A" else float(gpu_str)
                    cpu_val = None if cpu_str == "N/A" else float(cpu_str)
                    st.session_state.current_gpu_util = gpu_val
                    st.session_state.current_cpu_util = cpu_val
                    st.session_state.throttle_active = bool((gpu_val is not None and gpu_val > 80) or (cpu_val is not None and cpu_val > 85))
                    st.session_state.last_heartbeat_ts = time.time()
                    # Also surface a compact log line so users see live updates
                    gpu_disp = gpu_str if gpu_str != "N/A" else "N/A"
                    cpu_disp = cpu_str if cpu_str != "N/A" else "N/A"
                    st.session_state.log_messages.append(f"THROTTLE delay={float(delay_str):.1f}s GPU={gpu_disp}% CPU={cpu_disp}%")
                except (ValueError, IndexError):
                    pass
            elif line.startswith("CORTEX_HEARTBEAT::"):
                try:
                    _, gpu_str, cpu_str = line.split("::", 2)
                    gpu_val = None if gpu_str == "N/A" else float(gpu_str)
                    cpu_val = None if cpu_str == "N/A" else float(cpu_str)
                    st.session_state.current_gpu_util = gpu_val
                    st.session_state.current_cpu_util = cpu_val
                    st.session_state.last_heartbeat_ts = time.time()
                    # Log heartbeat visibly as a compact line
                    gpu_disp = gpu_str if gpu_str != "N/A" else "N/A"
                    cpu_disp = cpu_str if cpu_str != "N/A" else "N/A"
                    ts = time.strftime('%H:%M:%S')
                    st.session_state.log_messages.append(f"[{ts}] HEARTBEAT GPU={gpu_disp}% CPU={cpu_disp}%")
                except (ValueError, IndexError):
                    pass
            elif line.startswith("CORTEX_STAGE::FINALIZE_START"):
                st.session_state.log_messages.append("🔄 Starting finalization (embedding documents)...")
                st.session_state.finalize_started_detected = True
                # Don't clear analysis_done_detected - analysis DID complete, we just moved to next stage
            elif line.startswith("CORTEX_STAGE::FINALIZE_DONE"):
                st.session_state.log_messages.append("✅ Finalization completed successfully!")
                st.session_state.finalize_done_detected = True
                st.session_state.finalize_started_detected = False
                st.session_state.batch_auto_finalize_started = False
                st.session_state.batch_auto_processed = False
                st.session_state.batch_mode_active = False
                st.session_state.auto_finalize_triggered = False
            elif line.startswith("CORTEX_STAGE::ANALYSIS_DONE"):
                st.session_state.log_messages.append("✅ Analysis completed successfully!")
                st.session_state.analysis_done_detected = True
            else:
                st.session_state.log_messages.append(line)

        # Trim log to avoid unbounded growth
        if len(st.session_state.log_messages) > 1000:
            st.session_state.log_messages = st.session_state.log_messages[-1000:]

        # Clean up if process finished
        if not process_still_running:
            st.session_state.ingestion_process = None
            # Clear GPU/CPU metrics so they don't show stale values
            st.session_state.current_gpu_util = None
            st.session_state.current_cpu_util = None
            st.session_state.throttle_active = False
            # Signal reader thread to stop
            stop_ev = st.session_state.get("ingestion_reader_stop")
            if stop_ev:
                stop_ev.set()

    # Expandable log section (just for display now)
    # Keep expanded while process is running so user can monitor progress
    log_expanded = process_still_running or st.session_state.get('ingestion_process') is not None
    with st.expander("📋 Processing Log (click to expand/collapse)", expanded=log_expanded):
        log_container = st.container(height=400, border=True)
        with log_container:
            if st.session_state.log_messages:
                st.code("\n".join(st.session_state.log_messages[-100:]), language="log")
            else:
                st.info("No log messages yet...")

    # Handle completion or continue processing OUTSIDE the expander
    if not process_still_running and st.session_state.ingestion_process is None:
        # Process completed
        if st.session_state.get('finalize_done_detected', False):
            st.session_state.finalize_done_detected = False
            st.session_state.ingestion_stage = "config_done"
            progress_bar.progress(1.0, text="✅ Finalization Complete!")
            st.success("✅ Finalization complete! Your knowledge base has been updated.")
            st.rerun()
            return
        elif st.session_state.get('analysis_done_detected', False):
            # Only show "Analysis Complete!" if finalization hasn't started yet
            if not st.session_state.get('finalize_started_detected', False):
                progress_bar.progress(1.0, text="✅ Analysis Complete!")
                st.success("✅ Analysis complete!")
            else:
                # Finalization is running, don't show analysis complete
                pass
        else:
            progress_bar.progress(1.0, text="Process completed")
            st.info("Process completed.")
    elif process_still_running:
        # Process still running - schedule automatic refresh
        time.sleep(1)  # Wait 1 second before refreshing
        st.rerun()

    # Handle metadata review after process completes (only if not doing auto-finalize)
    if on_complete_stage == "metadata_review" and not process_still_running:
        # CRITICAL: This section handles auto-finalization after analysis completes
        st.markdown("---")
        st.markdown("### 📊 Analysis Complete")

        load_staged_files()

        # Check if we should automatically proceed to finalization
        # But only if analysis just completed (not if finalization already happened)
        should_finalize = should_auto_finalize()
        finalize_done = st.session_state.get('finalize_done_detected', False)
        retry_attempts = st.session_state.get('auto_finalize_retry_attempts', 0)

        logger.info(f"🔍 Auto-finalize check: on_complete_stage={on_complete_stage}, process_still_running={process_still_running}, should_finalize={should_finalize}, finalize_done={finalize_done}")

        # PROMINENT DEBUG: Show visible status at the top
        debug_col1, debug_col2, debug_col3 = st.columns(3)
        with debug_col1:
            st.metric("Should Finalize", "✅ Yes" if should_finalize else "❌ No")
        with debug_col2:
            st.metric("Already Done", "✅ Yes" if finalize_done else "❌ No")
        with debug_col3:
            st.metric("Stage", on_complete_stage)

        if should_finalize and not finalize_done:
            st.session_state.auto_finalize_retry_attempts = 0
            # Show info message to confirm auto-finalize is triggering
            st.success("✅ Analysis complete! Starting automatic finalization in 2 seconds...")
            logger.info("Analysis completed successfully - starting automatic finalization")
            time.sleep(2)  # Brief delay so user sees the message
            start_automatic_finalization()
            st.rerun()  # Immediate rerun to show finalization stage
            return  # Exit early to avoid setting stage to metadata_review
        else:
            if not should_finalize and not finalize_done and st.session_state.get('analysis_done_detected'):
                # Automatically poll for staging availability before surfacing warning
                if retry_attempts < MAX_AUTO_FINALIZE_RETRIES:
                    st.session_state.auto_finalize_retry_attempts = retry_attempts + 1
                    wait_time = min(2.0, 0.5 * st.session_state.auto_finalize_retry_attempts)
                    st.info(f"⏳ Waiting for staged documents to finish writing ({st.session_state.auto_finalize_retry_attempts}/{MAX_AUTO_FINALIZE_RETRIES})…")
                    time.sleep(wait_time)
                    st.rerun()
                    return
                else:
                    logger.warning("Auto-finalize gave up waiting for staging file after retries")
                    st.warning("⚠️ **Auto-finalize skipped:** Staged documents were not detected after multiple checks.")

            if finalize_done:
                logger.info("Auto-finalize skipped: finalization already completed")
                st.info("ℹ️ Finalization already completed")

        st.session_state.ingestion_stage = on_complete_stage
        st.session_state.auto_finalize_retry_attempts = 0
        time.sleep(1)
        st.rerun()

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
    
    # Check if documents should be automatically assigned to a target collection
    target_collection = st.session_state.get('target_collection_name', '')
    doc_ids = st.session_state.get('last_ingested_doc_ids', [])
    
    if doc_ids and target_collection:
        try:
            collection_mgr = WorkingCollectionManager()
            
            # Check if target collection needs to be created or already exists
            existing_collections = collection_mgr.get_collection_names()
            
            if target_collection not in existing_collections:
                # Create new collection
                if collection_mgr.create_collection(target_collection):
                    st.success(f"✅ Created new collection: **{target_collection}**")
                else:
                    st.error(f"❌ Failed to create collection: {target_collection}")
                    target_collection = ""  # Fall back to manual assignment
            
            if target_collection:
                # Assign documents to the target collection
                collection_mgr.add_docs_by_id_to_collection(target_collection, doc_ids)
                st.success(f"🎯 **{len(doc_ids)} documents automatically assigned** to collection: **{target_collection}**")
                st.session_state.last_ingested_doc_ids = []  # Clear after successful assignment
                st.session_state.target_collection_name = ""  # Clear target collection
                
        except Exception as e:
            st.error(f"❌ Failed to assign documents to collection '{target_collection}': {e}")
            logger.error(f"Collection assignment failed: {e}")
            # Fall through to manual assignment option
    
    # Manual collection creation option (if no automatic assignment or if it failed)
    if st.session_state.get('last_ingested_doc_ids'):
        with st.form("new_collection_from_ingest"):
            st.markdown("---")
            st.subheader("📂 Manual Collection Assignment")
            
            if target_collection:
                st.info(f"Collection assignment to '{target_collection}' failed. You can manually create a different collection below.")
            else:
                st.info(f"You've just added {len(st.session_state.last_ingested_doc_ids)} new documents. Save them as a new collection for easy access later.")
                
            collection_name = st.text_input("New Collection Name", placeholder="e.g., Project Phoenix Discovery")
            if st.form_submit_button("Create Collection", type="primary"):
                if collection_name:
                    collection_mgr = WorkingCollectionManager()
                    if collection_name in collection_mgr.get_collection_names():
                        st.error(f"Collection '{collection_name}' already exists.")
                    else:
                        collection_mgr.add_docs_by_id_to_collection(collection_name, st.session_state.last_ingested_doc_ids)
                        st.success(f"Successfully created collection '{collection_name}'!")
                        st.session_state.last_ingested_doc_ids = []
                else:
                    st.warning("Please provide a name for the collection.")
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
    st.header("Review AI-Generated Metadata")

    # If staging exists, offer a manual retry trigger for finalization
    try:
        if should_auto_finalize() and st.session_state.get('ingestion_stage') != 'finalizing':
            if st.button("🔁 Retry Finalization", key="retry_finalization", help="Run finalization again from staged results"):
                start_automatic_finalization()
                st.rerun()
    except Exception:
        pass

    if 'edited_staged_files' not in st.session_state or not st.session_state.edited_staged_files:
        initial_files = st.session_state.get('staged_files', [])
        failed_docs = {}
        
        for doc in initial_files:
            is_error = doc.get('rich_metadata', {}).get('summary', '').startswith("ERROR:")
            doc['exclude_from_final'] = is_error
            
            # In batch mode, log error documents separately
            if is_error and st.session_state.get("batch_ingest_mode", False):
                file_path = doc.get('doc_posix_path', 'Unknown')
                error_msg = doc.get('rich_metadata', {}).get('summary', 'Unknown error')
                failed_docs[file_path] = error_msg
        
        # Log failed documents if in batch mode
        if failed_docs and st.session_state.get("batch_ingest_mode", False):
            failure_log_path = log_failed_documents(failed_docs, st.session_state.db_path)
            if failure_log_path:
                st.warning(f"⚠️ {len(failed_docs)} documents failed processing and were logged to: `{failure_log_path}`")
        
        st.session_state.edited_staged_files = initial_files
        st.session_state.review_page = 0

    edited_files = st.session_state.edited_staged_files
    if not edited_files:
        st.success("Analysis complete, but no documents were staged for review.")
        # EXTRA DIAGNOSTICS (host): show staging path and parsed count
        try:
            from cortex_engine.ingest_cortex import get_staging_file_path
            wsl_db_path = convert_windows_to_wsl_path(st.session_state.get('db_path_input', st.session_state.get('db_path', '')))
            staging_path = Path(get_staging_file_path(wsl_db_path)) if wsl_db_path else None
            if staging_path:
                st.caption(f"Staging file: `{staging_path}` (exists={staging_path.exists()})")
                if staging_path.exists():
                    try:
                        with open(staging_path, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            staged_count = len(data.get('documents', []))
                        else:
                            staged_count = len(data)
                        st.caption(f"Parsed staged documents: {staged_count}")
                        if staged_count > 0:
                            st.info("Staging file contains documents. You can retry automatic finalization.")
                            if st.button("🚀 Retry Finalization", type="primary", key="retry_finalize_host"):
                                start_automatic_finalization()
                                st.stop()
                        with st.expander("Show staging JSON (first 2KB)", expanded=False):
                            preview = json.dumps(data, indent=2)
                            st.text_area("staging_ingestion.json", value=preview[:2048], height=200)
                    except Exception as pe:
                        st.warning(f"Could not read staging file: {pe}")
        except Exception:
            pass
        st.info("Check `logs/ingestion.log` for details.")
        if st.button("⬅️ Back to Configuration", key="back_config_no_staged_docs"): initialize_state(force_reset=True); st.rerun()
        return

    # In batch mode, automatically proceed with valid documents
    batch_mode = st.session_state.get("batch_ingest_mode", False) or st.session_state.get("batch_mode_active", False)
    
    # Debug info for troubleshooting
    if batch_mode:
        st.info(f"🔧 **Debug:** Batch mode detected - batch_ingest_mode: {st.session_state.get('batch_ingest_mode', False)}, batch_mode_active: {st.session_state.get('batch_mode_active', False)}")
    
    if batch_mode:
        # Check if we've already processed this batch (prevent re-processing on UI refresh)
        if not st.session_state.get("batch_auto_processed", False):
            valid_files = [doc for doc in edited_files if not doc.get('exclude_from_final', False)]
            excluded_count = len(edited_files) - len(valid_files)
            
            st.info(f"🚀 **Batch Mode:** Automatically processing {len(valid_files)} valid documents. {excluded_count} documents excluded due to errors.")
            
            if valid_files:
                # Auto-proceed to finalization
                st.session_state.last_ingested_doc_ids = [doc['doc_id'] for doc in valid_files if not doc.get('exclude_from_final')]
                # Write staging file to database directory, not project root
                container_db_path = get_runtime_db_path()
                staging_file_path = Path(container_db_path) / "staging_ingestion.json"
                payload = serialize_staging_payload(st.session_state.edited_staged_files, st.session_state.get('target_collection_name'))
                with open(staging_file_path, 'w') as f:
                    json.dump(payload, f, indent=2)
                st.session_state.staged_metadata = payload
                if not container_db_path or not Path(container_db_path).exists():
                    st.error(f"Database path is invalid or does not exist: {container_db_path}")
                    st.stop()

                st.session_state.log_messages = ["Finalizing batch ingestion..."]
                st.session_state.ingestion_stage = "finalizing"
                st.session_state.batch_auto_processed = True  # Mark as processed
                
                # Use direct script path to avoid module resolution confusion
                script_path = project_root / "cortex_engine" / "ingest_cortex.py"
                command = [sys.executable, str(script_path), "--finalize-from-staging", "--db-path", container_db_path]
                
                # Add skip image processing flag if enabled
                if st.session_state.get("skip_image_processing", False):
                    command.append("--skip-image-processing")
                
                st.session_state.ingestion_process = spawn_ingest(command)
                start_ingest_reader(st.session_state.ingestion_process)
                st.rerun()
            else:
                st.warning("No valid documents to process after filtering errors.")
                if st.button("⬅️ Back to Configuration", key="back_config_batch_mode"): 
                    initialize_state(force_reset=True)
                    st.rerun()
            return
        else:
            # Already processed, show progress or completion message
            st.info("🚀 **Batch Mode:** Processing has been initiated automatically.")
            return
    else:
        # Non-batch mode - show manual controls
        st.info(f"📋 **Manual Mode:** Please review the metadata below and proceed when ready.")

    st.info(f"Please review and approve the metadata for the **{len(edited_files)}** document(s) below.")
    
    # Add quick proceed option for users who want to skip review
    valid_docs_count = len([doc for doc in edited_files if not doc.get('exclude_from_final', False)])
    if valid_docs_count > 0:
        st.info(f"💡 **Quick Option:** Skip detailed review and proceed with {valid_docs_count} valid documents")
        if st.button("⚡ Skip Review & Proceed to Finalization", type="secondary", key="quick_proceed"):
            # Auto-proceed to finalization
            final_files_to_process = st.session_state.edited_staged_files
            doc_ids_to_ingest = [doc['doc_id'] for doc in final_files_to_process if not doc.get('exclude_from_final')]
            st.session_state.last_ingested_doc_ids = doc_ids_to_ingest
            # Write staging file to database directory, not project root
            container_db_path = get_runtime_db_path()
            staging_file_path = Path(container_db_path) / "staging_ingestion.json"
            payload = serialize_staging_payload(final_files_to_process, st.session_state.get('target_collection_name'))
            with open(staging_file_path, 'w') as f:
                json.dump(payload, f, indent=2)
            st.session_state.staged_metadata = payload

            container_db_path = get_runtime_db_path()
            if not container_db_path or not Path(container_db_path).exists():
                st.error(f"Database path is invalid or does not exist: {container_db_path}")
                st.stop()

            st.session_state.log_messages = ["Finalizing ingestion..."]
            st.session_state.ingestion_stage = "finalizing"
            command = [sys.executable, "-m", "cortex_engine.ingest_cortex", "--finalize-from-staging", "--db-path", container_db_path]
            
            if st.session_state.get("skip_image_processing", False):
                command.append("--skip-image-processing")
                
            st.session_state.ingestion_process = spawn_ingest(command)
            start_ingest_reader(st.session_state.ingestion_process)
            st.rerun()
    
    st.markdown("---")

    page = st.session_state.review_page
    start_idx, end_idx = page * REVIEW_PAGE_SIZE, (page + 1) * REVIEW_PAGE_SIZE
    paginated_files = edited_files[start_idx:end_idx]
    total_pages = -(-len(edited_files) // REVIEW_PAGE_SIZE) or 1

    def update_edited_state(index, field, value):
        if field == 'include': st.session_state.edited_staged_files[index]['exclude_from_final'] = not value
        elif field == 'thematic_tags': st.session_state.edited_staged_files[index]['rich_metadata'][field] = [tag.strip() for tag in value.split(',') if tag.strip()]
        else: st.session_state.edited_staged_files[index]['rich_metadata'][field] = value

    for i, doc in enumerate(paginated_files):
        absolute_index = start_idx + i
        rich_meta = doc.get('rich_metadata', {})
        is_included = not doc.get('exclude_from_final', False)
        checkbox_label = f"**{doc.get('file_name', 'N/A')}** - {rich_meta.get('summary', 'No summary available.')}"

        new_include_val = st.checkbox(checkbox_label, value=is_included, key=f"include_{absolute_index}")
        if new_include_val != is_included:
            update_edited_state(absolute_index, 'include', new_include_val); st.rerun()

        with st.expander("Edit Metadata & Preview"):
            # Auto-suggest document type based on filename
            filename = doc.get('file_name', '')
            doc_type_manager = get_document_type_manager()
            suggested_type = doc_type_manager.suggest_document_type(filename)
            
            # Show suggestion if it's different from current type
            current_doc_type = rich_meta.get('document_type', 'Other')
            if suggested_type != current_doc_type and suggested_type != 'Other':
                st.info(f"💡 **Auto-suggestion:** Based on the filename '{filename}', this document might be: **{suggested_type}**")
                if st.button(f"✅ Use '{suggested_type}'", key=f"suggest_{absolute_index}"):
                    update_edited_state(absolute_index, 'document_type', suggested_type)
                    st.rerun()
            
            try: doc_type_index = DOC_TYPE_OPTIONS.index(rich_meta.get('document_type'))
            except (ValueError, TypeError): doc_type_index = len(DOC_TYPE_OPTIONS) - 1
            try: outcome_index = PROPOSAL_OUTCOME_OPTIONS.index(rich_meta.get('proposal_outcome'))
            except (ValueError, TypeError): outcome_index = len(PROPOSAL_OUTCOME_OPTIONS) - 1

            # Document type selector with category context
            selected_doc_type = st.selectbox("Document Type", options=DOC_TYPE_OPTIONS, index=doc_type_index, key=f"dt_{absolute_index}", on_change=lambda idx=absolute_index: update_edited_state(idx, 'document_type', st.session_state[f"dt_{idx}"]))
            
            # Show which category this type belongs to
            if selected_doc_type and selected_doc_type != "Any":
                category = doc_type_manager.get_category_for_type(selected_doc_type)
                st.caption(f"📂 Category: {category}")
            st.selectbox("Proposal Outcome", options=PROPOSAL_OUTCOME_OPTIONS, index=outcome_index, key=f"oc_{absolute_index}", on_change=lambda idx=absolute_index: update_edited_state(idx, 'proposal_outcome', st.session_state[f"oc_{idx}"]))
            st.text_area("Summary", value=rich_meta.get('summary', ''), key=f"sm_{absolute_index}", height=100, on_change=lambda idx=absolute_index: update_edited_state(idx, 'summary', st.session_state[f"sm_{idx}"]))
            st.text_input("Thematic Tags (comma-separated)", value=", ".join(rich_meta.get('thematic_tags', [])), key=f"tg_{absolute_index}", on_change=lambda idx=absolute_index: update_edited_state(idx, 'thematic_tags', st.session_state[f"tg_{idx}"]))
            st.divider()
            st.text_area("File Content Preview", get_full_file_content(doc['doc_posix_path']), height=200, disabled=True, key=f"preview_{doc['doc_posix_path']}")

    st.divider()
    nav_cols = st.columns([1, 1, 5])
    if page > 0: nav_cols[0].button("⬅️ Previous", on_click=lambda: st.session_state.update(review_page=page - 1), use_container_width=True)
    if end_idx < len(edited_files): nav_cols[1].button("Next ➡️", on_click=lambda: st.session_state.update(review_page=page + 1), use_container_width=True)
    nav_cols[2].write(f"Page {page + 1} of {total_pages}")

    st.divider()
    
    # Show summary of valid vs error documents
    valid_docs = [doc for doc in edited_files if not doc.get('exclude_from_final', False)]
    error_docs = [doc for doc in edited_files if doc.get('exclude_from_final', False)]
    
    if error_docs:
        st.warning(f"⚠️ {len(error_docs)} documents had errors and will be excluded. {len(valid_docs)} documents are ready for ingestion.")
    else:
        st.success(f"✅ All {len(valid_docs)} documents are ready for ingestion.")
    
    action_cols = st.columns(2)
    if action_cols[0].button("⬅️ Cancel and Go Back", use_container_width=True): initialize_state(force_reset=True); st.rerun()
    
    # Always show finalize button if there are valid documents
    finalize_enabled = len(valid_docs) > 0
    button_text = f"✅ Finalize {len(valid_docs)} Approved Documents" if finalize_enabled else "❌ No Valid Documents to Finalize"
    
    if action_cols[1].button(button_text, use_container_width=True, type="primary", disabled=not finalize_enabled):
        final_files_to_process = st.session_state.edited_staged_files
        doc_ids_to_ingest = [doc['doc_id'] for doc in final_files_to_process if not doc.get('exclude_from_final')]
        st.session_state.last_ingested_doc_ids = doc_ids_to_ingest
        # Write staging file to database directory, not project root
        container_db_path = get_runtime_db_path()
        staging_file_path = Path(container_db_path) / "staging_ingestion.json"
        payload = serialize_staging_payload(final_files_to_process, st.session_state.get('target_collection_name'))
        with open(staging_file_path, 'w') as f:
            json.dump(payload, f, indent=2)
        st.session_state.staged_metadata = payload
        if not container_db_path or not Path(container_db_path).exists():
             st.error(f"Database path is invalid or does not exist: {container_db_path}"); st.stop()

        st.session_state.log_messages = ["Finalizing ingestion..."]
        st.session_state.ingestion_stage = "finalizing"
        command = [sys.executable, "-m", "cortex_engine.ingest_cortex", "--finalize-from-staging", "--db-path", container_db_path]
        
        # Add skip image processing flag if enabled
        if st.session_state.get("skip_image_processing", False):
            command.append("--skip-image-processing")
        st.session_state.ingestion_process = spawn_ingest(command)
        start_ingest_reader(st.session_state.ingestion_process)
        st.rerun()

def render_document_type_management():
    """Render the document type management interface."""
    st.header("📋 Document Type Management")
    st.markdown("Configure document categories, types, and keyword mappings for intelligent document classification.")
    
    doc_type_manager = get_document_type_manager()
    
    # Create tabs for different management functions
    tab1, tab2, tab3, tab4 = st.tabs(["📂 Categories", "🏷️ Type Mappings", "📊 Overview", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Document Categories")
        st.markdown("Organize document types into logical categories for better organization.")
        
        # Display existing categories
        categories = doc_type_manager.get_categories()
        
        for category_name, category_data in categories.items():
            with st.expander(f"📁 {category_name} ({len(category_data['types'])} types)", expanded=False):
                st.markdown(f"**Description:** {category_data['description']}")
                st.markdown(f"**Document Types:** {', '.join(category_data['types'])}")
                
                # Category management
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    new_type = st.text_input(f"Add new type to {category_name}:", key=f"new_type_{category_name}")
                    if st.button(f"➕ Add Type", key=f"add_type_{category_name}"):
                        if new_type and doc_type_manager.add_type_to_category(category_name, new_type):
                            st.success(f"Added '{new_type}' to {category_name}")
                            st.session_state.show_maintenance = True
                            st.rerun()
                        elif new_type:
                            st.warning(f"'{new_type}' already exists in {category_name}")
                
                with col2:
                    if category_data['types']:
                        type_to_remove = st.selectbox(f"Remove type from {category_name}:", 
                                                    [""] + category_data['types'], 
                                                    key=f"remove_type_{category_name}")
                        if type_to_remove and st.button(f"🗑️ Remove", key=f"remove_btn_{category_name}"):
                            if doc_type_manager.remove_type_from_category(category_name, type_to_remove):
                                st.success(f"Removed '{type_to_remove}' from {category_name}")
                                st.session_state.show_maintenance = True
                                st.rerun()
                
                with col3:
                    if category_name != "Other":  # Don't allow deleting "Other" category
                        if st.button(f"🗑️ Delete Category", key=f"delete_cat_{category_name}", type="secondary"):
                            if doc_type_manager.remove_category(category_name):
                                st.success(f"Deleted category '{category_name}'")
                                st.session_state.show_maintenance = True
                                st.rerun()
        
        # Add new category
        st.markdown("---")
        st.subheader("Add New Category")
        col1, col2 = st.columns(2)
        with col1:
            new_category_name = st.text_input("Category Name:", key="new_category_name")
        with col2:
            new_category_desc = st.text_input("Description:", key="new_category_desc")
        
        if st.button("➕ Create Category") and new_category_name and new_category_desc:
            if doc_type_manager.add_category(new_category_name, new_category_desc):
                st.success(f"Created category '{new_category_name}'")
                st.session_state.show_maintenance = True
                st.rerun()
            else:
                st.error(f"Category '{new_category_name}' already exists")
    
    with tab2:
        st.subheader("Keyword Mappings")
        st.markdown("Define keywords that automatically map to specific document types during ingestion.")
        
        # Display existing mappings
        mappings = doc_type_manager.get_type_mappings()
        all_types = doc_type_manager.get_all_document_types()
        
        if mappings:
            # Sorting controls
            st.markdown("**Current Mappings:**")
            sort_col1, sort_col2, sort_col3 = st.columns([2, 2, 2])
            
            with sort_col1:
                if st.button("🔤 Sort by Keyword", use_container_width=True, key="sort_by_keyword"):
                    st.session_state.mapping_sort_key = 'keyword'
                    st.session_state.mapping_sort_reverse = not st.session_state.get('mapping_sort_reverse', False)
                    # Keep maintenance mode active after rerun
                    st.session_state.show_maintenance = True
                    st.rerun()
            
            with sort_col2:
                if st.button("📋 Sort by Document Type", use_container_width=True, key="sort_by_doctype"):
                    st.session_state.mapping_sort_key = 'doctype'
                    st.session_state.mapping_sort_reverse = not st.session_state.get('mapping_sort_reverse', False)
                    # Keep maintenance mode active after rerun
                    st.session_state.show_maintenance = True
                    st.rerun()
            
            with sort_col3:
                # Filter by document type
                filter_type = st.selectbox("Filter by Type:", ["All"] + all_types, key="mapping_filter_type")
            
            # Apply sorting and filtering
            sorted_mappings = list(mappings.items())
            
            # Filter by document type if selected
            if filter_type != "All":
                sorted_mappings = [(k, v) for k, v in sorted_mappings if v == filter_type]
            
            # Apply sorting
            sort_key = st.session_state.get('mapping_sort_key', 'keyword')
            sort_reverse = st.session_state.get('mapping_sort_reverse', False)
            
            if sort_key == 'keyword':
                sorted_mappings.sort(key=lambda x: x[0].lower(), reverse=sort_reverse)
                sort_indicator = " 🔽" if sort_reverse else " 🔼"
                st.caption(f"Sorted by Keywords{sort_indicator}")
            elif sort_key == 'doctype':
                sorted_mappings.sort(key=lambda x: x[1].lower(), reverse=sort_reverse)
                sort_indicator = " 🔽" if sort_reverse else " 🔼"
                st.caption(f"Sorted by Document Types{sort_indicator}")
            
            if not sorted_mappings:
                st.info(f"No mappings found for document type '{filter_type}'")
            else:
                st.markdown("---")
                
                # Header row
                header_col1, header_col2, header_col3 = st.columns([2, 2, 1])
                with header_col1:
                    st.markdown("**Keyword**")
                with header_col2:
                    st.markdown("**Document Type**")
                with header_col3:
                    st.markdown("**Action**")
                
                st.markdown("---")
                
                # Display sorted mappings
                for i, (keyword, doc_type) in enumerate(sorted_mappings):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.code(keyword, language=None)
                    with col2:
                        # Get category for context
                        category = doc_type_manager.get_category_for_type(doc_type)
                        st.markdown(f"**{doc_type}**")
                        st.caption(f"📂 {category}")
                    with col3:
                        if st.button("🗑️", key=f"remove_mapping_{keyword}_{i}", help=f"Remove mapping for '{keyword}'"):
                            if doc_type_manager.remove_type_mapping(keyword):
                                st.success(f"Removed mapping for '{keyword}'")
                                st.session_state.show_maintenance = True
                                st.rerun()
                
                # Summary info
                st.markdown("---")
                total_mappings = len(mappings)
                shown_mappings = len(sorted_mappings)
                if filter_type != "All":
                    st.info(f"Showing {shown_mappings} of {total_mappings} mappings (filtered by '{filter_type}')")
                else:
                    st.info(f"Showing all {shown_mappings} mappings")
        else:
            st.info("No keyword mappings defined yet.")
        
        # Add new mapping
        st.markdown("---")
        st.subheader("Add New Mapping")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_keyword = st.text_input("Keyword (e.g., 'bio', 'agenda'):", key="new_keyword")
        with col2:
            new_mapping_type = st.selectbox("Maps to Document Type:", [""] + all_types, key="new_mapping_type")
        with col3:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("➕ Add Mapping") and new_keyword and new_mapping_type:
                if doc_type_manager.add_type_mapping(new_keyword, new_mapping_type):
                    st.success(f"Added mapping: '{new_keyword}' → '{new_mapping_type}'")
                    st.session_state.show_maintenance = True
                    st.rerun()
    
    with tab3:
        st.subheader("System Overview")
        
        categories = doc_type_manager.get_categories()
        mappings = doc_type_manager.get_type_mappings()
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Categories", len(categories))
        with col2:
            total_types = sum(len(cat['types']) for cat in categories.values())
            st.metric("Document Types", total_types)
        with col3:
            st.metric("Keyword Mappings", len(mappings))
        with col4:
            # Test suggestion
            st.metric("Auto-suggestions", "Active" if mappings else "None")
        
        # Category breakdown
        st.markdown("---")
        st.subheader("Category Breakdown")
        for category_name, category_data in categories.items():
            st.markdown(f"**{category_name}** ({len(category_data['types'])} types): {category_data['description']}")
            if category_data['types']:
                st.markdown(f"*Types:* {', '.join(category_data['types'])}")
        
        # Test the suggestion system
        st.markdown("---")
        st.subheader("Test Auto-Suggestion")
        test_filename = st.text_input("Test filename:", placeholder="e.g., john_smith_bio.pdf")
        if test_filename:
            suggested_type = doc_type_manager.suggest_document_type(test_filename)
            st.success(f"Suggested document type: **{suggested_type}**")
    
    with tab4:
        st.subheader("System Settings")
        
        # Export/Import configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Configuration**")
            if st.button("📥 Export Config"):
                config_json = doc_type_manager.export_config()
                st.download_button(
                    label="💾 Download Config JSON",
                    data=config_json,
                    file_name=f"document_types_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("**Import Configuration**")
            uploaded_file = st.file_uploader("Choose config file", type=['json'])
            if uploaded_file is not None:
                config_content = uploaded_file.read().decode('utf-8')
                if st.button("📤 Import Config"):
                    if doc_type_manager.import_config(config_content):
                        st.success("Configuration imported successfully!")
                        st.session_state.show_maintenance = True
                        st.rerun()
                    else:
                        st.error("Failed to import configuration. Please check the file format.")
        
        # Reset to defaults
        st.markdown("---")
        st.subheader("Reset to Defaults")
        st.warning("⚠️ This will reset all categories, types, and mappings to default values.")
        if st.button("🔄 Reset to Defaults", type="secondary"):
            if st.button("⚠️ Confirm Reset", type="primary"):
                if doc_type_manager.reset_to_defaults():
                    st.success("Configuration reset to defaults!")
                    st.session_state.show_maintenance = True
                    st.rerun()
    
    # Close button
    st.markdown("---")
    if st.button("✅ Close Document Type Management", use_container_width=True):
        st.session_state.show_maintenance = False
        st.rerun()

def check_recovery_needed():
    """Check if recovery is actually needed and return issues found."""
    return shared_check_recovery_needed(
        config_manager_cls=ConfigManager,
        recovery_manager_cls=IngestionRecoveryManager,
        ttl_seconds=120,
    )

def render_recovery_section():
    """Render the ingestion recovery and repair section only when needed."""
    shared_render_recovery_section(
        check_recovery_needed_fn=check_recovery_needed,
        config_manager_cls=ConfigManager,
        maintenance_page="pages/6_Maintenance.py",
        logger=logger,
    )

# --- Main App Logic ---
initialize_state()
st.title("2. Knowledge Ingest")
st.caption(f"Manage the knowledge base by ingesting new documents. App Version: {VERSION_STRING}")

# Add help system
help_system.show_help_menu()

# Check and display Ollama status (cached for 60 seconds to avoid slow UI)
try:
    from cortex_engine.utils.ollama_utils import check_ollama_service, get_ollama_status_message, get_ollama_instructions

    # Cache Ollama status check to avoid network calls on every interaction
    ollama_cache_key = "ollama_status_cache"
    ollama_cache_time_key = "ollama_status_cache_time"
    cache_ttl_seconds = 60  # Re-check every 60 seconds

    current_time = time.time()
    cached_time = st.session_state.get(ollama_cache_time_key, 0)

    if current_time - cached_time > cache_ttl_seconds:
        # Cache expired or first check - do the actual network call
        is_running, error_msg, resolved_url = check_ollama_service()
        st.session_state[ollama_cache_key] = (is_running, error_msg, resolved_url)
        st.session_state[ollama_cache_time_key] = current_time
    else:
        # Use cached result
        is_running, error_msg, resolved_url = st.session_state.get(ollama_cache_key, (False, "Not checked", None))

    if not is_running:
        st.warning(f"⚠️ {get_ollama_status_message(is_running, error_msg)}")
        with st.expander("ℹ️ **Important: Limited AI Functionality**", expanded=False):
            st.info("**Impact:** Documents will be processed with basic metadata only. AI-enhanced analysis, summaries, and tagging will be unavailable.")
            st.markdown(get_ollama_instructions())
    else:
        st.success("✅ Ollama service is running - Full AI capabilities available")
except Exception as e:
    st.warning(f"Unable to check Ollama status: {e}")

# Quick Recovery Button for Immediate Access
st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    if st.button("🚀 **Quick Recovery: Create Collection from Recent Ingest**", type="primary", use_container_width=True, key="quick_recovery_recent_ingest"):
        try:
            db_path = ConfigManager().get_config().get("ai_database_path", "")
            result = recover_collection_from_ingest_log(
                db_path=db_path,
                collection_name="recovered_recent_ingest",
                filter_valid_doc_ids_fn=filter_existing_doc_ids_for_collection,
                collection_manager_cls=WorkingCollectionManager,
            )
            if not result.get("ok"):
                st.error(result.get("error", "Recovery failed"))
            else:
                if result.get("created"):
                    st.success(f"Created collection '{result['collection_name']}'")
                st.success(
                    f"✅ **SUCCESS!** Recovered {result['recovered_doc_ids']} valid documents "
                    f"to '{result['collection_name']}' collection!"
                )
                if result.get("skipped_doc_ids", 0) > 0:
                    st.warning(
                        f"Skipped {result['skipped_doc_ids']} orphan/stale IDs that were not found in the vector store."
                    )
                st.info("📌 Go to Collection Management or Knowledge Search to access your recovered documents.")
        except Exception as recovery_error:
            st.error(f"Recovery failed: {recovery_error}")

with col2:
    if st.button("🔍 Check Ingestion Status", use_container_width=True):
        try:
            db_path = ConfigManager().get_config().get("ai_database_path", "")
            log_data, _, error = read_ingested_log(db_path)
            if error:
                if "not found" in error.lower():
                    st.warning(error)
                else:
                    st.error(error)
            else:
                st.success(f"📁 **{len(log_data)} files** have been ingested and are ready for recovery!")
                if log_data:
                    st.markdown("**Sample files:**")
                    for path in list(log_data.keys())[-5:]:
                        st.caption(f"• {os.path.basename(path)}")
        except Exception as e:
            st.error(f"Status check failed: {e}")

st.markdown("---")

# Add Ingestion Recovery & Repair Section
try:
    render_recovery_section()
except Exception as e:
    st.error(f"Recovery section failed to load: {e}")
    # Provide a basic recovery option as fallback
    with st.expander("🔧 Basic Recovery Tool", expanded=False):
        st.warning("Advanced recovery features unavailable. Using basic recovery.")
        if st.button("🚀 Create Collection from All Recent Ingests"):
            try:
                db_path = ConfigManager().get_config().get("ai_database_path", "")
                result = recover_collection_from_ingest_log(
                    db_path=db_path,
                    collection_name="recovered_ingestion",
                    filter_valid_doc_ids_fn=filter_existing_doc_ids_for_collection,
                    collection_manager_cls=WorkingCollectionManager,
                )
                if not result.get("ok"):
                    st.error(result.get("error", "Basic recovery failed"))
                else:
                    if result.get("created"):
                        st.success(f"Created collection '{result['collection_name']}'")
                    st.success(
                        f"✅ Recovered {result['recovered_doc_ids']} valid documents "
                        f"to '{result['collection_name']}' collection!"
                    )
                    if result.get("skipped_doc_ids", 0) > 0:
                        st.warning(
                            f"Skipped {result['skipped_doc_ids']} orphan/stale IDs that were not found in the vector store."
                        )
            except Exception as recovery_error:
                st.error(f"Basic recovery failed: {recovery_error}")

# Show help modal if requested
if st.session_state.get("show_help_modal", False):
    help_topic = st.session_state.get("help_topic", "overview")
    help_system.show_help_modal(help_topic)

# Show contextual help for this page
help_system.show_contextual_help("ingest")

# Check for existing batch state and show resume option
container_db_path = get_runtime_db_path()
batch_manager = BatchState(container_db_path)
set_runtime_db_path(str(batch_manager.db_path))
batch_status = batch_manager.get_status()
stage = st.session_state.get("ingestion_stage", "config")

# When a batch is actively processing (analysis or finalization), prefer the
# standard stage-based views so automatic progress/auto-finalize works.
if batch_status["active"] and stage not in {"analysis_running", "finalizing"}:
    # ACTIVE (BUT NOT CURRENTLY RUNNING) BATCH SECTION - management only
    render_active_batch_management(batch_manager, batch_status)
    
    # Add "Start Fresh" option
    st.markdown("---")
    if st.button("🆕 Start Fresh Ingestion", key="start_fresh", help="Clear current batch and start new ingestion"):
        batch_manager.clear_batch()
        st.success("Batch cleared. You can now start a fresh ingestion.")
        st.rerun()

else:
    # NO ACTIVE BATCH - Check for orphaned session first
    ingestion_log_path = Path(__file__).parent.parent / "logs" / "ingestion.log"
    orphaned_session = None

    if ingestion_log_path.exists():
        try:
            # Look for last progress indicator in log
            with open(ingestion_log_path, 'r') as log_file:
                lines = log_file.readlines()
                for line in reversed(lines[-1000:]):  # Check last 1000 lines
                    if "Analyzing:" in line and "(" in line and "/" in line:
                        # Extract progress info: (282/3983) 
                        import re
                        match = re.search(r'\((\d+)/(\d+)\)', line)
                        if match:
                            completed = int(match.group(1))
                            total = int(match.group(2))
                            remaining = total - completed
                            if remaining > 0:  # Only show if there are files remaining
                                orphaned_session = {
                                    'completed': completed,
                                    'total': total,
                                    'remaining': remaining,
                                    'progress_percent': round((completed / total) * 100, 1)
                                }
                            break
        except Exception as e:
            pass  # Ignore errors in log parsing

    if orphaned_session and stage in {"config", "pre_analysis"}:
        st.warning("⚠️ **Interrupted Processing Detected**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Previous Progress:** {orphaned_session['completed']}/{orphaned_session['total']} files ({orphaned_session['progress_percent']}%)  
            **Estimated Remaining:** {orphaned_session['remaining']} files  
            **Status:** Processing was interrupted - resume available with new batch system
            """)
        
        with col2:
            if st.button("🔄 Enable Resume Mode", type="primary", use_container_width=True, key="enable_resume_mode"):
                st.warning("⚠️ **Manual Resume Required**")
                st.info(f"""
                The interrupted session ({orphaned_session['completed']}/{orphaned_session['total']} files) was from an older version.
                
                **To resume:**
                1. Set up the same directories and filters below
                2. The system will skip the {orphaned_session['completed']} already processed files
                3. Future batches will have full automatic resume!
                """)
                
                # Store info for manual resume
                st.session_state.orphaned_session = orphaned_session
                st.session_state.resume_mode_enabled = True
        
        st.markdown("---")

    # Add maintenance access button (shown at top of ingestion workflow)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚙️ Document Type Management", use_container_width=True, help="Manage document categories and type mappings"):
            st.session_state.show_maintenance = not st.session_state.get("show_maintenance", False)
            st.rerun()

    st.markdown("---")

    # Check if maintenance mode is active
    if st.session_state.get("show_maintenance", False):
        render_document_type_management()
    else:
        # Normal ingestion workflow
        # Health check: prompt to migrate collections if needed
        show_collection_migration_healthcheck()

        # Render model configuration in sidebar
        render_sidebar_model_config()

        stage = st.session_state.get("ingestion_stage", "config")

        # DEBUG: Show stage transition info (temporary for troubleshooting)
        if stage in ["batch_processing", "analysis_running"]:
            batch_mode = st.session_state.get("batch_ingest_mode", False)
            st.info(f"🔍 **DEBUG:** Current stage: `{stage}` | Batch mode checkbox: `{batch_mode}`")

        if stage == "config": render_config_and_scan_ui()
        elif stage == "pre_analysis": render_pre_analysis_ui()
        elif stage == "batch_processing": render_batch_processing_ui()
        elif stage == "analysis_running": render_log_and_review_ui("Live Analysis Log", "metadata_review")
        elif stage == "metadata_review": render_metadata_review_ui()
        elif stage == "finalizing": render_log_and_review_ui("Live Finalization Log", "config_done")
        elif stage == "config_done": render_completion_screen()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
