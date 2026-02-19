"""
Pre-ingest organizer for fast repository triage before expensive RAG ingestion.

Phase 1 scope:
- Scan directories recursively
- Classify documents via deterministic heuristics
- Detect sensitivity/proprietary indicators
- Group versions and choose canonical candidates
- Emit manifest under external DB path: <db_path>/pre_ingest/pre_ingest_manifest.json
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .utils import (
    convert_source_path_to_docker_mount,
    convert_to_docker_mount_path,
    convert_windows_to_wsl_path,
    get_logger,
)
from .utils.file_utils import get_file_hash

logger = get_logger(__name__)
ProgressCallback = Callable[[str, Dict[str, Any]], None]
ControlCallback = Callable[[], None]


class PreIngestScanCancelled(RuntimeError):
    """Raised when a pre-ingest scan is stopped by operator action."""


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".pptx",
    ".csv",
    ".xlsx",
    ".json",
    ".xml",
    ".html",
    ".htm",
}

DEFAULT_CLIENT_MARKERS = [
    "client",
    "confidential",
    "restricted",
    "privileged",
    "nda",
]

DEFAULT_EXTERNAL_IP_OWNERS = [
    "deloitte",
    "mckinsey",
    "accenture",
    "pwc",
    "kpmg",
    "bcg",
    "bain",
]

DEFAULT_RESTRICTED_PATH_MARKERS = [
    "do_not_ingest",
    "restricted",
    "never_ingest",
    "blocked_ingest",
]


@dataclass
class OrganizerConfig:
    client_markers: List[str]
    external_ip_owners: List[str]
    restricted_path_markers: List[str]
    include_extensions: Optional[List[str]] = None
    max_file_count: int = 100000
    sample_char_limit: int = 2500

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OrganizerConfig":
        payload = data or {}
        include_extensions = payload.get("include_extensions")
        normalized_exts = None
        if isinstance(include_extensions, list):
            normalized_exts = sorted(
                {
                    ext if str(ext).startswith(".") else f".{ext}"
                    for ext in include_extensions
                    if str(ext).strip()
                }
            )

        return cls(
            client_markers=[str(v).lower().strip() for v in payload.get("client_markers", DEFAULT_CLIENT_MARKERS)],
            external_ip_owners=[str(v).lower().strip() for v in payload.get("external_ip_owners", DEFAULT_EXTERNAL_IP_OWNERS)],
            restricted_path_markers=[
                str(v).lower().strip() for v in payload.get("restricted_path_markers", DEFAULT_RESTRICTED_PATH_MARKERS)
            ],
            include_extensions=normalized_exts,
            max_file_count=max(1, int(payload.get("max_file_count", 100000))),
            sample_char_limit=max(200, int(payload.get("sample_char_limit", 2500))),
        )


def _normalized_source_path(path_value: str) -> str:
    if os.path.exists("/.dockerenv"):
        return path_value.strip()
    return convert_windows_to_wsl_path(path_value.strip())


def _extract_text_sample(file_path: Path, char_limit: int) -> str:
    ext = file_path.suffix.lower()
    try:
        if ext in {".txt", ".md", ".json", ".xml", ".html", ".htm", ".csv"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read(char_limit).strip()

        if ext == ".pdf":
            try:
                import PyPDF2  # type: ignore
            except Exception:
                return ""
            with open(file_path, "rb") as handle:
                reader = PyPDF2.PdfReader(handle)
                if not reader.pages:
                    return ""
                text = reader.pages[0].extract_text() or ""
                return text[:char_limit].strip()

        if ext == ".docx":
            try:
                from docx import Document  # type: ignore
            except Exception:
                return ""
            doc = Document(str(file_path))
            parts: List[str] = []
            for paragraph in doc.paragraphs:
                t = paragraph.text.strip()
                if t:
                    parts.append(t)
                if sum(len(p) for p in parts) >= char_limit:
                    break
            return "\n".join(parts)[:char_limit].strip()
    except Exception as exc:
        logger.debug(f"Text sample extraction failed for {file_path}: {exc}")
    return ""


def _classify_doc_class(file_path: Path, sample: str) -> str:
    low = f"{str(file_path).lower()} {sample.lower()}"
    if any(k in low for k in ("invoice", "timesheet", "receipt", "expense")):
        return "admin_finance"
    if any(k in low for k in ("contract", "statement of work", "sow", "agreement", "nda")):
        return "legal_contract"
    if any(k in low for k in ("draft", "v0", "working copy")):
        return "draft"
    if any(k in low for k in ("report", "presentation", "research", "analysis", "reference", "whitepaper")):
        return "work_knowledge"
    return "unknown"


def _extract_external_ip_owner(file_path: Path, sample: str, owners: List[str]) -> str:
    low = f"{str(file_path).lower()} {sample.lower()}"
    for owner in owners:
        if owner and owner in low:
            return owner.title()
    return ""


def _extract_client_related(file_path: Path, sample: str, client_markers: List[str]) -> List[str]:
    low = f"{str(file_path).lower()} {sample.lower()}"
    matches = [marker for marker in client_markers if marker and marker in low]
    return sorted(set(matches))


def _detect_sensitivity(file_path: Path, sample: str, config: OrganizerConfig) -> Tuple[str, str]:
    path_low = str(file_path).lower()
    text_low = sample.lower()
    combined = f"{path_low}\n{text_low}"

    if any(marker in path_low for marker in config.restricted_path_markers):
        return "restricted", "path_restricted_marker"
    if "strictly confidential" in combined or "client privileged" in combined:
        return "restricted", "strong_confidentiality_phrase"
    if any(token in combined for token in ("confidential", "internal use only", "proprietary", "privileged")):
        return "confidential", "confidentiality_marker"
    if any(token in combined for token in ("internal", "private", "client")):
        return "internal", "internal_marker"
    return "public", "no_sensitive_marker"


def _extract_version_signals(file_path: Path) -> Tuple[str, int, bool]:
    stem = file_path.stem
    low = stem.lower()

    base = re.sub(
        r"[\s_\-]*(v\d+(\.\d+)?|\(\d{3}\)|\d{8}|final|revised|draft|copy)\b",
        "",
        low,
        flags=re.IGNORECASE,
    )
    base = re.sub(r"\s+", " ", base).strip() or low

    version_score = 0
    v_match = re.search(r"\bv(\d+)\b", low)
    if v_match:
        version_score += int(v_match.group(1)) * 100
    n_match = re.search(r"\((\d{3})\)", low)
    if n_match:
        version_score += int(n_match.group(1))
    if "final" in low:
        version_score += 500
    if "revised" in low:
        version_score += 300
    if "draft" in low:
        version_score -= 200

    return base, version_score, "draft" in low


def _recommend_policy(
    *,
    doc_class: str,
    sensitivity_level: str,
    source_ownership: str,
    is_canonical_version: bool,
    restricted_path_markers_hit: bool,
    client_related: List[str],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if restricted_path_markers_hit:
        reasons.append("restricted_path_policy")
        return "do_not_ingest", reasons

    if not is_canonical_version:
        reasons.append("non_canonical_version")
        return "exclude", reasons

    if doc_class in {"admin_finance"}:
        reasons.append("administrative_document")
        return "exclude", reasons

    if source_ownership == "client_owned" or sensitivity_level in {"confidential", "restricted"}:
        reasons.append("sensitive_or_client_owned")
        return "review_required", reasons

    if source_ownership == "external_ip":
        reasons.append("external_ip_material")
        return "review_required", reasons

    if doc_class == "unknown" and client_related:
        reasons.append("unknown_class_with_client_markers")
        return "review_required", reasons

    if doc_class == "work_knowledge":
        reasons.append("work_knowledge_candidate")
        return "include", reasons

    reasons.append("default_review")
    return "review_required", reasons


def _detect_source_ownership(
    client_related: List[str],
    external_ip_owner: str,
    sensitivity_level: str,
) -> str:
    if external_ip_owner:
        return "external_ip"
    if client_related or sensitivity_level in {"confidential", "restricted"}:
        return "client_owned"
    return "first_party"


def _emit_progress(progress_callback: Optional[ProgressCallback], event: str, **data: Any) -> None:
    if not progress_callback:
        return
    try:
        progress_callback(event, data)
    except Exception:
        # Progress logging should never break scan execution.
        pass


def _check_control(control_callback: Optional[ControlCallback]) -> None:
    if not control_callback:
        return
    control_callback()


def _iter_candidate_files(
    source_dirs: List[str],
    config: OrganizerConfig,
    progress_callback: Optional[ProgressCallback] = None,
    control_callback: Optional[ControlCallback] = None,
) -> List[Path]:
    files: List[Path] = []
    include_exts = set(config.include_extensions or SUPPORTED_EXTENSIONS)

    for dir_idx, source_dir in enumerate(source_dirs, start=1):
        _check_control(control_callback)
        _emit_progress(
            progress_callback,
            "scan_dir_start",
            directory_index=dir_idx,
            directory_count=len(source_dirs),
            source_dir=source_dir,
        )
        resolved = convert_source_path_to_docker_mount(source_dir)
        path_obj = Path(resolved)
        if not path_obj.exists() or not path_obj.is_dir():
            logger.warning(f"Source directory not found for pre-ingest organizer: {source_dir} -> {resolved}")
            _emit_progress(
                progress_callback,
                "scan_dir_missing",
                source_dir=source_dir,
                resolved_path=resolved,
            )
            continue
        scanned_in_dir = 0
        for fp in path_obj.rglob("*"):
            _check_control(control_callback)
            if not fp.is_file():
                continue
            scanned_in_dir += 1
            if scanned_in_dir == 1 or scanned_in_dir % 100 == 0:
                _emit_progress(
                    progress_callback,
                    "scan_dir_progress",
                    source_dir=source_dir,
                    scanned_in_dir=scanned_in_dir,
                    discovered_total=len(files),
                    current_path=fp.as_posix(),
                )
            ext = fp.suffix.lower()
            if (not ext) or (ext not in include_exts):
                continue
            files.append(fp)
            if len(files) >= config.max_file_count:
                logger.warning(f"Max file count reached ({config.max_file_count}). Truncating scan.")
                _emit_progress(
                    progress_callback,
                    "scan_truncated",
                    max_file_count=config.max_file_count,
                    discovered_total=len(files),
                )
                return files
        _emit_progress(
            progress_callback,
            "scan_dir_done",
            source_dir=source_dir,
            scanned_in_dir=scanned_in_dir,
            discovered_total=len(files),
        )
    return files


def run_pre_ingest_organizer_scan(
    *,
    source_dirs: List[str],
    db_path: str,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    control_callback: Optional[ControlCallback] = None,
) -> Dict[str, Any]:
    """
    Execute pre-ingest scan and write manifest to external DB path.

    Returns a summary payload including manifest path and counts.
    """
    cfg = OrganizerConfig.from_dict(config)
    source_dirs = [s for s in source_dirs if str(s).strip()]
    if not source_dirs:
        raise ValueError("source_dirs is required")
    if not db_path or not str(db_path).strip():
        raise ValueError("db_path is required")

    runtime_db_path = convert_to_docker_mount_path(_normalized_source_path(db_path))
    pre_ingest_dir = Path(runtime_db_path) / "pre_ingest"
    pre_ingest_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(
        progress_callback,
        "scan_started",
        source_dir_count=len(source_dirs),
        db_path=runtime_db_path,
    )
    files = _iter_candidate_files(
        source_dirs,
        cfg,
        progress_callback=progress_callback,
        control_callback=control_callback,
    )
    _emit_progress(
        progress_callback,
        "scan_complete",
        discovered_total=len(files),
    )
    now = datetime.now(timezone.utc).isoformat()

    records: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    total_files = len(files)
    for idx, fp in enumerate(files, start=1):
        _check_control(control_callback)
        if idx == 1 or idx % 25 == 0 or idx == total_files:
            _emit_progress(
                progress_callback,
                "analyze_progress",
                processed=idx,
                total=total_files,
                current_path=fp.as_posix(),
            )
        sample = _extract_text_sample(fp, cfg.sample_char_limit)
        doc_class = _classify_doc_class(fp, sample)
        client_related = _extract_client_related(fp, sample, cfg.client_markers)
        external_ip_owner = _extract_external_ip_owner(fp, sample, cfg.external_ip_owners)
        sensitivity_level, sensitivity_reason = _detect_sensitivity(fp, sample, cfg)
        source_ownership = _detect_source_ownership(client_related, external_ip_owner, sensitivity_level)
        group_key_stem, version_score, is_draft_variant = _extract_version_signals(fp)
        group_key = f"{fp.parent.as_posix().lower()}::{group_key_stem}"
        restricted_hit = any(marker in str(fp).lower() for marker in cfg.restricted_path_markers)

        try:
            stat = fp.stat()
            size_bytes = int(stat.st_size)
            modified_ts = float(stat.st_mtime)
        except OSError:
            size_bytes = 0
            modified_ts = 0.0

        record = {
            "file_path": fp.as_posix(),
            "file_name": fp.name,
            "extension": fp.suffix.lower(),
            "size_bytes": size_bytes,
            "modified_at_epoch": modified_ts,
            "modified_at_iso": datetime.fromtimestamp(modified_ts, tz=timezone.utc).isoformat() if modified_ts else "",
            "file_hash": "",
            "doc_class": doc_class,
            "sensitivity_level": sensitivity_level,
            "sensitivity_reason": sensitivity_reason,
            "source_ownership": source_ownership,
            "client_related": client_related,
            "external_ip_owner": external_ip_owner,
            "version_group_id": group_key,
            "version_score": version_score,
            "is_draft_variant": is_draft_variant,
            "is_canonical_version": False,
            "ingest_policy_class": "review_required",
            "ingest_policy_confidence": 0.6,
            "ingest_policy_reasons": [],
            "scan_timestamp": now,
        }
        records.append(record)
        grouped[group_key].append(record)

    for _, group_items in grouped.items():
        canonical = max(
            group_items,
            key=lambda item: (
                int(item.get("version_score", 0)),
                float(item.get("modified_at_epoch", 0)),
                int(item.get("size_bytes", 0)),
            ),
        )
        canonical["is_canonical_version"] = True

    for rec in records:
        policy, reasons = _recommend_policy(
            doc_class=str(rec.get("doc_class")),
            sensitivity_level=str(rec.get("sensitivity_level")),
            source_ownership=str(rec.get("source_ownership")),
            is_canonical_version=bool(rec.get("is_canonical_version")),
            restricted_path_markers_hit=any(marker in str(rec.get("file_path", "")).lower() for marker in cfg.restricted_path_markers),
            client_related=list(rec.get("client_related", [])),
        )
        rec["ingest_policy_class"] = policy
        rec["ingest_policy_reasons"] = reasons
        if policy == "do_not_ingest":
            rec["ingest_policy_confidence"] = 0.95
        elif policy == "exclude":
            rec["ingest_policy_confidence"] = 0.9
        elif policy == "include":
            rec["ingest_policy_confidence"] = 0.8
        else:
            rec["ingest_policy_confidence"] = 0.65

        try:
            rec["file_hash"] = get_file_hash(rec["file_path"])
        except Exception:
            rec["file_hash"] = ""

    manifest = {
        "version": "1.0",
        "generated_at": now,
        "source_dirs": source_dirs,
        "db_path": runtime_db_path,
        "records": records,
        "summary": {
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
        },
    }

    manifest_path = pre_ingest_dir / "pre_ingest_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    _emit_progress(
        progress_callback,
        "manifest_written",
        manifest_path=manifest_path.as_posix(),
        total_files=len(records),
    )

    return {
        "manifest_path": manifest_path.as_posix(),
        "summary": manifest["summary"],
        "total_files": len(records),
    }
