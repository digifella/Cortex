# ## File: pages/7_Document_Extract.py
# Version: v5.8.0
# Date: 2026-01-29
# Purpose: Document extraction tools — Textifier (document to Markdown) and Anonymizer.

import streamlit as st
import sys
from pathlib import Path
import os
import shutil
import json
import re
import tempfile
import time
import zipfile
import io
import unicodedata
import subprocess
import hashlib
from datetime import datetime
from typing import Any, Callable, List, Dict, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from PIL import Image, ImageOps

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from cortex_engine.anonymizer import DocumentAnonymizer, AnonymizationMapping, AnonymizationOptions
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path, resolve_db_root_path
from cortex_engine.config_manager import ConfigManager
from cortex_engine.version_config import VERSION_STRING
from cortex_engine.journal_authority import classify_journal_authority
from cortex_engine.preface_classification import classify_credibility_tier
from cortex_engine.document_preface import add_document_preface
from cortex_engine.handoff_contract import validate_research_resolve_input
from cortex_engine.included_study_extractor import (
    IncludedStudyExtractorQuotaError,
    anthropic_key_source as included_study_anthropic_key_source,
    gemini_key_source,
    included_study_extractor_available,
    run_included_study_access_check,
    run_included_study_access_check_matrix,
    run_included_study_extractor_with_fallback,
)
from cortex_engine.research_resolve import (
    build_research_preferred_url_list,
    build_research_preview_rows,
    parse_research_spreadsheet_text,
    parse_research_spreadsheet_upload,
    run_research_resolve,
)
from cortex_engine.review_study_miner import (
    _NON_STUDY_ROW_LABELS,
    _normalize_text,
    _reference_number_pointers,
    ReviewMiningOptions,
    assess_review_documents,
    extract_review_reference_section,
    extract_review_table_blocks,
    mine_review_documents,
)
from cortex_engine.review_table_local_rescue import (
    annotate_local_candidate_completeness,
    local_table_rescue_available,
    local_table_rescue_host,
    merge_local_table_candidates,
    run_local_table_reconciliation,
    run_local_table_rescue,
)
from cortex_engine.review_table_rescue import anthropic_key_source, claude_table_rescue_available, run_claude_table_rescue
from cortex_engine.url_ingestor import URLIngestor
from cortex_engine.url_ingestor_ui import render_url_ingestor_ui

# Set up logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(page_title="Document & Photo Processing", layout="wide", page_icon="📝")

# Page metadata
PAGE_VERSION = VERSION_STRING
MAX_BATCH_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB


# ======================================================================
# Shared helpers
# ======================================================================


class _SessionUpload:
    """Persist uploaded file bytes across Streamlit reruns."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data or b""
        self.size = len(self._data)

    def getvalue(self) -> bytes:
        return self._data


def _resolve_db_root() -> Path:
    config = ConfigManager().get_config()
    raw_db = (config.get("ai_database_path") or "").strip()
    if not raw_db:
        raise ValueError("No ai_database_path configured. Set the database path first in Knowledge Ingest or Maintenance.")
    resolved = resolve_db_root_path(raw_db)
    if resolved:
        return Path(str(resolved))
    normalized = raw_db if os.path.exists("/.dockerenv") else convert_windows_to_wsl_path(raw_db)
    return Path(normalized)


def _editor_records(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict("records")
            return [dict(item) for item in records if isinstance(item, dict)]
        except Exception:
            return []
    return []


def _append_study_miner_local_log(message: str, *, placeholder=None) -> None:
    text = str(message or "").strip()
    if not text:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    lines = list(st.session_state.get("study_miner_local_log_lines") or [])
    lines.append(f"[{timestamp}] {text}")
    st.session_state["study_miner_local_log_lines"] = lines[-80:]
    if placeholder is not None:
        try:
            placeholder.code("\n".join(st.session_state["study_miner_local_log_lines"]), language="text")
        except Exception:
            pass


def _render_study_miner_local_log(expanded: bool = False) -> None:
    lines = list(st.session_state.get("study_miner_local_log_lines") or [])
    if not lines:
        return
    with st.expander("Local Rescue Log", expanded=expanded):
        st.code("\n".join(lines[-80:]), language="text")


def _merge_research_editor_rows(editor_rows: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_row_id = {str(item.get("row_id")): dict(item) for item in citations or []}
    merged: List[Dict[str, Any]] = []
    for row in editor_rows or []:
        if not bool(row.get("keep", True)):
            continue
        row_id = str(row.get("row_id"))
        base = dict(by_row_id.get(row_id) or {})
        if not base:
            continue
        for key in ("title", "authors", "year", "doi", "journal", "accession"):
            base[key] = str(row.get(key) or "").strip()
        merged.append(base)
    return merged


def _research_resolved_rows(resolved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in resolved or []:
        oa = item.get("open_access") or {}
        journal = item.get("journal") or {}
        retraction = item.get("retraction") or {}
        rows.append(
            {
                "row_id": item.get("row_id"),
                "input_title": item.get("input_title", ""),
                "resolved_url": item.get("resolved_url", ""),
                "resolved_doi": item.get("resolved_doi", ""),
                "confidence": item.get("confidence", ""),
                "open_access": oa.get("oa_status") if oa.get("is_oa") is not None else "unknown",
                "pdf_url": oa.get("pdf_url", ""),
                "journal": journal.get("name", ""),
                "sjr_quartile": journal.get("sjr_quartile", ""),
                "publisher": item.get("publisher", ""),
                "credibility_tier": item.get("credibility_tier", ""),
                "retracted": "yes" if retraction.get("is_retracted") else "",
            }
        )
    return rows


def _research_unresolved_rows(unresolved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in unresolved or []:
        best = ""
        if item.get("best_candidates"):
            first = item["best_candidates"][0]
            best = f"{first.get('title', '')} ({first.get('similarity', 0)})"
        rows.append(
            {
                "row_id": item.get("row_id"),
                "input_title": item.get("input_title", ""),
                "reason": item.get("reason", ""),
                "best_candidate": best,
            }
        )
    return rows


def _included_study_group_label(group: Dict[str, Any]) -> str:
    trial_label = str(group.get("trial_label") or "").strip()
    group_label = str(group.get("group_label") or "").strip()
    if trial_label and group_label:
        return f"{group_label} / {trial_label}"
    return trial_label or group_label


def _included_study_editor_rows(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    row_id = 1
    for table in tables or []:
        table_number = str(table.get("table_number") or "").strip()
        table_title = str(table.get("table_title") or "").strip()
        grouping_basis = str(table.get("grouping_basis") or "").strip()
        for group in list(table.get("groups") or []):
            combined_group = _included_study_group_label(group)
            for citation in list(group.get("citations") or []):
                resolved_title = str(citation.get("resolved_title") or "").strip()
                display = str(citation.get("display") or "").strip()
                resolved_authors = str(citation.get("resolved_authors") or citation.get("authors") or "").strip()
                resolved_year = str(citation.get("resolved_year") or citation.get("year") or "").strip()
                resolved_journal = str(citation.get("resolved_journal") or "").strip()
                resolved_doi = str(citation.get("resolved_doi") or "").strip()
                notes = str(citation.get("notes") or group.get("notes") or "").strip()
                needs_review = bool(citation.get("needs_review"))
                rows.append(
                    {
                        "keep": not needs_review,
                        "row_id": row_id,
                        "table_number": table_number,
                        "table_title": table_title,
                        "grouping_basis": grouping_basis,
                        "group_label": str(group.get("group_label") or "").strip(),
                        "trial_label": str(group.get("trial_label") or "").strip(),
                        "combined_group": combined_group,
                        "citation_display": display,
                        "title": resolved_title or display,
                        "authors": resolved_authors,
                        "year": resolved_year,
                        "doi": resolved_doi,
                        "journal": resolved_journal,
                        "reference_number": str(citation.get("reference_number") or "").strip(),
                        "notes": notes,
                        "needs_review": "yes" if needs_review else "",
                    }
                )
                row_id += 1
    return rows


def _merge_included_study_editor_rows(editor_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for row in editor_rows or []:
        if not bool(row.get("keep", True)):
            continue
        merged.append(
            {
                "row_id": int(row.get("row_id") or 0) or len(merged) + 1,
                "title": str(row.get("title") or row.get("citation_display") or "").strip(),
                "authors": str(row.get("authors") or "").strip(),
                "year": str(row.get("year") or "").strip(),
                "doi": str(row.get("doi") or "").strip(),
                "journal": str(row.get("journal") or "").strip(),
                "notes": str(row.get("notes") or "").strip(),
                "extra_fields": {
                    "table_number": str(row.get("table_number") or "").strip(),
                    "table_title": str(row.get("table_title") or "").strip(),
                    "grouping_basis": str(row.get("grouping_basis") or "").strip(),
                    "group_label": str(row.get("group_label") or "").strip(),
                    "trial_label": str(row.get("trial_label") or "").strip(),
                    "combined_group": str(row.get("combined_group") or "").strip(),
                    "reference_number": str(row.get("reference_number") or "").strip(),
                    "citation_display": str(row.get("citation_display") or "").strip(),
                    "needs_review": str(row.get("needs_review") or "").strip(),
                },
            }
        )
    return merged


def _set_included_study_keep_state(editor_rows: List[Dict[str, Any]], keep_value: bool) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for row in editor_rows or []:
        normalized = dict(row)
        normalized["keep"] = bool(keep_value)
        updated.append(normalized)
    return updated


def _study_miner_candidate_rows(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in candidates or []:
        extras = dict(item.get("extra_fields") or {})
        reference_number = str(item.get("reference_number") or extras.get("reference_number") or "").strip()
        reference_match_method = str(item.get("reference_match_method") or extras.get("reference_match_method") or "").strip()
        if not reference_number:
            inline_numbers = _reference_number_pointers(
                " | ".join(
                    [
                        str(extras.get("author_year") or "").strip(),
                        str(item.get("raw_citation") or "").strip(),
                        str(item.get("title") or "").strip(),
                    ]
                )
            )
            if inline_numbers:
                reference_number = str(inline_numbers[0] or "").strip()
                reference_match_method = reference_match_method or "table_inline_reference"
        reference_validation = str(item.get("reference_validation") or "").strip()
        needs_review = bool(item.get("needs_review"))
        review_warning = str(item.get("review_warning") or "").strip()
        group_parts = [str(extras.get("region") or "").strip(), str(extras.get("study") or "").strip()]
        rows.append(
            {
                "keep": bool(item.get("meets_criteria")) and not needs_review,
                "row_id": item.get("row_id"),
                "source_review_title": item.get("source_review_title") or item.get("source_name", ""),
                "table_index": int(extras.get("table_index") or 0),
                "table_label": str(extras.get("table_label") or "").strip(),
                "table_group": " / ".join(part for part in group_parts if part),
                "table_citation": str(extras.get("author_year") or "").strip(),
                "title": item.get("title", ""),
                "authors": item.get("authors", ""),
                "year": item.get("year", ""),
                "doi": item.get("doi", ""),
                "journal": item.get("journal", ""),
                "source_section": item.get("source_section", ""),
                "matches": "yes" if item.get("meets_criteria") else "",
                "score": int(item.get("relevance_score") or 0),
                "reference_number": reference_number,
                "reference_match_method": reference_match_method,
                "reference_validation": reference_validation,
                "reference_link": f"ref {reference_number} ({reference_match_method})" if reference_number and reference_match_method else "",
                "needs_review": "yes" if needs_review else "",
                "review_warning": review_warning,
                "design_matches": ", ".join(item.get("design_matches") or []),
                "outcome_matches": ", ".join(item.get("outcome_matches") or []),
            }
        )
    return _study_miner_harmonize_group_labels(rows)


def _is_structural_study_miner_candidate(item: Dict[str, Any]) -> bool:
    extras = dict(item.get("extra_fields") or {})
    texts = [
        str(item.get("title") or ""),
        str(item.get("raw_citation") or ""),
        str(extras.get("study") or ""),
        str(extras.get("author_year") or ""),
    ]
    normalized = " ".join(part for part in (_normalize_text(text) for text in texts) if part).strip()
    if not normalized:
        return False
    if normalized in _NON_STUDY_ROW_LABELS:
        return True
    if normalized in {
        "region study author year",
        "study author year",
        "author year",
        "region study author year study design study type",
    }:
        return True
    tokens = normalized.split()
    structural_tokens = {
        "region",
        "study",
        "author",
        "authors",
        "year",
        "design",
        "type",
        "outcomes",
        "outcome",
        "population",
        "treatment",
        "scale",
        "utility",
        "values",
        "source",
        "follow",
        "up",
        "followup",
        "assessment",
        "measures",
        "time",
        "point",
        "hrqol",
        "qol",
    }
    if tokens and len(tokens) <= 12 and set(tokens).issubset(structural_tokens):
        return True
    return False


def _filter_study_miner_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(item) for item in candidates or [] if not _is_structural_study_miner_candidate(item)]


def _study_miner_document_rows(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in documents or []:
        review_assessment = dict(item.get("review_assessment") or {})
        rows.append(
            {
                "confirm_review": bool(item.get("confirm_review")),
                "doc_id": int(item.get("doc_id") or 0),
                "source_name": item.get("source_name", ""),
                "review_title": item.get("review_title", ""),
                "systematic_review_likely": bool(item.get("systematic_review_likely")),
                "confidence": str(review_assessment.get("confidence") or ""),
                "score": int(review_assessment.get("score") or 0),
                "matched_signals": ", ".join(review_assessment.get("matched_signals") or []),
            }
        )
    return rows


def _merge_study_miner_document_rows(editor_rows: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_doc_id = {str(item.get("doc_id")): dict(item) for item in documents or []}
    merged: List[Dict[str, Any]] = []
    for row in editor_rows or []:
        base = dict(by_doc_id.get(str(row.get("doc_id"))) or {})
        if not base:
            continue
        base["confirm_review"] = bool(row.get("confirm_review"))
        base["review_title"] = str(row.get("review_title") or base.get("review_title") or "").strip()
        merged.append(base)
    return merged


def _merge_study_miner_editor_rows(editor_rows: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_row_id = {str(item.get("row_id")): dict(item) for item in candidates or []}
    selected: List[Dict[str, Any]] = []
    for row in editor_rows or []:
        if not bool(row.get("keep", True)):
            continue
        base = dict(by_row_id.get(str(row.get("row_id"))) or {})
        if not base:
            continue
        for key in ("title", "authors", "year", "doi", "journal"):
            base[key] = str(row.get(key) or "").strip()
        selected.append(base)
    return selected


def _build_study_miner_research_payload(
    candidates: List[Dict[str, Any]],
    *,
    check_open_access: bool,
    enrich_sjr: bool,
    unpaywall_email: str,
) -> Dict[str, Any]:
    return validate_research_resolve_input(
        {
            "citations": list(candidates or []),
            "options": {
                "check_open_access": bool(check_open_access),
                "enrich_sjr": bool(enrich_sjr),
                "unpaywall_email": str(unpaywall_email or "").strip(),
            },
        }
    )


def _run_study_miner_paper_retrieval(
    *,
    candidates: List[Dict[str, Any]],
    db_root: Path,
    resolver_options: Dict[str, Any],
    ingest_options: Dict[str, Any],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def _emit(message: str) -> None:
        text = str(message or "").strip()
        if text and progress_cb is not None:
            progress_cb(text)

    payload = _build_study_miner_research_payload(
        candidates,
        check_open_access=bool(resolver_options.get("check_open_access", True)),
        enrich_sjr=bool(resolver_options.get("enrich_sjr", True)),
        unpaywall_email=str(resolver_options.get("unpaywall_email") or "").strip(),
    )

    resolver_run_dir = Path(db_root) / "research_resolve" / time.strftime("%Y%m%d_%H%M%S")
    resolver_run_dir.mkdir(parents=True, exist_ok=True)
    _emit(f"Research Resolver: resolving {len(payload.get('citations') or [])} citation(s)")

    resolver_output = run_research_resolve(
        payload=payload,
        run_dir=resolver_run_dir,
        progress_cb=lambda pct, message, stage=None: _emit(
            f"Research Resolver: {message}" + (f" [{stage}]" if stage else "")
        ),
    )

    preferred_urls = build_research_preferred_url_list(list(resolver_output.get("resolved") or []))
    _emit(f"Research Resolver: found {len(preferred_urls)} preferred URL(s) for retrieval")

    url_results: List[Any] = []
    url_csv_path = ""
    url_json_path = ""
    url_zip_bytes = b""
    url_run_dir = None

    if preferred_urls:
        url_run_dir = Path(db_root) / "url_ingest" / time.strftime("%Y%m%d_%H%M%S")
        url_run_dir.mkdir(parents=True, exist_ok=True)
        ingestor = URLIngestor(url_run_dir, timeout=int(ingest_options.get("timeout_seconds") or 25))
        _emit(f"URL Ingestor: retrieving {len(preferred_urls)} URL(s)")
        url_results = ingestor.process_urls(
            urls=preferred_urls,
            convert_to_md=bool(ingest_options.get("convert_to_md", True)),
            use_vision_for_md=bool(ingest_options.get("use_vision_for_md", False)),
            textify_options=dict(ingest_options.get("textify_options") or {}),
            capture_web_md_on_no_pdf=bool(ingest_options.get("capture_web_md_on_no_pdf", True)),
            progress_cb=lambda done, total, message: _emit(f"URL Ingestor: {message} ({done}/{total})"),
            event_cb=lambda message: _emit(f"URL Ingestor: {message}"),
        )
        csv_path, json_path = ingestor.build_reports(url_results)
        url_csv_path = str(csv_path)
        url_json_path = str(json_path)
        url_zip_bytes = ingestor.build_zip_bytes(url_results, csv_path, json_path)

    return {
        "resolver_payload": payload,
        "resolver_output": resolver_output,
        "preferred_urls": preferred_urls,
        "resolver_run_dir": str(resolver_run_dir),
        "url_results": url_results,
        "url_csv_path": url_csv_path,
        "url_json_path": url_json_path,
        "url_zip_bytes": url_zip_bytes,
        "url_run_dir": str(url_run_dir) if url_run_dir else "",
    }


def _set_study_miner_keep_state(editor_rows: List[Dict[str, Any]], keep_value: bool) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for row in editor_rows or []:
        normalized = dict(row)
        normalized["keep"] = bool(keep_value)
        updated.append(normalized)
    return updated


def _study_miner_stats_rows(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"metric": "Tables detected", "value": int(stats.get("tables_detected") or 0)},
        {"metric": "Reference entries detected", "value": int(stats.get("reference_entries_detected") or 0)},
        {"metric": "Candidates total", "value": int(stats.get("candidates_total") or 0)},
        {"metric": "Candidates matching criteria", "value": int(stats.get("candidates_matching") or 0)},
        {"metric": "Table rows linked to bibliography", "value": int(stats.get("table_reference_links") or 0)},
        {"metric": "Candidates needing review", "value": int(stats.get("needs_review") or 0)},
        {"metric": "Reference mismatches", "value": int(stats.get("reference_mismatches") or 0)},
        {"metric": "Cloud rescue candidates", "value": int(stats.get("cloud_rescue_candidates") or 0)},
    ]


def _study_miner_review_rows(per_review: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in per_review or []:
        review_assessment = dict(item.get("review_assessment") or {})
        rows.append(
            {
                "doc_id": int(item.get("doc_id") or 0),
                "review_title": item.get("review_title", ""),
                "source_name": item.get("source_name", ""),
                "confidence": str(review_assessment.get("confidence") or ""),
                "score": int(review_assessment.get("score") or 0),
                "candidate_count": int(item.get("candidate_count") or 0),
                "matching_count": int(item.get("matching_count") or 0),
            }
        )
    return rows


def _render_study_miner_parse_evidence(documents: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> None:
    if not documents:
        return

    with st.expander("Parse Evidence", expanded=False):
        st.caption("Use this to compare what Cortex saw in the review tables against the candidate studies it parsed.")
        for document in documents:
            review_title = str(document.get("review_title") or document.get("source_name") or "Review").strip()
            source_name = str(document.get("source_name") or review_title).strip()
            review_candidates = [item for item in candidates if str(item.get("source_review") or item.get("source_name") or "").strip() == source_name]
            table_blocks = list(document.get("table_blocks") or [])
            table_snapshots = list(document.get("table_snapshots") or [])
            label = f"{review_title} ({len(review_candidates)} parsed candidate(s))"
            with st.container(border=True):
                st.markdown(f"**{label}**")
                left, right = st.columns([1, 1])

                with left:
                    st.caption("Detected table snapshots")
                    if table_snapshots:
                        for shot in table_snapshots:
                            caption = f"Page {int(shot.get('page_number') or 0)}"
                            if str(shot.get("text_sample") or "").strip():
                                caption += f" • {str(shot.get('text_sample') or '').strip()}"
                            st.image(shot.get("image_bytes"), caption=caption, use_container_width=True)
                    else:
                        st.info("No PDF table snapshot available for this review.")

                with right:
                    st.caption("Extracted markdown tables")
                    if table_blocks:
                        for block in table_blocks[:4]:
                            st.markdown(f"Table {int(block.get('table_index') or 0)} • {int(block.get('row_count') or 0)} row(s)")
                            if str(block.get("context_text") or "").strip():
                                st.caption("Nearby table context")
                                st.code(str(block.get("context_text") or ""), language="markdown")
                            st.code(str(block.get("markdown") or ""), language="markdown")
                    else:
                        st.info("No markdown table block detected in extracted review text.")

                st.caption("Parsed candidate rows from this review")
                if review_candidates:
                    st.dataframe(
                        _study_miner_candidate_rows(review_candidates),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No candidate rows parsed from this review.")


def _study_miner_candidate_key(item: Dict[str, Any]) -> str:
    extras = dict(item.get("extra_fields") or {})
    return " | ".join(
        [
            str(item.get("source_review") or item.get("source_name") or "").strip().lower(),
            str(extras.get("table_index") or "").strip(),
            str(item.get("doi") or "").strip().lower(),
            re.sub(r"\s+", " ", str(item.get("title") or "").strip().lower()),
            re.sub(r"\s+", " ", str(item.get("authors") or "").strip().lower()),
            str(item.get("year") or "").strip(),
            str(item.get("reference_number") or "").strip(),
        ]
    )


def _annotate_study_miner_slice_candidates(
    candidates: List[Dict[str, Any]],
    *,
    table_index: int,
    table_label: str,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    normalized_table_index = int(table_index or 0)
    normalized_table_label = str(table_label or "").strip()
    for item in candidates or []:
        candidate = dict(item)
        extras = dict(candidate.get("extra_fields") or {})
        extras["table_index"] = normalized_table_index
        extras["table_label"] = normalized_table_label
        candidate["extra_fields"] = extras
        annotated.append(candidate)
    return annotated


def _study_miner_group_looks_like_citation(study_value: str) -> bool:
    text = str(study_value or "").strip()
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if re.search(r"\b(?:19|20)\d{2}\b", text) and re.search(r"\[\d{1,3}\]", text):
        return True
    if " et al" in normalized:
        return True
    if normalized.startswith("clinicaltrials gov"):
        return True
    return False


def _study_miner_candidate_reference_key(item: Dict[str, Any]) -> str:
    extras = dict(item.get("extra_fields") or {})
    reference_number = str(item.get("reference_number") or extras.get("reference_number") or "").strip()
    return " | ".join(
        [
            _normalize_text(str(item.get("source_review") or item.get("source_name") or "")),
            str(extras.get("table_index") or "").strip(),
            reference_number,
            "" if reference_number else _normalize_text(str(item.get("raw_citation") or item.get("title") or "")),
        ]
    )


def _study_miner_candidate_score(item: Dict[str, Any]) -> int:
    extras = dict(item.get("extra_fields") or {})
    score = int(item.get("relevance_score") or 0)
    for field_name in ("reference_number", "title", "authors", "year"):
        if str(item.get(field_name) or "").strip():
            score += 1
    for field_name in (
        "study_design",
        "patient_population",
        "treatment",
        "followup_times_assessed",
        "assessment_method",
        "mapped_utility_measure",
    ):
        if str(extras.get(field_name) or "").strip():
            score += 1
    source_section = str(item.get("source_section") or "").strip()
    if "local_rescue" in source_section:
        score += 3
    if "local_reconciliation" in source_section:
        score += 1
    if str(extras.get("table_index") or "").strip():
        score += 1
    if str(extras.get("author_year") or "").strip().find("[") >= 0:
        score += 1
    if _study_miner_group_looks_like_citation(str(extras.get("study") or "")):
        score -= 4
    else:
        score += 2
    return score


def _merge_study_miner_candidate_records(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    preferred = incoming if _study_miner_candidate_score(incoming) > _study_miner_candidate_score(merged) else merged
    fallback = merged if preferred is incoming else incoming

    for field_name in (
        "source_section",
        "reference_match_method",
        "reference_validation",
        "review_warning",
    ):
        values: List[str] = []
        for raw in (merged.get(field_name, ""), incoming.get(field_name, "")):
            text = str(raw or "").strip()
            if not text:
                continue
            for part in [piece.strip() for piece in re.split(r"\s*;\s*|\s*,\s*", text) if piece.strip()]:
                if part not in values:
                    values.append(part)
        merged[field_name] = "; ".join(values)

    for field_name in ("needs_review", "meets_criteria"):
        merged[field_name] = bool(merged.get(field_name)) or bool(incoming.get(field_name))
    merged["relevance_score"] = max(int(merged.get("relevance_score") or 0), int(incoming.get("relevance_score") or 0))
    merged["design_matches"] = sorted(dict.fromkeys(list(merged.get("design_matches") or []) + list(incoming.get("design_matches") or [])))
    merged["outcome_matches"] = sorted(dict.fromkeys(list(merged.get("outcome_matches") or []) + list(incoming.get("outcome_matches") or [])))

    for field_name in (
        "title",
        "authors",
        "year",
        "doi",
        "journal",
        "raw_citation",
        "raw_excerpt",
        "reference_number",
        "source_review",
        "source_name",
        "source_review_title",
    ):
        merged[field_name] = preferred.get(field_name) or fallback.get(field_name) or merged.get(field_name)

    merged["row_id"] = preferred.get("row_id") or fallback.get("row_id") or merged.get("row_id")

    extras = dict(fallback.get("extra_fields") or {})
    preferred_extras = dict(preferred.get("extra_fields") or {})
    for field_name in (
        "region",
        "study",
        "author_year",
        "study_design",
        "patient_population",
        "treatment",
        "followup_times_assessed",
        "assessment_method",
        "scale",
        "baseline_mean_sd",
        "mean_change_last_followup_sd",
        "mapped_utility_measure",
        "table_index",
        "table_label",
        "source_review",
        "source_review_title",
        "source_doc_id",
    ):
        extras[field_name] = preferred_extras.get(field_name) or extras.get(field_name) or ""
    merged["extra_fields"] = extras
    return merged


def _dedupe_study_miner_candidates_by_table_reference(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    for item in candidates or []:
        candidate = dict(item)
        key = _study_miner_candidate_reference_key(candidate)
        if not key.strip(" |"):
            deduped.append(candidate)
            continue
        existing_idx = index_by_key.get(key)
        if existing_idx is None:
            deduped.append(candidate)
            index_by_key[key] = len(deduped) - 1
            continue
        deduped[existing_idx] = _merge_study_miner_candidate_records(deduped[existing_idx], candidate)
    return _filter_study_miner_candidates(deduped)


def _study_label_is_generic_bucket(study_value: str) -> bool:
    normalized = _normalize_text(study_value)
    return normalized in {"cua", "cea", "tto", "all tables", "table block", "global", "us", "uk", "nr"}


def _study_label_looks_specific_trial(study_value: str) -> bool:
    study_text = str(study_value or "").strip()
    normalized = _normalize_text(study_text)
    if not normalized or _study_label_is_generic_bucket(study_text) or _study_miner_group_looks_like_citation(study_text):
        return False
    if re.search(r"\d", study_text):
        return True
    upper_ratio = sum(1 for ch in study_text if ch.isupper()) / max(1, sum(1 for ch in study_text if ch.isalpha()))
    return upper_ratio >= 0.45 or len(normalized.split()) >= 2


def _normalize_study_miner_study_label(study_value: str) -> str:
    study_text = re.sub(r"\s+", " ", str(study_value or "").strip())
    if not study_text:
        return ""

    def _normalize_token(token: str) -> str:
        alpha_chars = [ch for ch in token if ch.isalpha()]
        if len(alpha_chars) < 6:
            return token
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / max(1, len(alpha_chars))
        if upper_ratio < 0.75:
            return token
        return re.sub(r"([A-Z])\1+", r"\1", token)

    return " ".join(_normalize_token(part) for part in study_text.split())


def _study_miner_split_group(row: Dict[str, Any]) -> tuple[str, str]:
    region = str(row.get("region") or "").strip()
    study = str(row.get("study") or "").strip()
    if region or study:
        return region, study
    group_text = str(row.get("table_group") or "").strip()
    if " / " in group_text:
        region_part, study_part = group_text.split(" / ", 1)
        return region_part.strip(), study_part.strip()
    return "", group_text


def _study_miner_group_quality(study_value: str) -> int:
    study_text = str(study_value or "").strip()
    compact = re.sub(r"[^a-z0-9]+", "", _normalize_text(study_text))
    score = len(compact)
    if re.search(r"\d", study_text):
        score += 6
    if len(study_text.split()) >= 2:
        score += 3
    if _study_label_looks_specific_trial(study_text):
        score += 4
    return score


def _study_miner_harmonize_group_labels(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_rows = [dict(item) for item in rows or []]
    group_refs: Dict[tuple[str, str, str], set[str]] = {}
    label_lookup: Dict[tuple[str, str, str], str] = {}

    for row in normalized_rows:
        region, study = _study_miner_split_group(row)
        normalized_study = _normalize_study_miner_study_label(study)
        if region or normalized_study:
            row["region"] = region
            row["study"] = normalized_study or study
            row["table_group"] = " / ".join(part for part in [region, row["study"]] if part)
        key = (
            str(row.get("source_review_title") or "").strip(),
            _normalize_text(region),
            _normalize_text(str(row.get("study") or study or "").strip()),
        )
        if not key[2]:
            continue
        label_lookup[key] = str(row.get("study") or study or "").strip()
        reference_number = str(row.get("reference_number") or "").strip()
        if reference_number:
            group_refs.setdefault(key, set()).add(reference_number)

    replacements: Dict[tuple[str, str, str], str] = {}
    grouped_by_review_region: Dict[tuple[str, str], List[tuple[str, set[str], str]]] = {}
    for key, refs in group_refs.items():
        review_title, region_key, study_key = key
        grouped_by_review_region.setdefault((review_title, region_key), []).append(
            (study_key, refs, label_lookup.get(key, ""))
        )

    for (review_title, region_key), study_items in grouped_by_review_region.items():
        for weak_key, weak_refs, weak_label in study_items:
            weak_compact = re.sub(r"[^a-z0-9]+", "", weak_key)
            if len(weak_compact) < 5:
                continue
            best_label = ""
            best_score = -1
            for strong_key, strong_refs, strong_label in study_items:
                if strong_key == weak_key:
                    continue
                if not (weak_refs & strong_refs):
                    continue
                strong_compact = re.sub(r"[^a-z0-9]+", "", strong_key)
                if not strong_compact.startswith(weak_compact):
                    continue
                if len(strong_compact) <= len(weak_compact):
                    continue
                score = _study_miner_group_quality(strong_label)
                if score > best_score:
                    best_score = score
                    best_label = strong_label
            if best_label:
                replacements[(review_title, region_key, weak_key)] = best_label

    if not replacements:
        return normalized_rows

    adjusted: List[Dict[str, Any]] = []
    for row in normalized_rows:
        region, study = _study_miner_split_group(row)
        key = (
            str(row.get("source_review_title") or "").strip(),
            _normalize_text(region),
            _normalize_text(study),
        )
        replacement = replacements.get(key)
        if replacement:
            row["region"] = region
            row["study"] = replacement
            row["table_group"] = " / ".join(part for part in [region, replacement] if part)
        adjusted.append(row)
    return adjusted


def _study_miner_treatment_tokens(text: str) -> set[str]:
    stopwords = {
        "adult",
        "adults",
        "and",
        "autologous",
        "based",
        "car",
        "cell",
        "cells",
        "chemotherapy",
        "clinical",
        "general",
        "health",
        "hypothetical",
        "in",
        "line",
        "lines",
        "of",
        "pathway",
        "patients",
        "population",
        "prior",
        "r",
        "r",
        "refractory",
        "relapsed",
        "salvage",
        "study",
        "systemic",
        "therapy",
        "treatment",
        "trial",
        "with",
    }
    return {
        token
        for token in _normalize_text(text).split()
        if token and token not in stopwords and len(token) > 2
    }


def _study_miner_row_looks_economic(candidate: Dict[str, Any]) -> bool:
    extras = dict(candidate.get("extra_fields") or {})
    haystack = " | ".join(
        [
            str(extras.get("study_design") or ""),
            str(extras.get("patient_population") or ""),
            str(extras.get("treatment") or ""),
            str(extras.get("assessment_method") or ""),
            str(candidate.get("raw_excerpt") or ""),
        ]
    )
    normalized = _normalize_text(haystack)
    return any(
        marker in normalized
        for marker in (
            "vignette",
            "general adult population",
            "hypothetical",
            "tto",
            "time trade off",
            "disutility",
        )
    )


def _study_miner_relabel_outlier_group_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for item in candidates or []:
        extras = dict(item.get("extra_fields") or {})
        table_index = str(extras.get("table_index") or "").strip()
        study = str(extras.get("study") or "").strip()
        grouped.setdefault((table_index, _normalize_text(study)), []).append(dict(item))

    adjusted: List[Dict[str, Any]] = []
    for (_table_index, _study_key), group_items in grouped.items():
        study_label = str(dict(group_items[0].get("extra_fields") or {}).get("study") or "").strip()
        if len(group_items) < 3 or not _study_label_looks_specific_trial(study_label):
            adjusted.extend(group_items)
            continue

        token_counts: Dict[str, int] = {}
        supported_rows = 0
        for item in group_items:
            if _study_miner_row_looks_economic(item):
                continue
            extras = dict(item.get("extra_fields") or {})
            tokens = _study_miner_treatment_tokens(str(extras.get("treatment") or ""))
            if not tokens:
                continue
            supported_rows += 1
            for token in tokens:
                token_counts[token] = int(token_counts.get(token) or 0) + 1
        dominant_tokens = {
            token
            for token, count in token_counts.items()
            if count >= max(2, (supported_rows + 1) // 2)
        }

        for item in group_items:
            candidate = dict(item)
            extras = dict(candidate.get("extra_fields") or {})
            author_year = str(extras.get("author_year") or candidate.get("raw_citation") or "").strip()
            treatment_tokens = _study_miner_treatment_tokens(str(extras.get("treatment") or ""))
            relabel_reason = ""
            replacement_study = ""
            if _study_miner_row_looks_economic(candidate):
                relabel_reason = "row looks like economic/vignette evidence rather than the trial group"
                assessment = _normalize_text(str(extras.get("assessment_method") or ""))
                replacement_study = "TTO" if "tto" in assessment or "time trade off" in assessment else author_year
            elif dominant_tokens and treatment_tokens and dominant_tokens.isdisjoint(treatment_tokens):
                relabel_reason = "treatment does not match dominant trial-group therapy"
                replacement_study = author_year

            if relabel_reason and replacement_study:
                extras["study"] = replacement_study
                candidate["extra_fields"] = extras
                candidate["needs_review"] = True
                warning = str(candidate.get("review_warning") or "").strip()
                candidate["review_warning"] = "; ".join(
                    part for part in [warning, relabel_reason] if part
                )
            adjusted.append(candidate)

    return _dedupe_study_miner_candidates_by_table_reference(adjusted)


def _reassign_candidates_to_matching_table_slices(
    candidates: List[Dict[str, Any]],
    table_slices: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    slice_kind_by_index: Dict[int, str] = {
        int(item.get("table_index") or 0): _classify_table_slice_kind(item)
        for item in table_slices or []
        if int(item.get("table_index") or 0) > 0
    }
    economic_slice_indices = [idx for idx, kind in slice_kind_by_index.items() if kind == "economic"]
    if not economic_slice_indices:
        return candidates
    target_economic_index = min(economic_slice_indices)
    target_economic_label = f"table {target_economic_index}"

    adjusted: List[Dict[str, Any]] = []
    for item in candidates or []:
        candidate = dict(item)
        extras = dict(candidate.get("extra_fields") or {})
        table_index = int(extras.get("table_index") or 0)
        current_kind = slice_kind_by_index.get(table_index, "")
        candidate_is_economic = _study_miner_row_looks_economic(candidate) or _study_label_is_generic_bucket(str(extras.get("study") or ""))
        if candidate_is_economic and current_kind and current_kind != "economic":
            extras["table_index"] = target_economic_index
            extras["table_label"] = target_economic_label
            candidate["extra_fields"] = extras
        adjusted.append(candidate)
    return adjusted


def _merge_study_miner_candidates(
    base_candidates: List[Dict[str, Any]],
    rescue_candidates: List[Dict[str, Any]],
    *,
    source_name: str,
    review_title: str,
    source_doc_id: str,
) -> List[Dict[str, Any]]:
    merged = [dict(item) for item in base_candidates or []]
    seen = {_study_miner_candidate_key(item) for item in merged}
    next_row_id = max([int(item.get("row_id") or 0) for item in merged] + [0]) + 1

    for item in rescue_candidates or []:
        normalized = dict(item)
        normalized["row_id"] = next_row_id
        normalized["source_name"] = source_name
        normalized["source_review"] = source_name
        normalized["source_review_title"] = review_title
        extras = dict(normalized.get("extra_fields") or {})
        extras["source_review"] = source_name
        extras["source_review_title"] = review_title
        extras["source_doc_id"] = source_doc_id
        normalized["extra_fields"] = extras
        key = _study_miner_candidate_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
        next_row_id += 1
    return _filter_study_miner_candidates(merged)


def _filter_base_table_candidates_after_local_rescue(
    candidates: List[Dict[str, Any]],
    *,
    review_source_name: str,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for item in candidates or []:
        source_review = str(item.get("source_review") or item.get("source_name") or "").strip()
        source_section = str(item.get("source_section") or "").strip()
        if source_review == review_source_name and source_section == "table":
            continue
        filtered.append(dict(item))
    return filtered


def _study_miner_export_rows(editor_rows: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_row_id = {str(item.get("row_id")): dict(item) for item in candidates or []}
    exported: List[Dict[str, Any]] = []
    for row in editor_rows or []:
        row_id = str(row.get("row_id") or "").strip()
        base = dict(by_row_id.get(row_id) or {})
        extras = dict(base.get("extra_fields") or {})
        structural_probe = {
            "title": row.get("title", base.get("title", "")),
            "raw_citation": base.get("raw_citation", row.get("table_citation", "")),
            "extra_fields": {
                "study": extras.get("study", ""),
                "author_year": extras.get("author_year", row.get("table_citation", "")),
            },
        }
        if _is_structural_study_miner_candidate(structural_probe):
            continue
        reference_number = str(row.get("reference_number", base.get("reference_number", extras.get("reference_number", ""))) or "").strip()
        reference_match_method = str(
            row.get("reference_match_method", base.get("reference_match_method", extras.get("reference_match_method", ""))) or ""
        ).strip()
        if not reference_number:
            inline_numbers = _reference_number_pointers(
                " | ".join(
                    [
                        str(row.get("table_citation", "")),
                        str(base.get("raw_citation", "")),
                        str(row.get("title", "")),
                    ]
                )
            )
            if inline_numbers:
                reference_number = str(inline_numbers[0] or "").strip()
                reference_match_method = reference_match_method or "table_inline_reference"
        reference_link = str(row.get("reference_link", "") or "").strip()
        if not reference_link and reference_number and reference_match_method:
            reference_link = f"ref {reference_number} ({reference_match_method})"
        exported.append(
            {
                "keep": bool(row.get("keep", True)),
                "row_id": row.get("row_id"),
                "source_review_title": row.get("source_review_title", ""),
                "table_index": int(row.get("table_index") or extras.get("table_index") or 0),
                "table_label": str(row.get("table_label") or extras.get("table_label") or "").strip(),
                "table_group": row.get("table_group", ""),
                "table_citation": row.get("table_citation", ""),
                "title": row.get("title", ""),
                "authors": row.get("authors", ""),
                "year": row.get("year", ""),
                "doi": row.get("doi", ""),
                "journal": row.get("journal", ""),
                "source_section": row.get("source_section", ""),
                "matches": row.get("matches", ""),
                "score": row.get("score", ""),
                "reference_number": reference_number,
                "reference_match_method": reference_match_method,
                "reference_validation": row.get("reference_validation", base.get("reference_validation", "")),
                "reference_link": reference_link,
                "needs_review": row.get("needs_review", ""),
                "review_warning": row.get("review_warning", ""),
                "design_matches": row.get("design_matches", ""),
                "outcome_matches": row.get("outcome_matches", ""),
                "region": extras.get("region", ""),
                "study": extras.get("study", ""),
                "study_design": extras.get("study_design", ""),
                "patient_population": extras.get("patient_population", ""),
                "treatment": extras.get("treatment", ""),
                "followup_times_assessed": extras.get("followup_times_assessed", ""),
                "assessment_method": extras.get("assessment_method", ""),
                "scale": extras.get("scale", ""),
                "baseline_mean_sd": extras.get("baseline_mean_sd", ""),
                "mean_change_last_followup_sd": extras.get("mean_change_last_followup_sd", ""),
                "mapped_utility_measure": extras.get("mapped_utility_measure", ""),
                "raw_citation": base.get("raw_citation", ""),
                "raw_excerpt": base.get("raw_excerpt", ""),
            }
        )
    return _study_miner_harmonize_group_labels(_dedupe_study_miner_export_rows(exported))


def _study_miner_export_key(row: Dict[str, Any]) -> str:
    reference_number = str(row.get("reference_number") or "").strip()
    return " | ".join(
        [
            _normalize_text(str(row.get("source_review_title") or "")),
            str(row.get("table_index") or "").strip(),
            reference_number,
            "" if reference_number else _normalize_text(str(row.get("table_citation") or row.get("title") or "")),
        ]
    )


def _study_miner_export_score(row: Dict[str, Any]) -> int:
    score = 0
    for field_name in (
        "reference_number",
        "authors",
        "year",
        "study_design",
        "patient_population",
        "treatment",
        "followup_times_assessed",
        "assessment_method",
        "baseline_mean_sd",
        "mean_change_last_followup_sd",
        "mapped_utility_measure",
    ):
        if str(row.get(field_name) or "").strip():
            score += 1
    if "[" in str(row.get("table_citation") or "") and "]" in str(row.get("table_citation") or ""):
        score += 2
    if "local_rescue" in str(row.get("source_section") or "").strip():
        score += 2
    if "local_reconciliation" in str(row.get("source_section") or "").strip():
        score += 1
    if str(row.get("table_index") or "").strip():
        score += 1
    if _study_miner_group_looks_like_citation(str(row.get("study") or row.get("table_group") or "")):
        score -= 4
    else:
        score += 2
    return score


def _merge_study_miner_export_rows(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for field_name in (
        "source_section",
        "reference_match_method",
        "reference_validation",
        "review_warning",
        "design_matches",
        "outcome_matches",
    ):
        values: List[str] = []
        for raw in (merged.get(field_name, ""), incoming.get(field_name, "")):
            text = str(raw or "").strip()
            if not text:
                continue
            for part in [piece.strip() for piece in re.split(r"\s*;\s*|\s*,\s*", text) if piece.strip()]:
                if part not in values:
                    values.append(part)
        merged[field_name] = "; ".join(values)

    for field_name in (
        "keep",
        "needs_review",
    ):
        merged[field_name] = bool(merged.get(field_name)) or bool(incoming.get(field_name))

    preferred = incoming if _study_miner_export_score(incoming) > _study_miner_export_score(merged) else merged
    fallback = merged if preferred is incoming else incoming

    for field_name in (
        "row_id",
        "source_review_title",
        "table_label",
        "table_group",
        "table_citation",
        "title",
        "authors",
        "year",
        "doi",
        "journal",
        "matches",
        "score",
        "reference_number",
        "reference_link",
        "region",
        "study",
        "study_design",
        "patient_population",
        "treatment",
        "followup_times_assessed",
        "assessment_method",
        "scale",
        "baseline_mean_sd",
        "mean_change_last_followup_sd",
        "mapped_utility_measure",
        "raw_citation",
        "raw_excerpt",
    ):
        merged[field_name] = str(preferred.get(field_name) or fallback.get(field_name) or "").strip()

    merged["table_index"] = int(preferred.get("table_index") or fallback.get("table_index") or 0)

    return merged


def _group_study_miner_export_rows_by_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in rows or []:
        review_title = str(row.get("source_review_title") or "").strip()
        table_index = int(row.get("table_index") or 0)
        table_label = str(row.get("table_label") or "").strip()
        key = (review_title, table_index, table_label)
        grouped.setdefault(key, []).append(dict(row))
    ordered: List[Dict[str, Any]] = []
    for (review_title, table_index, table_label), items in sorted(
        grouped.items(),
        key=lambda item: (
            _normalize_text(item[0][0]),
            int(item[0][1] or 0),
            _normalize_text(item[0][2]),
        ),
    ):
        label = table_label or (f"table {table_index}" if table_index > 0 else "all tables")
        ordered.append(
            {
                "source_review_title": review_title,
                "table_index": table_index,
                "table_label": label,
                "rows": items,
            }
        )
    return ordered


def _study_miner_export_group_is_low_value(group: Dict[str, Any]) -> bool:
    rows = [dict(item) for item in list(group.get("rows") or []) if isinstance(item, dict)]
    if not rows:
        return True
    reference_count = sum(1 for item in rows if str(item.get("reference_number") or "").strip())
    if reference_count > 0:
        return False
    return True


def _filter_study_miner_export_groups(
    groups: List[Dict[str, Any]],
    *,
    include_low_value: bool = False,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for group in groups or []:
        low_value = _study_miner_export_group_is_low_value(group)
        enriched = dict(group)
        enriched["low_value"] = low_value
        if low_value and not include_low_value:
            continue
        filtered.append(enriched)
    return filtered


def _dedupe_study_miner_export_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    for row in rows or []:
        normalized = dict(row)
        if _is_structural_study_miner_candidate(
            {
                "title": normalized.get("title", ""),
                "raw_citation": normalized.get("raw_citation", normalized.get("table_citation", "")),
                "extra_fields": {
                    "study": normalized.get("study", ""),
                    "author_year": normalized.get("table_citation", ""),
                },
            }
        ):
            continue
        if not str(normalized.get("table_group") or "").strip() and not str(normalized.get("reference_number") or "").strip():
            continue
        key = _study_miner_export_key(normalized)
        existing_idx = index_by_key.get(key)
        if existing_idx is None:
            deduped.append(normalized)
            index_by_key[key] = len(deduped) - 1
            continue
        deduped[existing_idx] = _merge_study_miner_export_rows(deduped[existing_idx], normalized)
    return deduped


def _refresh_study_miner_stats(
    candidates: List[Dict[str, Any]],
    base_stats: Dict[str, Any],
    cloud_rescue_runs: List[Dict[str, Any]],
    local_rescue_runs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    stats = dict(base_stats or {})
    stats["candidates_total"] = len(candidates)
    stats["candidates_matching"] = sum(1 for item in candidates if item.get("meets_criteria"))
    stats["table_candidates"] = sum(1 for item in candidates if item.get("source_section") == "table")
    stats["reference_candidates"] = sum(1 for item in candidates if item.get("source_section") == "references")
    stats["table_reference_links"] = sum(
        1 for item in candidates if item.get("source_section") == "table" and str(item.get("reference_match_method") or "").strip()
    )
    stats["needs_review"] = sum(1 for item in candidates if item.get("needs_review"))
    stats["reference_mismatches"] = sum(1 for item in candidates if str(item.get("reference_validation") or "") == "mismatch")
    stats["local_rescue_candidates"] = sum(len(item.get("candidates") or []) for item in local_rescue_runs or [])
    stats["cloud_rescue_candidates"] = sum(len(item.get("candidates") or []) for item in cloud_rescue_runs or [])
    return stats


def _study_miner_should_use_vision(selected: Any, vision_mode: str) -> bool:
    mode = str(vision_mode or "").strip().lower()
    if mode.startswith("on"):
        return True
    if mode.startswith("off"):
        return False
    file_paths = selected if isinstance(selected, list) else [selected]
    suffixes = {Path(str(item)).suffix.lower() for item in file_paths if str(item or "").strip()}
    return ".pdf" in suffixes


def _study_miner_cloud_rescue_recommended(result: Dict[str, Any], documents: List[Dict[str, Any]]) -> bool:
    stats = dict(result.get("stats") or {})
    candidates_total = int(stats.get("candidates_total") or 0)
    matching = int(stats.get("candidates_matching") or 0)
    links = int(stats.get("table_reference_links") or 0)
    table_blocks = sum(len(item.get("table_blocks") or []) for item in documents or [] if bool(item.get("confirm_review")))
    snapshots = sum(len(item.get("table_snapshots") or []) for item in documents or [] if bool(item.get("confirm_review")))
    if candidates_total >= 8 and matching == 0:
        return True
    if table_blocks >= 2 and snapshots == 0:
        return True
    if candidates_total >= 12 and links == 0:
        return True
    return False


def _study_miner_table_slices(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    table_blocks = [dict(item) for item in list(document.get("table_blocks") or []) if isinstance(item, dict)]
    table_snapshots = [dict(item) for item in list(document.get("table_snapshots") or []) if isinstance(item, dict)]
    if not table_blocks:
        return [{"label": "all tables", "table_index": 0, "table_blocks": [], "table_snapshots": table_snapshots}]

    def _table_block_context_number(block: Dict[str, Any]) -> int:
        haystack = " ".join(
            [
                str(block.get("context_before") or ""),
                str(block.get("context_text") or ""),
                str(block.get("context_after") or ""),
            ]
        )
        match = re.search(r"\btable\s+(\d{1,3})\b", haystack, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _table_block_header_signature(block: Dict[str, Any]) -> str:
        header = list(block.get("header") or [])
        text = " | ".join(str(item or "") for item in header[:3])
        return _normalize_text(text)

    def _table_block_is_continuation(block: Dict[str, Any]) -> bool:
        haystack = " ".join(
            [
                str(block.get("context_before") or ""),
                str(block.get("context_text") or ""),
                str(block.get("context_after") or ""),
            ]
        )
        return "continued" in _normalize_text(haystack)

    def _resolve_table_family_numbers(blocks: List[Dict[str, Any]]) -> List[int]:
        numbers = [_table_block_context_number(block) for block in blocks]
        signatures = [_table_block_header_signature(block) for block in blocks]
        continuations = [_table_block_is_continuation(block) for block in blocks]
        resolved = list(numbers)
        for idx in range(len(blocks)):
            if resolved[idx] > 0:
                continue
            if idx > 0 and resolved[idx - 1] > 0 and (
                signatures[idx] == signatures[idx - 1] or continuations[idx]
            ):
                resolved[idx] = resolved[idx - 1]
                continue
            if idx + 1 < len(blocks) and resolved[idx + 1] > 0 and signatures[idx] == signatures[idx + 1]:
                resolved[idx] = resolved[idx + 1]
        next_synthetic = max([value for value in resolved if value > 0] + [0]) + 1
        for idx in range(len(resolved)):
            if resolved[idx] > 0:
                continue
            if idx > 0 and signatures[idx] and signatures[idx] == signatures[idx - 1]:
                resolved[idx] = resolved[idx - 1]
                continue
            resolved[idx] = next_synthetic
            next_synthetic += 1
        return resolved

    family_numbers = _resolve_table_family_numbers(table_blocks)
    family_blocks: Dict[int, List[Dict[str, Any]]] = {}
    for block, family_number in zip(table_blocks, family_numbers):
        family_blocks.setdefault(int(family_number), []).append(dict(block))

    slices: List[Dict[str, Any]] = []
    for family_number, blocks in family_blocks.items():
        block_indices = {int(item.get("table_index") or 0) for item in blocks}
        block_snapshots = [
            dict(item)
            for item in table_snapshots
            if int(item.get("table_index") or 0) in block_indices or int(item.get("table_index") or 0) == family_number
        ]
        label = f"table {family_number}" if family_number > 0 else "table block"
        slices.append(
            {
                "label": label,
                "table_index": int(family_number),
                "block_count": len(blocks),
                "source_block_indices": sorted(block_indices),
                "table_blocks": blocks,
                "table_snapshots": block_snapshots,
            }
        )
    if not slices:
        return [{"label": "all tables", "table_index": 0, "table_blocks": table_blocks, "table_snapshots": table_snapshots}]
    return slices


def _estimate_slice_page_numbers(pdf_path: str, table_slices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_slices = [dict(item) for item in table_slices or []]
    if not normalized_slices:
        return []
    if not str(pdf_path or "").strip().lower().endswith(".pdf"):
        return normalized_slices
    if any(list(item.get("table_snapshots") or []) for item in normalized_slices):
        return normalized_slices

    try:
        import fitz

        with fitz.open(str(pdf_path)) as doc:
            total_pages = int(len(doc) or 0)
    except Exception:
        return normalized_slices

    if total_pages <= 0:
        return normalized_slices

    total_weight = sum(max(1, int(item.get("block_count") or len(list(item.get("table_blocks") or [])) or 1)) for item in normalized_slices)
    allow_overlap = len(normalized_slices) == 1
    consumed = 0
    for idx, item in enumerate(normalized_slices):
        weight = max(1, int(item.get("block_count") or len(list(item.get("table_blocks") or [])) or 1))
        start_ratio = consumed / float(total_weight)
        consumed += weight
        end_ratio = consumed / float(total_weight)
        start_page = int(start_ratio * total_pages) + 1
        end_page = max(start_page, int((end_ratio * total_pages) + 0.9999))
        if allow_overlap and idx > 0:
            start_page = max(1, start_page - 1)
        if allow_overlap and idx < len(normalized_slices) - 1:
            end_page = min(total_pages, end_page + 1)
        item["heuristic_page_numbers"] = list(range(start_page, min(total_pages, end_page) + 1))
    return normalized_slices


def _slice_has_explicit_table_context(table_slice: Dict[str, Any]) -> bool:
    table_index = int(table_slice.get("table_index") or 0)
    if table_index <= 0:
        return True
    pattern = re.compile(rf"\btable\s+{table_index}\b", flags=re.IGNORECASE)
    for block in list(table_slice.get("table_blocks") or []):
        haystack = " ".join(
            [
                str(block.get("context_before") or ""),
                str(block.get("context_text") or ""),
                str(block.get("context_after") or ""),
            ]
        )
        if pattern.search(haystack):
            return True
    return False


def _classify_table_slice_kind(table_slice: Dict[str, Any]) -> str:
    context_text = " ".join(
        " ".join(
            [
                str(block.get("context_before") or ""),
                str(block.get("context_text") or ""),
                str(block.get("context_after") or ""),
            ]
        )
        for block in list(table_slice.get("table_blocks") or [])
    )
    header_text = " ".join(
        " ".join(str(item or "") for item in list(block.get("header") or []))
        for block in list(table_slice.get("table_blocks") or [])
    )
    normalized_context = _normalize_text(context_text)
    normalized_headers = _normalize_text(header_text)
    if any(
        marker in normalized_context
        for marker in (
            "economic studies",
            "hta report",
            "hta reports",
            "cost utility",
            "cost effectiveness",
        )
    ) or any(
        marker in normalized_headers
        for marker in (
            "cua",
            "cea",
            "tto",
            "source of utility values",
            "utility values reported",
            "utility values in remission",
        )
    ):
        return "economic"
    if any(marker in normalized_context for marker in ("hrqol", "quality of life")) or any(
        marker in normalized_headers
        for marker in (
            "fact",
            "eortc",
            "sf 36",
            "eq 5d",
            "author year",
            "utilities",
        )
    ):
        return "hrqol"
    return ""


def _run_study_miner_cloud_rescue(
    result: Dict[str, Any],
    documents: List[Dict[str, Any]],
    *,
    design_query: str,
    outcome_query: str,
) -> Dict[str, Any]:
    rescue_runs: List[Dict[str, Any]] = []
    merged_candidates = list(result.get("candidates") or [])
    for document in documents:
        if not bool(document.get("confirm_review")):
            continue
        rescue_result = run_claude_table_rescue(
            review_title=str(document.get("review_title") or document.get("source_name") or "").strip(),
            design_query=design_query,
            outcome_query=outcome_query,
            table_snapshots=list(document.get("table_snapshots") or []),
            table_blocks=list(document.get("table_blocks") or []),
            references_text=extract_review_reference_section(str(document.get("text") or "")),
            pdf_path=str(document.get("file_path") or ""),
        )
        rescue_runs.append(
            {
                "source_name": str(document.get("source_name") or "").strip(),
                "review_title": str(document.get("review_title") or "").strip(),
                **rescue_result,
            }
        )
        merged_candidates = _merge_study_miner_candidates(
            merged_candidates,
            list(rescue_result.get("candidates") or []),
            source_name=str(document.get("source_name") or "").strip(),
            review_title=str(document.get("review_title") or "").strip(),
            source_doc_id=str(document.get("doc_id") or ""),
        )
    updated_result = dict(result)
    updated_result["candidates"] = merged_candidates
    updated_result["cloud_rescue_runs"] = rescue_runs
    updated_result["cloud_rescue_attempted"] = True
    updated_result["stats"] = _refresh_study_miner_stats(
        merged_candidates,
        dict(result.get("stats") or {}),
        rescue_runs,
        list(updated_result.get("local_rescue_runs") or []),
    )
    return updated_result


def _run_study_miner_local_rescue(
    result: Dict[str, Any],
    documents: List[Dict[str, Any]],
    *,
    design_query: str,
    outcome_query: str,
    model: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    rescue_runs: List[Dict[str, Any]] = []
    merged_candidates = list(result.get("candidates") or [])
    for document in documents:
        if not bool(document.get("confirm_review")):
            continue
        review_name = str(document.get("review_title") or document.get("source_name") or "Review").strip()
        references_text = extract_review_reference_section(str(document.get("text") or ""))
        table_slices = _estimate_slice_page_numbers(
            str(document.get("file_path") or "").strip(),
            _study_miner_table_slices(document),
        )
        explicit_table_slices = [dict(item) for item in table_slices if _slice_has_explicit_table_context(item)]
        if explicit_table_slices:
            table_slices = explicit_table_slices
        slice_results: List[Dict[str, Any]] = []
        pdf_path = str(document.get("file_path") or "").strip()
        for table_slice in table_slices:
            slice_label = str(table_slice.get("label") or "table block").strip()
            table_index = int(table_slice.get("table_index") or 0)
            if log_callback:
                log_callback(f"{review_name}: starting local table rescue for {slice_label} with model `{model}`")
            rescue_result = run_local_table_rescue(
                review_title=review_name,
                design_query=design_query,
                outcome_query=outcome_query,
                table_snapshots=list(table_slice.get("table_snapshots") or []),
                table_blocks=list(table_slice.get("table_blocks") or []),
                references_text=references_text,
                pdf_path=pdf_path,
                page_numbers=list(table_slice.get("heuristic_page_numbers") or []),
                model=model,
            )
            rescue_candidates = _annotate_study_miner_slice_candidates(
                list(rescue_result.get("candidates") or []),
                table_index=table_index,
                table_label=slice_label,
            )
            rescue_result = {
                **rescue_result,
                "candidates": rescue_candidates,
            }
            if log_callback:
                log_callback(
                    f"{review_name}: {slice_label} table pass returned {len(rescue_result.get('candidates') or [])} grouped candidate(s) across "
                    f"{len(list(rescue_result.get('pages_used') or []))} page(s)"
                )
            final_local_result = rescue_result
            if pdf_path.lower().endswith(".pdf") and list(rescue_result.get("candidates") or []):
                if log_callback:
                    log_callback(f"{review_name}: starting full-PDF reconciliation for {slice_label}")
                reconciliation_result = run_local_table_reconciliation(
                    review_title=review_name,
                    design_query=design_query,
                    outcome_query=outcome_query,
                    provisional_candidates=list(rescue_result.get("candidates") or []),
                    references_text=references_text,
                    review_text=str(document.get("text") or ""),
                    pdf_path=pdf_path,
                    page_numbers=list(rescue_result.get("pages_used") or []),
                    preserve_group_labels=True,
                    model=model,
                )
                reconciliation_candidates = _annotate_study_miner_slice_candidates(
                    list(reconciliation_result.get("candidates") or []),
                    table_index=table_index,
                    table_label=slice_label,
                )
                reconciliation_result = {
                    **reconciliation_result,
                    "candidates": reconciliation_candidates,
                }
                if log_callback:
                    log_callback(
                        f"{review_name}: {slice_label} reconciliation returned {len(reconciliation_result.get('candidates') or [])} candidate(s) across "
                        f"{len(list(reconciliation_result.get('pages_used') or []))} page(s)"
                    )
                if list(reconciliation_result.get("candidates") or []):
                    merged_local_candidates = merge_local_table_candidates(
                        list(rescue_result.get("candidates") or []),
                        list(reconciliation_result.get("candidates") or []),
                    )
                    final_local_result = {
                        **reconciliation_result,
                        "candidates": merged_local_candidates,
                        "provisional_candidates": list(rescue_result.get("candidates") or []),
                        "reconciliation_candidates": list(reconciliation_result.get("candidates") or []),
                        "reconciled_with_full_pdf": True,
                        "reconciliation_warnings": list(reconciliation_result.get("warnings") or []),
                        "reconciliation_pages_used": list(reconciliation_result.get("pages_used") or []),
                    }
                else:
                    final_local_result = {
                        **rescue_result,
                        "reconciled_with_full_pdf": False,
                        "reconciliation_warnings": list(reconciliation_result.get("warnings") or []),
                    }
                    if log_callback:
                        log_callback(f"{review_name}: {slice_label} reconciliation did not add candidates; keeping table-pass result")
            slice_results.append(
                {
                    **final_local_result,
                    "slice_label": slice_label,
                    "table_index": table_index,
                }
            )
        merged_local_candidates = merge_local_table_candidates(*(list(item.get("candidates") or []) for item in slice_results))
        merged_local_candidates = _reassign_candidates_to_matching_table_slices(
            merged_local_candidates,
            table_slices,
        )
        merged_local_candidates = _study_miner_relabel_outlier_group_candidates(merged_local_candidates)
        merged_local_candidates = annotate_local_candidate_completeness(
            merged_local_candidates,
            references_text=references_text,
        )
        merged_local_candidates = _dedupe_study_miner_candidates_by_table_reference(merged_local_candidates)
        final_local_result = {
            "provider": "ollama",
            "model": model,
            "used_page_images": any(bool(item.get("used_page_images")) for item in slice_results),
            "pages_used": sorted(
                {
                    int(page)
                    for item in slice_results
                    for page in list(item.get("pages_used") or [])
                    if int(page or 0) > 0
                }
            ),
            "candidates": merged_local_candidates,
            "warnings": [warning for item in slice_results for warning in list(item.get("warnings") or [])],
            "raw_response": "\n\n".join(
                str(item.get("raw_response") or "").strip()
                for item in slice_results
                if str(item.get("raw_response") or "").strip()
            ),
            "provisional_candidates": [candidate for item in slice_results for candidate in list(item.get("provisional_candidates") or item.get("candidates") or [])],
            "reconciliation_candidates": [candidate for item in slice_results for candidate in list(item.get("reconciliation_candidates") or [])],
            "reconciled_with_full_pdf": any(bool(item.get("reconciled_with_full_pdf")) for item in slice_results),
            "reconciliation_warnings": [warning for item in slice_results for warning in list(item.get("reconciliation_warnings") or [])],
            "reconciliation_pages_used": sorted(
                {
                    int(page)
                    for item in slice_results
                    for page in list(item.get("reconciliation_pages_used") or [])
                    if int(page or 0) > 0
                }
            ),
            "slice_runs": slice_results,
        }
        if log_callback:
            review_needed = sum(1 for item in merged_local_candidates if item.get("needs_review"))
            log_callback(
                f"{review_name}: merged local result now has {len(merged_local_candidates)} candidate(s); "
                f"{review_needed} flagged for review"
            )
        raw_preview = str(final_local_result.get("raw_response") or "").strip()
        if log_callback and raw_preview:
            preview = raw_preview[:700].replace("\n", " ")
            log_callback(f"{review_name}: local model output preview: {preview}")
        rescue_runs.append(
            {
                "source_name": str(document.get("source_name") or "").strip(),
                "review_title": str(document.get("review_title") or "").strip(),
                **final_local_result,
            }
        )
        merged_candidates = _merge_study_miner_candidates(
            merged_candidates,
            list(final_local_result.get("candidates") or []),
            source_name=str(document.get("source_name") or "").strip(),
            review_title=str(document.get("review_title") or "").strip(),
            source_doc_id=str(document.get("doc_id") or ""),
        )
        merged_candidates = _filter_base_table_candidates_after_local_rescue(
            merged_candidates,
            review_source_name=str(document.get("source_name") or "").strip(),
        )
        merged_candidates = _dedupe_study_miner_candidates_by_table_reference(merged_candidates)
    updated_result = dict(result)
    updated_result["candidates"] = merged_candidates
    updated_result["local_rescue_runs"] = rescue_runs
    updated_result["local_rescue_attempted"] = True
    updated_result["stats"] = _refresh_study_miner_stats(
        merged_candidates,
        dict(result.get("stats") or {}),
        list(updated_result.get("cloud_rescue_runs") or []),
        rescue_runs,
    )
    return updated_result


def _extract_study_miner_documents(
    selected: Any,
    *,
    use_vision: bool,
    pdf_mode_label: str,
    docling_timeout_seconds: float,
    image_timeout_seconds: float,
    image_budget_seconds: float,
) -> List[Dict[str, Any]]:
    file_paths = selected if isinstance(selected, list) else [selected]
    file_paths = [str(item) for item in file_paths if str(item or "").strip()]
    if not file_paths:
        return []

    mode_map = {
        "Hybrid (Recommended): Docling first, Qwen enhancement, fallback on timeout": "hybrid",
        "Docling only (best layout/tables)": "docling",
        "Qwen 30B cleanup (LLM-first, no Docling)": "qwen30b",
    }
    textifier_options = {
        "use_vision": use_vision,
        "pdf_strategy": mode_map.get(pdf_mode_label, "hybrid"),
        "cleanup_provider": "lmstudio",
        "cleanup_model": "qwen2.5:32b",
        "docling_timeout_seconds": float(docling_timeout_seconds),
        "image_description_timeout_seconds": float(image_timeout_seconds),
        "image_enrich_max_seconds": float(image_budget_seconds),
    }

    progress = st.progress(0.0, text="Preparing review documents...")
    documents: List[Dict[str, Any]] = []
    total_files = max(1, len(file_paths))

    for idx, file_path in enumerate(file_paths):
        visible_name = _user_visible_filename(file_path)
        base_frac = idx / float(total_files)
        span = 1.0 / float(total_files)
        suffix = Path(file_path).suffix.lower()

        if suffix in {".md", ".txt"}:
            progress.progress(base_frac, text=f"Reading {visible_name}...")
            review_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            progress.progress(min(1.0, base_frac + span), text=f"Loaded {visible_name}")
        else:
            from cortex_engine.textifier import DocumentTextifier

            def _on_progress(frac, msg, _base=base_frac, _span=span, _name=visible_name):
                clamped = max(0.0, min(1.0, float(frac or 0.0)))
                progress.progress(min(1.0, _base + (_span * clamped)), text=f"{_name}: {msg}")

            review_text = DocumentTextifier.from_options(textifier_options, on_progress=_on_progress).textify_file(file_path)
            progress.progress(min(1.0, base_frac + span), text=f"Extracted {visible_name}")

        documents.append(
            {
                "source_name": visible_name,
                "review_title": _user_visible_stem(file_path),
                "file_path": file_path,
                "text": review_text,
                "table_blocks": extract_review_table_blocks(review_text),
                "table_snapshots": _extract_study_miner_table_snapshots(file_path, suffix),
            }
        )

    progress.progress(1.0, text=f"Prepared {len(documents)} review document(s)")
    return documents


def _extract_study_miner_table_snapshots(file_path: str, suffix: str) -> List[Dict[str, Any]]:
    if suffix != ".pdf":
        return []
    try:
        from cortex_engine.docling_reader import DoclingDocumentReader

        reader = DoclingDocumentReader(skip_vlm_processing=True)
        if not getattr(reader, "is_available", False) or not reader.can_process_file(file_path):
            return []
        docs = reader.load_data(file_path)
        if not docs:
            return []
        metadata = docs[0].metadata or {}
        provenance = metadata.get("docling_provenance") or {}
        elements = list(provenance.get("elements") or [])
        table_elements = [item for item in elements if str(item.get("type") or "").strip().lower() == "table"]
        if not table_elements:
            return []

        import fitz

        snapshots: List[Dict[str, Any]] = []
        seen: set[str] = set()
        with fitz.open(file_path) as doc:
            for idx, item in enumerate(table_elements, start=1):
                if len(snapshots) >= 6:
                    break
                bbox = item.get("bbox")
                page_raw = item.get("page")
                try:
                    page_idx = int(page_raw)
                    if page_idx >= 1:
                        page_idx -= 1
                except Exception:
                    continue
                if page_idx < 0 or page_idx >= len(doc):
                    continue
                page = doc[page_idx]
                clip = None
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        rect = fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                        rect = rect & page.rect
                        if not rect.is_empty and rect.width >= 8 and rect.height >= 8:
                            clip = rect
                    except Exception:
                        clip = None
                key = f"{page_idx}:{tuple(round(float(v), 1) for v in bbox)}" if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else f"{page_idx}:page"
                if key in seen:
                    continue
                seen.add(key)
                pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                snapshots.append(
                    {
                        "table_index": idx,
                        "page_number": page_idx + 1,
                        "bbox": list(bbox) if isinstance(bbox, (list, tuple)) else [],
                        "text_sample": str(item.get("text_sample") or "").strip(),
                        "image_bytes": pix.tobytes("png"),
                    }
                )
        return snapshots
    except Exception:
        return []

def _get_knowledge_base_files(extensions: List[str]) -> List[Path]:
    """Return files from knowledge base directories matching given extensions."""
    config_manager = ConfigManager()
    config = config_manager.get_config()

    possible_dirs = []
    if config.get("ai_database_path"):
        base_path = Path(convert_windows_to_wsl_path(config["ai_database_path"]))
        possible_dirs.extend([
            base_path / "documents",
            base_path / "source_documents",
            base_path.parent / "documents",
            base_path.parent / "source_documents",
        ])
    possible_dirs.extend([
        project_root / "documents",
        project_root / "source_documents",
        project_root / "test_documents",
    ])

    files = []
    for dir_path in possible_dirs:
        if dir_path.exists():
            for fp in dir_path.glob("**/*"):
                if fp.is_file() and fp.suffix.lower() in extensions:
                    files.append(fp)
    return files


def _file_input_widget(key_prefix: str, allowed_types: List[str], label: str = "Choose a document:"):
    """Render upload / browse KB widget. Returns selected file path or None."""
    # Use a version counter so "Clear All Files" can reset the uploader widget
    if f"{key_prefix}_upload_version" not in st.session_state:
        st.session_state[f"{key_prefix}_upload_version"] = 0
    upload_version = st.session_state[f"{key_prefix}_upload_version"]

    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Browse Knowledge Base"],
        key=f"{key_prefix}_method",
    )

    selected_file = None

    if input_method == "Upload File":
        uploaded = st.file_uploader(label, type=allowed_types,
                                    key=f"{key_prefix}_upload_v{upload_version}",
                                    accept_multiple_files=(st.session_state.get(f"{key_prefix}_batch", False)))
        if uploaded:
            files = uploaded if isinstance(uploaded, list) else [uploaded]
            accepted_files = []
            accepted_bytes = 0
            for uf in files:
                size_bytes = int(getattr(uf, "size", 0) or 0)
                if size_bytes <= 0:
                    try:
                        size_bytes = len(uf.getvalue())
                    except Exception:
                        size_bytes = 0
                if accepted_bytes + size_bytes > MAX_BATCH_UPLOAD_BYTES:
                    break
                accepted_files.append(uf)
                accepted_bytes += size_bytes
            if not accepted_files:
                st.error("Upload exceeds 1GB total limit. Select fewer/smaller files.")
                return None
            if len(accepted_files) < len(files):
                st.warning(
                    f"Maximum 1GB total upload for this batch — only the first "
                    f"{len(accepted_files)} of {len(files)} files will be processed."
                )
            files = accepted_files
            temp_dir = Path(tempfile.gettempdir()) / f"cortex_{key_prefix}"
            temp_dir.mkdir(exist_ok=True, mode=0o755)
            paths = []
            for uf in files:
                dest = str(temp_dir / f"upload_{int(time.time())}_{uf.name}")
                with open(dest, "wb") as f:
                    f.write(uf.getvalue())
                os.chmod(dest, 0o644)
                paths.append(dest)
            if len(paths) == 1:
                selected_file = paths[0]
                st.success(f"Uploaded: {files[0].name}")
            else:
                selected_file = paths  # list for batch
                st.success(f"Uploaded {len(paths)} files")
    else:
        knowledge_files = _get_knowledge_base_files([f".{t}" for t in allowed_types])
        if knowledge_files:
            names = [f"{f.name} ({f.parent.name})" for f in knowledge_files]
            idx = st.selectbox("Select document:", range(len(names)),
                               format_func=lambda x: names[x], index=None,
                               placeholder="Choose a document...", key=f"{key_prefix}_kb")
            if idx is not None:
                selected_file = str(knowledge_files[idx])
                st.success(f"Selected: {knowledge_files[idx].name}")
        else:
            st.warning("No documents found in knowledge base directories")
            st.info("Try uploading a file instead")

    return selected_file


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _user_visible_filename(file_path: str) -> str:
    """Strip internal upload prefixes from temp files for display/metadata."""
    name = Path(file_path).name
    m = re.match(r"^upload_\d+_(.+)$", name)
    return m.group(1) if m else name


def _user_visible_stem(file_path: str) -> str:
    return Path(_user_visible_filename(file_path)).stem


def _read_photo_metadata_preview(file_path: str) -> dict:
    """Read existing photo metadata fields for preview (keywords/description/location)."""
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        return {"available": False, "reason": "exiftool not found on PATH"}
    try:
        result = subprocess.run(
            [
                exiftool_path,
                "-json",
                "-XMP-dc:Subject",
                "-IPTC:Keywords",
                "-XMP-dc:Description",
                "-IPTC:Caption-Abstract",
                "-EXIF:ImageDescription",
                "-XMP-photoshop:City",
                "-IPTC:City",
                "-XMP-photoshop:State",
                "-IPTC:Province-State",
                "-XMP-photoshop:Country",
                "-IPTC:Country-PrimaryLocationName",
                "-GPSLatitude",
                "-GPSLongitude",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "reason": result.stderr.strip() or "exiftool read failed"}

        payload = json.loads(result.stdout)
        if not payload:
            return {"available": False, "reason": "No metadata found"}
        row = payload[0]

        keywords = []
        for field in ("Subject", "Keywords"):
            val = row.get(field, [])
            if isinstance(val, str):
                val = [val]
            for item in val:
                v = (item or "").strip()
                if v:
                    keywords.append(v)
        keywords = list(dict.fromkeys(keywords))

        description = (
            (row.get("Description") or "").strip()
            or (row.get("Caption-Abstract") or "").strip()
            or (row.get("ImageDescription") or "").strip()
        )
        city = (row.get("City") or "").strip()
        state = (row.get("State") or row.get("Province-State") or "").strip()
        country = (row.get("Country") or row.get("Country-PrimaryLocationName") or "").strip()

        gps_lat = row.get("GPSLatitude")
        gps_lon = row.get("GPSLongitude")
        gps = None
        if gps_lat not in (None, "") and gps_lon not in (None, ""):
            gps = f"{gps_lat}, {gps_lon}"

        return {
            "available": True,
            "keywords": keywords,
            "description": description,
            "city": city,
            "state": state,
            "country": country,
            "gps": gps,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _write_photo_metadata_quick_edit(
    file_path: str,
    keywords: List[str],
    description: str,
    city: str,
    state: str,
    country: str,
) -> dict:
    """Apply quick metadata edits (replace keywords/description/location fields)."""
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        return {"success": False, "message": "exiftool not found on PATH"}
    try:
        cmd = [
            exiftool_path,
            "-overwrite_original",
            # Clear existing keyword/caption/location fields so this acts as an explicit edit.
            "-XMP-dc:Subject=",
            "-IPTC:Keywords=",
            "-XMP-dc:Description=",
            "-IPTC:Caption-Abstract=",
            "-EXIF:ImageDescription=",
            "-IPTC:Country-PrimaryLocationName=",
            "-XMP-photoshop:Country=",
            "-IPTC:Province-State=",
            "-XMP-photoshop:State=",
            "-IPTC:City=",
            "-XMP-photoshop:City=",
        ]
        for kw in keywords:
            cmd.append(f"-XMP-dc:Subject+={kw}")
            cmd.append(f"-IPTC:Keywords+={kw}")
        desc = (description or "").strip()
        if desc:
            cmd.append(f"-XMP-dc:Description={desc}")
            cmd.append(f"-IPTC:Caption-Abstract={desc}")
            cmd.append(f"-EXIF:ImageDescription={desc}")
        if country.strip():
            cmd.append(f"-IPTC:Country-PrimaryLocationName={country.strip()}")
            cmd.append(f"-XMP-photoshop:Country={country.strip()}")
        if state.strip():
            cmd.append(f"-IPTC:Province-State={state.strip()}")
            cmd.append(f"-XMP-photoshop:State={state.strip()}")
        if city.strip():
            cmd.append(f"-IPTC:City={city.strip()}")
            cmd.append(f"-XMP-photoshop:City={city.strip()}")
        cmd.append(file_path)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return {"success": True, "message": result.stdout.strip()}
        return {"success": False, "message": result.stderr.strip() or "metadata write failed"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def _halftone_strength_label(strength: float) -> str:
    value = float(strength or 0.0)
    if value < 34:
        return "Light"
    if value < 67:
        return "Medium"
    return "Strong"


def _zoom_crop_image(image_path: str, zoom: float, focus_x: int, focus_y: int) -> Optional[Image.Image]:
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in {"RGB", "RGBA", "L"}:
                img = img.convert("RGB")
            if float(zoom) <= 1.05:
                return img.copy()

            width, height = img.size
            crop_width = max(1, int(round(width / float(zoom))))
            crop_height = max(1, int(round(height / float(zoom))))

            center_x = int(round((max(0, min(100, int(focus_x))) / 100.0) * width))
            center_y = int(round((max(0, min(100, int(focus_y))) / 100.0) * height))

            left = max(0, min(width - crop_width, center_x - crop_width // 2))
            top = max(0, min(height - crop_height, center_y - crop_height // 2))
            box = (left, top, left + crop_width, top + crop_height)
            return img.crop(box).copy()
    except Exception:
        return None


def _resize_preview_image(image: Image.Image, target_width: int = 1400) -> Image.Image:
    if image.width <= 0 or image.height <= 0:
        return image
    if image.width >= target_width:
        return image
    scale = float(target_width) / float(image.width)
    target_height = max(1, int(round(image.height * scale)))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _build_ab_window_image(
    original_image: Image.Image,
    repaired_image: Image.Image,
    split_pct: int,
    target_width: int = 1400,
) -> Optional[Image.Image]:
    try:
        left = original_image.convert("RGB")
        right = repaired_image.convert("RGB")
        if left.size != right.size:
            right = right.resize(left.size, Image.Resampling.LANCZOS)

        if left.width < target_width:
            left = _resize_preview_image(left, target_width=target_width)
            right = right.resize(left.size, Image.Resampling.LANCZOS)

        split_x = int(round((max(0, min(100, int(split_pct))) / 100.0) * left.width))
        merged = Image.new("RGB", left.size)
        if split_x > 0:
            merged.paste(left.crop((0, 0, split_x, left.height)), (0, 0))
        if split_x < left.width:
            merged.paste(right.crop((split_x, 0, right.width, right.height)), (split_x, 0))

        band_half = 2
        for offset in range(-band_half, band_half + 1):
            x = split_x + offset
            if 0 <= x < merged.width:
                color = (255, 255, 255) if offset == 0 else (0, 0, 0)
                for y in range(merged.height):
                    merged.putpixel((x, y), color)
        return merged
    except Exception:
        return None


def _render_halftone_ab_compare(
    original_path: str,
    repaired_path: str,
    strength: float,
    widget_prefix: str,
    heading: str = "A/B Window",
) -> None:
    if not original_path or not repaired_path:
        return
    if not Path(original_path).exists() or not Path(repaired_path).exists():
        return

    zoom = st.slider(
        f"{heading} zoom",
        min_value=1.0,
        max_value=12.0,
        value=3.0,
        step=0.25,
        key=f"{widget_prefix}_zoom",
    )
    focus_cols = st.columns(3)
    with focus_cols[0]:
        focus_x = st.slider("Focus X (%)", 0, 100, 50, 1, key=f"{widget_prefix}_focus_x")
    with focus_cols[1]:
        focus_y = st.slider("Focus Y (%)", 0, 100, 50, 1, key=f"{widget_prefix}_focus_y")
    with focus_cols[2]:
        split_position = st.slider("A/B split (%)", 0, 100, 50, 1, key=f"{widget_prefix}_split")

    original_zoom = _zoom_crop_image(original_path, zoom, focus_x, focus_y)
    repaired_zoom = _zoom_crop_image(repaired_path, zoom, focus_x, focus_y)

    if original_zoom and repaired_zoom:
        ab_window = _build_ab_window_image(
            original_zoom,
            repaired_zoom,
            split_pct=split_position,
            target_width=1600,
        )
        if ab_window:
            st.markdown(
                f"**{heading}**  \nZoom: {zoom:.2f}x · Strength: {int(round(float(strength)))} · {_halftone_strength_label(strength)}"
            )
            st.image(ab_window, use_column_width=True)
            st.caption("Left of the divider is original. Right is repaired.")

    detail_tabs = st.tabs(["Zoomed Original", "Zoomed Repair", "Full Image A/B"])
    with detail_tabs[0]:
        if original_zoom:
            st.image(_resize_preview_image(original_zoom, target_width=1400), use_column_width=True)
    with detail_tabs[1]:
        if repaired_zoom:
            st.image(_resize_preview_image(repaired_zoom, target_width=1400), use_column_width=True)
    with detail_tabs[2]:
        full_original = _zoom_crop_image(original_path, 1.0, 50, 50)
        full_repaired = _zoom_crop_image(repaired_path, 1.0, 50, 50)
        if full_original and full_repaired:
            full_ab = _build_ab_window_image(
                full_original,
                full_repaired,
                split_pct=split_position,
                target_width=1400,
            )
            if full_ab:
                st.image(full_ab, use_column_width=True)


def _ascii_fold(text: str) -> str:
    """Fold unicode text to ASCII for robust search/parsing."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_ascii_text(text: str) -> str:
    """Normalize arbitrary text to ASCII-safe form."""
    cleaned = _clean_line(_ascii_fold(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def _normalize_person_name_ascii(name: str) -> str:
    """Normalize person names to ASCII-safe form while preserving punctuation."""
    normalized = _normalize_ascii_text(name)
    # Remove affiliation markers/superscripts commonly embedded in author lists.
    normalized = re.sub(r"\b\d+\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,;:-*")
    return normalized


def _first_nonempty_lines(text: str, limit: int = 30) -> List[str]:
    lines = [_clean_line(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if x]
    return lines[:limit]


def _looks_like_person_name(candidate: str) -> bool:
    c = _normalize_ascii_text(_clean_line(re.sub(r"\d+", "", candidate)))
    if not c or len(c) < 5 or len(c) > 80:
        return False
    low = c.lower()
    disallow = [
        "university", "department", "school", "faculty", "institute", "hospital", "campus",
        "australia", "india", "srilanka", "melbourne", "brisbane", "gold coast", "corresponding author",
        "author:", "keywords", "abstract", "study", "comprehensive", "licensed", "creative commons",
        "lecturer", "postgraduate", "associate professor", "professor",
        "management", "countries", "country", "licensee", "republic", "sciences",
        "introduction", "change management",
    ]
    if any(x in low for x in disallow):
        return False
    if "@" in c or "http" in low:
        return False
    c = re.sub(r"\b(Dr|Prof)\.?\s+", "", c, flags=re.IGNORECASE)
    c = re.sub(r"\b(PhD|MD|MSc|BSc|BA|MA|MPhil|DPhil|MBA|BHlthSc|Mast\s+Nutr&Diet)\b\.?", "", c, flags=re.IGNORECASE)
    c = re.sub(r"\s+", " ", c).strip(" ,;:-")
    parts = c.split()
    if len(parts) < 2 or len(parts) > 5:
        return False
    for p in parts:
        if not re.match(r"^(?:[A-Z]\.?|[A-Z][A-Za-z'\-]+)$", p):
            return False
    return True


def _extract_names_from_author_block(raw: str) -> List[str]:
    """Extract person-name candidates from a likely author block."""
    text = _clean_line(raw)
    if not text:
        return []

    # Remove common non-author prefixes and obvious affiliation tails.
    text = re.sub(r"(?i)^authors?\s*[:\-]\s*", "", text)
    text = re.sub(r"(?i)^for referencing,\s*please use:\s*", "", text)
    text = re.sub(r"(?i)\b(university|department|faculty|institute|hospital)\b.*$", "", text)
    text = re.sub(r"[*†‡§]", " ", text)
    text = re.sub(r"\(\s*[^)]*\)", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")

    parts = [p.strip() for p in re.split(r",|;|\band\b", text, flags=re.IGNORECASE) if p.strip()]
    names: List[str] = []
    for part in parts:
        cleaned = _normalize_ascii_text(part).strip(" ,;:-")
        # Handle citation forms like "Surname, A.B." by flipping to "A.B. Surname".
        m = re.match(r"^([A-Z][A-Za-z'\-]+)\s*,\s*([A-Z](?:\.[A-Z])*\.?)$", cleaned)
        if m:
            cleaned = f"{m.group(2)} {m.group(1)}"
        if _looks_like_person_name(cleaned):
            names.append(cleaned)
    return list(dict.fromkeys(names))


def _photo_description_issue(description: str) -> Optional[str]:
    """Return a user-facing issue for placeholder image descriptions."""
    desc = (description or "").strip()
    if not desc.startswith("[Image:"):
        return None
    low = desc.lower()
    if "timed out" in low:
        return "Image description timed out. The vision model may be overloaded."
    if "unavailable" in low:
        return "Image description was skipped because the vision model was unavailable."
    if "error" in low:
        return "Image description failed due to a vision model error."
    if "logo/icon omitted" in low:
        return "Image looked like a logo or icon, so description was intentionally skipped."
    return desc


def _extract_available_at(md_content: str) -> str:
    text = md_content or ""
    doi_url = re.search(r"https?://(?:dx\.)?doi\.org/\S+", text, flags=re.IGNORECASE)
    if doi_url:
        return doi_url.group(0).rstrip(").,;]")

    doi = re.search(r"\bdoi\s*[:]\s*(10\.\d{4,9}/[^\s;,)\]]+)", text, flags=re.IGNORECASE)
    if doi:
        return f"https://doi.org/{doi.group(1)}"

    doi_bare = re.search(r"\b(10\.\d{4,9}/[^\s;,)\]]+)", text)
    if doi_bare:
        return f"https://doi.org/{doi_bare.group(1)}"

    url = re.search(r"https?://\S+", text, flags=re.IGNORECASE)
    if url:
        return url.group(0).rstrip(").,;]")

    return "Unknown"


def _sanitize_markdown_for_preface(md_content: str) -> str:
    """Remove known conversion-error boilerplate from markdown before LLM/keyword extraction."""
    def _is_logo_or_icon_caption(caption_text: str) -> bool:
        c = _clean_line(caption_text)
        if not c:
            return True
        low = c.lower()
        marker_hits = sum(
            1
            for marker in [
                "logo", "icon", "icons", "watermark", "emblem", "symbol",
                "badge", "seal", "silhouette", "letters", "initials",
            ]
            if marker in low
        )
        visual_hits = sum(
            1
            for marker in [
                "black and white", "circular", "circle", "strip", "background",
                "left icon", "right icon", "main colors", "main colours",
            ]
            if marker in low
        )
        return (marker_hits >= 1 and visual_hits >= 1) or marker_hits >= 2

    cleaned_lines = []
    error_markers = [
        "image could not be described",
        "vision model",
        "vlm processing failed",
        "image processing failed",
        "source_type': 'image_error'",
        "source_type: image_error",
    ]
    for raw_line in (md_content or "").splitlines():
        line = raw_line.strip()
        low = line.lower()
        if any(marker in low for marker in error_markers):
            continue
        # Drop inline image-caption blocks that are logo/icon noise.
        image_caption_match = re.match(r"^\s*>\s*\*\*\[Image[^\]]*\]\*\*:\s*(.+)$", line, flags=re.IGNORECASE)
        if image_caption_match:
            caption_body = image_caption_match.group(1).strip()
            if _is_logo_or_icon_caption(caption_body):
                continue
        # Drop markdown image embeds that mostly carry filenames/noise for metadata extraction.
        if line.startswith("![") and "](" in line:
            continue
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines)


def _extract_authors_from_markdown(md_content: str) -> List[str]:
    pre_abstract = re.split(r"(?im)^\s*abstract\b", md_content or "", maxsplit=1)[0][:12000]
    lines = _first_nonempty_lines(pre_abstract, limit=120)
    authors: List[str] = []

    # 1) Prefer explicit author/citation blocks first.
    explicit_patterns = [
        r"(?is)\bauthors?\b\s*[:\-]?\s*(.+?)(?=\n\s*(university|affiliations?|author contribution|doi|abstract|keywords?|for referencing|\Z))",
        r"(?is)for referencing,\s*please use:\s*(.+?)(?=\n|$)",
    ]
    for pat in explicit_patterns:
        for match in re.finditer(pat, pre_abstract, flags=re.IGNORECASE):
            authors.extend(_extract_names_from_author_block(match.group(1)))
    if authors:
        return list(dict.fromkeys(authors))[:12]

    # 2) Fall back to line-based heuristics.
    for line in lines:
        if len(line) > 260:
            continue
        if re.match(r"^\d+\s", line):
            continue
        # Skip likely title/topic lines unless they look like a name list delimiter line.
        if "," not in line and ";" not in line and " and " not in line.lower():
            continue
        cleaned = re.sub(r"([A-Za-z])\d+\b", r"\1", line)
        cleaned = re.sub(r"\band\b", ",", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(Dr|Prof)\.?\s+", "", cleaned, flags=re.IGNORECASE)
        parts = [p.strip() for p in re.split(r",|;", cleaned) if p.strip()]
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip(" ,;:-")
            if _looks_like_person_name(part):
                authors.append(part)
        if len(authors) >= 12:
            break

    unique = list(dict.fromkeys(authors))
    return unique[:12]


def _guess_title_from_markdown(md_content: str, file_path: str) -> str:
    lines = _first_nonempty_lines(md_content, limit=120)
    skip_exact = {
        "viewpoint", "introduction", "abstract", "background", "keywords", "key words"
    }
    skip_contains = [
        "please cite this publication as",
        "this work is published under",
        "opinions expressed and arguments employed",
        "revised version",
        "corrigenda",
    ]
    # Academic fallback: title is often the line immediately before author names.
    for i in range(1, min(80, len(lines))):
        if _extract_authors_from_markdown(lines[i]):
            prev = _clean_line(lines[i - 1])
            if (
                len(prev) >= 20
                and not re.match(r"^page\s+\d+", prev, flags=re.IGNORECASE)
                and "licensed" not in prev.lower()
                and "creative commons" not in prev.lower()
                and not any(s in prev.lower() for s in skip_contains)
            ):
                return prev[:180]
    for i, line in enumerate(lines[:40]):
        low = line.lower().strip()
        if low in skip_exact and i + 1 < len(lines):
            nxt = lines[i + 1]
            if len(nxt) > 15:
                return nxt[:180]
        if re.match(r"^#+\s*page\s+\d+", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^page\s+\d+", low):
            continue
        if line.startswith("#"):
            title = _clean_line(line.lstrip("#"))
            if title and not re.match(r"^page\s+\d+", title, flags=re.IGNORECASE):
                return title[:180]
    for line in lines[:60]:
        low = line.lower()
        if re.match(r"^page\s+\d+", low):
            continue
        if len(line) < 15:
            continue
        if len(line) > 180:
            continue
        if any(x in low for x in skip_contains):
            continue
        if any(x in low for x in ["university", "hospital", "school of", "email:", "doi:", "journal", "www."]):
            continue
        if any(x in low for x in ["licensed", "creative commons", "corresponding author"]):
            continue
        if low in skip_exact:
            continue
        # Prefer sentence-case/title-like lines as title candidates.
        if len(re.findall(r"[A-Za-z]", line)) >= 12:
            return line[:180]
    return Path(file_path).stem


def _detect_source_type_hint(file_path: str, md_content: str) -> str:
    text = f"{_user_visible_filename(file_path)}\n{md_content[:20000]}".lower()

    academic_patterns = [
        r"\belsevier\b", r"\bspringer\b", r"\bwiley\b", r"\bieee\b",
        r"\bdoi\b", r"\bjournal\b", r"\bproceedings\b",
    ]
    consulting_patterns = [
        r"\bdeloitte\b", r"\bmckinsey\b", r"\bbain\b", r"\bbcg\b",
        r"\bkpmg\b", r"\bpwc\b", r"\baccenture\b", r"\bconsulting\b",
        r"\bernst\s*&\s*young\b", r"\bey\b",
    ]
    institutional_patterns = [
        r"\badha\b", r"\baidh\b", r"\baustralian digital health agency\b",
        r"\baustralasian institute of digital health\b",
        r"\bdepartment of health\b", r"\bministry\b", r"\bgovernment\b",
        r"\bwho\b", r"\boecd\b", r"\bworld bank\b", r"\bunited nations\b",
        r"\bnhs\b", r"\bcdc\b", r"\buniversity\b", r"\binstitute\b",
    ]
    ai_patterns = [
        r"\bperplexity\b", r"\bchatgpt\b", r"\bopenai\b", r"\bclaude\b", r"\bgemini\b",
        r"\bdeep research\b", r"\bdeep report\b", r"\bdeep report into\b",
        r"\bgenerated by ai\b", r"\bai-generated\b",
    ]

    def _has_any(patterns: List[str]) -> bool:
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    has_academic = _has_any(academic_patterns)
    has_consulting = _has_any(consulting_patterns)
    has_institutional = _has_any(institutional_patterns)
    has_ai_markers = _has_any(ai_patterns)
    em_dash_count = text.count("—") + text.count("–")

    if has_academic:
        return "Academic"
    # Explicit institutional signals should not be overwritten by consulting hints.
    if has_institutional and not has_consulting:
        return "Other"
    if has_consulting:
        return "Consulting Company"
    # If not strongly human-sourced, bias toward AI when AI-style signals are present.
    if has_ai_markers or (em_dash_count >= 8 and not (has_academic or has_institutional or has_consulting)):
        return "AI Generated Report"
    return "Other"


def _infer_publisher_from_text(md_text: str, source_hint: str) -> str:
    text = (md_text or "").lower()
    publisher_patterns = [
        (r"\baustralian digital health agency\b|\badha\b", "Australian Digital Health Agency"),
        (r"\baustralasian institute of digital health\b|\baidh\b", "Australasian Institute of Digital Health"),
        (r"\bworld health organization\b|\bwho\b", "World Health Organization"),
        (r"\boecd\b", "OECD"),
        (r"\bworld bank\b", "World Bank"),
        (r"\bunited nations\b|\bun\b", "United Nations"),
        (r"\belsevier\b", "Elsevier"),
        (r"\bspringer\b", "Springer"),
        (r"\bwiley\b", "Wiley"),
        (r"\bieee\b", "IEEE"),
        (r"\bdeloitte\b", "Deloitte"),
        (r"\bmckinsey\b", "McKinsey"),
        (r"\bbain\b", "Bain"),
        (r"\bbcg\b", "BCG"),
        (r"\bkpmg\b", "KPMG"),
        (r"\bernst\s*&\s*young\b|\bey\b", "EY"),
        (r"\bpwc\b", "PwC"),
        (r"\baccenture\b", "Accenture"),
        (r"\bperplexity\b", "Perplexity"),
        (r"\bopenai\b", "OpenAI"),
        (r"\bclaude\b|\banthropic\b", "Anthropic"),
        (r"\bgemini\b|\bgoogle\b", "Google"),
    ]
    for pattern, label in publisher_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    if source_hint == "AI Generated Report":
        return "Unknown AI"
    return "Unknown"


def _extract_publishing_date(md_text: str, file_path: str) -> str:
    text = (md_text or "")[:12000]
    text_ascii = _ascii_fold(text)

    # Strong citation-line signal, e.g. "... (2024), ... DOI ..."
    citation_line = re.search(
        r"(?is)\bplease\s+cite\b[^\n]{0,300}?\((19|20)\d{2}\)",
        text_ascii,
        flags=re.IGNORECASE,
    )
    if citation_line:
        m = re.search(r"\((19|20)\d{2}\)", citation_line.group(0))
        if m:
            return m.group(0).strip("()")

    revised_pattern = re.search(
        r"(?i)\brevised\s+version\b[^A-Za-z0-9]{0,8}(?:,\s*)?([A-Za-z]{3,9}\s+20\d{2}|20\d{2})",
        text_ascii,
    )
    if revised_pattern:
        return _clean_line(revised_pattern.group(1))

    # Prefer explicitly labeled publication/update lines before generic year-only matches.
    label_pattern = (
        r"(?i)\b(?:published|publication date|date published|released|issued|last updated|updated)\b"
        r"[^A-Za-z0-9]{0,12}"
        r"(20\d{2}[-/][01]?\d[-/][0-3]?\d|[0-3]?\d\s+[A-Za-z]{3,9}\s+20\d{2}|[A-Za-z]{3,9}\s+[0-3]?\d,\s+20\d{2}|20\d{2})"
    )
    m = re.search(label_pattern, text_ascii)
    if m:
        return _clean_line(m.group(1))

    # Context-aware year extraction from citation/publication lines.
    scored_year = ""
    scored_value = -999
    for raw_line in text_ascii.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue
        years = re.findall(r"\b((?:19|20)\d{2})\b", line)
        if not years:
            continue

        low = line.lower()
        score = 0
        if "please cite" in low or "cite this publication" in low:
            score += 8
        if "revised version" in low:
            score += 7
        if any(k in low for k in ["published", "publication", "issued", "release", "copyright"]):
            score += 6
        if any(k in low for k in ["isbn", "issn", "doi"]):
            score += 5
        if re.search(r"\((?:19|20)\d{2}\)", line):
            score += 4
        if low.startswith("> **[image"):
            score -= 5
        if "photo credits" in low:
            score -= 3

        candidate_year = years[0]
        if score > scored_value:
            scored_value = score
            scored_year = candidate_year

    if scored_year and scored_value >= 3:
        return scored_year

    generic_pattern = r"(20\d{2}[-/][01]?\d[-/][0-3]?\d|[0-3]?\d\s+[A-Za-z]{3,9}\s+20\d{2}|[A-Za-z]{3,9}\s+[0-3]?\d,\s+20\d{2})"
    m = re.search(generic_pattern, text_ascii)
    if m:
        return _clean_line(m.group(1))

    year_match = re.search(r"\b((?:19|20)\d{2})\b", text_ascii)
    if year_match:
        return year_match.group(1)

    # Last fallback: year in filename.
    filename = _user_visible_filename(file_path)
    m = re.search(r"\b(19|20)\d{2}\b", filename)
    if m:
        return m.group(0)
    return "Unknown"


def _extract_json_block(raw: str) -> Optional[dict]:
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_preface_metadata_with_llm(file_path: str, md_content: str, source_hint: str) -> Optional[dict]:
    try:
        from cortex_engine.llm_interface import LLMInterface
        llm = LLMInterface(model="mistral:latest", temperature=0.1)
    except Exception as e:
        logger.warning(f"Could not initialize LLM for preface extraction: {e}")
        return None

    snippet = _sanitize_markdown_for_preface(md_content)[:18000]
    prompt = f"""
You extract publication metadata from markdown content.
Return STRICT JSON only with keys:
- title (string)
- source_type (one of: Academic, Consulting Company, AI Generated Report, Other)
- publisher (string)
- publishing_date (string)
- authors (array of strings)
- available_at (string)
- abstract (string)
- keywords (array of strings)
- credibility_tier_value (integer: 0..5)
- credibility_tier_key (one of: peer-reviewed, institutional, pre-print, editorial, commentary, unclassified)
- credibility_tier_label (one of: Peer-Reviewed, Institutional, Pre-Print, Editorial, Commentary, Unclassified)
- credibility (human-readable string, e.g. "Draft Institutional Report")

Rules:
- Use source hint: "{source_hint}" unless content strongly indicates a different source_type.
- If AI Generated Report and publisher cannot be identified, set publisher to "Unknown AI".
- If abstract is not explicitly present, generate a concise abstract from the document.
- If abstract exists, extract the full abstract paragraph(s), not just one line.
- For authors, include only person names and exclude affiliations/locations/institutions.
- For title, prefer the document title and never use abstract opening text or citation boilerplate (e.g. "Please cite this publication as").
- For available_at, prefer DOI URL if present, else canonical report URL, else "Unknown".
- For publishing_date, prioritize explicit year/date from citation/publishing lines, including patterns like "(2024)" and "Revised version, November 2024".
- Provide 5-12 useful keywords, with each keyword at most TWO WORDS.
- Credibility tiers:
  5 / peer-reviewed / Peer-Reviewed: NLM/PubMed, Nature, The Lancet, JAMA, BMJ
  4 / institutional / Institutional: WHO, UN/IPCC, OECD, World Bank, ABS, government depts, universities/institutes
  3 / pre-print / Pre-Print: arXiv, SSRN, bioRxiv, ResearchGate
  2 / editorial / Editorial: Scientific American, The Conversation, HBR
  1 / commentary / Commentary: blogs, newsletters, consulting reports, opinion
  0 / unclassified / Unclassified: not yet assessed
- If source_type is AI Generated Report, default to 0 / unclassified.
- Treat titles like "Deep report into..." and strongly AI-styled writing (frequent em-dashes) as AI-generated unless clear human institutional/academic markers are present.
- Ignore tiny/logo/icon-only figure captions when determining title, abstract, keywords, source_type, and date.
- If a field is unknown use "Unknown" (or [] for authors/keywords).

File name: {_user_visible_filename(file_path)}

Markdown content:
{snippet}
"""
    try:
        response = llm.generate(prompt, max_tokens=900)
        return _extract_json_block(response)
    except Exception as e:
        logger.warning(f"LLM preface extraction failed: {e}")
        return None


def _detect_document_stage(file_path: str, title: str, md_content: str) -> str:
    text = f"{_user_visible_filename(file_path)}\n{title}\n{md_content[:30000]}".lower()
    return "Draft" if re.search(r"\bdraft\b", text) else "Final"


def _check_url_availability(url: str, timeout: float = 8.0) -> Dict[str, str]:
    """Probe URL availability without downloading full content."""
    checked_at = datetime.now().strftime("%Y-%m-%d")
    if not url or url == "Unknown":
        return {
            "availability_status": "unknown",
            "availability_http_code": "",
            "availability_checked_at": checked_at,
            "availability_note": "No canonical source URL provided.",
        }

    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {
            "availability_status": "unknown",
            "availability_http_code": "",
            "availability_checked_at": checked_at,
            "availability_note": f"URL appears invalid: {url}",
        }

    headers = {
        "User-Agent": "CortexSuite/DocumentExtract (+source-integrity-check)",
        "Accept": "*/*",
    }
    methods = ("HEAD", "GET")
    last_http_error = None

    for method in methods:
        req_headers = dict(headers)
        if method == "GET":
            req_headers["Range"] = "bytes=0-0"
        req = Request(url, headers=req_headers, method=method)
        try:
            with urlopen(req, timeout=timeout) as resp:
                code = int(getattr(resp, "status", 200) or 200)
            if 200 <= code < 400:
                return {
                    "availability_status": "available",
                    "availability_http_code": str(code),
                    "availability_checked_at": checked_at,
                    "availability_note": f"Available as at {checked_at}.",
                }
            if code == 404:
                return {
                    "availability_status": "not_found",
                    "availability_http_code": "404",
                    "availability_checked_at": checked_at,
                    "availability_note": f"Previously available at: {url} but no longer available as at: {checked_at}.",
                }
            if code == 410:
                return {
                    "availability_status": "gone",
                    "availability_http_code": "410",
                    "availability_checked_at": checked_at,
                    "availability_note": f"Previously available at: {url} but has been removed (HTTP 410) as at: {checked_at}.",
                }
            if 400 <= code < 500:
                return {
                    "availability_status": "client_error",
                    "availability_http_code": str(code),
                    "availability_checked_at": checked_at,
                    "availability_note": f"Source URL returned HTTP {code} as at {checked_at}; verify whether the source moved, is access-restricted, or withdrawn.",
                }
            if code >= 500:
                return {
                    "availability_status": "server_error",
                    "availability_http_code": str(code),
                    "availability_checked_at": checked_at,
                    "availability_note": f"Source host returned HTTP {code} as at {checked_at}; availability could not be confirmed.",
                }
        except HTTPError as e:
            code = int(e.code)
            if code == 404:
                return {
                    "availability_status": "not_found",
                    "availability_http_code": "404",
                    "availability_checked_at": checked_at,
                    "availability_note": f"Previously available at: {url} but no longer available as at: {checked_at}.",
                }
            if code == 410:
                return {
                    "availability_status": "gone",
                    "availability_http_code": "410",
                    "availability_checked_at": checked_at,
                    "availability_note": f"Previously available at: {url} but has been removed (HTTP 410) as at: {checked_at}.",
                }
            if code == 405 and method == "HEAD":
                # Some hosts disallow HEAD; retry with ranged GET.
                continue
            last_http_error = e
        except URLError as e:
            return {
                "availability_status": "unreachable",
                "availability_http_code": "",
                "availability_checked_at": checked_at,
                "availability_note": f"Source URL could not be reached as at {checked_at}: {e.reason}",
            }
        except Exception as e:
            return {
                "availability_status": "unreachable",
                "availability_http_code": "",
                "availability_checked_at": checked_at,
                "availability_note": f"Source URL check failed as at {checked_at}: {e}",
            }

    if last_http_error is not None:
        code = int(last_http_error.code)
        return {
            "availability_status": "client_error" if 400 <= code < 500 else "server_error",
            "availability_http_code": str(code),
            "availability_checked_at": checked_at,
            "availability_note": f"Source URL returned HTTP {code} as at {checked_at}; verify whether the source moved, is access-restricted, or withdrawn.",
        }
    return {
        "availability_status": "unknown",
        "availability_http_code": "",
        "availability_checked_at": checked_at,
        "availability_note": f"Source URL availability is unknown as at {checked_at}.",
    }


def _classify_credibility_tier(
    file_path: str,
    source_type: str,
    publisher: str,
    available_at: str,
    md_content: str,
    availability_status: str = "unknown",
) -> Dict[str, str]:
    text = f"{_user_visible_filename(file_path)}\n{publisher}\n{available_at}\n{md_content[:50000]}".lower()
    tier_value, key, label = classify_credibility_tier(
        text=text,
        source_type=source_type,
        availability_status=availability_status,
    )
    stage = _detect_document_stage(file_path, "", md_content)
    if availability_status in {"not_found", "gone"}:
        credibility_text = f"{stage} {label} Report (Source Link Unavailable)"
    elif availability_status in {"client_error", "server_error", "unreachable"}:
        credibility_text = f"{stage} {label} Report (Source Link Unverified)"
    else:
        credibility_text = f"{stage} {label} Report"
    return {
        "credibility_tier_value": tier_value,
        "credibility_tier_key": key,
        "credibility_tier_label": label,
        "credibility": credibility_text,
    }


def _clean_keywords(keywords: List[str]) -> List[str]:
    banned = {
        "image", "vision", "model", "error", "could", "described", "failed",
        "processing", "source", "type", "unknown", "document",
    }
    cleaned: List[str] = []
    for keyword in keywords:
        k = re.sub(r"\s+", " ", str(keyword or "").strip().lower())
        if not k or len(k) < 3:
            continue
        if k in banned:
            continue
        if len(k) > 60:
            continue
        if len(k.split()) > 2:
            continue
        if "image" in k and "health" not in k:
            continue
        if "error" in k or "vision model" in k:
            continue
        cleaned.append(k)
    # preserve order while deduping
    return list(dict.fromkeys(cleaned))[:8]


def _fallback_preface_metadata(file_path: str, md_content: str, source_hint: str) -> dict:
    md_clean = _sanitize_markdown_for_preface(md_content)
    title = _guess_title_from_markdown(md_clean, file_path)
    lines = _first_nonempty_lines(md_clean, limit=120)
    text_lower = md_clean.lower()

    publisher = _infer_publisher_from_text(md_clean, source_hint)
    publishing_date = _extract_publishing_date(md_clean, file_path)
    available_at = _extract_available_at(md_clean)

    authors = _extract_authors_from_markdown(md_clean)

    abstract = ""
    abstract_match = re.search(
        r"(?is)\babstract\b\s*[:\-]?\s*(.+?)(?=\n\s*(keywords?|key words?|introduction|background|methods?|j\s+med|doi:|##\s*page|\Z))",
        md_clean,
    )
    if abstract_match:
        abstract = _clean_line(abstract_match.group(1))
    if not abstract:
        # Fallback summary from first meaningful lines.
        abstract = " ".join([l for l in lines if len(l) > 30][:4])[:900] or "Summary not available."

    keywords = []
    kw_match = re.search(
        r"(?is)\bkeywords?\b\s*[:\-]?\s*(.+?)(?=\n\s*(introduction|background|methods?|##\s*page|\Z))",
        md_content[:20000],
        flags=re.IGNORECASE,
    )
    if kw_match:
        kw_raw = _clean_line(kw_match.group(1))
        keywords = [k.strip().lower() for k in re.split(r",|;|\|", kw_raw) if k.strip()][:12]
    if not keywords:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z\-]{3,}\b", md_clean[:8000])
        freq = {}
        stop = {
            "this", "that", "with", "from", "were", "have", "been", "into", "their", "about", "which",
            "document", "page", "pages", "report", "using", "used", "study", "journal", "research",
            "university", "australia", "gold", "coast"
        }
        for t in tokens:
            low = t.lower()
            if low in stop:
                continue
            freq[low] = freq.get(low, 0) + 1
        keywords = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]]

    availability = _check_url_availability(available_at)
    credibility = _classify_credibility_tier(
        file_path=file_path,
        source_type=source_hint or "Other",
        publisher=publisher,
        available_at=available_at,
        availability_status=availability.get("availability_status", "unknown"),
        md_content=md_clean,
    )

    return {
        "title": title or _user_visible_stem(file_path),
        "source_type": source_hint or "Other",
        "publisher": publisher,
        "publishing_date": publishing_date,
        "authors": authors,
        "available_at": available_at,
        "availability_status": availability.get("availability_status", "unknown"),
        "availability_http_code": availability.get("availability_http_code", ""),
        "availability_checked_at": availability.get("availability_checked_at", ""),
        "availability_note": availability.get("availability_note", ""),
        "abstract": abstract,
        "keywords": _clean_keywords(keywords),
        **credibility,
    }


def _normalize_preface_metadata(file_path: str, source_hint: str, raw_meta: Optional[dict], fallback_meta: dict, md_content: str) -> dict:
    data = raw_meta or {}
    # Back-compat aliases from older systems.
    if "credibility_tier_value" not in data and "credibility_value" in data:
        data["credibility_tier_value"] = data.get("credibility_value")
    if "credibility_tier_key" not in data and "credibility_source" in data:
        data["credibility_tier_key"] = str(data.get("credibility_source", "")).strip().lower()
    title = _normalize_ascii_text(_clean_line(str(data.get("title", "")))) or _normalize_ascii_text(fallback_meta["title"])
    source_type = _clean_line(str(data.get("source_type", ""))) or source_hint or fallback_meta["source_type"]
    if source_type not in {"Academic", "Consulting Company", "AI Generated Report", "Other"}:
        source_type = source_hint if source_hint in {"Academic", "Consulting Company", "AI Generated Report", "Other"} else "Other"

    publisher = _normalize_ascii_text(_clean_line(str(data.get("publisher", "")))) or _normalize_ascii_text(fallback_meta["publisher"])
    if not publisher or publisher.lower() == "unknown":
        publisher = _infer_publisher_from_text(_sanitize_markdown_for_preface(md_content), source_type)
    if source_type == "AI Generated Report" and (not publisher or publisher.lower() == "unknown"):
        publisher = "Unknown AI"

    publishing_date = _clean_line(str(data.get("publishing_date", ""))) or fallback_meta["publishing_date"] or "Unknown"
    if not publishing_date or publishing_date == "Unknown":
        publishing_date = _extract_publishing_date(_sanitize_markdown_for_preface(md_content), file_path)
    available_at = _clean_line(str(data.get("available_at", ""))) or fallback_meta.get("available_at", "Unknown")
    if available_at != "Unknown":
        available_at = _extract_available_at(available_at)
    availability = _check_url_availability(available_at)

    # Final deterministic source-type pass using consolidated title/publisher/content.
    # This guards against LLM drift and false consulting classifications.
    consolidated_text = f"{title}\n{publisher}\n{_sanitize_markdown_for_preface(md_content)}"
    deterministic_source = _detect_source_type_hint(file_path, consolidated_text)
    if deterministic_source == "AI Generated Report":
        source_type = "AI Generated Report"
    elif source_type == "Consulting Company" and deterministic_source == "Other":
        # Prefer neutral "Other" when strong consulting markers are absent.
        source_type = "Other"

    authors_raw = data.get("authors", fallback_meta.get("authors", []))
    if isinstance(authors_raw, str):
        authors = [a.strip() for a in re.split(r",|;", authors_raw) if a.strip()]
    elif isinstance(authors_raw, list):
        authors = [str(a).strip() for a in authors_raw if str(a).strip()]
    else:
        authors = []
    if not authors:
        authors = fallback_meta.get("authors", [])
    authors = [_normalize_person_name_ascii(a) for a in authors if _looks_like_person_name(a)]
    authors = [a for a in authors if _looks_like_person_name(a)]
    authors = list(dict.fromkeys(authors))

    keywords_raw = data.get("keywords", fallback_meta.get("keywords", []))
    if isinstance(keywords_raw, str):
        keywords = [k.strip() for k in re.split(r",|;|\|", keywords_raw) if k.strip()]
    elif isinstance(keywords_raw, list):
        keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
    else:
        keywords = []
    if not keywords:
        keywords = fallback_meta.get("keywords", [])
    keywords = _clean_keywords([_normalize_ascii_text(k.lower()) for k in keywords if k])

    abstract = _clean_line(str(data.get("abstract", ""))) or fallback_meta["abstract"] or "Summary not available."

    cred_value_raw = data.get("credibility_tier_value", fallback_meta.get("credibility_tier_value", 0))
    try:
        cred_value = int(cred_value_raw)
    except Exception:
        cred_value = int(fallback_meta.get("credibility_tier_value", 0))
    if cred_value not in _CREDIBILITY_TIERS:
        cred_value = int(fallback_meta.get("credibility_tier_value", 0))
    # Deterministic policy alignment with credibility_spec.md.
    deterministic = _classify_credibility_tier(
        file_path=file_path,
        source_type=source_type,
        publisher=publisher,
        available_at=available_at,
        availability_status=availability.get("availability_status", "unknown"),
        md_content=_sanitize_markdown_for_preface(md_content),
    )
    det_value = int(deterministic.get("credibility_tier_value", 0))
    if source_type == "AI Generated Report" or det_value > 0:
        cred_value = det_value
    cred_key, cred_label = _CREDIBILITY_TIERS.get(cred_value, _CREDIBILITY_TIERS[0])
    stage = _detect_document_stage(file_path, title, _sanitize_markdown_for_preface(md_content))
    availability_status = availability.get("availability_status", "unknown")
    if availability_status in {"not_found", "gone"}:
        credibility_text = f"{stage} {cred_label} Report (Source Link Unavailable)"
    elif availability_status in {"client_error", "server_error", "unreachable"}:
        credibility_text = f"{stage} {cred_label} Report (Source Link Unverified)"
    else:
        credibility_text = f"{stage} {cred_label} Report"

    journal_authority = classify_journal_authority(
        title=title,
        text=_sanitize_markdown_for_preface(md_content),
    )

    source_integrity_flag = "ok"
    if availability_status in {"not_found", "gone"}:
        source_integrity_flag = "deprecated_or_removed"
    elif availability_status in {"client_error", "server_error", "unreachable", "unknown"}:
        source_integrity_flag = "unverified"

    return {
        "title": title or _user_visible_stem(file_path),
        "source_type": source_type,
        "publisher": publisher or "Unknown",
        "publishing_date": publishing_date,
        "authors": authors[:20],
        "available_at": available_at or "Unknown",
        "availability_status": availability_status,
        "availability_http_code": availability.get("availability_http_code", ""),
        "availability_checked_at": availability.get("availability_checked_at", ""),
        "availability_note": availability.get("availability_note", ""),
        "source_integrity_flag": source_integrity_flag,
        "keywords": keywords[:8],
        "abstract": abstract,
        "credibility_tier_value": cred_value,
        "credibility_tier_key": cred_key,
        "credibility_tier_label": cred_label,
        "credibility": credibility_text,
        **journal_authority,
    }


def _yaml_escape(value: str) -> str:
    v = str(value or "").replace("'", "''")
    return f"'{v}'"


def _build_preface(md_meta: dict) -> str:
    authors = md_meta.get("authors") or []
    keywords = md_meta.get("keywords") or []
    authors_yaml = "[" + ", ".join(_yaml_escape(a) for a in authors) + "]" if authors else "[]"
    keywords_yaml = "[" + ", ".join(_yaml_escape(k) for k in keywords) + "]" if keywords else "[]"
    lines = [
        "---",
        "preface_schema: '1.0'",
        f"title: {_yaml_escape(md_meta['title'])}",
        f"source_type: {_yaml_escape(md_meta['source_type'])}",
        f"publisher: {_yaml_escape(md_meta['publisher'])}",
        f"publishing_date: {_yaml_escape(md_meta['publishing_date'])}",
        f"authors: {authors_yaml}",
        f"available_at: {_yaml_escape(md_meta.get('available_at', 'Unknown'))}",
        f"availability_status: {_yaml_escape(md_meta.get('availability_status', 'unknown'))}",
        f"availability_http_code: {_yaml_escape(md_meta.get('availability_http_code', ''))}",
        f"availability_checked_at: {_yaml_escape(md_meta.get('availability_checked_at', ''))}",
        f"availability_note: {_yaml_escape(md_meta.get('availability_note', ''))}",
        f"source_integrity_flag: {_yaml_escape(md_meta.get('source_integrity_flag', 'unverified'))}",
        f"credibility_tier_value: {_yaml_escape(md_meta.get('credibility_tier_value', 0))}",
        f"credibility_tier_key: {_yaml_escape(md_meta.get('credibility_tier_key', 'unclassified'))}",
        f"credibility_tier_label: {_yaml_escape(md_meta.get('credibility_tier_label', 'Unclassified'))}",
        f"credibility: {_yaml_escape(md_meta.get('credibility', 'Unclassified Report'))}",
        f"journal_ranking_source: {_yaml_escape(md_meta.get('journal_ranking_source', 'scimagojr_2024'))}",
        f"journal_sourceid: {_yaml_escape(md_meta.get('journal_sourceid', ''))}",
        f"journal_title: {_yaml_escape(md_meta.get('journal_title', ''))}",
        f"journal_issn: {_yaml_escape(md_meta.get('journal_issn', ''))}",
        f"journal_sjr: {_yaml_escape(md_meta.get('journal_sjr', 0.0))}",
        f"journal_quartile: {_yaml_escape(md_meta.get('journal_quartile', ''))}",
        f"journal_rank_global: {_yaml_escape(md_meta.get('journal_rank_global', 0))}",
        f"journal_categories: {_yaml_escape(md_meta.get('journal_categories', ''))}",
        f"journal_areas: {_yaml_escape(md_meta.get('journal_areas', ''))}",
        f"journal_high_ranked: {_yaml_escape(md_meta.get('journal_high_ranked', False))}",
        f"journal_match_method: {_yaml_escape(md_meta.get('journal_match_method', 'none'))}",
        f"journal_match_confidence: {_yaml_escape(md_meta.get('journal_match_confidence', 0.0))}",
        f"keywords: {keywords_yaml}",
        f"abstract: {_yaml_escape(md_meta['abstract'])}",
        "---",
        "",
    ]
    return "\n".join(lines)


def _add_document_preface(file_path: str, md_content: str) -> str:
    # Shared engine routine used by both Textifier UI and URL-ingestor PDF->MD flow.
    return add_document_preface(file_path, md_content)


# ======================================================================
# Textifier tab
# ======================================================================

def _render_textifier_tab():
    """Render the Textifier tool UI."""
    st.markdown("Convert PDF, DOCX, PPTX, or image files (PNG/JPG) to rich Markdown with optional AI image descriptions.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")

        use_vision = st.toggle("Use Vision Model for images", value=True, key="txt_vision")
        pdf_mode_label = st.selectbox(
            "PDF processing mode",
            options=[
                "Hybrid (Recommended): Docling first, Qwen enhancement, fallback on timeout",
                "Docling only (best layout/tables)",
                "Qwen 30B cleanup (LLM-first, no Docling)",
            ],
            index=0,
            key="txt_pdf_mode",
            help="Hybrid runs Docling first, then enriches with vision/LLM, and falls back if timeout limits are hit.",
        )
        cleanup_provider = st.selectbox(
            "Cleanup LLM provider",
            options=["lmstudio", "ollama"],
            index=0,
            key="txt_cleanup_provider",
        )
        cleanup_model = st.text_input(
            "Cleanup model",
            value="qwen2.5:32b",
            key="txt_cleanup_model",
            help="Model name for markdown cleanup when using Qwen/Hybrid modes.",
        )
        st.caption("Timeout safeguards")
        docling_timeout_seconds = st.number_input(
            "Docling timeout (seconds)",
            min_value=30,
            max_value=1200,
            value=240,
            step=10,
            key="txt_docling_timeout_s",
            help="If Docling exceeds this time, Textifier falls back to legacy extraction.",
        )
        image_timeout_seconds = st.number_input(
            "Per-image vision timeout (seconds)",
            min_value=3,
            max_value=180,
            value=20,
            step=1,
            key="txt_image_timeout_s",
            help="Maximum time allowed for each image description.",
        )
        image_budget_seconds = st.number_input(
            "Total image-description budget (seconds)",
            min_value=10,
            max_value=1800,
            value=120,
            step=10,
            key="txt_image_budget_s",
            help="When budget is exceeded, remaining images are skipped with placeholders.",
        )
        batch_mode = st.toggle("Batch mode (multi-file)", value=False, key="txt_batch_toggle")
        st.session_state["textifier_batch"] = batch_mode
        batch_cooldown_seconds = st.slider(
            "Cooldown between files (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="txt_batch_cooldown_s",
            help="Adds a short pause between files to reduce system load during batch processing.",
        )

        selected = _file_input_widget("textifier", ["pdf", "docx", "pptx", "png", "jpg", "jpeg"])

        # Clear all uploaded files button
        if st.button("Clear All Files", key="txt_clear_all", use_container_width=True):
            # Bump upload widget version so Streamlit creates a fresh uploader
            ver = st.session_state.get("textifier_upload_version", 0)
            # Clear all textifier-related state except the version we're about to set
            for key in list(st.session_state.keys()):
                if key.startswith("textifier_"):
                    del st.session_state[key]
            if "textifier_results" in st.session_state:
                del st.session_state["textifier_results"]
            st.session_state["textifier_upload_version"] = ver + 1
            # Clean temp files
            temp_dir = Path(tempfile.gettempdir()) / "cortex_textifier"
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            st.rerun()

    with col2:
        st.header("Output")

        if selected:
            files_to_process = selected if isinstance(selected, list) else [selected]
            total_files = len(files_to_process)

            if st.button("Convert to Markdown", type="primary", use_container_width=True):
                from cortex_engine.textifier import DocumentTextifier

                results = {}
                progress = st.progress(0.0, "Starting conversion...")
                status_text = st.empty()

                for file_idx, fpath in enumerate(files_to_process):
                    fname = _user_visible_stem(fpath)
                    file_base = file_idx / total_files
                    file_span = 1.0 / total_files

                    def _on_progress(frac, msg, _base=file_base, _span=file_span, _name=_user_visible_filename(fpath)):
                        overall = min(_base + frac * _span, 1.0)
                        label = f"[{_name}] {msg}" if total_files > 1 else msg
                        progress.progress(overall, label)

                    if total_files > 1:
                        status_text.info(f"File {file_idx + 1}/{total_files}: {_user_visible_filename(fpath)}")

                    mode_map = {
                        "Hybrid (Recommended): Docling first, Qwen enhancement, fallback on timeout": "hybrid",
                        "Docling only (best layout/tables)": "docling",
                        "Qwen 30B cleanup (LLM-first, no Docling)": "qwen30b",
                    }
                    textifier_options = {
                        "use_vision": use_vision,
                        "pdf_strategy": mode_map.get(pdf_mode_label, "hybrid"),
                        "cleanup_provider": cleanup_provider,
                        "cleanup_model": cleanup_model,
                        "docling_timeout_seconds": float(docling_timeout_seconds),
                        "image_description_timeout_seconds": float(image_timeout_seconds),
                        "image_enrich_max_seconds": float(image_budget_seconds),
                    }
                    textifier = DocumentTextifier.from_options(textifier_options, on_progress=_on_progress)
                    try:
                        md_content = textifier.textify_file(fpath)
                    except Exception as e:
                        st.error(f"Failed to convert {_user_visible_filename(fpath)}: {e}")
                        logger.error(f"Textifier conversion error for {fpath}: {e}", exc_info=True)
                        continue

                    # Preface enrichment is best-effort. Never block markdown download on metadata issues.
                    try:
                        _on_progress(1.0, "Generating metadata preface...")
                        md_content = _add_document_preface(fpath, md_content)
                    except Exception as e:
                        st.warning(
                            f"Converted {_user_visible_filename(fpath)} but metadata preface failed; "
                            "download includes markdown without preface."
                        )
                        logger.warning(f"Textifier preface enrichment failed for {fpath}: {e}", exc_info=True)

                    results[f"{fname}_textified.md"] = md_content

                    if batch_cooldown_seconds > 0 and total_files > 1 and file_idx < total_files - 1:
                        progress.progress(
                            min((file_idx + 1) / total_files, 1.0),
                            f"Cooling down for {batch_cooldown_seconds:.1f}s before next file..."
                        )
                        time.sleep(batch_cooldown_seconds)

                progress.progress(1.0, "Done!")
                status_text.empty()

                if results:
                    st.session_state["textifier_results"] = results

        # Display results
        results = st.session_state.get("textifier_results")
        if results:
            st.divider()
            st.subheader("Results")

            if len(results) == 1:
                name, content = next(iter(results.items()))
                st.download_button("Download Markdown", content, file_name=name,
                                   mime="text/markdown", use_container_width=True)
                with st.expander("Preview", expanded=True):
                    st.markdown(content[:5000] + ("\n\n*... truncated ...*" if len(content) > 5000 else ""))
            else:
                # Zip download for batch
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, content in results.items():
                        zf.writestr(name, content)
                buf.seek(0)
                st.download_button("Download All (ZIP)", buf.getvalue(),
                                   file_name="textified_documents.zip",
                                   mime="application/zip", use_container_width=True)
                for name, content in results.items():
                    with st.expander(name):
                        st.markdown(content[:3000] + ("\n\n*... truncated ...*" if len(content) > 3000 else ""))
        elif selected:
            st.info("Click **Convert to Markdown** to process your document")
        else:
            st.info("Select a document from the left panel to get started")


# ======================================================================
# Anonymizer tab (original logic preserved)
# ======================================================================

def _render_anonymizer_tab():
    """Render the Anonymizer tool UI (original Document Anonymizer logic)."""
    st.markdown("Replace identifying information with generic placeholders for privacy protection.")

    if "anonymizer_results" not in st.session_state:
        st.session_state.anonymizer_results = {}
    if "current_anonymization" not in st.session_state:
        st.session_state.current_anonymization = None

    with st.expander("About Document Anonymizer", expanded=False):
        st.markdown("""
        **Protect sensitive information** by replacing identifying details with generic placeholders.

        **Features:**
        - **Smart Entity Detection**: Automatically finds people, companies, and locations
        - **Consistent Replacement**: Same entity always gets the same placeholder
        - **Multiple Formats**: PDF, Word, and text file support

        **Replacement Examples:**
        - People: John Smith -> Person A
        - Companies: Acme Corp -> Company 1
        - Contact Info: emails -> [EMAIL], phones -> [PHONE]
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Document Input")
        batch_mode = st.toggle("Batch mode", value=False, key="anonymizer_batch_toggle")
        st.session_state["anonymizer_batch"] = batch_mode
        selected_file = _file_input_widget("anonymizer", ["pdf", "docx", "txt"])

        has_files = selected_file is not None
        file_list = selected_file if isinstance(selected_file, list) else ([selected_file] if selected_file else [])

        if has_files:
            st.divider()
            st.subheader("Anonymization Settings")
            confidence_threshold = st.slider(
                "Entity Detection Confidence:",
                min_value=0.1, max_value=0.9, value=0.3, step=0.1,
                help="Lower values detect more entities (may include false positives)",
            )
            st.session_state.confidence_threshold = confidence_threshold
            st.caption("Granular redaction controls")
            redact_people = st.checkbox("Redact people", value=True, key="anon_opt_people")
            redact_organizations = st.checkbox("Redact organizations", value=True, key="anon_opt_orgs")
            redact_projects = st.checkbox("Redact projects", value=True, key="anon_opt_projects")
            redact_locations = st.checkbox("Redact locations", value=True, key="anon_opt_locations")
            redact_personal_pronouns = st.checkbox(
                "Redact personal pronouns (he/she/they/etc.)",
                value=False,
                key="anon_opt_pronouns",
            )
            redact_company_names = st.checkbox(
                "Redact custom company names",
                value=False,
                key="anon_opt_company_names",
            )
            preserve_source_formatting = st.checkbox(
                "Preserve source formatting",
                value=True,
                key="anon_opt_preserve_formatting",
                help="Remove <CR>/<CRLF> markers and reflow hard-wrapped lines while keeping lists/headings/table-like lines.",
            )
            custom_company_names_text = st.text_area(
                "Custom company names (comma-separated)",
                value="",
                key="anon_opt_company_names_text",
                help="Optional deterministic masking list for known organization names.",
                disabled=not redact_company_names,
            )
            anonymization_options = AnonymizationOptions(
                redact_people=redact_people,
                redact_organizations=redact_organizations,
                redact_projects=redact_projects,
                redact_locations=redact_locations,
                redact_personal_pronouns=redact_personal_pronouns,
                redact_company_names=redact_company_names,
                custom_company_names=[
                    name.strip() for name in custom_company_names_text.split(",") if name.strip()
                ],
                preserve_source_formatting=preserve_source_formatting,
            )
            st.session_state.anonymization_options = anonymization_options

    with col2:
        st.header("Anonymization Process")

        if has_files:
            if len(file_list) == 1:
                st.markdown(f"**File:** `{Path(file_list[0]).name}`")
            else:
                st.info(f"{len(file_list)} document(s) selected")

            if st.button("Start Anonymization", type="primary", use_container_width=True):
                progress_bar = st.progress(0, "Initializing anonymization...")

                # Shared mapping across batch so entities stay consistent
                anonymizer = DocumentAnonymizer()
                mapping = AnonymizationMapping()
                batch_results = []

                for idx, fpath in enumerate(file_list):
                    fname = Path(fpath).name
                    base_pct = idx / len(file_list)
                    try:
                        progress_bar.progress(
                            min(base_pct + 0.02, 1.0),
                            f"[{idx+1}/{len(file_list)}] Reading {fname}..."
                        )

                        result_path, result_mapping = anonymizer.anonymize_single_file(
                            input_path=fpath,
                            output_path=None,
                            mapping=mapping,
                            confidence_threshold=st.session_state.confidence_threshold,
                            options=st.session_state.get("anonymization_options"),
                        )
                        # Re-use the returned mapping for next file (consistent entities)
                        mapping = result_mapping

                        batch_results.append({
                            "original_file": fpath,
                            "anonymized_file": result_path,
                            "mapping": result_mapping,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })

                    except Exception as e:
                        st.error(f"**Failed:** {fname}: {e}")
                        logger.error(f"Anonymization error for {fpath}: {e}", exc_info=True)

                progress_bar.progress(1.0, "Anonymization complete!")

                if batch_results:
                    # For single file, keep backward compat
                    st.session_state.current_anonymization = batch_results[0] if len(batch_results) == 1 else None
                    st.session_state.anonymization_batch_results = batch_results

                    # Summary metrics (use the final mapping which has all entities)
                    final_mapping = batch_results[-1]["mapping"]
                    st.success(f"**Anonymized {len(batch_results)} document(s) successfully!**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("People", len([k for k, v in final_mapping.mappings.items() if v.startswith("Person")]))
                    with col_b:
                        st.metric("Companies", len([k for k, v in final_mapping.mappings.items() if v.startswith("Company")]))
                    with col_c:
                        st.metric("Projects", len([k for k, v in final_mapping.mappings.items() if v.startswith("Project")]))

        # --- Display results ---
        batch_results = st.session_state.get("anonymization_batch_results")
        single_result = st.session_state.get("current_anonymization")

        if batch_results and len(batch_results) > 1:
            # Batch results
            st.divider()
            st.subheader("Anonymization Results")

            # Zip download for all anonymized files
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in batch_results:
                    zf.write(r["anonymized_file"], Path(r["anonymized_file"]).name)
            buf.seek(0)
            st.download_button(
                f"Download All {len(batch_results)} Anonymized Documents",
                buf.getvalue(),
                file_name="anonymized_documents.zip",
                mime="application/zip",
                use_container_width=True,
            )

            # Mapping report (shared across batch)
            mapping_content = _generate_mapping_report(batch_results[-1]["mapping"])
            st.download_button(
                label="Download Mapping Reference",
                data=mapping_content,
                file_name=f"anonymization_mapping_{int(time.time())}.txt",
                mime="text/plain",
                help="Reference file showing original -> anonymized mappings (keep secure!)",
            )

            # Per-file expanders
            for r in batch_results:
                orig_name = Path(r["original_file"]).name
                anon_name = Path(r["anonymized_file"]).name
                with st.expander(f"{orig_name} -> {anon_name}", expanded=False):
                    try:
                        with open(r["anonymized_file"], "r", encoding="utf-8") as f:
                            content = f.read()
                        preview = content[:2000]
                        if len(content) > 2000:
                            preview += "\n\n... [Content truncated for preview] ..."
                        st.text_area("Preview:", preview, height=200, key=f"anon_preview_{orig_name}")
                        st.download_button(
                            f"Download {anon_name}",
                            content,
                            file_name=anon_name,
                            mime="text/plain",
                            key=f"anon_dl_{orig_name}",
                        )
                    except Exception as e:
                        st.error(f"Could not load: {e}")

            # Entity mappings table
            final_mapping = batch_results[-1]["mapping"]
            if final_mapping.mappings:
                with st.expander("Entity Mappings", expanded=False):
                    import pandas as pd
                    rows = [{"Original": orig, "Anonymized": anon, "Type": _get_entity_type(anon)}
                            for orig, anon in final_mapping.mappings.items()]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        elif single_result or (batch_results and len(batch_results) == 1):
            result = single_result or batch_results[0]
            st.divider()
            st.subheader("Anonymization Results")

            col_orig, col_anon = st.columns(2)
            with col_orig:
                st.markdown("**Original File:**")
                st.code(Path(result["original_file"]).name)
            with col_anon:
                st.markdown("**Anonymized File:**")
                st.code(Path(result["anonymized_file"]).name)

            st.markdown("### Download Results")
            try:
                with open(result["anonymized_file"], "r", encoding="utf-8") as f:
                    anonymized_content = f.read()

                st.download_button(
                    label="Download Anonymized Document",
                    data=anonymized_content,
                    file_name=Path(result["anonymized_file"]).name,
                    mime="text/plain",
                    use_container_width=True,
                )

                mapping_content = _generate_mapping_report(result["mapping"])
                st.download_button(
                    label="Download Mapping Reference",
                    data=mapping_content,
                    file_name=f"anonymization_mapping_{int(time.time())}.txt",
                    mime="text/plain",
                    help="Reference file showing original -> anonymized mappings (keep secure!)",
                )

                with st.expander("Preview Anonymized Content", expanded=False):
                    preview = anonymized_content[:2000]
                    if len(anonymized_content) > 2000:
                        preview += "\n\n... [Content truncated for preview] ..."
                    st.text_area("Preview:", preview, height=300)

                if result["mapping"].mappings:
                    with st.expander("Entity Mappings", expanded=False):
                        import pandas as pd
                        rows = []
                        for original, anon in result["mapping"].mappings.items():
                            rows.append({"Original": original, "Anonymized": anon,
                                         "Type": _get_entity_type(anon)})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Could not load results: {str(e)}")

        elif has_files:
            st.info("Click **Start Anonymization** to process your document(s)")
        else:
            st.info("Select a document from the left panel to get started")


def _generate_mapping_report(mapping: AnonymizationMapping) -> str:
    """Generate a formatted mapping report."""
    report = [
        "ANONYMIZATION MAPPING REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "WARNING: KEEP THIS FILE SECURE AND SEPARATE FROM ANONYMIZED DOCUMENTS",
        "",
    ]
    if not mapping.mappings:
        report.append("No entity mappings found.")
        return "\n".join(report)

    groups = {"Person": [], "Company": [], "Project": []}
    other = []
    for original, anon in mapping.mappings.items():
        placed = False
        for prefix in groups:
            if anon.startswith(prefix):
                groups[prefix].append((original, anon))
                placed = True
                break
        if not placed:
            other.append((original, anon))

    labels = {"Person": "PEOPLE", "Company": "COMPANIES", "Project": "PROJECTS"}
    for prefix, items in groups.items():
        if items:
            report.append(f"{labels[prefix]}:")
            for orig, anon in sorted(items):
                report.append(f"  {orig} -> {anon}")
            report.append("")
    if other:
        report.append("OTHER:")
        for orig, anon in sorted(other):
            report.append(f"  {orig} -> {anon}")

    return "\n".join(report)


def _get_entity_type(anonymized: str) -> str:
    for prefix, label in [("Person", "Person"), ("Company", "Company"),
                          ("Project", "Project"), ("Location", "Location")]:
        if anonymized.startswith(prefix):
            return label
    return "Other"


# ======================================================================
# PDF Image Extractor tab
# ======================================================================

def _render_pdf_image_extract_tab():
    """Render the PDF image extraction utility."""
    st.subheader("PDF Image Extractor")
    with st.expander("About PDF Image Extractor", expanded=False):
        st.write(
            "Extracts likely photographic image blocks from PDFs into JPG files. "
            "This is useful for scanned newsletters, reports, and old flatbed PDF scans where "
            "photos need to be pulled out for separate processing."
        )
        st.caption(
            "Filtering is heuristic-based: small graphics, header/footer art, and icons can be "
            "excluded using the size and edge controls below."
        )

    col1, col2 = st.columns([1, 1.2])

    with col1:
        batch_mode = st.toggle("Batch mode (multi-file)", value=False, key="pdfimg_batch_toggle")
        st.session_state["pdfimg_batch"] = batch_mode

        min_width_px = st.slider(
            "Minimum source image width (px)",
            min_value=100,
            max_value=3000,
            value=500,
            step=50,
            key="pdfimg_min_width_px",
            help="Ignore small embedded graphics narrower than this.",
        )
        min_height_px = st.slider(
            "Minimum source image height (px)",
            min_value=100,
            max_value=3000,
            value=500,
            step=50,
            key="pdfimg_min_height_px",
            help="Ignore small embedded graphics shorter than this.",
        )
        min_page_coverage_pct = st.slider(
            "Minimum on-page coverage (%)",
            min_value=0.0,
            max_value=40.0,
            value=2.0,
            step=0.5,
            key="pdfimg_min_coverage_pct",
            help="Filters out very small items even if their source image dimensions are large.",
        )
        ignore_edge_decorations = st.checkbox(
            "Ignore likely header/footer and edge decorations",
            value=True,
            key="pdfimg_ignore_edge",
            help="Skips small banner/icon-like images close to the page edges.",
        )
        edge_margin_pct = st.slider(
            "Edge margin exclusion band (%)",
            min_value=2.0,
            max_value=20.0,
            value=8.0,
            step=1.0,
            key="pdfimg_edge_margin_pct",
            disabled=not ignore_edge_decorations,
            help="The top, bottom, left, and right edge band used for decoration filtering.",
        )
        render_scale = st.slider(
            "Output render scale",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.25,
            key="pdfimg_render_scale",
            help="Higher values create larger JPG extractions but use more memory.",
        )
        split_full_page_scans = st.checkbox(
            "Split full-page scans into separate photo crops",
            value=False,
            key="pdfimg_split_scans",
            help="When a PDF page is one large scanned image, try to detect multiple photo regions inside it.",
        )
        split_min_crop_coverage_pct = st.slider(
            "Minimum split crop area (%)",
            min_value=1.0,
            max_value=25.0,
            value=6.0,
            step=0.5,
            key="pdfimg_split_min_crop_pct",
            disabled=not split_full_page_scans,
            help="Smaller values detect more candidate crops but can pick up layout noise.",
        )

        selected = _file_input_widget("pdfimg", ["pdf"], label="Choose PDF scan:")

        if st.button("Clear Extractor Files", key="pdfimg_clear_all", use_container_width=True):
            ver = st.session_state.get("pdfimg_upload_version", 0)
            for key in list(st.session_state.keys()):
                if key.startswith("pdfimg_"):
                    del st.session_state[key]
            st.session_state["pdfimg_upload_version"] = ver + 1
            st.rerun()

    with col2:
        st.header("Output")

        if selected:
            files_to_process = selected if isinstance(selected, list) else [selected]
            st.info(f"{len(files_to_process)} PDF file(s) ready for image extraction.")

            if st.button("Extract Images to ZIP", type="primary", use_container_width=True):
                from cortex_engine.textifier import DocumentTextifier

                extractor = DocumentTextifier(use_vision=False)
                all_images: List[Dict[str, object]] = []
                extraction_rows: List[Dict[str, object]] = []
                progress = st.progress(0.0, "Starting image scan...")

                for file_idx, fpath in enumerate(files_to_process):
                    visible_name = _user_visible_filename(fpath)
                    file_base = file_idx / len(files_to_process)
                    file_span = 1.0 / len(files_to_process)

                    def _on_progress(frac, msg, _base=file_base, _span=file_span, _name=visible_name):
                        overall = min(_base + frac * _span, 1.0)
                        label = f"[{_name}] {msg}" if len(files_to_process) > 1 else msg
                        progress.progress(overall, label)

                    extractor.on_progress = _on_progress
                    try:
                        result = extractor.extract_pdf_images(
                            fpath,
                            min_width_px=int(min_width_px),
                            min_height_px=int(min_height_px),
                            min_page_coverage_pct=float(min_page_coverage_pct),
                            ignore_edge_decorations=bool(ignore_edge_decorations),
                            edge_margin_pct=float(edge_margin_pct),
                            render_scale=float(render_scale),
                            split_full_page_scans=bool(split_full_page_scans),
                            split_min_crop_coverage_pct=float(split_min_crop_coverage_pct),
                        )
                    except Exception as e:
                        st.error(f"Failed to scan {_user_visible_filename(fpath)}: {e}")
                        logger.error(f"PDF image extraction failed for {fpath}: {e}", exc_info=True)
                        continue

                    extraction_rows.append(
                        {
                            "pdf": visible_name,
                            "pages": int(result.get("pages_scanned", 0) or 0),
                            "detected": int(result.get("detected_blocks", 0) or 0),
                            "extracted": len(result.get("images", []) or []),
                            "skipped_small": int(result.get("skipped_small", 0) or 0),
                            "skipped_edge": int(result.get("skipped_edge", 0) or 0),
                            "skipped_invalid": int(result.get("skipped_invalid", 0) or 0),
                            "split_generated": int(result.get("split_generated", 0) or 0),
                            "message": str(result.get("message", "") or ""),
                        }
                    )

                    for image in result.get("images", []) or []:
                        image_record = dict(image)
                        image_record["zip_path"] = f"{Path(visible_name).stem}/{image_record['file_name']}"
                        all_images.append(image_record)

                progress.progress(1.0, "Done!")

                st.session_state["pdfimg_results"] = extraction_rows
                if all_images:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        summary_lines = [
                            "Cortex PDF Image Extractor",
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "",
                        ]
                        for row in extraction_rows:
                            summary_lines.append(
                                f"{row['pdf']}: extracted {row['extracted']} of {row['detected']} detected "
                                f"image block(s); skipped small={row['skipped_small']}, "
                                f"edge={row['skipped_edge']}, invalid={row['skipped_invalid']}."
                            )
                        summary_lines.append("")
                        summary_lines.append("Extracted files:")
                        for image in all_images:
                            zf.writestr(str(image["zip_path"]), image["bytes"])
                            summary_lines.append(
                                f"- {image['zip_path']} | page {image['page']} | "
                                f"source {image['intrinsic_width']}x{image['intrinsic_height']} px | "
                                f"coverage {image['coverage_pct']}% | bbox {image['bbox']}"
                            )
                        zf.writestr("extraction_summary.txt", "\n".join(summary_lines) + "\n")

                    zip_name = "pdf_extracted_images.zip" if len(files_to_process) > 1 else (
                        f"{Path(_user_visible_filename(files_to_process[0])).stem}_images.zip"
                    )
                    st.session_state["pdfimg_zip_bytes"] = buf.getvalue()
                    st.session_state["pdfimg_zip_name"] = zip_name
                else:
                    st.session_state.pop("pdfimg_zip_bytes", None)
                    st.session_state.pop("pdfimg_zip_name", None)

        rows = st.session_state.get("pdfimg_results")
        zip_bytes = st.session_state.get("pdfimg_zip_bytes")
        zip_name = st.session_state.get("pdfimg_zip_name", "pdf_extracted_images.zip")

        if rows:
            st.divider()
            st.subheader("Extraction Results")

            total_detected = sum(int(row.get("detected", 0) or 0) for row in rows)
            total_extracted = sum(int(row.get("extracted", 0) or 0) for row in rows)
            total_split = sum(int(row.get("split_generated", 0) or 0) for row in rows)
            metric_cols = st.columns(4)
            metric_cols[0].metric("PDFs Scanned", len(rows))
            metric_cols[1].metric("Image Blocks Found", total_detected)
            metric_cols[2].metric("JPGs Extracted", total_extracted)
            metric_cols[3].metric("Split Crops", total_split)

            for row in rows:
                st.write(
                    f"**{row['pdf']}**: extracted {row['extracted']} / {row['detected']} "
                    f"(small skipped: {row['skipped_small']}, edge skipped: {row['skipped_edge']}, "
                    f"invalid skipped: {row['skipped_invalid']}, split crops: {row['split_generated']})"
                )

            if zip_bytes:
                st.download_button(
                    "Download Extracted JPGs (ZIP)",
                    zip_bytes,
                    file_name=zip_name,
                    mime="application/zip",
                    use_container_width=True,
                )
            else:
                st.warning(
                    "No images met the current thresholds. Lower the size or coverage filters "
                    "if this PDF contains smaller photos."
                )


# ======================================================================
# Photo Processor tab
# ======================================================================

def _render_photo_keywords_tab():
    """Render the Photo Processor tool for batch resize and photo metadata workflows."""
    st.markdown(
        "Process photos in batch: resize for gallery use, generate AI keywords, "
        "clean sensitive tags, and write EXIF/XMP ownership metadata."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")

        st.session_state["photokw_batch"] = True  # always batch-capable

        # Upload version counter for clear button
        if "photokw_upload_version" not in st.session_state:
            st.session_state["photokw_upload_version"] = 0
        ver = st.session_state["photokw_upload_version"]

        uploaded_input = st.file_uploader(
            "Drop photos here:",
            type=["png", "jpg", "jpeg", "tiff", "webp", "gif", "bmp"],
            accept_multiple_files=True,
            key=f"photokw_upload_v{ver}",
        )
        upload_cache_key = "photokw_uploaded_cache"
        if uploaded_input:
            uploaded = [_SessionUpload(uf.name, uf.getvalue()) for uf in uploaded_input]
            st.session_state[upload_cache_key] = [
                {"name": uf.name, "data": uf.getvalue()}
                for uf in uploaded_input
            ]
        else:
            cached_uploads = st.session_state.get(upload_cache_key) or []
            uploaded = [
                _SessionUpload(str(item.get("name") or ""), item.get("data") or b"")
                for item in cached_uploads
                if isinstance(item, dict) and item.get("name")
            ]

        write_to_original = st.toggle(
            "Write to original files",
            value=False,
            key="photokw_write_original",
            help="When OFF, keywords are written to copies in a temp folder (originals untouched). "
                 "When ON, keywords are written directly to the uploaded files.",
        )

        city_radius = st.slider(
            "City location radius",
            min_value=1, max_value=50, value=5, step=1,
            key="photokw_city_radius",
            help="Radius (km) for city-level reverse geocoding of GPS coordinates. "
                 "Larger values may match broader city names for rural locations.",
        )

        clear_keywords = st.checkbox(
            "Clear existing keywords/tags first",
            value=False,
            key="photokw_clear_keywords",
            help="Remove all existing XMP Subject and IPTC Keywords before writing new ones.",
        )
        clear_location = st.checkbox(
            "Clear existing location fields first",
            value=False,
            key="photokw_clear_location",
            help="Remove existing Country, State, and City EXIF fields and rebuild from GPS/location hints.",
        )
        generate_description = st.checkbox(
            "Generate AI description + keywords",
            value=True,
            key="photokw_generate_description",
            help="Writes a fresh description and any new keywords. Turn this off to update only location/GPS metadata.",
        )
        populate_location = st.checkbox(
            "Fill location and GPS metadata",
            value=True,
            key="photokw_populate_location",
            help="Completes City/State/Country from GPS, or derives GPS from City/Country hints when GPS is missing.",
        )
        fallback_city = st.text_input(
            "Fallback city (optional)",
            value="",
            key="photokw_fallback_city",
            disabled=not populate_location,
            help="Used only when a photo has no GPS and no embedded location fields.",
        )
        fallback_country = st.text_input(
            "Fallback country (optional)",
            value="",
            key="photokw_fallback_country",
            disabled=not populate_location,
            help="If only a country is provided, Cortex will try that country's capital city.",
        )
        resize_profile = st.selectbox(
            "Resize profile",
            options=["Keep original dimensions", "Low (1920 x 1080)", "Medium (2560 x 1440)"],
            index=0,
            key="photokw_resize_profile",
            help="Maximum output dimensions. Photos already below the selected profile are not resized.",
        )
        no_resize_selected = resize_profile == "Keep original dimensions"
        convert_to_jpg = st.checkbox(
            "Convert resized output to JPG",
            value=False,
            key="photokw_convert_to_jpg",
            disabled=no_resize_selected,
            help="Only non-JPG sources are converted. Existing JPG files stay JPG and are only resized when needed.",
        )
        jpg_quality = st.slider(
            "JPG quality",
            min_value=60,
            max_value=100,
            value=90,
            step=1,
            key="photokw_jpg_quality",
            disabled=(not convert_to_jpg) or no_resize_selected,
            help="Only applies when JPG conversion is enabled.",
        )
        halftone_strength = st.slider(
            "Halftone repair strength",
            min_value=0,
            max_value=100,
            value=42,
            step=1,
            key="photokw_halftone_strength",
            help="Lower values are gentler. Higher values remove more screen pattern but can soften detail.",
        )
        st.caption(f"Current repair profile: {_halftone_strength_label(halftone_strength)}")
        halftone_preserve_color = st.checkbox(
            "Preserve colour during halftone repair",
            value=True,
            key="photokw_halftone_preserve_color",
            help="When enabled, Cortex repairs the luminance channel and keeps colour information where possible.",
        )

        anonymize_keywords = st.checkbox(
            "Anonymize sensitive keywords",
            value=False,
            key="photokw_anonymize_keywords",
            help="Remove personal/sensitive tags from generated keywords using the blocked list below.",
        )
        blocked_keywords_text = st.text_input(
            "Blocked keywords (comma-separated)",
            value="friends,family,paul,paul_c,jacqui",
            key="photokw_blocked_keywords",
            help="These keywords are removed when anonymization is enabled.",
        )

        apply_ownership = st.checkbox(
            "Insert ownership info",
            value=True,
            key="photokw_apply_ownership",
            help="Write ownership/copyright metadata fields in EXIF/IPTC/XMP.",
        )
        ownership_notice = st.text_area(
            "Ownership notice",
            value="All rights (c) Longboardfella. Contact longboardfella.com for info on use of photos.",
            key="photokw_ownership_notice",
            height=90,
        )
        photokw_batch_cooldown_seconds = st.slider(
            "Cooldown between photos (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="photokw_batch_cooldown_s",
            help="Adds a short pause between photos to keep batch processing more responsive.",
        )

        if st.button("Clear All Photos", key="photokw_clear", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("photokw_") and key != "photokw_upload_version":
                    del st.session_state[key]
            st.session_state["photokw_upload_version"] = ver + 1
            if "photokw_results" in st.session_state:
                del st.session_state["photokw_results"]
            temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw"
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            st.rerun()

    with col2:
        st.header("Results")

        if uploaded:
            accepted_uploads = []
            accepted_bytes = 0
            for uf in uploaded:
                size_bytes = int(getattr(uf, "size", 0) or 0)
                if size_bytes <= 0:
                    try:
                        size_bytes = len(uf.getvalue())
                    except Exception:
                        size_bytes = 0
                if accepted_bytes + size_bytes > MAX_BATCH_UPLOAD_BYTES:
                    break
                accepted_uploads.append(uf)
                accepted_bytes += size_bytes
            if not accepted_uploads:
                st.error("Selected photos exceed the 1GB total upload limit.")
                return
            if len(accepted_uploads) < len(uploaded):
                st.warning(
                    f"Maximum 1GB total upload per photo batch — only the first "
                    f"{len(accepted_uploads)} of {len(uploaded)} photos will be processed."
                )
            uploaded = accepted_uploads
            st.session_state[upload_cache_key] = [
                {"name": uf.name, "data": uf.getvalue()}
                for uf in uploaded
            ]
            total = len(uploaded)
            st.info(f"{total} photo(s) selected ({accepted_bytes / (1024 * 1024):.1f} MB total)")

            # Single-photo metadata preview for quick testing before processing.
            if total == 1:
                preview_photo = uploaded[0]
                preview_bytes = preview_photo.getvalue()
                preview_temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw_preview"
                preview_temp_dir.mkdir(exist_ok=True, mode=0o755)

                # Keep a stable working copy path for this uploaded file across reruns,
                # so quick metadata edits are not lost.
                preview_sig = f"{preview_photo.name}:{len(preview_bytes)}:{hashlib.md5(preview_bytes).hexdigest()}"
                existing_sig = st.session_state.get("photokw_single_upload_sig")
                existing_path = st.session_state.get("photokw_single_working_path")
                if preview_sig != existing_sig or not existing_path or not Path(existing_path).exists():
                    preview_path = preview_temp_dir / preview_photo.name
                    with open(preview_path, "wb") as f:
                        f.write(preview_bytes)
                    os.chmod(str(preview_path), 0o644)
                    st.session_state["photokw_single_upload_sig"] = preview_sig
                    st.session_state["photokw_single_working_path"] = str(preview_path)
                    st.session_state.pop("photokw_vlm_probe", None)
                else:
                    preview_path = Path(existing_path)

                probe_col, refresh_col = st.columns(2)
                run_probe = probe_col.button("Run VLM Diagnostic Probe", key="photokw_vlm_probe_btn", use_container_width=True)
                if refresh_col.button("Refresh Preview", key="photokw_preview_refresh", use_container_width=True):
                    st.rerun()

                if run_probe:
                    from cortex_engine.textifier import DocumentTextifier

                    probe_result = DocumentTextifier(use_vision=True).probe_image_vlm(preview_bytes, simple_prompt=True)
                    st.session_state["photokw_vlm_probe"] = probe_result

                preview_meta = _read_photo_metadata_preview(str(preview_path))
                probe_result = st.session_state.get("photokw_vlm_probe")

                with st.expander("Single Photo Metadata Preview", expanded=True):
                    st.image(preview_bytes, caption=preview_photo.name, width=420)
                    if probe_result:
                        st.markdown("**VLM Diagnostic Probe**")
                        st.json(probe_result)
                        st.caption(
                            "This shows the raw response shape from Ollama for the current photo using the simple prompt."
                        )
                        st.divider()
                    if preview_meta.get("available"):
                        description = preview_meta.get("description", "")
                        keywords = preview_meta.get("keywords", [])
                        city = preview_meta.get("city", "")
                        state = preview_meta.get("state", "")
                        country = preview_meta.get("country", "")
                        gps = preview_meta.get("gps")

                        if description:
                            st.markdown(f"**Description:** {description}")
                        else:
                            st.caption("No existing description found in metadata.")

                        if keywords:
                            st.markdown(f"**Keywords ({len(keywords)}):** {', '.join(keywords)}")
                        else:
                            st.caption("No existing keywords found in metadata.")

                        location_parts = [v for v in [city, state, country] if v]
                        if location_parts:
                            st.markdown(f"**Location fields:** {', '.join(location_parts)}")
                        else:
                            st.caption("No existing City/State/Country metadata found.")
                        if gps:
                            st.caption(f"GPS: {gps}")

                        st.divider()
                        st.markdown("**Quick Edit Metadata**")
                        edit_keywords = st.text_area(
                            "Edit keywords (comma-separated)",
                            value=", ".join(keywords),
                            key="photokw_edit_keywords",
                            height=90,
                        )
                        edit_description = st.text_area(
                            "Edit description",
                            value=description,
                            key="photokw_edit_description",
                            height=90,
                        )
                        ec1, ec2, ec3 = st.columns(3)
                        with ec1:
                            edit_city = st.text_input("City", value=city, key="photokw_edit_city")
                        with ec2:
                            edit_state = st.text_input("State", value=state, key="photokw_edit_state")
                        with ec3:
                            edit_country = st.text_input("Country", value=country, key="photokw_edit_country")

                        if st.button("Apply Quick Metadata Edits", key="photokw_apply_quick_edit", use_container_width=True):
                            edited_keywords = [k.strip() for k in edit_keywords.split(",") if k.strip()]
                            write_result = _write_photo_metadata_quick_edit(
                                str(preview_path),
                                keywords=edited_keywords,
                                description=edit_description,
                                city=edit_city,
                                state=edit_state,
                                country=edit_country,
                            )
                            if write_result.get("success"):
                                st.success("Metadata edits applied.")
                                st.rerun()
                            else:
                                st.error(f"Could not apply metadata edits: {write_result.get('message', 'Unknown error')}")
                    else:
                        st.info(f"Metadata preview unavailable: {preview_meta.get('reason', 'Unknown reason')}")

                halftone_preview_state = st.session_state.get("photokw_halftone_preview") or {}
                halftone_preview_matches = (
                    halftone_preview_state.get("source_sig") == preview_sig
                    and float(halftone_preview_state.get("strength", -1)) == float(halftone_strength)
                    and bool(halftone_preview_state.get("preserve_color")) == bool(halftone_preserve_color)
                    and Path(str(halftone_preview_state.get("output_path") or "")).exists()
                )

                with st.expander("Halftone Repair Preview", expanded=True):
                    st.caption(
                        "Generate a repaired preview for the current strength, then inspect the full image and a zoomed crop before batch processing."
                    )
                    preview_button_cols = st.columns(2)
                    generate_halftone_preview = preview_button_cols[0].button(
                        "Generate Halftone Preview",
                        key="photokw_generate_halftone_preview",
                        use_container_width=True,
                    )
                    clear_halftone_preview = preview_button_cols[1].button(
                        "Clear Preview",
                        key="photokw_clear_halftone_preview",
                        use_container_width=True,
                    )

                    if clear_halftone_preview:
                        existing_preview = Path(str(halftone_preview_state.get("output_path") or ""))
                        if existing_preview.exists():
                            try:
                                existing_preview.unlink()
                            except Exception:
                                pass
                        st.session_state.pop("photokw_halftone_preview", None)
                        halftone_preview_state = {}
                        halftone_preview_matches = False

                    if generate_halftone_preview:
                        from cortex_engine.textifier import DocumentTextifier

                        preview_output_path = preview_temp_dir / (
                            f"{preview_path.stem}_halftone_preview_{int(halftone_strength)}"
                            f"{'_color' if halftone_preserve_color else '_mono'}{preview_path.suffix}"
                        )
                        shutil.copy2(preview_path, preview_output_path)
                        os.chmod(str(preview_output_path), 0o644)

                        with st.spinner("Generating halftone preview..."):
                            preview_result = DocumentTextifier(use_vision=False).repair_halftone_image(
                                str(preview_output_path),
                                strength=halftone_strength,
                                preserve_color=halftone_preserve_color,
                                convert_to_jpg=False,
                            )
                        preview_info = preview_result.get("halftone_repair_info", {})
                        if preview_info.get("repaired"):
                            st.session_state["photokw_halftone_preview"] = {
                                "source_sig": preview_sig,
                                "strength": float(halftone_strength),
                                "preserve_color": bool(halftone_preserve_color),
                                "output_path": str(preview_result.get("output_path") or preview_output_path),
                            }
                            halftone_preview_state = st.session_state["photokw_halftone_preview"]
                            halftone_preview_matches = True
                        else:
                            st.error(f"Preview generation failed: {preview_info.get('error', 'Unknown error')}")

                    if halftone_preview_state and not halftone_preview_matches:
                        st.info("Preview settings changed. Generate a new preview to match the current strength and colour options.")

                    if halftone_preview_matches:
                        preview_output_path = str(halftone_preview_state["output_path"])
                        _render_halftone_ab_compare(
                            str(preview_path),
                            preview_output_path,
                            strength=float(halftone_strength),
                            widget_prefix="photokw_halftone_preview_compare",
                            heading="Preview A/B Window",
                        )

            resolution_map = {
                "Keep original dimensions": (None, None),
                "Low (1920 x 1080)": (1920, 1080),
                "Medium (2560 x 1440)": (2560, 1440),
            }
            max_width, max_height = resolution_map.get(resize_profile, (None, None))

            action_cols = st.columns(3)
            do_resize_only = action_cols[0].button("Resize Photos Only", use_container_width=True)
            do_halftone_repair = action_cols[1].button("Repair Halftone Artefacts", use_container_width=True)
            do_keywords = action_cols[2].button("Process Selected Metadata", type="primary", use_container_width=True)

            if do_resize_only or do_halftone_repair or do_keywords:
                if do_keywords and not any([generate_description, populate_location, apply_ownership]):
                    st.warning("Select at least one metadata action before processing.")
                    return

                from cortex_engine.textifier import DocumentTextifier

                # Save uploads to temp dir
                temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw"
                temp_dir.mkdir(exist_ok=True, mode=0o755)
                file_paths = []
                if total == 1 and st.session_state.get("photokw_single_working_path"):
                    working_path = st.session_state.get("photokw_single_working_path")
                    if working_path and Path(working_path).exists():
                        dest = temp_dir / uploaded[0].name
                        shutil.copy2(working_path, dest)
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))
                    else:
                        uf = uploaded[0]
                        dest = temp_dir / uf.name
                        with open(dest, "wb") as f:
                            f.write(uf.getvalue())
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))
                else:
                    for uf in uploaded:
                        dest = temp_dir / uf.name
                        with open(dest, "wb") as f:
                            f.write(uf.getvalue())
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))

                textifier = DocumentTextifier(use_vision=True)
                results = []
                if do_resize_only:
                    mode = "resize_only"
                    progress_message = "Starting resize..."
                elif do_halftone_repair:
                    mode = "halftone_repair"
                    progress_message = "Starting halftone repair..."
                else:
                    mode = "keyword_metadata"
                    progress_message = "Starting metadata processing..."
                progress = st.progress(0.0, progress_message)
                blocked_keywords = [k.strip().lower() for k in blocked_keywords_text.split(",") if k.strip()]

                for idx, fpath in enumerate(file_paths):
                    fname = Path(fpath).name

                    def _on_progress(frac, msg, _idx=idx, _total=total, _name=fname):
                        overall = min((_idx + frac) / _total, 1.0)
                        progress.progress(overall, f"[{_name}] {msg}")

                    textifier.on_progress = _on_progress
                    try:
                        if do_resize_only:
                            if max_width is None or max_height is None:
                                result = {
                                    "file_name": Path(fpath).name,
                                    "output_path": fpath,
                                    "resize_info": {
                                        "resized": False,
                                        "metadata_preserved": True,
                                        "skipped_resize": True,
                                    },
                                }
                            else:
                                result = textifier.resize_image_only(
                                    fpath,
                                    max_width=max_width,
                                    max_height=max_height,
                                    convert_to_jpg=convert_to_jpg,
                                    jpg_quality=jpg_quality,
                                )
                            output_path = str(result.get("output_path", fpath))
                            file_paths[idx] = output_path
                            if anonymize_keywords:
                                result["keyword_anonymize_result"] = textifier.anonymize_existing_photo_keywords(
                                    output_path, blocked_keywords=blocked_keywords
                                )
                            if apply_ownership and ownership_notice.strip():
                                result["ownership_result"] = textifier.write_ownership_metadata(
                                    output_path, ownership_notice.strip()
                                )
                        elif do_halftone_repair:
                            result = textifier.repair_halftone_image(
                                fpath,
                                strength=halftone_strength,
                                preserve_color=halftone_preserve_color,
                                convert_to_jpg=convert_to_jpg,
                                jpg_quality=jpg_quality,
                            )
                            output_path = str(result.get("output_path", fpath))
                            file_paths[idx] = output_path
                            if apply_ownership and ownership_notice.strip():
                                result["ownership_result"] = textifier.write_ownership_metadata(
                                    output_path, ownership_notice.strip()
                                )
                        else:
                            result = textifier.keyword_image(
                                fpath, city_radius_km=city_radius,
                                clear_keywords=(clear_keywords if generate_description else False),
                                clear_location=(clear_location if populate_location else False),
                                generate_description=generate_description,
                                populate_location=populate_location,
                                anonymize_keywords=anonymize_keywords,
                                blocked_keywords=blocked_keywords,
                                fallback_city=fallback_city,
                                fallback_country=fallback_country,
                                ownership_notice=(ownership_notice.strip() if apply_ownership else ""),
                            )
                        results.append(result)
                    except Exception as e:
                        st.error(f"Failed: {fname}: {e}")
                        logger.error(f"Photo keyword error for {fpath}: {e}", exc_info=True)
                    if photokw_batch_cooldown_seconds > 0 and total > 1 and idx < total - 1:
                        progress.progress(
                            min((idx + 1) / total, 1.0),
                            f"Cooling down for {photokw_batch_cooldown_seconds:.1f}s before next photo..."
                        )
                        time.sleep(photokw_batch_cooldown_seconds)

                progress.progress(1.0, "Done!")

                # If writing to originals, user needs to copy back — but since
                # we're working on uploaded copies in temp, the writes already happened.
                # The user downloads the processed files.
                if results:
                    st.session_state["photokw_results"] = results
                    st.session_state["photokw_paths"] = file_paths
                    st.session_state["photokw_mode"] = mode

        # Display results
        results = st.session_state.get("photokw_results")
        file_paths = st.session_state.get("photokw_paths", [])
        photokw_mode = st.session_state.get("photokw_mode", "keyword_metadata")

        if results:
            st.divider()

            if photokw_mode == "resize_only":
                resized_count = sum(1 for r in results if r.get("resize_info", {}).get("resized"))
                total_removed = sum(len(r.get("keyword_anonymize_result", {}).get("removed_keywords", [])) for r in results)
                ownership_ok = sum(
                    1 for r in results
                    if not r.get("ownership_result") or r.get("ownership_result", {}).get("success")
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Photos Processed", len(results))
                with mc2:
                    st.metric("Resized", f"{resized_count}/{len(results)}")
                with mc3:
                    st.metric("Sensitive Tags Removed", total_removed)
                with mc4:
                    st.metric("Ownership Written", f"{ownership_ok}/{len(results)}")
            elif photokw_mode == "halftone_repair":
                repaired_count = sum(1 for r in results if r.get("halftone_repair_info", {}).get("repaired"))
                converted_count = sum(1 for r in results if r.get("halftone_repair_info", {}).get("converted_to_jpg"))
                ownership_ok = sum(
                    1 for r in results
                    if not r.get("ownership_result") or r.get("ownership_result", {}).get("success")
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Photos Processed", len(results))
                with mc2:
                    st.metric("Repaired", f"{repaired_count}/{len(results)}")
                with mc3:
                    st.metric("Converted To JPG", converted_count)
                with mc4:
                    st.metric("Ownership Written", f"{ownership_ok}/{len(results)}")
            else:
                # Summary metrics
                total_new = sum(len(r.get("new_keywords", [])) for r in results)
                total_existing = sum(len(r.get("existing_keywords", [])) for r in results)
                total_removed = sum(len(r.get("removed_sensitive_keywords", [])) for r in results)
                successful = sum(1 for r in results if r["exif_result"]["success"])
                location_written = sum(
                    1 for r in results if r.get("location_result", {}).get("location_written")
                )
                gps_written = sum(
                    1 for r in results if r.get("location_result", {}).get("gps_written")
                )
                ownership_ok = sum(
                    1 for r in results
                    if not r.get("ownership_result") or r.get("ownership_result", {}).get("success")
                )
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                with mc1:
                    st.metric("Photos Processed", len(results))
                with mc2:
                    st.metric("Existing Tags", total_existing)
                with mc3:
                    st.metric("New Tags Added", total_new)
                with mc4:
                    st.metric("Sensitive Tags Removed", total_removed)
                with mc5:
                    st.metric("Metadata Written", f"{successful}/{len(results)}")
                st.caption(
                    f"Location fields written: {location_written}/{len(results)} | "
                    f"GPS written: {gps_written}/{len(results)} | "
                    f"Ownership metadata written: {ownership_ok}/{len(results)}"
                )

            # Download — single file direct, multiple as zip
            if file_paths:
                if len(file_paths) == 1:
                    fpath = file_paths[0]
                    fname = Path(fpath).name
                    mime = "image/jpeg" if fname.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    with open(fpath, "rb") as dl_f:
                        st.download_button(
                            f"Download {fname}",
                            dl_f.read(),
                            file_name=fname,
                            mime=mime,
                            use_container_width=True,
                        )
                else:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fpath in file_paths:
                            zf.write(fpath, Path(fpath).name)
                    buf.seek(0)
                    st.download_button(
                        f"Download All {len(file_paths)} Photos",
                        buf.getvalue(),
                        file_name="processed_photos.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

            # Per-image details
            if len(results) == 1:
                # Single photo — show inline preview (like Textifier)
                r = results[0]
                resize_info = r.get("resize_info", {})
                repair_info = r.get("halftone_repair_info", {})
                ownership_result = r.get("ownership_result")
                if photokw_mode == "resize_only":
                    if resize_info.get("skipped_resize"):
                        st.info(f"Left dimensions unchanged for {r['file_name']}")
                    elif resize_info.get("resized"):
                        st.success(
                            f"Resized {r['file_name']}: "
                            f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                            f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                        )
                    elif resize_info.get("converted_to_jpg"):
                        st.success(f"Converted to JPG: {r['file_name']}")
                    else:
                        st.info(f"No resize needed for {r['file_name']}")
                    if ownership_result:
                        if ownership_result.get("success"):
                            st.success("Ownership metadata written")
                        else:
                            st.warning(f"Ownership metadata write failed: {ownership_result.get('message', '')}")
                elif photokw_mode == "halftone_repair":
                    if repair_info.get("repaired"):
                        st.success(
                            f"Applied halftone repair strength {int(round(float(repair_info.get('strength', 0))))} "
                            f"to {r['file_name']}"
                        )
                    else:
                        st.error(f"Halftone repair failed: {repair_info.get('error', 'Unknown error')}")
                    if ownership_result:
                        if ownership_result.get("success"):
                            st.success("Ownership metadata written")
                        else:
                            st.warning(f"Ownership metadata write failed: {ownership_result.get('message', '')}")
                else:
                    exif = r["exif_result"]
                    desc_issue = _photo_description_issue(r.get("description", ""))
                    if exif.get("message") == "Keyword generation skipped":
                        st.info("Description/keyword generation skipped")
                    elif exif["success"]:
                        st.success(f"EXIF written: {exif['keywords_written']} keywords to {r['file_name']}")
                    else:
                        st.error(f"EXIF write failed: {exif['message']}")
                    if desc_issue:
                        st.warning(desc_issue)
                    if ownership_result:
                        if ownership_result.get("success"):
                            st.success("Ownership metadata written")
                        else:
                            st.warning(f"Ownership metadata write failed: {ownership_result.get('message', '')}")

                    # GPS / location feedback
                    location_result = r.get("location_result", {})
                    if not location_result.get("enabled"):
                        st.caption("Location/GPS processing was skipped.")
                    elif r.get("location") and any((r.get("location") or {}).values()):
                        loc = r["location"]
                        parts = [v for v in [loc.get("city"), loc.get("state"), loc.get("country")] if v]
                        if parts:
                            st.info(f"Location: **{', '.join(parts)}**")
                        if r.get("gps"):
                            st.caption(f"GPS: {r['gps'][0]:.5f}, {r['gps'][1]:.5f}")
                        if location_result.get("gps_written"):
                            st.success("GPS coordinates were derived and written.")
                    else:
                        st.warning(
                            f"No GPS or usable location hint was found for **{r['file_name']}**. "
                            "Add fallback City/Country to auto-fill empty photos."
                        )

                with st.expander("Preview", expanded=True):
                    # Show thumbnail of the photo
                    if file_paths and Path(file_paths[0]).exists():
                        st.image(file_paths[0], caption=r["file_name"], width=400)
                    if resize_info.get("resized"):
                        st.caption(
                            "Resized: "
                            f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                            f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                        )
                    if photokw_mode == "resize_only":
                        st.markdown(
                            f"**Metadata preserved after resize:** "
                            f"{'Yes' if resize_info.get('metadata_preserved') else 'Partial/Unknown'}"
                        )
                        if resize_info.get("skipped_resize"):
                            st.caption("Dimensions were left unchanged by request.")
                    elif photokw_mode == "halftone_repair":
                        strength_value = float(repair_info.get("strength", 0))
                        st.markdown(
                            f"**Repair strength:** {int(round(strength_value))} ({_halftone_strength_label(strength_value)})  \n"
                            f"**Preserve colour:** {'Yes' if repair_info.get('preserve_color') else 'No'}  \n"
                            f"**Metadata preserved after repair:** {'Yes' if repair_info.get('metadata_preserved') else 'Partial/Unknown'}"
                        )
                        if repair_info.get("converted_to_jpg"):
                            st.caption("Converted repaired output to JPG.")
                        original_compare_path = st.session_state.get("photokw_single_working_path")
                        if original_compare_path and Path(str(original_compare_path)).exists() and file_paths:
                            st.divider()
                            _render_halftone_ab_compare(
                                str(original_compare_path),
                                str(file_paths[0]),
                                strength=strength_value,
                                widget_prefix="photokw_halftone_result_compare",
                                heading="Result A/B Window",
                            )
                    else:
                        desc = r["description"] or "(no description generated)"
                        st.markdown(f"**Description:**\n\n{desc}")
                        desc_issue = _photo_description_issue(desc)
                        if desc_issue:
                            st.warning(desc_issue)
                        st.divider()
                        # Location fields
                        if r.get("location") and any(r["location"].values()):
                            loc = r["location"]
                            st.markdown(
                                f"**Location:** {loc.get('city', '')} · "
                                f"{loc.get('state', '')} · {loc.get('country', '')}"
                            )
                            if r.get("gps"):
                                st.caption(f"GPS: {r['gps'][0]:.5f}, {r['gps'][1]:.5f}")
                            st.divider()
                        existing = r.get("existing_keywords", [])
                        new_kw = r.get("new_keywords", [])
                        removed_kw = r.get("removed_sensitive_keywords", [])
                        if existing:
                            st.markdown(f"**Existing tags ({len(existing)}):** {', '.join(existing)}")
                        if new_kw:
                            st.markdown(f"**New tags added ({len(new_kw)}):** {', '.join(new_kw)}")
                        elif not existing:
                            st.warning("No keywords generated — the vision model may have failed to describe this image.")
                        if removed_kw:
                            st.caption(f"Removed sensitive tags: {', '.join(removed_kw)}")
                        st.markdown(f"**Combined keywords ({len(r['keywords'])}):**")
                        if r["keywords"]:
                            st.markdown(", ".join(r["keywords"]))
            else:
                # Batch mode
                if photokw_mode != "resize_only":
                    no_gps = [
                        r for r in results
                        if "nogps" in [kw.lower() for kw in r.get("keywords", [])]
                    ]
                    if no_gps:
                        st.warning(
                            f"**{len(no_gps)} photo(s) have no GPS data** — tagged with "
                            f"'nogps' for easy filtering: "
                            f"{', '.join(r['file_name'] for r in no_gps)}"
                        )
                for r in results:
                    resize_info = r.get("resize_info", {})
                    loc = r.get("location")
                    loc_label = ""
                    if loc and any(loc.values()):
                        parts = [v for v in [loc.get("city"), loc.get("state"), loc.get("country")] if v]
                        loc_label = f"  —  {', '.join(parts)}"
                    with st.expander(f"{r['file_name']}{loc_label}", expanded=False):
                        # Show thumbnail in batch mode too
                        idx = next((i for i, fp in enumerate(file_paths) if Path(fp).name == r["file_name"]), None)
                        if idx is not None and Path(file_paths[idx]).exists():
                            st.image(file_paths[idx], caption=r["file_name"], width=300)
                        if resize_info.get("resized"):
                            st.caption(
                                "Resized: "
                                f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                                f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                            )
                        if photokw_mode == "resize_only":
                            st.caption(
                                "Metadata preserved after resize: "
                                f"{'Yes' if resize_info.get('metadata_preserved') else 'Partial/Unknown'}"
                            )
                            if resize_info.get("converted_to_jpg"):
                                st.caption("Converted to JPG")
                            anon_result = r.get("keyword_anonymize_result")
                            if anon_result:
                                if anon_result.get("success"):
                                    removed = anon_result.get("removed_keywords", [])
                                    if removed:
                                        st.caption(f"Removed sensitive tags: {', '.join(removed)}")
                                    else:
                                        st.caption("No sensitive tags removed.")
                                else:
                                    st.warning(
                                        f"Keyword anonymization failed: {anon_result.get('message', 'Unknown error')}"
                                    )
                        elif photokw_mode == "halftone_repair":
                            repair_info = r.get("halftone_repair_info", {})
                            if repair_info.get("repaired"):
                                strength_value = float(repair_info.get("strength", 0))
                                st.caption(
                                    f"Repair strength: {int(round(strength_value))} ({_halftone_strength_label(strength_value)}) | "
                                    f"Preserve colour: {'Yes' if repair_info.get('preserve_color') else 'No'} | "
                                    f"Metadata preserved: {'Yes' if repair_info.get('metadata_preserved') else 'Partial/Unknown'}"
                                )
                                if repair_info.get("converted_to_jpg"):
                                    st.caption("Converted repaired output to JPG")
                            else:
                                st.error(f"Repair failed: {repair_info.get('error', 'Unknown error')}")
                        else:
                            desc = r.get('description') or '(no description)'
                            st.markdown(f"**Description:** {desc}")
                            desc_issue = _photo_description_issue(desc)
                            if desc_issue:
                                st.warning(desc_issue)
                            if loc and any(loc.values()):
                                st.markdown(
                                    f"**Location:** {loc.get('city', '')} · "
                                    f"{loc.get('state', '')} · {loc.get('country', '')}"
                                )
                            elif not r.get("has_gps"):
                                st.caption("No GPS data — tagged 'nogps'")
                            existing = r.get("existing_keywords", [])
                            new_kw = r.get("new_keywords", [])
                            removed_kw = r.get("removed_sensitive_keywords", [])
                            if existing:
                                st.caption(f"Existing: {', '.join(existing)}")
                            if new_kw:
                                st.caption(f"Added: {', '.join(new_kw)}")
                            if removed_kw:
                                st.caption(f"Removed: {', '.join(removed_kw)}")
                            st.markdown(f"**Keywords ({len(r['keywords'])}):** {', '.join(r['keywords'])}")
                            exif = r["exif_result"]
                            if exif.get("message") == "Keyword generation skipped":
                                st.info("Description/keyword generation skipped")
                            elif exif["success"]:
                                st.success(f"EXIF written: {exif['keywords_written']} new keywords")
                            else:
                                st.error(f"EXIF write failed: {exif['message']}")

        elif uploaded:
            st.info("Choose an action: **Resize Photos Only**, **Repair Halftone Artefacts**, or **Process Selected Metadata**")
        else:
            st.info("Upload photos from the left panel to get started")


# ======================================================================
# Included Study Extractor tab
# ======================================================================

def _render_included_study_extractor_tab():
    st.markdown(
        "Use a large multimodal model to extract only the included-study tables from a review PDF, keep the grouped trial/table structure, "
        "and surface bibliography-linked papers for retrieval."
    )

    left, right = st.columns([1, 2])

    with left:
        st.header("PDF Input")
        st.session_state["included_study_batch"] = False
        selected = _file_input_widget("included_study", ["pdf"], label="Choose review PDF:")

        provider = st.selectbox(
            "Provider",
            options=["gemini", "anthropic"],
            index=0,
            key="included_study_provider",
        )
        default_model = "gemini-2.5-flash" if provider == "gemini" else "claude-sonnet-4-6"
        model = st.text_input(
            "Model",
            value=st.session_state.get("included_study_model", default_model) or default_model,
            key="included_study_model",
        )
        fallback_to_anthropic = False
        fallback_model = "claude-sonnet-4-6"
        if provider == "gemini":
            fallback_to_anthropic = st.checkbox(
                "Fallback to Claude if Gemini quota/rate limit is hit",
                value=bool(st.session_state.get("included_study_auto_fallback_to_claude", False)),
                key="included_study_auto_fallback_to_claude",
                disabled=not included_study_extractor_available("anthropic"),
            )
            if included_study_extractor_available("anthropic"):
                fallback_model = st.text_input(
                    "Fallback Claude model",
                    value=st.session_state.get("included_study_fallback_model", "claude-sonnet-4-6") or "claude-sonnet-4-6",
                    key="included_study_fallback_model",
                    disabled=not fallback_to_anthropic,
                )
            else:
                st.caption("Anthropic fallback unavailable: `ANTHROPIC_API_KEY` not currently configured.")
        key_source = gemini_key_source() if provider == "gemini" else included_study_anthropic_key_source()
        if key_source:
            st.caption(f"API key source: `{key_source}`")
        if not included_study_extractor_available(provider):
            missing_name = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
            st.info(f"{missing_name} is not currently available to Streamlit.")

        if provider == "gemini":
            access_c1, access_c2 = st.columns(2)
            with access_c1:
                if st.button(
                    "Test Gemini Access",
                    use_container_width=True,
                    key="included_study_test_gemini_access_btn",
                    disabled=not included_study_extractor_available("gemini"),
                ):
                    try:
                        with st.spinner("Testing Gemini access with a tiny prompt..."):
                            access_result = run_included_study_access_check(provider="gemini", model=model)
                        st.session_state["included_study_access_check_result"] = access_result
                        st.success("Gemini access test succeeded.")
                    except IncludedStudyExtractorQuotaError as e:
                        st.session_state["included_study_access_check_result"] = {
                            "provider": "gemini",
                            "model": model,
                            "ok": False,
                            "preview": "",
                            "error": str(e),
                        }
                        st.error(f"Gemini access test failed: {e}")
                    except Exception as e:
                        st.session_state["included_study_access_check_result"] = {
                            "provider": "gemini",
                            "model": model,
                            "ok": False,
                            "preview": "",
                            "error": str(e),
                        }
                        st.error(f"Gemini access test failed: {e}")
            with access_c2:
                if st.button(
                    "Test Common Gemini Models",
                    use_container_width=True,
                    key="included_study_test_gemini_matrix_btn",
                    disabled=not included_study_extractor_available("gemini"),
                ):
                    with st.spinner("Testing common Gemini models with tiny prompts..."):
                        st.session_state["included_study_access_check_matrix"] = run_included_study_access_check_matrix()

        access_result = st.session_state.get("included_study_access_check_result") or {}
        if provider == "gemini" and access_result:
            access_provider = str(access_result.get("provider") or "").strip()
            access_model = str(access_result.get("model") or "").strip()
            access_preview = str(access_result.get("preview") or "").strip()
            access_error = str(access_result.get("error") or "").strip()
            access_ok = bool(access_result.get("ok"))
            if access_ok:
                st.caption(f"Gemini access OK: `{access_provider}` / `{access_model}` -> `{access_preview or 'ACCESS_OK'}`")
            elif access_error:
                st.caption(f"Last Gemini access test failed for `{access_model}`: {access_error}")
        access_matrix = list(st.session_state.get("included_study_access_check_matrix") or [])
        if provider == "gemini" and access_matrix:
            matrix_rows = []
            for item in access_matrix:
                matrix_rows.append(
                    {
                        "model": str(item.get("model") or "").strip(),
                        "ok": "yes" if bool(item.get("ok")) else "no",
                        "preview": str(item.get("preview") or "").strip(),
                        "error": str(item.get("error") or "").strip(),
                    }
                )
            st.dataframe(matrix_rows, use_container_width=True, hide_index=True)

        if selected and st.button(
            "Extract Included-Study Tables",
            type="primary",
            use_container_width=True,
            key="included_study_extract_btn",
            disabled=not included_study_extractor_available(provider),
        ):
            try:
                review_title = _user_visible_stem(selected)
                with st.spinner("Calling the large-model table extractor..."):
                    result = run_included_study_extractor_with_fallback(
                        pdf_path=str(selected),
                        provider=provider,
                        model=model,
                        review_title=review_title,
                        fallback_provider="anthropic" if (provider == "gemini" and fallback_to_anthropic) else "",
                        fallback_model=fallback_model if (provider == "gemini" and fallback_to_anthropic) else "",
                    )
                st.session_state["included_study_result"] = result
                st.session_state["included_study_editor_rows"] = _included_study_editor_rows(result.get("tables") or [])
                st.success("Included-study extraction complete.")
            except IncludedStudyExtractorQuotaError as e:
                st.error(
                    f"Included-study extraction failed: {e}. "
                    "Gemini quota was exhausted for this PDF. Enable Claude fallback or switch provider."
                )
            except Exception as e:
                st.error(f"Included-study extraction failed: {e}")

    with right:
        st.header("Grouped Output")
        result = st.session_state.get("included_study_result") or {}
        tables = list(result.get("tables") or [])
        if result:
            provider_label = str(result.get("provider") or "").strip()
            model_label = str(result.get("model") or "").strip()
            requested_provider = str(result.get("requested_provider") or "").strip()
            requested_model = str(result.get("requested_model") or "").strip()
            if requested_provider and requested_provider != provider_label:
                st.caption(
                    f"Requested: `{requested_provider}` / `{requested_model or '?'}`\n"
                    f"Used: `{provider_label}` / `{model_label}`"
                )
            else:
                st.caption(f"Provider: `{provider_label}` • Model: `{model_label}`")
            for warning in list(result.get("warnings") or []):
                st.warning(warning)

            t1, t2, t3 = st.columns(3)
            t1.metric("Included-study tables", len(tables))
            total_groups = sum(len(list(item.get("groups") or [])) for item in tables)
            total_citations = sum(
                len(list(group.get("citations") or []))
                for item in tables
                for group in list(item.get("groups") or [])
            )
            t2.metric("Groups", total_groups)
            t3.metric("Cited papers", total_citations)

            if not tables:
                st.warning("The extractor run completed, but no included-study tables were parsed into structured output.")
            else:
                for table in tables:
                    label = f"Table {str(table.get('table_number') or '?').strip()}: {str(table.get('table_title') or '').strip()}"
                    with st.expander(label, expanded=False):
                        grouping_basis = str(table.get("grouping_basis") or "").strip()
                        if grouping_basis:
                            st.caption(f"Grouping basis: {grouping_basis}")
                        for group in list(table.get("groups") or []):
                            combined_group = _included_study_group_label(group)
                            st.markdown(f"**{combined_group or 'Group'}**")
                            citations = list(group.get("citations") or [])
                            for citation in citations:
                                display = str(citation.get("display") or "").strip()
                                resolved_title = str(citation.get("resolved_title") or "").strip()
                                notes = str(citation.get("notes") or "").strip()
                                line = display
                                if resolved_title:
                                    line += f" -> {resolved_title}"
                                if notes:
                                    line += f" [{notes}]"
                                st.write(line)

                editor_source = st.session_state.get("included_study_editor_rows") or _included_study_editor_rows(tables)
                bulk_left, bulk_mid, bulk_right = st.columns([1, 1, 4])
                with bulk_left:
                    if st.button("Select All", use_container_width=True, key="included_study_select_all"):
                        st.session_state["included_study_editor_rows"] = _set_included_study_keep_state(editor_source, True)
                        st.rerun()
                with bulk_mid:
                    if st.button("Deselect All", use_container_width=True, key="included_study_deselect_all"):
                        st.session_state["included_study_editor_rows"] = _set_included_study_keep_state(editor_source, False)
                        st.rerun()
                with bulk_right:
                    st.caption("Selection is at the paper level, not the raw table-row level.")

                edited = st.data_editor(
                    editor_source,
                    use_container_width=True,
                    hide_index=True,
                    key="included_study_editor",
                    column_config={
                        "keep": st.column_config.CheckboxColumn("Keep", default=True),
                        "row_id": st.column_config.NumberColumn("Row", format="%d"),
                        "table_number": st.column_config.TextColumn("Table", width="small"),
                        "table_title": st.column_config.TextColumn("Table title", width="medium"),
                        "combined_group": st.column_config.TextColumn("Grouped under", width="medium"),
                        "citation_display": st.column_config.TextColumn("Table citation", width="medium"),
                        "title": st.column_config.TextColumn("Resolved title", width="large"),
                        "authors": st.column_config.TextColumn("Authors", width="medium"),
                        "year": st.column_config.TextColumn("Year", width="small"),
                        "doi": st.column_config.TextColumn("DOI", width="medium"),
                        "journal": st.column_config.TextColumn("Journal", width="medium"),
                        "reference_number": st.column_config.TextColumn("Ref", width="small"),
                        "needs_review": st.column_config.TextColumn("Needs review", width="small"),
                        "notes": st.column_config.TextColumn("Notes", width="medium"),
                    },
                    disabled=[
                        "row_id",
                        "table_number",
                        "table_title",
                        "combined_group",
                        "citation_display",
                        "reference_number",
                        "needs_review",
                    ],
                )
                editor_rows = _editor_records(edited)
                if editor_rows:
                    st.session_state["included_study_editor_rows"] = editor_rows

                selected_count = sum(1 for item in editor_rows if bool(item.get("keep", True)))
                st.caption(f"{selected_count} paper(s) selected for resolution/retrieval.")

                export_rows = _merge_included_study_editor_rows(editor_rows)
                if export_rows:
                    import pandas as pd

                    export_csv = pd.DataFrame(editor_rows).to_csv(index=False)
                    st.download_button(
                        "Download Included-Study Selection CSV",
                        data=export_csv,
                        file_name=f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_included_study_extractor.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="included_study_download_csv",
                    )

                if st.button(
                    "Send Selected to Research Resolver",
                    use_container_width=True,
                    key="included_study_send_to_resolver",
                    disabled=(selected_count == 0),
                ):
                    selected_citations = _merge_included_study_editor_rows(editor_rows)
                    st.session_state["research_parse_result"] = {
                        "source_name": "Included Study Extractor",
                        "citations": selected_citations,
                        "detected_fields": ["title", "authors", "year", "doi", "journal", "notes"],
                        "warnings": [],
                    }
                    st.session_state["research_editor_rows"] = build_research_preview_rows(selected_citations)
                    st.success("Selected papers copied into Research Resolver.")

                with st.expander("Paper Retrieval", expanded=False):
                    retrieval_c1, retrieval_c2 = st.columns(2)
                    with retrieval_c1:
                        retrieval_check_oa = st.checkbox(
                            "Check Open Access via Unpaywall",
                            value=bool(st.session_state.get("research_check_oa", True)),
                            key="included_study_retrieval_check_oa",
                        )
                        retrieval_enrich_sjr = st.checkbox(
                            "Enrich journal rankings via SJR",
                            value=bool(st.session_state.get("research_enrich_sjr", True)),
                            key="included_study_retrieval_enrich_sjr",
                        )
                        retrieval_unpaywall_email = st.text_input(
                            "Unpaywall contact email",
                            value=st.session_state.get("research_unpaywall_email", ""),
                            key="included_study_retrieval_unpaywall_email",
                        )
                    with retrieval_c2:
                        retrieval_convert_to_md = st.checkbox(
                            "Convert retrieved PDFs to Markdown",
                            value=True,
                            key="included_study_retrieval_convert_to_md",
                        )
                        retrieval_use_vision = st.checkbox(
                            "Use vision during PDF->MD conversion",
                            value=bool(st.session_state.get("url_ingestor_use_vision", False)),
                            key="included_study_retrieval_use_vision",
                        )
                        retrieval_capture_web = st.checkbox(
                            "Capture web page as Markdown when PDF unavailable",
                            value=True,
                            key="included_study_retrieval_capture_web",
                        )
                        retrieval_timeout_seconds = st.number_input(
                            "Request timeout (seconds)",
                            min_value=5,
                            max_value=120,
                            value=int(st.session_state.get("url_ingestor_timeout_seconds", 25) or 25),
                            step=5,
                            key="included_study_retrieval_timeout_seconds",
                        )

                    if st.button(
                        "Resolve + Retrieve Selected Papers",
                        use_container_width=True,
                        key="included_study_retrieve_papers_btn",
                        disabled=(selected_count == 0),
                    ):
                        try:
                            db_root = _resolve_db_root()
                            selected_citations = _merge_included_study_editor_rows(editor_rows)
                            retrieval_log_placeholder = st.empty()
                            retrieval_log_lines: List[str] = []

                            def _retrieval_log(message: str) -> None:
                                stamp = time.strftime("%H:%M:%S")
                                retrieval_log_lines.append(f"{stamp} {message}")
                                retrieval_log_placeholder.text_area(
                                    "Paper retrieval log",
                                    value="\n".join(retrieval_log_lines[-300:]),
                                    height=220,
                                    disabled=True,
                                )

                            with st.spinner("Resolving citations and retrieving papers..."):
                                retrieval_output = _run_study_miner_paper_retrieval(
                                    candidates=selected_citations,
                                    db_root=db_root,
                                    resolver_options={
                                        "check_open_access": retrieval_check_oa,
                                        "enrich_sjr": retrieval_enrich_sjr,
                                        "unpaywall_email": retrieval_unpaywall_email,
                                    },
                                    ingest_options={
                                        "convert_to_md": retrieval_convert_to_md,
                                        "use_vision_for_md": retrieval_use_vision,
                                        "capture_web_md_on_no_pdf": retrieval_capture_web,
                                        "timeout_seconds": retrieval_timeout_seconds,
                                        "textify_options": {"pdf_strategy": "hybrid"},
                                    },
                                    progress_cb=_retrieval_log,
                                )
                            st.session_state["research_parse_result"] = {
                                "source_name": "Included Study Extractor",
                                "citations": retrieval_output.get("resolver_payload", {}).get("citations") or selected_citations,
                                "detected_fields": ["title", "authors", "year", "doi", "journal", "notes"],
                                "warnings": [],
                            }
                            st.session_state["research_editor_rows"] = build_research_preview_rows(
                                retrieval_output.get("resolver_payload", {}).get("citations") or selected_citations
                            )
                            st.session_state["research_resolve_output"] = retrieval_output.get("resolver_output") or {}
                            st.session_state["research_resolve_run_dir"] = str(retrieval_output.get("resolver_run_dir") or "")
                            st.session_state["research_resolve_log_lines"] = list(retrieval_log_lines)
                            st.session_state["url_ingestor_input"] = "\n".join(retrieval_output.get("preferred_urls") or [])
                            st.session_state["url_ingestor_results"] = list(retrieval_output.get("url_results") or [])
                            st.session_state["url_ingestor_csv_path"] = str(retrieval_output.get("url_csv_path") or "")
                            st.session_state["url_ingestor_json_path"] = str(retrieval_output.get("url_json_path") or "")
                            st.session_state["url_ingestor_zip_bytes"] = retrieval_output.get("url_zip_bytes") or b""
                            st.session_state["url_ingestor_run_dir"] = str(retrieval_output.get("url_run_dir") or "")
                            st.session_state["url_ingestor_event_log"] = list(retrieval_log_lines)
                            st.success(
                                f"Resolved {len((retrieval_output.get('resolver_output') or {}).get('resolved') or [])} citation(s) "
                                f"and queued {len(retrieval_output.get('preferred_urls') or [])} URL(s) for retrieval."
                            )
                        except Exception as e:
                            st.error(f"Paper retrieval failed: {e}")

            with st.expander("Raw Model Output", expanded=False):
                raw_text = str(result.get("raw_response") or "").strip()
                st.caption(f"Raw output length: {len(raw_text)} characters")
                if raw_text:
                    st.code(raw_text, language="json")
                    st.download_button(
                        "Download Raw Model Output",
                        data=raw_text,
                        file_name=f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_included_study_raw_output.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="included_study_download_raw_output",
                    )
                else:
                    st.info("No raw model output was returned for this run.")
        else:
            st.info("Upload a review PDF and run the extractor to get grouped included-study tables.")


# ======================================================================
# Study Miner tab
# ======================================================================

def _render_study_miner_tab():
    st.markdown(
        "Screen one or more review documents for likely systematic reviews, confirm them, then mine included-study citations that can be handed into Research Resolver."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Review Input")
        st.session_state["study_miner_batch"] = True
        selected = _file_input_widget("study_miner", ["pdf", "docx", "pptx", "md", "txt"], label="Choose review document(s):")
        st.caption("Upload File mode accepts multiple reviews so you can screen a batch before mining.")

        vision_mode = st.selectbox(
            "Vision assist",
            options=["Auto (Recommended)", "On", "Off"],
            index=0,
            key="study_miner_vision_mode",
            help="Auto turns vision on for PDF reviews so the system can recover more context from difficult layouts.",
        )
        pdf_mode_label = st.selectbox(
            "Extraction mode",
            options=[
                "Hybrid (Recommended): Docling first, Qwen enhancement, fallback on timeout",
                "Docling only (best layout/tables)",
                "Qwen 30B cleanup (LLM-first, no Docling)",
            ],
            index=0,
            key="study_miner_pdf_mode",
        )
        docling_timeout_seconds = st.number_input(
            "Docling timeout (seconds)",
            min_value=30,
            max_value=1200,
            value=240,
            step=10,
            key="study_miner_docling_timeout_s",
        )
        image_timeout_seconds = st.number_input(
            "Per-image timeout (seconds)",
            min_value=3,
            max_value=180,
            value=20,
            step=1,
            key="study_miner_image_timeout_s",
        )
        image_budget_seconds = st.number_input(
            "Total image budget (seconds)",
            min_value=10,
            max_value=1800,
            value=90,
            step=10,
            key="study_miner_image_budget_s",
        )

        st.divider()
        st.subheader("Criteria")
        design_query = st.text_input(
            "Study design criteria",
            value="RCT, clinical trial, randomised, randomized",
            key="study_miner_design_query",
        )
        outcome_query = st.text_input(
            "Outcome criteria",
            value="health-related quality of life, HRQoL, quality of life, patient-reported outcome",
            key="study_miner_outcome_query",
        )
        require_all = st.checkbox(
            "Require both design and outcome criteria",
            value=True,
            key="study_miner_require_all",
        )
        include_references = st.checkbox(
            "Also scan reference list for title-level matches",
            value=True,
            key="study_miner_include_references",
        )

    with col2:
        st.header("Review Screening")

        if selected and st.button("Scan Reviews", type="primary", use_container_width=True, key="study_miner_scan_btn"):
            try:
                use_vision = _study_miner_should_use_vision(selected, vision_mode)
                source_documents = _extract_study_miner_documents(
                    selected,
                    use_vision=use_vision,
                    pdf_mode_label=pdf_mode_label,
                    docling_timeout_seconds=float(docling_timeout_seconds),
                    image_timeout_seconds=float(image_timeout_seconds),
                    image_budget_seconds=float(image_budget_seconds),
                )
                assessed = assess_review_documents(source_documents)
                st.session_state["study_miner_documents"] = list(assessed.get("documents") or [])
                st.session_state["study_miner_document_rows"] = _study_miner_document_rows(list(assessed.get("documents") or []))
                st.session_state["study_miner_effective_use_vision"] = use_vision
                st.session_state.pop("study_miner_result", None)
                st.session_state.pop("study_miner_editor_rows", None)
            except Exception as e:
                st.error(f"Review screening failed: {e}")

        source_documents = list(st.session_state.get("study_miner_documents") or [])
        document_rows = st.session_state.get("study_miner_document_rows") or _study_miner_document_rows(source_documents)
        if document_rows:
            if "study_miner_effective_use_vision" in st.session_state:
                effective_label = "enabled" if bool(st.session_state.get("study_miner_effective_use_vision")) else "disabled"
                st.caption(f"Effective vision assist for the last scan: {effective_label}.")
            likely_count = sum(1 for item in source_documents if bool(item.get("systematic_review_likely")))
            confirmed_default = sum(1 for item in source_documents if bool(item.get("confirm_review")))
            d1, d2, d3 = st.columns(3)
            d1.metric("Documents", len(source_documents))
            d2.metric("Likely systematic reviews", likely_count)
            d3.metric("Initially confirmed", confirmed_default)

            edited_documents = st.data_editor(
                document_rows,
                use_container_width=True,
                hide_index=True,
                key="study_miner_documents_editor",
                column_config={
                    "confirm_review": st.column_config.CheckboxColumn("Mine", default=False),
                    "doc_id": st.column_config.NumberColumn("Doc", format="%d"),
                    "source_name": st.column_config.TextColumn("File", width="medium"),
                    "review_title": st.column_config.TextColumn("Review title", width="medium"),
                    "systematic_review_likely": st.column_config.CheckboxColumn("Likely SR"),
                    "confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "score": st.column_config.NumberColumn("Score", format="%d"),
                    "matched_signals": st.column_config.TextColumn("Signals", width="large"),
                },
                disabled=["doc_id", "source_name", "systematic_review_likely", "confidence", "score", "matched_signals"],
            )
            document_editor_rows = _editor_records(edited_documents)
            if document_editor_rows:
                st.session_state["study_miner_document_rows"] = document_editor_rows

            confirmed_count = sum(1 for item in document_editor_rows if bool(item.get("confirm_review")))
            st.caption(f"{confirmed_count} review document(s) marked for study mining.")

            if st.button("Mine Confirmed Reviews", use_container_width=True, key="study_miner_run_btn"):
                merged_documents = _merge_study_miner_document_rows(document_editor_rows, source_documents)
                confirmed_doc_ids = [int(item.get("doc_id") or 0) for item in merged_documents if bool(item.get("confirm_review"))]
                if not confirmed_doc_ids:
                    st.warning("Select at least one likely systematic review before mining included studies.")
                else:
                    try:
                        result = mine_review_documents(
                            merged_documents,
                            options=ReviewMiningOptions(
                                design_query=design_query,
                                outcome_query=outcome_query,
                                require_all_criteria=require_all,
                                include_reference_list_scan=include_references,
                            ),
                            confirmed_doc_ids=confirmed_doc_ids,
                        )
                        st.session_state["study_miner_documents"] = list(result.get("documents") or merged_documents)
                        st.session_state["study_miner_document_rows"] = _study_miner_document_rows(
                            list(result.get("documents") or merged_documents)
                        )
                        st.session_state["study_miner_result"] = result
                        st.session_state["study_miner_editor_rows"] = _study_miner_candidate_rows(result.get("candidates") or [])
                    except Exception as e:
                        st.error(f"Study mining failed: {e}")
        elif selected:
            st.info("Click Scan Reviews to classify the uploaded review documents before mining.")
        else:
            st.info("Upload or choose one or more reviews, then scan them for systematic-review signals.")

        result = st.session_state.get("study_miner_result") or {}
        candidates = list(result.get("candidates") or [])
        evidence_documents = list(result.get("documents") or st.session_state.get("study_miner_documents") or [])
        rescue_recommended = bool(result) and _study_miner_cloud_rescue_recommended(result, evidence_documents)
        default_local_model = st.session_state.get(
            "study_miner_local_rescue_model",
            os.environ.get("CORTEX_REVIEW_TABLE_OLLAMA_MODEL", "qwen3.5:35b-a3b"),
        )
        auto_local_rescue = st.session_state.get("study_miner_auto_local_rescue", True)
        local_available = local_table_rescue_available()
        if (
            result
            and rescue_recommended
            and auto_local_rescue
            and local_available
            and not bool(result.get("local_rescue_attempted"))
        ):
            auto_log_placeholder = st.empty()
            st.session_state["study_miner_local_log_lines"] = []
            _append_study_miner_local_log(f"Auto local rescue queued with model `{default_local_model}`", placeholder=auto_log_placeholder)
            with st.spinner("Local parse looks confused. Running Ollama table rescue..."):
                updated_result = _run_study_miner_local_rescue(
                    result,
                    evidence_documents,
                    design_query=design_query,
                    outcome_query=outcome_query,
                    model=default_local_model,
                    log_callback=lambda message: _append_study_miner_local_log(message, placeholder=auto_log_placeholder),
                )
            st.session_state["study_miner_result"] = updated_result
            st.session_state["study_miner_editor_rows"] = _study_miner_candidate_rows(updated_result.get("candidates") or [])
            result = updated_result
            candidates = list(updated_result.get("candidates") or [])
            rescue_recommended = bool(result) and _study_miner_cloud_rescue_recommended(result, evidence_documents)
        if candidates:
            st.divider()
            st.subheader("Mined Study Candidates")
            stats = result.get("stats") or {}
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Confirmed reviews mined", int(stats.get("reviews_mined") or 0))
            s2.metric("Candidates", int(stats.get("candidates_total") or 0))
            s3.metric("Matching criteria", int(stats.get("candidates_matching") or 0))
            s4.metric("Table refs linked", int(stats.get("table_reference_links") or 0))

            per_review = list(result.get("per_review") or [])
            if per_review:
                with st.expander("Per-review mining summary", expanded=False):
                    st.dataframe(_study_miner_review_rows(per_review), use_container_width=True, hide_index=True)
            show_parse_evidence = st.checkbox(
                "Show parse evidence",
                value=st.session_state.get("study_miner_show_parse_evidence", False),
                key="study_miner_show_parse_evidence",
                help="Displays rendered table snapshots and extracted markdown. Keep this off while selecting rows to avoid slow reruns.",
            )
            if show_parse_evidence:
                _render_study_miner_parse_evidence(evidence_documents, candidates)
            else:
                st.caption("Parse evidence is hidden to keep candidate selection responsive.")
            _render_study_miner_local_log(expanded=False)

            with st.expander("Local Table Rescue", expanded=False):
                st.caption("Preferred fallback for multi-page continuation tables. Uses a local Ollama vision model on the actual table pages, then links the extracted rows back to references locally.")
                st.checkbox(
                    "Auto-run local Ollama rescue when the parse looks confused",
                    value=True,
                    key="study_miner_auto_local_rescue",
                )
                local_model = st.text_input(
                    "Ollama vision model",
                    value=default_local_model,
                    key="study_miner_local_rescue_model",
                )
                st.caption(f"Ollama endpoint: `{local_table_rescue_host()}`")
                if rescue_recommended:
                    st.warning("This review looks like a continuation-table case. Local Ollama rescue is recommended.")
                if not local_available:
                    st.info("Ollama is not reachable. Start the local server and ensure a vision model such as `qwen3.5:35b-a3b` is available, then run local rescue.")
                elif st.button("Run Local Table Rescue", use_container_width=True, key="study_miner_local_rescue_btn"):
                    local_log_placeholder = st.empty()
                    st.session_state["study_miner_local_log_lines"] = []
                    _append_study_miner_local_log(f"Manual local rescue queued with model `{local_model}`", placeholder=local_log_placeholder)
                    with st.spinner("Running Ollama table rescue..."):
                        updated_result = _run_study_miner_local_rescue(
                            result,
                            evidence_documents,
                            design_query=design_query,
                            outcome_query=outcome_query,
                            model=local_model,
                            log_callback=lambda message: _append_study_miner_local_log(message, placeholder=local_log_placeholder),
                        )
                    st.session_state["study_miner_result"] = updated_result
                    st.session_state["study_miner_editor_rows"] = _study_miner_candidate_rows(updated_result.get("candidates") or [])
                    result = updated_result
                    candidates = list(updated_result.get("candidates") or [])
                    stats = updated_result["stats"]
                    rescue_recommended = bool(result) and _study_miner_cloud_rescue_recommended(result, evidence_documents)
                    st.success("Local table rescue candidates added to Study Miner.")

                local_runs = list((st.session_state.get("study_miner_result") or {}).get("local_rescue_runs") or [])
                if local_runs:
                    for run in local_runs:
                        warnings = list(run.get("warnings") or [])
                        model_name = str(run.get("model") or "").strip()
                        review_name = str(run.get("review_title") or run.get("source_name") or "Review").strip()
                        st.markdown(f"**{review_name}** via `{model_name}`")
                        pages_used = [int(page) for page in list(run.get("pages_used") or []) if int(page or 0) > 0]
                        if pages_used:
                            st.caption(f"Pages used: {', '.join(str(page) for page in pages_used)}")
                        if bool(run.get("reconciled_with_full_pdf")):
                            reconciliation_pages = [int(page) for page in list(run.get("reconciliation_pages_used") or []) if int(page or 0) > 0]
                            st.caption("Full-PDF reconciliation: yes")
                            if reconciliation_pages:
                                st.caption(f"Reconciliation pages: {', '.join(str(page) for page in reconciliation_pages)}")
                        if warnings:
                            for warning in warnings:
                                st.warning(warning)
                        st.caption(f"{len(run.get('candidates') or [])} local candidate(s) returned.")

            with st.expander("Cloud Table Rescue", expanded=False):
                st.caption("Optional paid fallback if the local rescue still misses rows. Use this only after trying the local LM Studio rescue.")
                rescue_enabled = st.checkbox(
                    "Enable Claude Sonnet rescue for complex tables",
                    value=False,
                    key="study_miner_cloud_rescue_enabled",
                )
                key_source = anthropic_key_source()
                if key_source:
                    st.caption(f"Anthropic key source: `{key_source}`")
                if rescue_recommended:
                    st.warning("This review still looks like a complex table case. Claude rescue remains available as a secondary fallback.")
                if not claude_table_rescue_available():
                    st.info("ANTHROPIC_API_KEY is not available to Streamlit. Put it in `.env`, export it before launch, or keep it in `worker/config.env`.")
                elif rescue_enabled and st.button("Run Claude Table Rescue", use_container_width=True, key="study_miner_cloud_rescue_btn"):
                    with st.spinner("Running Claude Sonnet table rescue..."):
                        updated_result = _run_study_miner_cloud_rescue(
                            result,
                            evidence_documents,
                            design_query=design_query,
                            outcome_query=outcome_query,
                        )
                    st.session_state["study_miner_result"] = updated_result
                    st.session_state["study_miner_editor_rows"] = _study_miner_candidate_rows(updated_result.get("candidates") or [])
                    result = updated_result
                    candidates = list(updated_result.get("candidates") or [])
                    stats = updated_result["stats"]
                    st.success("Claude table rescue candidates added to Study Miner.")

                cloud_runs = list((st.session_state.get("study_miner_result") or {}).get("cloud_rescue_runs") or [])
                if cloud_runs:
                    for run in cloud_runs:
                        warnings = list(run.get("warnings") or [])
                        model_name = str(run.get("model") or "").strip()
                        review_name = str(run.get("review_title") or run.get("source_name") or "Review").strip()
                        st.markdown(f"**{review_name}** via `{model_name}`")
                        st.caption(
                            "Evidence mode: "
                            + ("full attached PDF" if bool(run.get("used_full_pdf")) else "extracted tables only")
                        )
                        if warnings:
                            for warning in warnings:
                                st.warning(warning)
                        st.caption(f"{len(run.get('candidates') or [])} cloud candidate(s) returned.")

            editor_source = st.session_state.get("study_miner_editor_rows") or _study_miner_candidate_rows(candidates)
            bulk_left, bulk_mid, bulk_right = st.columns([1, 1, 4])
            with bulk_left:
                if st.button("Select All", use_container_width=True, key="study_miner_select_all"):
                    st.session_state["study_miner_editor_rows"] = _set_study_miner_keep_state(editor_source, True)
                    st.rerun()
            with bulk_mid:
                if st.button("Deselect All", use_container_width=True, key="study_miner_deselect_all"):
                    st.session_state["study_miner_editor_rows"] = _set_study_miner_keep_state(editor_source, False)
                    st.rerun()
            with bulk_right:
                st.caption("Bulk selection updates the cached candidate table directly so you do not need to click through every row.")
            edited = st.data_editor(
                editor_source,
                use_container_width=True,
                hide_index=True,
                key="study_miner_editor",
                column_config={
                    "keep": st.column_config.CheckboxColumn("Keep", default=True),
                    "row_id": st.column_config.NumberColumn("Row", format="%d"),
                    "source_review_title": st.column_config.TextColumn("Review", width="medium"),
                    "table_index": st.column_config.NumberColumn("Table", format="%d"),
                    "table_label": st.column_config.TextColumn("Table slice", width="small"),
                    "table_group": st.column_config.TextColumn("Table group", width="medium"),
                    "table_citation": st.column_config.TextColumn("Table citation", width="medium"),
                    "title": st.column_config.TextColumn("Title / Citation", width="large"),
                    "authors": st.column_config.TextColumn("Authors", width="medium"),
                    "year": st.column_config.TextColumn("Year", width="small"),
                    "doi": st.column_config.TextColumn("DOI", width="medium"),
                    "journal": st.column_config.TextColumn("Journal", width="medium"),
                    "source_section": st.column_config.TextColumn("Source", width="small"),
                    "matches": st.column_config.TextColumn("Matches", width="small"),
                    "score": st.column_config.NumberColumn("Score", format="%d"),
                    "reference_link": st.column_config.TextColumn("Reference link", width="medium"),
                    "needs_review": st.column_config.TextColumn("Needs review", width="small"),
                    "review_warning": st.column_config.TextColumn("Review warning", width="large"),
                    "design_matches": st.column_config.TextColumn("Design hits", width="medium"),
                    "outcome_matches": st.column_config.TextColumn("Outcome hits", width="medium"),
                },
                disabled=["row_id", "source_review_title", "table_index", "table_label", "table_group", "table_citation", "source_section", "matches", "score", "reference_link", "needs_review", "review_warning", "design_matches", "outcome_matches"],
            )
            editor_rows = _editor_records(edited)
            if editor_rows:
                st.session_state["study_miner_editor_rows"] = editor_rows

            selected_count = sum(1 for item in editor_rows if bool(item.get("keep", True)))
            st.caption(f"{selected_count} candidate citation(s) selected.")
            if int(stats.get("needs_review") or 0) > 0:
                st.warning("Some table rows were linked with inconsistencies or low confidence. Review those rows before sending them onward.")

            export_rows = _study_miner_export_rows(editor_rows, candidates)
            raw_grouped_export_rows = [
                item
                for item in _group_study_miner_export_rows_by_table(export_rows)
                if int(item.get("table_index") or 0) > 0
            ]
            include_low_value_tables = st.checkbox(
                "Include low-value table slices in exports",
                value=False,
                key="study_miner_include_low_value_tables",
                help="Show table slices that produced no explicit reference numbers. These are usually low-value for bibliography reconciliation.",
            )
            grouped_export_rows = _filter_study_miner_export_groups(
                raw_grouped_export_rows,
                include_low_value=include_low_value_tables,
            )
            export_table_keys = {
                (
                    str(item.get("source_review_title") or "").strip(),
                    int(item.get("table_index") or 0),
                )
                for item in grouped_export_rows
            }
            filtered_export_rows = [
                dict(item)
                for item in export_rows
                if (
                    str(item.get("source_review_title") or "").strip(),
                    int(item.get("table_index") or 0),
                )
                in export_table_keys
            ]
            suppressed_groups = [
                item
                for item in raw_grouped_export_rows
                if (
                    str(item.get("source_review_title") or "").strip(),
                    int(item.get("table_index") or 0),
                )
                not in export_table_keys
            ]
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if filtered_export_rows:
                    import pandas as pd

                    study_miner_csv = pd.DataFrame(filtered_export_rows).to_csv(index=False)
                    st.download_button(
                        "Download Study Miner CSV",
                        data=study_miner_csv,
                        file_name=f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_study_miner_export.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="study_miner_download_csv",
                    )
            with export_col2:
                st.caption("CSV export includes explicit `table_index`, `table_label`, `reference_number`, `reference_match_method`, and reconciliation fields for bibliography lookup.")
                if suppressed_groups and not include_low_value_tables:
                    suppressed_labels = ", ".join(str(item.get("table_label") or "table").strip() for item in suppressed_groups)
                    st.caption(f"Low-value table slices hidden by default: {suppressed_labels}")

            if grouped_export_rows:
                st.caption("Per-table CSV downloads use the detected table slices so you can inspect each table separately.")
                table_download_columns = st.columns(min(3, max(1, len(grouped_export_rows))))
                for idx, group in enumerate(grouped_export_rows):
                    with table_download_columns[idx % len(table_download_columns)]:
                        import pandas as pd

                        group_csv = pd.DataFrame(group.get("rows") or []).to_csv(index=False)
                        review_slug = re.sub(r"[^A-Za-z0-9]+", "_", str(group.get("source_review_title") or "review").strip()).strip("_") or "review"
                        label_slug = re.sub(r"[^A-Za-z0-9]+", "_", str(group.get("table_label") or "table").strip()).strip("_") or "table"
                        st.download_button(
                            f"Download {group.get('table_label')} CSV",
                            data=group_csv,
                            file_name=(
                                f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_{review_slug}_"
                                f"{label_slug}_study_miner_export.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True,
                            key=f"study_miner_download_csv_{idx}",
                        )

            if st.button("Send Selected to Research Resolver", use_container_width=True, key="study_miner_send_to_resolver"):
                selected_candidates = _merge_study_miner_editor_rows(editor_rows, candidates)
                st.session_state["research_parse_result"] = {
                    "source_name": "Study Miner",
                    "citations": selected_candidates,
                    "detected_fields": ["title", "authors", "year", "doi", "journal", "notes"],
                    "warnings": [],
                }
                st.session_state["research_editor_rows"] = build_research_preview_rows(selected_candidates)
                st.success("Selected candidates copied into Research Resolver.")

            with st.expander("Paper Retrieval", expanded=False):
                st.caption("Resolve the selected Study Miner bibliography rows and immediately attempt OA PDF/web retrieval for each matched paper.")
                retrieval_c1, retrieval_c2 = st.columns(2)
                with retrieval_c1:
                    retrieval_check_oa = st.checkbox(
                        "Check Open Access via Unpaywall",
                        value=bool(st.session_state.get("research_check_oa", True)),
                        key="study_miner_retrieval_check_oa",
                    )
                    retrieval_enrich_sjr = st.checkbox(
                        "Enrich journal rankings via SJR",
                        value=bool(st.session_state.get("research_enrich_sjr", True)),
                        key="study_miner_retrieval_enrich_sjr",
                    )
                    retrieval_unpaywall_email = st.text_input(
                        "Unpaywall contact email",
                        value=st.session_state.get("research_unpaywall_email", ""),
                        key="study_miner_retrieval_unpaywall_email",
                    )
                with retrieval_c2:
                    retrieval_convert_to_md = st.checkbox(
                        "Convert retrieved PDFs to Markdown",
                        value=True,
                        key="study_miner_retrieval_convert_to_md",
                    )
                    retrieval_use_vision = st.checkbox(
                        "Use vision during PDF->MD conversion",
                        value=bool(st.session_state.get("url_ingestor_use_vision", False)),
                        key="study_miner_retrieval_use_vision",
                    )
                    retrieval_capture_web = st.checkbox(
                        "Capture web page as Markdown when PDF unavailable",
                        value=True,
                        key="study_miner_retrieval_capture_web",
                    )
                    retrieval_timeout_seconds = st.number_input(
                        "Request timeout (seconds)",
                        min_value=5,
                        max_value=120,
                        value=int(st.session_state.get("url_ingestor_timeout_seconds", 25) or 25),
                        step=5,
                        key="study_miner_retrieval_timeout_seconds",
                    )

                if retrieval_check_oa and not str(retrieval_unpaywall_email or "").strip():
                    st.info("Open-access enrichment will be limited until an Unpaywall email is provided.")

                if st.button(
                    "Resolve + Retrieve Selected Papers",
                    use_container_width=True,
                    key="study_miner_retrieve_papers_btn",
                    disabled=(selected_count == 0),
                ):
                    try:
                        db_root = _resolve_db_root()
                        selected_candidates = _merge_study_miner_editor_rows(editor_rows, candidates)
                        retrieval_log_placeholder = st.empty()
                        retrieval_log_lines: List[str] = []

                        def _retrieval_log(message: str) -> None:
                            stamp = time.strftime("%H:%M:%S")
                            retrieval_log_lines.append(f"{stamp} {message}")
                            if len(retrieval_log_lines) > 300:
                                del retrieval_log_lines[:-300]
                            retrieval_log_placeholder.text_area(
                                "Paper retrieval log",
                                value="\n".join(retrieval_log_lines),
                                height=220,
                                disabled=True,
                            )

                        with st.spinner("Resolving citations and retrieving papers..."):
                            retrieval_output = _run_study_miner_paper_retrieval(
                                candidates=selected_candidates,
                                db_root=db_root,
                                resolver_options={
                                    "check_open_access": retrieval_check_oa,
                                    "enrich_sjr": retrieval_enrich_sjr,
                                    "unpaywall_email": retrieval_unpaywall_email,
                                },
                                ingest_options={
                                    "convert_to_md": retrieval_convert_to_md,
                                    "use_vision_for_md": retrieval_use_vision,
                                    "capture_web_md_on_no_pdf": retrieval_capture_web,
                                    "timeout_seconds": retrieval_timeout_seconds,
                                    "textify_options": {"pdf_strategy": "hybrid"},
                                },
                                progress_cb=_retrieval_log,
                            )

                        st.session_state["research_parse_result"] = {
                            "source_name": "Study Miner",
                            "citations": retrieval_output.get("resolver_payload", {}).get("citations") or selected_candidates,
                            "detected_fields": ["title", "authors", "year", "doi", "journal", "notes"],
                            "warnings": [],
                        }
                        st.session_state["research_editor_rows"] = build_research_preview_rows(
                            retrieval_output.get("resolver_payload", {}).get("citations") or selected_candidates
                        )
                        st.session_state["research_resolve_output"] = retrieval_output.get("resolver_output") or {}
                        st.session_state["research_resolve_run_dir"] = str(retrieval_output.get("resolver_run_dir") or "")
                        st.session_state["research_resolve_log_lines"] = list(retrieval_log_lines)
                        st.session_state["url_ingestor_input"] = "\n".join(retrieval_output.get("preferred_urls") or [])
                        st.session_state["url_ingestor_results"] = list(retrieval_output.get("url_results") or [])
                        st.session_state["url_ingestor_csv_path"] = str(retrieval_output.get("url_csv_path") or "")
                        st.session_state["url_ingestor_json_path"] = str(retrieval_output.get("url_json_path") or "")
                        st.session_state["url_ingestor_zip_bytes"] = retrieval_output.get("url_zip_bytes") or b""
                        st.session_state["url_ingestor_run_dir"] = str(retrieval_output.get("url_run_dir") or "")
                        st.session_state["url_ingestor_event_log"] = list(retrieval_log_lines)
                        st.success(
                            f"Resolved {len((retrieval_output.get('resolver_output') or {}).get('resolved') or [])} citation(s) "
                            f"and queued {len(retrieval_output.get('preferred_urls') or [])} URL(s) for retrieval."
                        )
                    except Exception as e:
                        st.error(f"Paper retrieval failed: {e}")

            with st.expander("Mining stats", expanded=False):
                st.dataframe(_study_miner_stats_rows(stats), use_container_width=True, hide_index=True)
        elif result and rescue_recommended:
            st.warning("The local parser looks confused on this review. Local Ollama rescue is recommended.")
            _render_study_miner_local_log(expanded=False)
            with st.expander("Local Table Rescue", expanded=True):
                st.checkbox(
                    "Auto-run local Ollama rescue when the parse looks confused",
                    value=True,
                    key="study_miner_auto_local_rescue_empty",
                )
                local_model = st.text_input(
                    "Ollama vision model",
                    value=default_local_model,
                    key="study_miner_local_rescue_model_empty",
                )
                st.caption(f"Ollama endpoint: `{local_table_rescue_host()}`")
                if not local_available:
                    st.info("Ollama is not reachable. Start the local server and ensure a vision model is available, then run local rescue.")
                elif st.button("Run Local Table Rescue", use_container_width=True, key="study_miner_local_rescue_btn_empty"):
                    local_log_placeholder = st.empty()
                    st.session_state["study_miner_local_log_lines"] = []
                    _append_study_miner_local_log(f"Manual local rescue queued with model `{local_model}`", placeholder=local_log_placeholder)
                    with st.spinner("Running Ollama table rescue..."):
                        updated_result = _run_study_miner_local_rescue(
                            result,
                            evidence_documents,
                            design_query=design_query,
                            outcome_query=outcome_query,
                            model=local_model,
                            log_callback=lambda message: _append_study_miner_local_log(message, placeholder=local_log_placeholder),
                        )
                    st.session_state["study_miner_result"] = updated_result
                    st.session_state["study_miner_editor_rows"] = _study_miner_candidate_rows(updated_result.get("candidates") or [])
                    st.success("Local table rescue candidates added to Study Miner.")
        elif result:
            st.info("No candidate studies matched the confirmed review documents and current criteria.")


# ======================================================================
# Research Resolver tab
# ======================================================================

def _render_research_resolver_tab():
    st.markdown(
        "Resolve academic citation spreadsheets from pasted TSV/CSV text or uploaded CSV/TSV/XLSX files. "
        "This local Streamlit harness is aimed at the same `research_resolve` contract used by the website queue worker."
    )

    try:
        db_root = _resolve_db_root()
        st.caption(f"Resolver outputs are written under: {db_root / 'research_resolve'}")
    except Exception as e:
        db_root = None
        st.warning(str(e))

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")
        input_mode = st.radio(
            "Choose input method:",
            ["Paste TSV/CSV", "Upload spreadsheet"],
            key="research_input_mode",
        )

        pasted_text = ""
        uploaded_sheet = None
        if input_mode == "Paste TSV/CSV":
            pasted_text = st.text_area(
                "Paste tab-separated or comma-separated citations with headers",
                height=260,
                placeholder="Title\tAuthors\tYear\tDOI\tJournal\nPatient-Reported Outcomes...\tBarot, S. V.; Patel, B. J.\t2020\t10.1007/s11899-020-00562-9\tCurr Hematol Malig Rep",
                key="research_pasted_text",
            )
        else:
            uploaded_sheet = st.file_uploader(
                "Upload CSV, TSV, TXT, or XLSX",
                type=["csv", "tsv", "txt", "xlsx"],
                key="research_upload",
            )
            if uploaded_sheet is not None:
                st.caption(f"Loaded `{uploaded_sheet.name}` ({int(getattr(uploaded_sheet, 'size', 0) or 0):,} bytes)")

        st.divider()
        st.subheader("Resolver Options")
        check_open_access = st.checkbox("Check Open Access via Unpaywall", value=True, key="research_check_oa")
        enrich_sjr = st.checkbox("Enrich journal rankings via SJR", value=True, key="research_enrich_sjr")
        unpaywall_email = st.text_input(
            "Unpaywall contact email",
            value=st.session_state.get("research_unpaywall_email", ""),
            key="research_unpaywall_email",
            help="Required for live Unpaywall OA lookups. Leave blank to skip OA enrichment.",
        )
        if check_open_access and not unpaywall_email.strip():
            st.info("Open-access enrichment will be skipped until an Unpaywall email is provided.")

        if st.button("Parse Spreadsheet", type="primary", use_container_width=True, key="research_parse_btn"):
            try:
                if input_mode == "Paste TSV/CSV":
                    if not pasted_text.strip():
                        st.warning("Paste citation text first.")
                    else:
                        parsed = parse_research_spreadsheet_text(pasted_text, source_name="Pasted text")
                        st.session_state["research_parse_result"] = parsed
                        st.session_state["research_editor_rows"] = build_research_preview_rows(parsed.get("citations") or [])
                        st.session_state.pop("research_resolve_output", None)
                else:
                    if uploaded_sheet is None:
                        st.warning("Upload a spreadsheet first.")
                    else:
                        parsed = parse_research_spreadsheet_upload(uploaded_sheet.name, uploaded_sheet.getvalue())
                        st.session_state["research_parse_result"] = parsed
                        st.session_state["research_editor_rows"] = build_research_preview_rows(parsed.get("citations") or [])
                        st.session_state.pop("research_resolve_output", None)
            except Exception as e:
                st.error(f"Spreadsheet parse failed: {e}")

    with col2:
        st.header("Preview & Results")
        parse_result = st.session_state.get("research_parse_result") or {}
        citations = list(parse_result.get("citations") or [])

        if citations:
            warnings = list(parse_result.get("warnings") or [])
            detected_fields = list(parse_result.get("detected_fields") or [])
            if detected_fields:
                st.caption(f"Detected fields: {', '.join(detected_fields)}")
            for warning in warnings:
                st.warning(warning)

            total = len(citations)
            doi_count = sum(1 for item in citations if str(item.get("doi") or "").strip())
            amber_count = sum(1 for item in citations if item.get("preview_confidence") == "amber")
            red_count = sum(1 for item in citations if item.get("preview_confidence") == "red")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Parsed citations", total)
            m2.metric("Has DOI", doi_count)
            m3.metric("Title+author+year", amber_count)
            m4.metric("Title only", red_count)

            editor_source = st.session_state.get("research_editor_rows") or build_research_preview_rows(citations)
            edited = st.data_editor(
                editor_source,
                use_container_width=True,
                hide_index=True,
                key="research_editor",
                column_config={
                    "keep": st.column_config.CheckboxColumn("Keep", default=True),
                    "row_id": st.column_config.NumberColumn("Row", format="%d"),
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "authors": st.column_config.TextColumn("Authors", width="medium"),
                    "year": st.column_config.TextColumn("Year", width="small"),
                    "doi": st.column_config.TextColumn("DOI", width="medium"),
                    "journal": st.column_config.TextColumn("Journal", width="medium"),
                    "accession": st.column_config.TextColumn("Accession", width="small"),
                    "confidence": st.column_config.TextColumn("Preview", width="small"),
                },
                disabled=["row_id", "confidence"],
            )
            editor_rows = _editor_records(edited)
            if editor_rows:
                st.session_state["research_editor_rows"] = editor_rows

            selected_count = sum(1 for item in editor_rows if bool(item.get("keep", True)))
            st.caption(f"{selected_count} citation(s) selected for resolution.")

            if st.button(
                "Resolve Citations",
                type="primary",
                use_container_width=True,
                key="research_resolve_btn",
                disabled=(selected_count == 0 or db_root is None),
            ):
                try:
                    selected_citations = _merge_research_editor_rows(editor_rows, citations)
                    payload = validate_research_resolve_input(
                        {
                            "citations": selected_citations,
                            "options": {
                                "check_open_access": check_open_access,
                                "enrich_sjr": enrich_sjr,
                                "unpaywall_email": unpaywall_email,
                            },
                        }
                    )
                    run_dir = db_root / "research_resolve" / time.strftime("%Y%m%d_%H%M%S")
                    run_dir.mkdir(parents=True, exist_ok=True)

                    progress = st.progress(0.0, text="Starting citation resolution...")
                    log_box = st.empty()
                    log_lines: List[str] = []

                    def _progress_cb(progress_pct: float, message: str, stage: Optional[str] = None) -> None:
                        frac = max(0.0, min(1.0, float(progress_pct or 0.0) / 100.0))
                        progress.progress(frac, text=message)
                        stamp = time.strftime("%H:%M:%S")
                        stage_label = f"[{stage}] " if stage else ""
                        log_lines.append(f"{stamp} {stage_label}{message}")
                        if len(log_lines) > 300:
                            del log_lines[:-300]
                        log_box.text_area("Resolver log", value="\n".join(log_lines), height=180, disabled=True)

                    output = run_research_resolve(
                        payload=payload,
                        run_dir=run_dir,
                        progress_cb=_progress_cb,
                    )
                    progress.progress(1.0, text="Citation resolution complete")
                    st.session_state["research_resolve_output"] = output
                    st.session_state["research_resolve_run_dir"] = str(run_dir)
                    st.session_state["research_resolve_log_lines"] = list(log_lines)
                except Exception as e:
                    st.error(f"Research resolve failed: {e}")

        elif parse_result:
            for warning in parse_result.get("warnings") or []:
                st.warning(warning)
            st.info("No valid citations were parsed from the supplied spreadsheet.")
        else:
            st.info("Parse a citation spreadsheet from the left panel to preview and resolve rows here.")

        output = st.session_state.get("research_resolve_output") or {}
        if output:
            st.divider()
            st.subheader("Resolved Results")
            stats = output.get("stats") or {}
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total", int(stats.get("total") or 0))
            r2.metric("High confidence", int(stats.get("resolved_high") or 0))
            r3.metric("Low confidence", int(stats.get("resolved_low") or 0))
            r4.metric("Unresolved", int(stats.get("unresolved") or 0))

            oa1, oa2 = st.columns(2)
            oa1.metric("Open access", int(stats.get("open_access") or 0))
            oa2.metric("Closed access", int(stats.get("closed_access") or 0))

            resolved_rows = _research_resolved_rows(output.get("resolved") or [])
            unresolved_rows = _research_unresolved_rows(output.get("unresolved") or [])

            if resolved_rows:
                st.dataframe(
                    resolved_rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "resolved_url": st.column_config.LinkColumn("Resolved URL", display_text="open"),
                        "pdf_url": st.column_config.LinkColumn("PDF URL", display_text="pdf"),
                    },
                )
            if unresolved_rows:
                with st.expander("Unresolved rows", expanded=True):
                    st.dataframe(unresolved_rows, use_container_width=True, hide_index=True)

            preferred_urls = build_research_preferred_url_list(output.get("resolved") or [])
            if preferred_urls:
                st.caption("Preferred ingest URLs use OA PDF links when available, otherwise DOI URLs.")
                st.text_area(
                    "Preferred URL list",
                    value="\n".join(preferred_urls),
                    height=180,
                    key="research_preferred_urls",
                )
                c1, c2 = st.columns(2)
                c1.download_button(
                    "Download URL List",
                    data="\n".join(preferred_urls),
                    file_name="research_resolve_urls.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                if c2.button("Send URLs to URL Ingestor Tab", use_container_width=True, key="research_send_to_url_ingestor"):
                    st.session_state["url_ingestor_input"] = "\n".join(preferred_urls)
                    st.success("Preferred URLs copied into the URL Ingestor tab input.")

            result_json = json.dumps(output, ensure_ascii=False, indent=2)
            st.download_button(
                "Download Full Result JSON",
                data=result_json,
                file_name="research_resolve_result.json",
                mime="application/json",
                use_container_width=True,
            )
            run_dir = st.session_state.get("research_resolve_run_dir")
            if run_dir:
                st.caption(f"Saved run output: {run_dir}")


# ======================================================================
# URL Ingestor tab
# ======================================================================

def _render_url_ingestor_tab():
    render_url_ingestor_ui(standalone=False)


# ======================================================================
# Main
# ======================================================================

def main():
    st.title("Document or Photo Processing")
    st.caption(f"Version: {PAGE_VERSION} • Document conversion, review mining, citation resolution, and privacy tools")

    tab_textifier, tab_included_study, tab_study_miner, tab_research, tab_url_ingest, tab_pdfimg, tab_photo, tab_anonymizer = st.tabs(
        ["Textifier", "Included Study Extractor", "Study Miner", "Research Resolver", "URL PDF Ingestor", "PDF Image Extractor", "Photo Processor", "Anonymizer"]
    )

    with tab_textifier:
        _render_textifier_tab()

    with tab_included_study:
        _render_included_study_extractor_tab()

    with tab_study_miner:
        _render_study_miner_tab()

    with tab_research:
        _render_research_resolver_tab()

    with tab_url_ingest:
        _render_url_ingestor_tab()

    with tab_pdfimg:
        _render_pdf_image_extract_tab()

    with tab_photo:
        _render_photo_keywords_tab()

    with tab_anonymizer:
        _render_anonymizer_tab()


if __name__ == "__main__":
    main()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
