from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import fitz

from cortex_engine.research_resolve import _extract_year, _normalize_doi
from cortex_engine.review_study_miner import (
    _DESIGN_EXPANSIONS,
    _NON_STUDY_ROW_LABELS,
    _OUTCOME_EXPANSIONS,
    _expand_keywords,
    _extract_authors_and_year,
    _extract_journal,
    _find_matches,
    _normalize_text,
    _parse_reference_entries,
    _reference_number_pointers,
    _reference_match_from_row,
    _validate_linked_reference,
)
from cortex_engine.review_table_rescue import parse_review_table_rescue_response


_DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip() or "http://127.0.0.1:11434"
_DEFAULT_OLLAMA_MODEL = os.environ.get("CORTEX_REVIEW_TABLE_OLLAMA_MODEL", "").strip() or "qwen3.5:35b-a3b"
_OLLAMA_MODEL_CANDIDATES = (
    "qwen3.5:35b-a3b",
    "qwen3-vl:8b",
    "llava:7b",
    "llava:latest",
)
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_ASSESSMENT_HINTS = (
    "fact",
    "eortc",
    "qlq",
    "sf 36",
    "sf36",
    "eq 5d",
    "global health",
    "subscale",
    "toi",
    "pcs",
    "mcs",
    "ts",
)
_MISSING_VALUE_HINTS = {"", "nr", "n r", "na", "n a", "not reported", "unknown"}
_AUTHOR_YEAR_START_RE = re.compile(r"\b[A-Z][A-Za-z' -]+(?:\s+et al\.)?[\s,;:/-]*(?:\(|\[)?(?:19|20)\d{2}")
_GROUP_TOKEN_STOPWORDS = {
    "adult",
    "adults",
    "aged",
    "assessment",
    "baseline",
    "better",
    "cells",
    "cycle",
    "cycles",
    "design",
    "diffuse",
    "disease",
    "end",
    "follow",
    "global",
    "group",
    "health",
    "higher",
    "hours",
    "large",
    "life",
    "line",
    "lines",
    "lymphoma",
    "measure",
    "measures",
    "month",
    "months",
    "outcome",
    "outcomes",
    "patient",
    "patients",
    "phase",
    "prior",
    "quality",
    "reflect",
    "reported",
    "responders",
    "review",
    "scores",
    "single",
    "status",
    "study",
    "systematic",
    "table",
    "therapy",
    "treatment",
    "trial",
    "value",
    "values",
}


def local_table_rescue_host() -> str:
    return _DEFAULT_OLLAMA_HOST.rstrip("/")


def _ollama_tags_url(host: str = "") -> str:
    target = str(host or local_table_rescue_host()).strip().rstrip("/")
    return f"{target}/api/tags"


def _ollama_chat_url(host: str = "") -> str:
    target = str(host or local_table_rescue_host()).strip().rstrip("/")
    return f"{target}/api/chat"


def local_table_rescue_available(host: str = "") -> bool:
    target = str(host or local_table_rescue_host()).strip().rstrip("/")
    if not target:
        return False
    try:
        with urlopen(_ollama_tags_url(target), timeout=1.5) as response:
            return int(getattr(response, "status", 200) or 200) < 500
    except URLError:
        return False
    except Exception:
        return False


def _list_ollama_models(host: str = "") -> List[str]:
    try:
        with urlopen(_ollama_tags_url(host), timeout=2.0) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        models = payload.get("models") or []
        return [str(item.get("name") or "").strip() for item in models if str(item.get("name") or "").strip()]
    except Exception:
        return []


def _resolve_ollama_model_name(requested: str = "", host: str = "") -> str:
    available = _list_ollama_models(host)
    ordered = [str(requested or "").strip(), _DEFAULT_OLLAMA_MODEL, *_OLLAMA_MODEL_CANDIDATES]
    for candidate in ordered:
        if not candidate:
            continue
        for available_name in available:
            if available_name == candidate:
                return available_name
        for available_name in available:
            if available_name.lower().startswith(f"{candidate.lower()}:"):
                return available_name
    return str(requested or "").strip() or _DEFAULT_OLLAMA_MODEL


def _coerce_list_of_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _informative_tokens(*texts: str) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in _normalize_text(text).split():
            if len(token) < 4:
                continue
            if token in _GROUP_TOKEN_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def _looks_like_assessment_label(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized in _NON_STUDY_ROW_LABELS:
        return True
    return any(hint in normalized for hint in _ASSESSMENT_HINTS)


def _looks_like_missing_value(text: str) -> bool:
    return _normalize_text(text) in _MISSING_VALUE_HINTS


def _looks_like_author_year_blob(text: str) -> bool:
    blob = str(text or "").strip()
    if not blob:
        return False
    authors, year = _extract_authors_and_year(blob)
    return bool(authors and year)


def _same_or_missing_region(current_region: str, previous_region: str) -> bool:
    current = str(current_region or "").strip()
    previous = str(previous_region or "").strip()
    if not current or _looks_like_missing_value(current):
        return True
    if not previous or _looks_like_missing_value(previous):
        return True
    return _normalize_text(current) == _normalize_text(previous)


def _should_carry_forward_study(
    *,
    study: str,
    author_year: str,
    region: str,
    previous: Dict[str, Any],
) -> bool:
    if not previous:
        return False
    previous_study = str(previous.get("study") or "").strip()
    previous_author_year = str(previous.get("author_year") or "").strip()
    previous_region = str(previous.get("region") or "").strip()
    if not study:
        if not author_year:
            return True
        if author_year == previous_author_year:
            return True
        return _same_or_missing_region(region, previous_region)
    if study == previous_study:
        return True
    return bool(author_year and author_year == previous_author_year)


def _normalize_local_rescue_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_rows: List[Dict[str, Any]] = []
    previous: Dict[str, Any] = {}

    for raw_item in list(rows or []):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)

        for key, value in list(item.items()):
            if isinstance(value, str):
                item[key] = re.sub(r"\s+", " ", value).strip()

        study = str(item.get("study") or item.get("study_label") or "").strip()
        region = str(item.get("region") or "").strip()
        author_year = str(item.get("author_year") or item.get("citation") or item.get("authorYear") or "").strip()
        assessment_method = str(item.get("assessment_method") or "").strip()

        if study and not author_year and _looks_like_author_year_blob(study):
            item["author_year"] = study
            author_year = study
            item["study"] = str(previous.get("study") or "").strip()
            study = str(item.get("study") or "").strip()

        if study and _looks_like_assessment_label(study):
            if not assessment_method:
                item["assessment_method"] = study
            if previous.get("study"):
                item["study"] = str(previous.get("study") or "").strip()
                study = str(item.get("study") or "").strip()
            if not author_year and previous.get("author_year"):
                item["author_year"] = str(previous.get("author_year") or "").strip()
                author_year = str(item.get("author_year") or "").strip()

        continuation = _should_carry_forward_study(
            study=study,
            author_year=author_year,
            region=region,
            previous=previous,
        )
        if continuation:
            for field_name in (
                "study",
                "author_year",
                "study_design",
                "patient_population",
                "followup_times_assessed",
            ):
                if not str(item.get(field_name) or "").strip() and str(previous.get(field_name) or "").strip():
                    item[field_name] = str(previous.get(field_name) or "").strip()
            if _looks_like_missing_value(str(item.get("treatment") or "")) and str(previous.get("treatment") or "").strip():
                item["treatment"] = str(previous.get("treatment") or "").strip()

        normalized_rows.append(item)

        if str(item.get("study") or "").strip() or str(item.get("author_year") or "").strip():
            previous = item

    return normalized_rows


def _split_row_citations(author_year: str, study: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(author_year or "").strip())
    if not text:
        return [str(study or "").strip()] if str(study or "").strip() else []

    numbered_chunks = [
        chunk.strip(" ,;")
        for chunk in re.findall(r"[^,;]+?\[\d{1,3}\]", text)
        if chunk.strip(" ,;")
    ]
    if numbered_chunks:
        return list(dict.fromkeys(numbered_chunks))

    author_year_chunks = [
        chunk.strip(" ,;")
        for chunk in re.findall(r"[A-Z][^,;]*?(?:19|20)\d{2}", text)
        if chunk.strip(" ,;")
    ]
    if author_year_chunks:
        return list(dict.fromkeys(author_year_chunks))

    return [text]


def _strip_leading_study_prefix(text: str, study: str) -> str:
    raw = str(text or "").strip()
    study_text = str(study or "").strip()
    if not raw or not study_text:
        return raw
    pattern = re.compile(rf"^\s*{re.escape(study_text)}\s*(?:[:|,;/.-]+)?\s*(.+)$", flags=re.IGNORECASE)
    match = pattern.match(raw)
    if match:
        return str(match.group(1) or "").strip()
    return raw


def _normalize_citation_chunk(text: str, study: str = "") -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip()).strip(" ,;|")
    if not raw:
        return ""
    raw = re.sub(r"^(?:citations?|references?)\s+", "", raw, flags=re.IGNORECASE).strip()
    raw = _strip_leading_study_prefix(raw, study)
    match = _AUTHOR_YEAR_START_RE.search(raw)
    if not match:
        return raw
    prefix = raw[: match.start()].strip(" ,;:/|-")
    if prefix:
        prefix_tokens = [token for token in re.findall(r"[A-Za-z0-9]+", prefix) if token]
        if prefix_tokens and all(token.isdigit() or not re.search(r"[a-z]", token) for token in prefix_tokens):
            raw = raw[match.start() :].strip(" ,;:/|-")
    return raw


def _reference_display_author(authors_text: str) -> str:
    text = re.sub(r"\s+", " ", str(authors_text or "").strip()).strip(" ,;")
    if not text:
        return ""
    if re.search(r"clinicaltrials\.gov", text, flags=re.IGNORECASE):
        return "ClinicalTrials.gov"
    head = re.split(r"\s*(?:,|;|\bet al\b)\s*", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if not head:
        return ""
    if "," in head:
        head = head.split(",", 1)[0].strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z'`.-]*", head)
    if not tokens:
        return head
    return tokens[0]


def _canonical_citation_display(
    citation_text: str,
    linked_reference: Dict[str, Any] | None,
    reference_number: str = "",
) -> str:
    raw = re.sub(r"\s+", " ", str(citation_text or "").strip()).strip(" ,;")
    if not linked_reference:
        return raw
    display_author = _reference_display_author(str(linked_reference.get("authors") or ""))
    display_year = str(linked_reference.get("year") or "").strip()
    display_ref = str(linked_reference.get("reference_number") or reference_number or "").strip()
    if display_author and display_year:
        display = f"{display_author} {display_year}"
        if display_ref:
            display += f" [{display_ref}]"
        return display
    return raw


def _citation_variants_from_notes(notes: str, study: str = "") -> List[str]:
    text = re.sub(r"\s+", " ", str(notes or "").strip())
    if not text:
        return []
    normalized = _normalize_text(text)
    if "citation" not in normalized and "grouped under" not in normalized:
        return []
    numbered_chunks = [
        _normalize_citation_chunk(chunk.strip(" ,;"), study)
        for chunk in re.findall(r"[^,;]+?\[\d{1,3}\]", text)
        if chunk.strip(" ,;")
    ]
    numbered_chunks = [chunk for chunk in numbered_chunks if chunk]
    if len(numbered_chunks) >= 2:
        return list(dict.fromkeys(numbered_chunks))
    return []


def _candidate_dedupe_key(item: Dict[str, Any]) -> str:
    return " | ".join(
        [
            _normalize_text(str(item.get("reference_number") or "")),
            _normalize_text(str(item.get("doi") or "")),
            _normalize_text(str(item.get("title") or "")),
            _normalize_text(str(item.get("authors") or "")),
            _normalize_text(str(item.get("year") or "")),
        ]
    )


def _merge_candidate_values(existing: str, incoming: str) -> str:
    merged: List[str] = []
    for raw in (existing, incoming):
        text = str(raw or "").strip()
        if not text:
            continue
        for part in re.split(r"\s*[;|]\s*|\s*,\s*(?=[A-Z0-9])", text):
            cleaned = part.strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
    return "; ".join(merged)


def _candidate_quality_score(item: Dict[str, Any]) -> int:
    extras = dict(item.get("extra_fields") or {})
    study = str(extras.get("study") or "").strip()
    region = str(extras.get("region") or "").strip()
    citation = str(extras.get("author_year") or item.get("raw_citation") or "").strip()
    score = int(item.get("relevance_score") or 0)
    if study and _normalize_text(study) not in _MISSING_VALUE_HINTS and not _looks_like_assessment_label(study):
        score += 4
    if region and _normalize_text(region) not in _MISSING_VALUE_HINTS and not _looks_like_assessment_label(region):
        score += 1
    if _looks_like_author_year_blob(citation):
        score += 2
    if _reference_number_pointers(citation):
        score += 2
    if str(item.get("reference_validation") or "") == "mismatch":
        score -= 5
    if bool(item.get("needs_review")):
        score -= 2
    return score


def _dedupe_local_rescue_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}

    for item in candidates or []:
        candidate = dict(item)
        key = _candidate_dedupe_key(candidate)
        if not key.strip(" |"):
            continue
        existing_idx = index_by_key.get(key)
        if existing_idx is None:
            deduped.append(candidate)
            index_by_key[key] = len(deduped) - 1
            continue

        existing = deduped[existing_idx]
        if _candidate_quality_score(candidate) > _candidate_quality_score(existing):
            existing, candidate = candidate, existing
            deduped[existing_idx] = existing
        existing["relevance_score"] = max(int(existing.get("relevance_score") or 0), int(candidate.get("relevance_score") or 0))
        existing["meets_criteria"] = bool(existing.get("meets_criteria")) or bool(candidate.get("meets_criteria"))
        existing["needs_review"] = bool(existing.get("needs_review")) or bool(candidate.get("needs_review"))
        existing["review_warning"] = _merge_candidate_values(existing.get("review_warning", ""), candidate.get("review_warning", ""))
        existing["raw_excerpt"] = _merge_candidate_values(existing.get("raw_excerpt", ""), candidate.get("raw_excerpt", ""))
        existing["design_matches"] = sorted(dict.fromkeys(list(existing.get("design_matches") or []) + list(candidate.get("design_matches") or [])))
        existing["outcome_matches"] = sorted(dict.fromkeys(list(existing.get("outcome_matches") or []) + list(candidate.get("outcome_matches") or [])))

        extras = dict(existing.get("extra_fields") or {})
        incoming_extras = dict(candidate.get("extra_fields") or {})
        for field_name in ("assessment_method", "mapped_utility_measure", "source_pages", "author_year"):
            extras[field_name] = _merge_candidate_values(extras.get(field_name, ""), incoming_extras.get(field_name, ""))
        existing["extra_fields"] = extras

    return deduped


def merge_local_table_candidates(*candidate_groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for group in candidate_groups:
        merged.extend(dict(item) for item in list(group or []) if isinstance(item, dict))
    return _dedupe_local_rescue_candidates(merged)


def _candidate_group_key(item: Dict[str, Any]) -> str:
    extras = dict(item.get("extra_fields") or {})
    return " | ".join(
        [
            _normalize_text(str(extras.get("region") or "")),
            _normalize_text(str(extras.get("study") or "")),
        ]
    ).strip(" |")


def _align_reconciled_candidates_to_provisional_groups(
    provisional_candidates: Sequence[Dict[str, Any]],
    reconciled_candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    provisional_group_keys = {
        _candidate_group_key(item)
        for item in provisional_candidates or []
        if isinstance(item, dict) and _candidate_group_key(item)
    }
    provisional_by_reference: Dict[str, Dict[str, Any]] = {}
    for item in provisional_candidates or []:
        if not isinstance(item, dict):
            continue
        reference_number = str(item.get("reference_number") or "").strip()
        if not reference_number:
            continue
        existing = provisional_by_reference.get(reference_number)
        if existing is None or _candidate_quality_score(item) > _candidate_quality_score(existing):
            provisional_by_reference[reference_number] = dict(item)

    aligned: List[Dict[str, Any]] = []
    for item in reconciled_candidates or []:
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        extras = dict(candidate.get("extra_fields") or {})
        reference_number = str(candidate.get("reference_number") or extras.get("reference_number") or "").strip()
        matched_provisional = provisional_by_reference.get(reference_number)
        if matched_provisional:
            matched_extras = dict(matched_provisional.get("extra_fields") or {})
            if str(matched_extras.get("region") or "").strip():
                extras["region"] = str(matched_extras.get("region") or "").strip()
            if str(matched_extras.get("study") or "").strip():
                extras["study"] = str(matched_extras.get("study") or "").strip()
            candidate["extra_fields"] = extras
            aligned.append(candidate)
            continue
        if _candidate_group_key(candidate) in provisional_group_keys:
            candidate["extra_fields"] = extras
            aligned.append(candidate)

    return _dedupe_local_rescue_candidates(aligned)


def annotate_local_candidate_completeness(
    candidates: Sequence[Dict[str, Any]],
    *,
    references_text: str,
) -> List[Dict[str, Any]]:
    annotated = [dict(item) for item in candidates or []]
    reference_entries = _parse_reference_entries(references_text)
    groups: Dict[str, List[int]] = {}

    for idx, item in enumerate(annotated):
        extras = dict(item.get("extra_fields") or {})
        region = str(extras.get("region") or "").strip()
        study = str(extras.get("study") or "").strip()
        key = " | ".join([region, study]).strip()
        if not key:
            continue
        groups.setdefault(key, []).append(idx)

    for group_key, indices in groups.items():
        group_candidates = [annotated[idx] for idx in indices]
        existing_refs = {str(item.get("reference_number") or "").strip() for item in group_candidates if str(item.get("reference_number") or "").strip()}
        surname_keys = {
            _normalize_text(str(item.get("authors") or item.get("raw_citation") or "")).split(" ")[0]
            for item in group_candidates
            if _normalize_text(str(item.get("authors") or item.get("raw_citation") or "")).split(" ")
        }
        surname_keys.discard("")
        extras = dict(group_candidates[0].get("extra_fields") or {})
        group_tokens = _informative_tokens(
            str(extras.get("region") or ""),
            str(extras.get("study") or ""),
            str(extras.get("treatment") or ""),
            " ".join(str(item.get("title") or "") for item in group_candidates),
            " ".join(str(item.get("raw_excerpt") or "")[:400] for item in group_candidates),
        )

        missing_refs: List[str] = []
        for ref in reference_entries:
            ref_number = str(ref.get("reference_number") or "").strip()
            if not ref_number or ref_number in existing_refs:
                continue
            ref_author_key = _normalize_text(str(ref.get("authors") or "")).split(" ")[0]
            ref_tokens = _informative_tokens(
                str(ref.get("entry_text") or ""),
                str(ref.get("title") or ""),
                str(ref.get("authors") or ""),
            )
            score = 0
            if ref_author_key and ref_author_key in surname_keys:
                score += 3
            overlap = group_tokens & ref_tokens
            if len(overlap) >= 2:
                score += 2
            if len(overlap) >= 4:
                score += 1
            if score >= 5:
                missing_refs.append(ref_number)

        missing_refs = list(dict.fromkeys(missing_refs))
        if not missing_refs:
            continue

        warning = f"group may be incomplete; possible additional references: {', '.join(f'[{ref}]' for ref in missing_refs)}"
        for idx in indices:
            item = annotated[idx]
            item["needs_review"] = True
            combined_warning = "; ".join(
                part for part in [str(item.get("review_warning") or "").strip(), warning] if part
            )
            item["review_warning"] = combined_warning
            extras = dict(item.get("extra_fields") or {})
            extras["needs_review"] = "yes"
            extras["review_warning"] = combined_warning
            item["extra_fields"] = extras

    return annotated


def _render_pdf_pages_for_local_rescue(
    pdf_path: str,
    table_snapshots: Sequence[Dict[str, Any]],
    *,
    page_numbers: Sequence[int] | None = None,
    max_pages: int = 6,
    zoom: float = 1.35,
) -> List[Dict[str, Any]]:
    path_text = str(pdf_path or "").strip()
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return []

    def _snapshot_rect_for_page(doc_page: fitz.Page, page_number: int) -> fitz.Rect | None:
        rects: List[fitz.Rect] = []
        page_width = float(doc_page.rect.width)
        page_height = float(doc_page.rect.height)
        for item in list(table_snapshots or []):
            try:
                item_page = int(item.get("page_number") or 0)
            except Exception:
                item_page = 0
            if item_page != page_number:
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                rect = fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])) & doc_page.rect
            except Exception:
                continue
            if rect.is_empty or rect.width < 8 or rect.height < 8:
                continue
            rects.append(rect)

        if not rects:
            return None

        merged = fitz.Rect(rects[0])
        for rect in rects[1:]:
            merged.include_rect(rect)

        pad_x = max(12.0, page_width * 0.02)
        pad_top = max(24.0, page_height * 0.04)
        pad_bottom = max(18.0, page_height * 0.03)
        expanded = fitz.Rect(
            max(0.0, merged.x0 - pad_x),
            max(0.0, merged.y0 - pad_top),
            min(page_width, merged.x1 + pad_x),
            min(page_height, merged.y1 + pad_bottom),
        ) & doc_page.rect
        return expanded if not expanded.is_empty else None

    def _fallback_table_rect(doc_page: fitz.Page) -> fitz.Rect:
        page_width = float(doc_page.rect.width)
        page_height = float(doc_page.rect.height)
        return fitz.Rect(
            page_width * 0.03,
            page_height * 0.08,
            page_width * 0.97,
            page_height * 0.95,
        ) & doc_page.rect

    with fitz.open(path) as doc:
        snapshot_page_numbers = [int(item.get("page_number") or 0) for item in list(table_snapshots or []) if int(item.get("page_number") or 0) > 0]
        ordered_pages = list(dict.fromkeys(snapshot_page_numbers))
        preferred_page_numbers = [int(page) for page in list(page_numbers or []) if int(page or 0) > 0]
        if not ordered_pages:
            ordered_pages = list(dict.fromkeys(preferred_page_numbers))
        if not ordered_pages:
            ordered_pages = list(range(1, min(len(doc), max_pages) + 1))
        ordered_pages = ordered_pages[:max_pages]

        rendered: List[Dict[str, Any]] = []
        for page_number in ordered_pages:
            if page_number < 1 or page_number > len(doc):
                continue
            page = doc[page_number - 1]
            clip_rect = _snapshot_rect_for_page(page, page_number) or _fallback_table_rect(page)
            pix = page.get_pixmap(clip=clip_rect, matrix=fitz.Matrix(float(zoom), float(zoom)), alpha=False)
            rendered.append(
                {
                    "page_number": page_number,
                    "bbox": [round(float(value), 2) for value in clip_rect],
                    "image_bytes": pix.tobytes("png"),
                    "text_sample": " ".join(page.get_text("text").split())[:400],
                }
            )
        return rendered


def _render_full_pdf_pages_for_reconciliation(
    pdf_path: str,
    *,
    max_pages: int = 24,
    zoom: float = 0.85,
) -> List[Dict[str, Any]]:
    path_text = str(pdf_path or "").strip()
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return []

    with fitz.open(path) as doc:
        rendered: List[Dict[str, Any]] = []
        for page_index in range(min(len(doc), max_pages)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(float(zoom), float(zoom)), alpha=False)
            rendered.append(
                {
                    "page_number": page_index + 1,
                    "image_bytes": pix.tobytes("png"),
                    "text_sample": " ".join(page.get_text("text").split())[:400],
                }
            )
        return rendered


def _build_page_hints(rendered_pages: Sequence[Dict[str, Any]], table_blocks: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for page in list(rendered_pages or [])[:6]:
        sample = str(page.get("text_sample") or "").strip()
        part = f"Page {int(page.get('page_number') or 0)}"
        if sample:
            part += f": {sample[:400]}"
        parts.append(part)

    for block in list(table_blocks or [])[:4]:
        header = _coerce_list_of_strings(block.get("header"))
        context = str(block.get("context_text") or "").strip()
        detail_lines = [f"Detected table block {int(block.get('table_index') or 0)}"]
        if header:
            detail_lines.append(f"Columns: {', '.join(header[:12])}")
        if context:
            detail_lines.append(f"Nearby context: {context[:700]}")
        parts.append("\n".join(detail_lines))

    return "\n\n".join(parts).strip()


def _build_provisional_candidate_summary(candidates: Sequence[Dict[str, Any]]) -> str:
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in candidates or []:
        extras = dict(item.get("extra_fields") or {})
        region = str(extras.get("region") or "").strip()
        study = str(extras.get("study") or "").strip()
        key = " | ".join([region, study]).strip()
        if not key:
            continue
        bucket = grouped.setdefault(
            key,
            {
                "region": region,
                "study": study,
                "citations": [],
                "assessment_methods": [],
                "treatment": str(extras.get("treatment") or "").strip(),
                "study_design": str(extras.get("study_design") or "").strip(),
                "patient_population": str(extras.get("patient_population") or "").strip(),
                "followup": str(extras.get("followup_times_assessed") or "").strip(),
            },
        )
        citation = str(extras.get("author_year") or item.get("raw_citation") or "").strip()
        if citation and citation not in bucket["citations"]:
            bucket["citations"].append(citation)
        assessment_method = str(extras.get("assessment_method") or "").strip()
        if assessment_method and assessment_method not in bucket["assessment_methods"]:
            bucket["assessment_methods"].append(assessment_method)

    lines: List[str] = []
    for bucket in grouped.values():
        line = f"- {bucket['region']} / {bucket['study']}"
        if bucket["citations"]:
            line += f" | provisional citations: {', '.join(bucket['citations'])}"
        if bucket["assessment_methods"]:
            line += f" | measures: {', '.join(bucket['assessment_methods'])}"
        if bucket["treatment"]:
            line += f" | treatment: {bucket['treatment']}"
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_rows_payload(text: str) -> Dict[str, Any]:
    parsed = parse_review_table_rescue_response(text)
    if parsed:
        return parsed
    raw = str(text or "").strip()
    if not raw:
        return {}
    fenced = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        maybe = json.loads(fenced)
        if isinstance(maybe, dict):
            return maybe
        if isinstance(maybe, list):
            return {"rows": maybe, "warnings": []}
    except Exception:
        pass
    match = _JSON_ARRAY_RE.search(raw)
    if match:
        try:
            maybe = json.loads(match.group(0))
            if isinstance(maybe, list):
                return {"rows": maybe, "warnings": []}
        except Exception:
            return {}
    return {}


def _normalize_reconciled_candidates(
    payload: Dict[str, Any],
    *,
    design_query: str,
    outcome_query: str,
    references_text: str,
) -> List[Dict[str, Any]]:
    reference_entries = _parse_reference_entries(references_text)
    numbered_map = {
        str(item.get("reference_number") or "").strip(): item
        for item in reference_entries
        if str(item.get("reference_number") or "").strip()
    }
    design_keywords = _expand_keywords(design_query, _DESIGN_EXPANSIONS)
    outcome_keywords = _expand_keywords(outcome_query, _OUTCOME_EXPANSIONS)
    candidates: List[Dict[str, Any]] = []

    for item in list(payload.get("candidates") or payload.get("rows") or []):
        if not isinstance(item, dict):
            continue
        region = str(item.get("region") or "").strip()
        study = str(item.get("study") or "").strip()
        author_year = str(item.get("author_year") or item.get("citation") or "").strip()
        reference_number = str(item.get("reference_number") or "").strip()
        study_design = str(item.get("study_design") or "").strip()
        patient_population = str(item.get("patient_population") or "").strip()
        treatment = str(item.get("treatment") or "").strip()
        followup = str(item.get("followup_times_assessed") or "").strip()
        notes = str(item.get("notes") or "").strip()
        review_warning = str(item.get("review_warning") or "").strip()
        needs_review = bool(item.get("needs_review"))
        assessment_methods = _coerce_list_of_strings(item.get("assessment_methods"))
        mapped_utility_measures = _coerce_list_of_strings(item.get("mapped_utility_measures"))
        source_pages = [int(page) for page in list(item.get("source_pages") or []) if str(page).strip().isdigit()]

        citation_variants = _citation_variants_from_notes(notes, study)
        if not citation_variants:
            citation_variants = [_normalize_citation_chunk(author_year or study, study)]
        citation_variants = [variant for variant in citation_variants if variant]
        if not citation_variants:
            continue
        for citation_cell in citation_variants:
            candidate_reference_number = str(reference_number or "").strip()
            candidate_needs_review = needs_review
            candidate_review_warning = review_warning
            inline_reference_numbers = _reference_number_pointers(citation_cell)
            linked_reference = numbered_map.get(candidate_reference_number) if candidate_reference_number else None
            if linked_reference:
                linked_reference = {**linked_reference, "match_method": "reference_number"}
            if not linked_reference:
                linked_reference = _reference_match_from_row(
                    citation_cell,
                    " | ".join([region, study, treatment, " ".join(assessment_methods)]),
                    reference_entries,
                )
            if not linked_reference and inline_reference_numbers and not candidate_reference_number:
                candidate_reference_number = str(inline_reference_numbers[0] or "").strip()

            title = citation_cell
            authors, year = _extract_authors_and_year(citation_cell)
            journal = ""
            doi = _normalize_doi(str(item.get("doi") or "").strip())
            reference_match_method = ""
            reference_validation = ""
            citation_display = citation_cell

            if linked_reference:
                title = str(linked_reference.get("title") or title or "").strip() or title
                authors = str(linked_reference.get("authors") or authors or "").strip() or authors
                year = str(linked_reference.get("year") or year or "").strip() or year
                journal = str(linked_reference.get("journal") or journal or "").strip() or journal
                doi = str(linked_reference.get("doi") or doi or "").strip() or doi
                candidate_reference_number = str(linked_reference.get("reference_number") or candidate_reference_number or "").strip()
                reference_match_method = str(linked_reference.get("match_method") or "").strip()
                reference_validation, linked_needs_review, linked_warning = _validate_linked_reference(
                    citation_cell,
                    " | ".join([region, study, treatment, " ".join(assessment_methods)]),
                    linked_reference,
                )
                if linked_needs_review:
                    candidate_needs_review = True
                if linked_warning:
                    candidate_review_warning = "; ".join(
                        part for part in [candidate_review_warning, linked_warning] if part
                    )
                citation_display = _canonical_citation_display(citation_cell, linked_reference, candidate_reference_number)
            elif candidate_reference_number:
                citation_display = citation_cell

            scoring_text = " | ".join(
                part
                for part in [
                    region,
                    study,
                    citation_display,
                    study_design,
                    patient_population,
                    treatment,
                    followup,
                    "; ".join(assessment_methods),
                    "; ".join(mapped_utility_measures),
                    notes,
                    str(linked_reference.get("entry_text") or "").strip() if linked_reference else "",
                ]
                if part
            )
            design_matches = _find_matches(scoring_text, design_keywords)
            outcome_matches = _find_matches(scoring_text, outcome_keywords)
            relevance_score = (len(design_matches) * 2) + (len(outcome_matches) * 2)
            if candidate_reference_number:
                relevance_score += 2
            if study_design:
                relevance_score += 1
            if assessment_methods:
                relevance_score += 1

            candidates.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": year or _extract_year(citation_cell),
                    "doi": doi,
                    "journal": journal,
                    "abstract": "",
                    "volume": "",
                    "issue": "",
                    "pages": "",
                    "accession": "",
                    "aim": "",
                    "notes": "local table reconciliation",
                    "source_section": "local_reconciliation",
                    "meets_criteria": bool(design_matches) and bool(outcome_matches),
                    "relevance_score": relevance_score,
                    "design_matches": design_matches,
                    "outcome_matches": outcome_matches,
                    "raw_citation": citation_cell,
                    "raw_excerpt": scoring_text[:1000],
                    "reference_number": candidate_reference_number,
                    "reference_match_method": reference_match_method or ("table_inline_reference" if candidate_reference_number else "local_reconciliation"),
                    "reference_validation": reference_validation,
                    "needs_review": candidate_needs_review,
                    "review_warning": candidate_review_warning,
                    "extra_fields": {
                        "region": region,
                        "study": study,
                        "author_year": citation_display,
                        "study_design": study_design,
                        "patient_population": patient_population,
                        "treatment": treatment,
                        "followup_times_assessed": followup,
                        "assessment_method": "; ".join(assessment_methods),
                        "mapped_utility_measure": "; ".join(mapped_utility_measures),
                        "source_pages": ", ".join(str(page) for page in source_pages),
                        "reference_number": candidate_reference_number,
                        "reference_match_method": reference_match_method or ("table_inline_reference" if candidate_reference_number else "local_reconciliation"),
                        "reference_validation": reference_validation,
                        "needs_review": "yes" if candidate_needs_review else "",
                        "review_warning": candidate_review_warning,
                    },
                }
            )

    return _dedupe_local_rescue_candidates(candidates)


def _normalize_local_rescue_candidates(
    payload: Dict[str, Any],
    *,
    design_query: str,
    outcome_query: str,
    references_text: str,
) -> List[Dict[str, Any]]:
    reference_entries = _parse_reference_entries(references_text)
    design_keywords = _expand_keywords(design_query, _DESIGN_EXPANSIONS)
    outcome_keywords = _expand_keywords(outcome_query, _OUTCOME_EXPANSIONS)
    candidates: List[Dict[str, Any]] = []

    for item in _normalize_local_rescue_rows(payload.get("rows") or []):
        region = str(item.get("region") or "").strip()
        study = str(item.get("study") or item.get("study_label") or "").strip()
        author_year = str(item.get("author_year") or item.get("citation") or item.get("authorYear") or "").strip()
        study_design = str(item.get("study_design") or item.get("study_type") or "").strip()
        patient_population = str(item.get("patient_population") or "").strip()
        treatment = str(item.get("treatment") or "").strip()
        followup = str(item.get("followup_times_assessed") or item.get("follow_up_times_assessed") or "").strip()
        assessment_method = str(item.get("assessment_method") or "").strip()
        scale = str(item.get("scale") or "").strip()
        baseline = str(item.get("baseline_mean_sd") or item.get("baseline") or "").strip()
        mean_change = str(item.get("mean_change_last_followup_sd") or item.get("mean_change") or "").strip()
        mapped_measure = str(
            item.get("hrqol_measures_mappable_to_direct_utility_values")
            or item.get("mapped_utility_measure")
            or item.get("mapped_measure")
            or ""
        ).strip()
        notes = str(item.get("notes") or "").strip()
        source_pages = [int(page) for page in list(item.get("source_pages") or []) if str(page).strip().isdigit()]

        field_parts = [
            region,
            study,
            author_year,
            study_design,
            patient_population,
            treatment,
            followup,
            assessment_method,
            scale,
            baseline,
            mean_change,
            mapped_measure,
            notes,
        ]
        joined = " | ".join(part for part in field_parts if part)
        if not joined:
            continue

        citation_variants = [_normalize_citation_chunk(chunk, study) for chunk in _split_row_citations(author_year, study)]
        citation_variants = [chunk for chunk in citation_variants if chunk]
        if not citation_variants:
            citation_variants = [_normalize_citation_chunk(author_year or study, study)]

        for citation_cell in citation_variants:
            citation_cell = str(citation_cell or "").strip()
            if not citation_cell:
                continue

            linked_reference = _reference_match_from_row(citation_cell, joined, reference_entries)
            reference_numbers = _reference_number_pointers(citation_cell)
            if not linked_reference and reference_numbers:
                numbered_map = {
                    str(item.get("reference_number") or "").strip(): item
                    for item in reference_entries
                    if str(item.get("reference_number") or "").strip()
                }
                matched = numbered_map.get(reference_numbers[0])
                if matched:
                    linked_reference = {**matched, "match_method": "reference_number"}

            authors, year = _extract_authors_and_year(citation_cell or joined)
            title = citation_cell
            journal = _extract_journal(joined)
            doi = _normalize_doi(str(item.get("doi") or "").strip())
            reference_number = ""
            reference_match_method = ""
            reference_validation = ""
            needs_review = bool(item.get("needs_review"))
            review_warning = str(item.get("review_warning") or "").strip()

            citation_display = citation_cell
            if linked_reference:
                title = str(linked_reference.get("title") or title or "").strip() or title
                authors = str(linked_reference.get("authors") or authors or "").strip() or authors
                year = str(linked_reference.get("year") or year or "").strip() or year
                journal = str(linked_reference.get("journal") or journal or "").strip() or journal
                doi = str(linked_reference.get("doi") or doi or "").strip() or doi
                reference_number = str(linked_reference.get("reference_number") or "").strip()
                reference_match_method = str(linked_reference.get("match_method") or "").strip()
                reference_validation, linked_needs_review, linked_warning = _validate_linked_reference(citation_cell, joined, linked_reference)
                if linked_needs_review:
                    needs_review = True
                if linked_warning:
                    review_warning = "; ".join(part for part in [review_warning, linked_warning] if part)
                citation_display = _canonical_citation_display(citation_cell, linked_reference, reference_number)
            elif reference_numbers:
                reference_number = str(reference_numbers[0] or "").strip()
                reference_match_method = "table_inline_reference"
            if reference_numbers and reference_number and reference_number not in reference_numbers:
                needs_review = True
                mismatch_warning = (
                    f"table reference number(s) {', '.join(reference_numbers)} do not match linked reference {reference_number}"
                )
                review_warning = "; ".join(part for part in [review_warning, mismatch_warning] if part)
                reference_validation = "mismatch"

            scoring_text = joined
            if linked_reference:
                scoring_text = " | ".join(
                    part
                    for part in [
                        joined,
                        str(linked_reference.get("entry_text") or "").strip(),
                        str(linked_reference.get("title") or "").strip(),
                    ]
                    if part
                )
            design_matches = _find_matches(scoring_text, design_keywords)
            outcome_matches = _find_matches(scoring_text, outcome_keywords)
            relevance_score = (len(design_matches) * 2) + (len(outcome_matches) * 2)
            if reference_number:
                relevance_score += 1
            if study_design:
                relevance_score += 1
            if assessment_method or scale:
                relevance_score += 1

            candidates.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": year or _extract_year(joined),
                    "doi": doi,
                    "journal": journal,
                    "abstract": "",
                    "volume": "",
                    "issue": "",
                    "pages": "",
                    "accession": "",
                    "aim": "",
                    "notes": "local table rescue",
                    "source_section": "local_rescue",
                    "meets_criteria": bool(design_matches) and bool(outcome_matches),
                    "relevance_score": relevance_score,
                    "design_matches": design_matches,
                    "outcome_matches": outcome_matches,
                    "raw_citation": citation_cell,
                    "raw_excerpt": scoring_text[:1000],
                    "reference_number": reference_number,
                    "reference_match_method": reference_match_method or "local_rescue",
                    "reference_validation": reference_validation,
                    "needs_review": needs_review,
                    "review_warning": review_warning,
                    "extra_fields": {
                        "region": region,
                        "study": study,
                        "author_year": citation_display,
                        "study_design": study_design,
                        "patient_population": patient_population,
                        "treatment": treatment,
                        "followup_times_assessed": followup,
                        "assessment_method": assessment_method,
                        "scale": scale,
                        "baseline_mean_sd": baseline,
                        "mean_change_last_followup_sd": mean_change,
                        "mapped_utility_measure": mapped_measure,
                        "source_pages": ", ".join(str(page) for page in source_pages),
                        "reference_number": reference_number,
                        "reference_match_method": reference_match_method or "local_rescue",
                        "reference_validation": reference_validation,
                        "needs_review": "yes" if needs_review else "",
                        "review_warning": review_warning,
                    },
                }
            )

    return _dedupe_local_rescue_candidates(candidates)


def _ollama_chat(model: str, prompt: str, image_bytes_list: Sequence[bytes], host: str = "") -> str:
    encoded_images = [base64.b64encode(item).decode("utf-8") for item in image_bytes_list if item]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": encoded_images,
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 8000,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        _ollama_chat_url(host),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=240) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    message = parsed.get("message") or {}
    content = str(message.get("content") or parsed.get("response") or "").strip()
    if content:
        return content
    return str(message.get("thinking") or "").strip()


def run_local_table_rescue(
    *,
    review_title: str,
    design_query: str,
    outcome_query: str,
    table_snapshots: Sequence[Dict[str, Any]],
    table_blocks: Sequence[Dict[str, Any]],
    references_text: str,
    pdf_path: str = "",
    page_numbers: Sequence[int] | None = None,
    model: str = "",
) -> Dict[str, Any]:
    rendered_pages = _render_pdf_pages_for_local_rescue(pdf_path, table_snapshots, page_numbers=page_numbers)
    if not rendered_pages and not table_snapshots:
        return {
            "provider": "ollama",
            "model": model or _DEFAULT_OLLAMA_MODEL,
            "candidates": [],
            "warnings": ["No table page evidence available for local rescue."],
            "raw_response": "",
            "used_page_images": False,
        }

    model_name = _resolve_ollama_model_name(model)
    evidence_pages = rendered_pages or list(table_snapshots or [])[:6]
    page_hints = _build_page_hints(rendered_pages, table_blocks)

    prompt = (
        "You are reading page images from a multi-page continuation table in a systematic review.\n\n"
        "Extract the table rows into JSON. Ignore page headers, page numbers, and repeated column headers.\n"
        "When a row has blank cells that clearly inherit from the preceding row or study block, fill them forward.\n"
        "Keep the study/trial label and the author/year citation separate when possible.\n"
        "Focus on visible evidence only. If uncertain, set needs_review=true.\n\n"
        "Return compact JSON only. Do not add markdown fences, commentary, or prose.\n\n"
        f"Review title: {review_title}\n"
        f"Design criteria: {design_query}\n"
        f"Outcome criteria: {outcome_query}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "rows": [\n'
        "    {\n"
        '      "region": "",\n'
        '      "study": "",\n'
        '      "author_year": "",\n'
        '      "study_design": "",\n'
        '      "patient_population": "",\n'
        '      "treatment": "",\n'
        '      "followup_times_assessed": "",\n'
        '      "assessment_method": "",\n'
        '      "scale": "",\n'
        '      "baseline_mean_sd": "",\n'
        '      "mean_change_last_followup_sd": "",\n'
        '      "hrqol_measures_mappable_to_direct_utility_values": "",\n'
        '      "source_pages": [],\n'
        '      "needs_review": false,\n'
        '      "review_warning": ""\n'
        "    }\n"
        "  ],\n"
        '  "warnings": []\n'
        "}\n\n"
        "Hints from the PDF/text layer:\n"
        f"{page_hints[:4000]}\n\n"
        "Reference excerpt for citation linking:\n"
        f"{str(references_text or '').strip()[:5000]}"
    )

    raw_response = _ollama_chat(
        model_name,
        prompt,
        [page.get("image_bytes") for page in evidence_pages[:6] if page.get("image_bytes")],
    )
    parsed = _extract_rows_payload(raw_response)
    warnings = _coerce_list_of_strings(parsed.get("warnings"))
    return {
        "provider": "ollama",
        "model": model_name,
        "used_page_images": bool(evidence_pages),
        "pages_used": [int(page.get("page_number") or 0) for page in evidence_pages if int(page.get("page_number") or 0) > 0],
        "candidates": _normalize_local_rescue_candidates(
            parsed,
            design_query=design_query,
            outcome_query=outcome_query,
            references_text=references_text,
        ),
        "warnings": warnings,
        "raw_response": raw_response,
    }


def run_local_table_reconciliation(
    *,
    review_title: str,
    design_query: str,
    outcome_query: str,
    provisional_candidates: Sequence[Dict[str, Any]],
    references_text: str,
    review_text: str = "",
    pdf_path: str = "",
    page_numbers: Sequence[int] | None = None,
    preserve_group_labels: bool = False,
    model: str = "",
) -> Dict[str, Any]:
    rendered_pages = _render_full_pdf_pages_for_reconciliation(pdf_path)
    scoped_page_numbers = [int(page) for page in list(page_numbers or []) if int(page or 0) > 0]
    if scoped_page_numbers:
        rendered_pages = [page for page in rendered_pages if int(page.get("page_number") or 0) in scoped_page_numbers]
    model_name = _resolve_ollama_model_name(model)
    provisional_summary = _build_provisional_candidate_summary(provisional_candidates)

    if not provisional_summary:
        return {
            "provider": "ollama",
            "model": model_name,
            "used_page_images": bool(rendered_pages),
            "pages_used": [int(page.get("page_number") or 0) for page in rendered_pages if int(page.get("page_number") or 0) > 0],
            "candidates": [],
            "warnings": ["No provisional table candidates were available for reconciliation."],
            "raw_response": "",
        }

    prompt = (
        "You are reconciling study citations from a full systematic-review PDF after an initial table-extraction pass.\n\n"
        "Use the full review PDF, the provisional grouped candidate list, and the extracted reference section.\n"
        "Your job is to produce one consolidated candidate per cited paper/reference within each study/trial group.\n"
        "Do not return one row per outcome measure. Merge multiple measures for the same cited paper into assessment_methods.\n"
        "Add missing cited papers when they are clearly present in the table or reference list.\n"
        "Do not mix rows from different tables or unrelated sections of the review.\n"
        "Stay anchored to the provisional group labels and page scope supplied here.\n"
        "Keep the provisional region/study grouping unless the table itself clearly shows a better label.\n"
        "If the table's reference number appears wrong, keep the table citation text, link the best bibliography match, and set needs_review=true with a short review_warning.\n"
        "Return compact JSON only. Do not add markdown fences or prose.\n\n"
        f"Review title: {review_title}\n"
        f"Design criteria: {design_query}\n"
        f"Outcome criteria: {outcome_query}\n\n"
        f"Scoped table pages: {', '.join(str(page) for page in scoped_page_numbers) if scoped_page_numbers else 'all rendered pages'}\n"
        f"Preserve provisional group labels: {'yes' if preserve_group_labels else 'no'}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "region": "",\n'
        '      "study": "",\n'
        '      "author_year": "",\n'
        '      "reference_number": "",\n'
        '      "study_design": "",\n'
        '      "patient_population": "",\n'
        '      "treatment": "",\n'
        '      "followup_times_assessed": "",\n'
        '      "assessment_methods": [],\n'
        '      "mapped_utility_measures": [],\n'
        '      "source_pages": [],\n'
        '      "notes": "",\n'
        '      "needs_review": false,\n'
        '      "review_warning": ""\n'
        "    }\n"
        "  ],\n"
        '  "warnings": []\n'
        "}\n\n"
        "Provisional grouped candidates from the first pass:\n"
        f"{provisional_summary[:6000]}\n\n"
        "Extracted reference section:\n"
        f"{str(references_text or '').strip()[:18000]}\n\n"
        "Extracted review text excerpt:\n"
        f"{str(review_text or '').strip()[:18000]}"
    )

    raw_response = _ollama_chat(
        model_name,
        prompt,
        [page.get("image_bytes") for page in rendered_pages if page.get("image_bytes")],
    )
    parsed = _extract_rows_payload(raw_response)
    warnings = _coerce_list_of_strings(parsed.get("warnings"))
    return {
        "provider": "ollama",
        "model": model_name,
        "used_page_images": bool(rendered_pages),
        "pages_used": [int(page.get("page_number") or 0) for page in rendered_pages if int(page.get("page_number") or 0) > 0],
        "candidates": (
            _align_reconciled_candidates_to_provisional_groups(
                provisional_candidates,
                _normalize_reconciled_candidates(
                    parsed,
                    design_query=design_query,
                    outcome_query=outcome_query,
                    references_text=references_text,
                ),
            )
            if preserve_group_labels
            else _normalize_reconciled_candidates(
                parsed,
                design_query=design_query,
                outcome_query=outcome_query,
                references_text=references_text,
            )
        ),
        "warnings": warnings,
        "raw_response": raw_response,
    }


lmstudio_table_rescue_base_url = local_table_rescue_host
lmstudio_table_rescue_available = local_table_rescue_available
run_lmstudio_table_rescue = run_local_table_rescue
