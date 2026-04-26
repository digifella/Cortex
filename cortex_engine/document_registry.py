from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, Optional

from cortex_engine.stakeholder_signal_matcher import normalize_lookup

STRICT_DOC_TYPES = {"annual_report", "strategic_plan"}
RELAXED_DOC_TYPES = {"org_chart"}
DOC_INGEST_POLICIES = {
    "annual_report": "strict",
    "strategic_plan": "strict",
    "org_chart": "relaxed",
}

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*(?:to|[-–])\s*(20\d{2})\b", re.IGNORECASE)
_YEAR_RANGE_COMPACT_RE = re.compile(r"(?<!\d)(20\d{2})\s*[/\-_]\s*(\d{2,4})(?!\d)")


def canonical_org_slug(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", normalize_lookup(value))
    return "-".join(tokens[:12]).strip("-")


def infer_ingest_policy(doc_type: str) -> str:
    return DOC_INGEST_POLICIES.get(str(doc_type or "").strip().lower(), "relaxed")


def _format_period_range(left: str, right: str, doc_type: str) -> str:
    lowered_doc_type = str(doc_type or "").strip().lower()
    if lowered_doc_type == "annual_report":
        if len(right) == 4 and left[:2] == right[:2]:
            return f"{left}-{right[-2:]}"
        return f"{left}-{right}"
    return f"{left}-{right}"


def derive_period_label(doc_type: str, *values: Any) -> str:
    lowered_doc_type = str(doc_type or "").strip().lower()
    text_values = [str(value or "") for value in values if str(value or "").strip()]
    for text in text_values:
        match = _YEAR_RANGE_RE.search(text)
        if match:
            return _format_period_range(match.group(1), match.group(2), lowered_doc_type)
    for text in text_values:
        match = _YEAR_RANGE_COMPACT_RE.search(text)
        if not match:
            continue
        left = match.group(1)
        right = match.group(2)
        if len(right) == 2:
            right = left[:2] + right
        return _format_period_range(left, right, lowered_doc_type)
    if lowered_doc_type == "strategic_plan":
        for text in text_values:
            match = re.search(r"\b(20\d{2})\b", text)
            if match:
                years = [year.group(1) for year in _YEAR_RE.finditer(text)]
                if len(years) >= 2:
                    return f"{years[0]}-{years[-1]}"
                return match.group(1)
    for text in text_values:
        match = _YEAR_RE.search(text)
        if match:
            return match.group(1)
    for text in text_values:
        match = re.search(r"(20\d{2})", text)
        if match:
            return match.group(1)
    return ""


def build_content_fingerprint(
    *,
    attachment_fingerprints: Optional[Iterable[str]] = None,
    text: str = "",
    source_url: str = "",
) -> str:
    attachment_values = [str(item).strip() for item in (attachment_fingerprints or []) if str(item).strip()]
    if attachment_values:
        basis = "||".join(sorted(set(attachment_values)))
        return "sha1:" + hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()
    normalized_text = " ".join(str(text or "").split())
    if normalized_text:
        return "sha1:" + hashlib.sha1(normalized_text.encode("utf-8", "ignore")).hexdigest()
    normalized_url = str(source_url or "").strip().lower()
    if normalized_url:
        return "sha1:" + hashlib.sha1(normalized_url.encode("utf-8", "ignore")).hexdigest()
    return ""


def build_canonical_doc_key(
    target_org_name: str,
    doc_type: str,
    period_label: str = "",
    title: str = "",
    source_url: str = "",
) -> str:
    org_slug = canonical_org_slug(target_org_name) or "unknown-org"
    doc_slug = normalize_lookup(doc_type).replace(" ", "_") or "document"
    period = str(period_label or "").strip()
    if not period:
        fallback = normalize_lookup(title or source_url).replace(" ", "-").strip("-")
        period = fallback[:80] or "undated"
    return f"{org_slug}|{doc_slug}|{period}"


def build_document_meta(
    *,
    doc_type: str,
    target_org_name: str,
    title: str = "",
    period_label: str = "",
    published_at: str = "",
    content_fingerprint: str = "",
    source_url: str = "",
    source_label: str = "",
    status: str = "",
) -> Dict[str, Any]:
    lowered_doc_type = str(doc_type or "").strip().lower()
    resolved_period = str(period_label or "").strip()
    if not resolved_period:
        resolved_period = derive_period_label(lowered_doc_type, title, published_at, source_url, source_label)
    meta = {
        "doc_type": lowered_doc_type,
        "target_org_name": str(target_org_name or "").strip(),
        "title": str(title or "").strip(),
        "period_label": resolved_period,
        "published_at": str(published_at or "").strip(),
        "canonical_doc_key": build_canonical_doc_key(
            target_org_name=str(target_org_name or "").strip(),
            doc_type=lowered_doc_type,
            period_label=resolved_period,
            title=str(title or "").strip(),
            source_url=str(source_url or "").strip(),
        ),
        "content_fingerprint": str(content_fingerprint or "").strip(),
        "source_url": str(source_url or "").strip(),
        "source_label": str(source_label or "").strip(),
        "status": str(status or "").strip() or "new_document",
        "ingest_policy": infer_ingest_policy(lowered_doc_type),
    }
    meta["ingest_recommendation"] = ingest_recommendation(meta["status"], meta["ingest_policy"])
    return meta


def ingest_recommendation(status: str, ingest_policy: str) -> str:
    normalized_status = str(status or "").strip().lower()
    normalized_policy = str(ingest_policy or "").strip().lower()
    if normalized_policy == "strict" and normalized_status == "known_same":
        return "skip"
    return "ingest"


def classify_document(existing: Optional[Dict[str, Any]], document_meta: Dict[str, Any]) -> str:
    if not existing:
        return "new_document"
    existing_fp = str(existing.get("content_fingerprint") or "").strip()
    new_fp = str(document_meta.get("content_fingerprint") or "").strip()
    if existing_fp and new_fp and existing_fp == new_fp:
        return "known_same"
    if existing_fp or new_fp:
        return "changed_document"
    return "known_same"
