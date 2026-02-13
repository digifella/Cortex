"""
Shared document preface generation for markdown artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .llm_interface import LLMInterface
from .preface_classification import classify_credibility_tier_with_reason
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _user_visible_filename(file_path: str) -> str:
    name = Path(file_path).name
    m = re.match(r"^upload_\d+_(.+)$", name)
    return m.group(1) if m else name


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


def _extract_available_at(md_content: str) -> str:
    text = md_content or ""
    doi_match = re.search(r"(https?://(?:dx\.)?doi\.org/[^\s)\]]+)", text, re.IGNORECASE)
    if doi_match:
        return doi_match.group(1).strip(" .,;)")
    url_match = re.search(r"(https?://[^\s)\]]+)", text, re.IGNORECASE)
    if url_match:
        return url_match.group(1).strip(" .,;)")
    return "Unknown"


def _detect_source_type_hint(file_path: str, md_content: str) -> str:
    text = f"{_user_visible_filename(file_path)}\n{md_content[:20000]}".lower()
    if any(x in text for x in ["arxiv", "ssrn", "biorxiv", "preprint"]):
        return "Academic"
    if any(x in text for x in ["who", "un.org", ".gov", ".edu", "oecd", "world bank", "institute"]):
        return "Consulting Company"
    if any(x in text for x in ["chatgpt", "claude", "gemini", "perplexity", "ai generated"]):
        return "AI Generated Report"
    return "Other"


def _guess_title_from_markdown(md_content: str, file_path: str) -> str:
    for raw in (md_content or "").splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if len(line) > 8 and len(line) < 200 and not line.lower().startswith(("source:", "http://", "https://")):
            return line
    return Path(_user_visible_filename(file_path)).stem or "Unknown"


def _extract_keywords_simple(md_content: str, limit: int = 8) -> List[str]:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", md_content or "").lower()
    words = [w for w in text.split() if len(w) > 3]
    stop = {
        "this", "that", "with", "from", "have", "were", "their", "about", "into", "while",
        "where", "which", "there", "these", "those", "http", "https", "source", "page",
        "report", "study", "article", "journal", "paper", "figure",
    }
    seen = set()
    out: List[str] = []
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def _extract_preface_metadata_with_llm(file_path: str, md_content: str, source_hint: str) -> Optional[dict]:
    try:
        llm = LLMInterface(model="mistral:latest", temperature=0.1, request_timeout=60.0)
    except Exception as e:
        logger.warning(f"Could not initialize LLM for preface extraction: {e}")
        return None

    snippet = (md_content or "")[:18000]
    prompt = f"""
Extract metadata from markdown and return STRICT JSON only with keys:
title, source_type, publisher, publishing_date, authors, available_at, abstract, keywords.

Rules:
- source_type one of: Academic, Consulting Company, AI Generated Report, Other
- Prefer source hint "{source_hint}" unless strong contrary evidence.
- authors and keywords must be arrays.
- If unknown, use "Unknown" or [].

File name: {_user_visible_filename(file_path)}
Markdown:
{snippet}
"""
    try:
        response = llm.generate(prompt, max_tokens=700)
        return _extract_json_block(response)
    except Exception as e:
        logger.warning(f"LLM preface extraction failed: {e}")
        return None


def _normalize_preface_metadata(file_path: str, source_hint: str, raw_meta: Optional[dict], md_content: str) -> dict:
    raw_meta = raw_meta or {}
    title = _clean_line(str(raw_meta.get("title") or "")) or _guess_title_from_markdown(md_content, file_path)
    source_type = _clean_line(str(raw_meta.get("source_type") or "")) or source_hint or "Other"
    if source_type not in {"Academic", "Consulting Company", "AI Generated Report", "Other"}:
        source_type = source_hint or "Other"

    available_at = _clean_line(str(raw_meta.get("available_at") or "")) or _extract_available_at(md_content)
    publisher = _clean_line(str(raw_meta.get("publisher") or ""))
    if not publisher and available_at not in {"", "Unknown"}:
        publisher = urlparse(available_at).netloc or "Unknown"
    publisher = publisher or "Unknown"

    publishing_date = _clean_line(str(raw_meta.get("publishing_date") or "")) or "Unknown"
    authors = raw_meta.get("authors") if isinstance(raw_meta.get("authors"), list) else []
    authors = [_clean_line(str(a)) for a in authors if _clean_line(str(a))]

    abstract = _clean_line(str(raw_meta.get("abstract") or ""))
    if not abstract:
        abstract = " ".join((_clean_line(x) for x in (md_content or "").splitlines() if _clean_line(x)))[:1200] or "Unknown"

    keywords = raw_meta.get("keywords") if isinstance(raw_meta.get("keywords"), list) else []
    keywords = [_clean_line(str(k)).lower() for k in keywords if _clean_line(str(k))]
    if not keywords:
        keywords = _extract_keywords_simple(md_content)

    tier_value, tier_key, tier_label, tier_reason = classify_credibility_tier_with_reason(
        text=f"{publisher}\n{available_at}\n{md_content[:50000]}",
        source_type=source_type,
        availability_status="available" if available_at != "Unknown" else "unknown",
    )

    return {
        "title": title,
        "source_type": source_type,
        "publisher": publisher,
        "publishing_date": publishing_date,
        "authors": authors[:8],
        "available_at": available_at,
        "availability_status": "available" if available_at != "Unknown" else "unknown",
        "availability_http_code": "200" if available_at != "Unknown" else "",
        "availability_checked_at": "",
        "availability_note": "",
        "source_integrity_flag": "verified" if available_at != "Unknown" else "unverified",
        "keywords": keywords[:8],
        "abstract": abstract,
        "credibility_tier_value": tier_value,
        "credibility_tier_key": tier_key,
        "credibility_tier_label": tier_label,
        "credibility_reason": tier_reason,
        "credibility": f"Final {tier_label} Report",
    }


def _yaml_escape(value: object) -> str:
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
        f"credibility_reason: {_yaml_escape(md_meta.get('credibility_reason', 'unknown'))}",
        f"credibility: {_yaml_escape(md_meta.get('credibility', 'Unclassified Report'))}",
        "journal_ranking_source: 'n/a'",
        "journal_sourceid: ''",
        "journal_title: ''",
        "journal_issn: ''",
        "journal_sjr: '0.0'",
        "journal_quartile: ''",
        "journal_rank_global: '0'",
        "journal_categories: ''",
        "journal_areas: ''",
        "journal_high_ranked: 'False'",
        "journal_match_method: 'none'",
        "journal_match_confidence: '0.0'",
        f"keywords: {keywords_yaml}",
        f"abstract: {_yaml_escape(md_meta['abstract'])}",
        "---",
        "",
    ]
    return "\n".join(lines)


def add_document_preface(file_path: str, md_content: str) -> str:
    source_hint = _detect_source_type_hint(file_path, md_content)
    raw_meta = _extract_preface_metadata_with_llm(file_path, md_content, source_hint)
    meta = _normalize_preface_metadata(file_path, source_hint, raw_meta, md_content)
    preface = _build_preface(meta)
    return preface + md_content
