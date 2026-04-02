from __future__ import annotations

import csv
import io
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import requests

from cortex_engine.journal_authority import classify_journal_authority
from cortex_engine.preface_classification import classify_credibility_tier_with_reason
from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PMID_RE = re.compile(r"^\d{1,8}$")
_PREPRINT_MARKERS = (
    "preprint",
    "pre-print",
    "arxiv",
    "biorxiv",
    "medrxiv",
    "ssrn",
    "research square",
)
_RESEARCH_COLUMN_ALIASES = {
    "title": ["title", "article title", "paper title"],
    "authors": ["author", "authors", "author s", "author(s)"],
    "year": ["year", "published year", "pub year", "date"],
    "doi": ["doi", "digital object identifier"],
    "journal": ["journal", "source", "publication", "periodical"],
    "abstract": ["abstract", "summary"],
    "volume": ["volume", "vol"],
    "issue": ["issue", "no"],
    "pages": ["pages", "page range"],
    "accession": ["accession number", "pmid", "pubmed id"],
    "aim": ["aim", "aim objective", "objective", "purpose"],
    "notes": ["notes", "tags", "study", "ref"],
}


def _normalize_text(text: str) -> str:
    base = unicodedata.normalize("NFKD", str(text or ""))
    ascii_text = base.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    return re.sub(r"\s+", " ", cleaned)


def _normalize_header_key(text: str) -> str:
    return _normalize_text(str(text or "").replace("&", " and "))


def _preview_confidence(citation: Dict[str, Any]) -> str:
    if str(citation.get("doi") or "").strip():
        return "green"
    if str(citation.get("title") or "").strip() and str(citation.get("authors") or "").strip() and _extract_year(citation.get("year")):
        return "amber"
    if str(citation.get("title") or "").strip():
        return "red"
    return "invalid"


def _detect_column_mapping(headers: List[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    used_fields: set[str] = set()
    normalized_headers = [_normalize_header_key(header) for header in headers]

    for idx, header in enumerate(normalized_headers):
        for canonical, aliases in _RESEARCH_COLUMN_ALIASES.items():
            if canonical in used_fields:
                continue
            if header == canonical or header in aliases:
                mapping[idx] = canonical
                used_fields.add(canonical)
                break

    for idx, header in enumerate(normalized_headers):
        if idx in mapping or not header:
            continue
        for canonical, aliases in _RESEARCH_COLUMN_ALIASES.items():
            if canonical in used_fields:
                continue
            choices = [canonical, *aliases]
            if any(header in choice or choice in header for choice in choices):
                mapping[idx] = canonical
                used_fields.add(canonical)
                break

    return mapping


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _build_citation_from_row(
    row_values: List[Any],
    headers: List[str],
    mapping: Dict[int, str],
    row_id: int,
) -> Optional[Dict[str, Any]]:
    values = [_stringify_cell(value) for value in row_values]
    if not any(values):
        return None

    citation = {
        "row_id": row_id,
        "title": "",
        "authors": "",
        "year": "",
        "doi": "",
        "journal": "",
        "abstract": "",
        "volume": "",
        "issue": "",
        "pages": "",
        "accession": "",
        "aim": "",
        "notes": "",
        "extra_fields": {},
    }
    extras: Dict[str, str] = {}
    for idx, value in enumerate(values):
        if not value:
            continue
        canonical = mapping.get(idx)
        if canonical:
            citation[canonical] = value
        else:
            header = str(headers[idx] or f"column_{idx + 1}").strip()
            extras[header] = value

    citation["doi"] = _normalize_doi(citation.get("doi") or "")
    citation["year"] = _extract_year(citation.get("year"))
    citation["extra_fields"] = extras
    if not str(citation.get("title") or "").strip():
        return None
    citation["preview_confidence"] = _preview_confidence(citation)
    return citation


def _parse_delimited_text(raw_text: str) -> List[List[str]]:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not text.strip():
        return []
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    if delimiter == ",":
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
            delimiter = getattr(dialect, "delimiter", ",") or ","
        except Exception:
            delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [list(row) for row in reader]


def _parse_xlsx_bytes(data: bytes) -> List[List[str]]:
    try:
        from openpyxl import load_workbook
    except Exception as exc:
        raise RuntimeError("openpyxl is required for XLSX parsing") from exc

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    sheet = workbook.active
    rows: List[List[str]] = []
    for row in sheet.iter_rows(values_only=True):
        rows.append([_stringify_cell(cell) for cell in row])
    return rows


def _build_parse_result(rows: List[List[str]], source_name: str) -> Dict[str, Any]:
    if not rows:
        return {
            "source_name": source_name,
            "citations": [],
            "detected_fields": [],
            "warnings": ["No rows detected."],
        }

    headers = [_stringify_cell(cell) or f"column_{idx + 1}" for idx, cell in enumerate(rows[0])]
    mapping = _detect_column_mapping(headers)
    citations: List[Dict[str, Any]] = []
    warnings: List[str] = []
    seen_dois: set[str] = set()
    duplicate_dois: List[str] = []

    for offset, row in enumerate(rows[1:], start=1):
        citation = _build_citation_from_row(row, headers, mapping, row_id=offset)
        if not citation:
            continue
        doi = str(citation.get("doi") or "").lower()
        if doi:
            if doi in seen_dois:
                duplicate_dois.append(citation["doi"])
                continue
            seen_dois.add(doi)
        citations.append(citation)

    if duplicate_dois:
        warnings.append(
            f"Skipped {len(duplicate_dois)} duplicate DOI row(s): {', '.join(sorted(dict.fromkeys(duplicate_dois))[:5])}"
        )
    if "title" not in mapping.values():
        warnings.append("No title column detected. Resolution requires a title column.")

    return {
        "source_name": source_name,
        "citations": citations,
        "detected_fields": [field for field in _RESEARCH_COLUMN_ALIASES.keys() if field in mapping.values()],
        "warnings": warnings,
    }


def parse_research_spreadsheet_text(raw_text: str, source_name: str = "Pasted text") -> Dict[str, Any]:
    return _build_parse_result(_parse_delimited_text(raw_text), source_name)


def parse_research_spreadsheet_upload(filename: str, data: bytes) -> Dict[str, Any]:
    suffix = Path(str(filename or "upload")).suffix.lower()
    if suffix == ".xlsx":
        rows = _parse_xlsx_bytes(data)
    else:
        rows = _parse_delimited_text(data.decode("utf-8", errors="ignore"))
    return _build_parse_result(rows, Path(str(filename or "upload")).name)


def build_research_preview_rows(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for citation in citations or []:
        rows.append(
            {
                "keep": True,
                "row_id": citation.get("row_id"),
                "title": citation.get("title", ""),
                "authors": citation.get("authors", ""),
                "year": citation.get("year", ""),
                "doi": citation.get("doi", ""),
                "journal": citation.get("journal", ""),
                "accession": citation.get("accession", ""),
                "confidence": citation.get("preview_confidence") or _preview_confidence(citation),
            }
        )
    return rows


def build_research_preferred_url_list(resolved: List[Dict[str, Any]]) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()
    for item in resolved or []:
        open_access = item.get("open_access") or {}
        candidate = str(open_access.get("pdf_url") or item.get("resolved_url") or "").strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            urls.append(candidate)
    return urls


def _title_similarity(left: str, right: str) -> float:
    a = _normalize_text(left)
    b = _normalize_text(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(a=a, b=b).ratio()
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    jaccard = (len(a_tokens & b_tokens) / len(a_tokens | b_tokens)) if (a_tokens or b_tokens) else 0.0
    return round((ratio * 0.7) + (jaccard * 0.3), 3)


def _extract_first_author(authors: str) -> str:
    text = str(authors or "").strip()
    if not text:
        return ""
    first = re.split(r";|\band\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if "," in first:
        return first.split(",", 1)[0].strip()
    parts = [part for part in re.split(r"\s+", first) if part]
    return parts[-1] if parts else ""


def _normalize_doi(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("https://dx.doi.org/", "").replace("http://dx.doi.org/", "")
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    match = _DOI_RE.search(text)
    return match.group(0).rstrip(" .;,)") if match else text.strip().rstrip(" .;,)")


def _extract_year(raw: Any) -> str:
    text = str(raw or "").strip()
    match = _YEAR_RE.search(text)
    return match.group(0) if match else ""


def _pick_primary_url(message: Dict[str, Any], doi: str) -> str:
    candidates = [
        str(message.get("URL") or "").strip(),
        str(message.get("link") or "").strip(),
        f"https://doi.org/{doi}" if doi else "",
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _extract_issn(message: Dict[str, Any]) -> str:
    for key in ("ISSN", "issn-type"):
        value = message.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    issn = str(item.get("value") or "").strip()
                    if issn:
                        return issn
                else:
                    issn = str(item or "").strip()
                    if issn:
                        return issn
        issn = str(value or "").strip()
        if issn:
            return issn
    return ""


def _extract_crossref_year(message: Dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "issued", "created"):
        section = message.get(key) or {}
        parts = section.get("date-parts") if isinstance(section, dict) else None
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            value = str(parts[0][0]).strip()
            if _YEAR_RE.fullmatch(value):
                return value
    return ""


def _retraction_info(message: Dict[str, Any]) -> Dict[str, Any]:
    updates = message.get("update-to") or []
    if not isinstance(updates, list):
        updates = [updates]
    for item in updates:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("type") or "").strip().lower()
        if "retract" in label:
            return {
                "is_retracted": True,
                "warning": "CrossRef metadata indicates a retraction update.",
            }
    return {"is_retracted": False, "warning": ""}


class ResearchResolver:
    def __init__(
        self,
        *,
        options: Dict[str, Any],
        progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
        is_cancelled_cb: Optional[Callable[[], bool]] = None,
        session: Optional[requests.Session] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.options = dict(options or {})
        self.progress_cb = progress_cb
        self.is_cancelled_cb = is_cancelled_cb
        self.sleep_fn = sleep_fn
        self.session = session or requests.Session()
        polite_email = str(
            self.options.get("unpaywall_email")
            or self.options.get("crossref_mailto")
            or ""
        ).strip()
        ua = "CortexSuiteResearchResolve/1.0"
        if polite_email:
            ua = f"{ua} (mailto:{polite_email})"
        self.session.headers.update(
            {
                "User-Agent": ua,
                "Accept": "application/json",
            }
        )
        self._last_crossref_at = 0.0

    def resolve_all(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(citations)
        resolved: List[Dict[str, Any]] = []
        unresolved: List[Dict[str, Any]] = []

        if self.progress_cb:
            self.progress_cb(5, f"Resolving {total} citations", "research_resolve_start")

        for idx, citation in enumerate(citations, start=1):
            self._check_cancel()
            row_id = citation.get("row_id", idx)
            self._emit_progress(idx - 1, total, f"Resolving row {row_id}/{total}", "research_resolve_lookup")
            result = self.resolve_one(citation)
            if result.get("status") == "resolved":
                resolved.append(result["payload"])
            else:
                unresolved.append(result["payload"])
            self._emit_progress(idx, total, f"Resolved row {row_id}/{total}", "research_resolve_lookup")

        stats = {
            "total": total,
            "resolved_high": sum(1 for item in resolved if item.get("confidence") == "high"),
            "resolved_low": sum(1 for item in resolved if item.get("confidence") == "low"),
            "unresolved": len(unresolved),
            "open_access": sum(1 for item in resolved if (item.get("open_access") or {}).get("is_oa") is True),
            "closed_access": sum(1 for item in resolved if (item.get("open_access") or {}).get("is_oa") is False),
        }

        if self.progress_cb:
            self.progress_cb(100, "Research resolve complete", "done")

        return {
            "status": "completed",
            "resolved": resolved,
            "unresolved": unresolved,
            "stats": stats,
        }

    def resolve_one(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        doi = _normalize_doi(citation.get("doi") or "")
        accession = str(citation.get("accession") or "").strip()

        if doi:
            resolved = self._resolve_by_doi(doi, citation)
            if resolved:
                return {"status": "resolved", "payload": resolved}

        if accession:
            resolved = self._resolve_by_accession(accession, citation)
            if resolved:
                return {"status": "resolved", "payload": resolved}

        searched = self._resolve_by_search(citation)
        if searched:
            return {"status": "resolved", "payload": searched}

        unresolved = {
            "row_id": citation.get("row_id"),
            "input_title": str(citation.get("title") or "").strip(),
            "reason": "No CrossRef match found",
            "best_candidates": self._best_candidates_for_unresolved(citation),
        }
        return {"status": "unresolved", "payload": unresolved}

    def _check_cancel(self) -> None:
        if self.is_cancelled_cb and self.is_cancelled_cb():
            raise RuntimeError("Cancelled by operator")

    def _emit_progress(self, done: int, total: int, message: str, stage: str) -> None:
        if not self.progress_cb:
            return
        frac = 0.0 if total <= 0 else min(1.0, max(0.0, done / float(total)))
        pct = 10 + int(frac * 85)
        self.progress_cb(pct, message, stage)

    def _respect_crossref_delay(self) -> None:
        elapsed = time.monotonic() - self._last_crossref_at
        delay = max(0.0, 0.5 - elapsed)
        if delay > 0:
            self.sleep_fn(delay)

    def _request_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        retryable_statuses = {429, 500, 502, 503, 504}
        for attempt, delay in enumerate((0.0, 2.0, 4.0, 8.0), start=1):
            self._check_cancel()
            if source == "crossref":
                self._respect_crossref_delay()
            try:
                response = self.session.get(url, params=params, timeout=30)
                if source == "crossref":
                    self._last_crossref_at = time.monotonic()
            except requests.RequestException as exc:
                if attempt >= 4:
                    logger.warning("%s request failed for %s: %s", source, url, exc)
                    return None
                self.sleep_fn(delay)
                continue

            if response.status_code == 404:
                return None
            if response.status_code in retryable_statuses:
                if attempt >= 4:
                    logger.warning("%s request failed for %s with status %s", source, url, response.status_code)
                    return None
                self.sleep_fn(delay)
                continue
            if response.status_code >= 400:
                logger.warning("%s request failed for %s with status %s", source, url, response.status_code)
                return None
            try:
                return response.json()
            except Exception as exc:
                logger.warning("%s JSON parse failed for %s: %s", source, url, exc)
                return None
        return None

    def _crossref_works(self, *, doi: str = "", params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if doi:
            encoded = quote(doi, safe="")
            payload = self._request_json(f"https://api.crossref.org/works/{encoded}", source="crossref")
            if isinstance(payload, dict):
                return payload.get("message") if isinstance(payload.get("message"), dict) else None
            return None

        payload = self._request_json("https://api.crossref.org/works", params=params, source="crossref")
        if not isinstance(payload, dict):
            return None
        message = payload.get("message")
        if not isinstance(message, dict):
            return None
        items = message.get("items")
        if isinstance(items, list):
            return {"items": items}
        return None

    def _resolve_by_doi(self, doi: str, citation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        message = self._crossref_works(doi=doi)
        if not message:
            return None
        return self._build_resolved_payload(
            citation=citation,
            message=message,
            source_api="crossref_doi",
            resolution_method="doi_direct",
            confidence="high",
            similarity=1.0,
        )

    def _resolve_by_accession(self, accession: str, citation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _PMID_RE.fullmatch(accession):
            return None

        crossref = self._crossref_works(params={"filter": f"alternative-id:{accession}", "rows": 5})
        items = crossref.get("items") if crossref else []
        if items:
            candidate = self._score_candidates(
                citation,
                items,
                source_api="crossref_alt_id",
                resolution_method="accession_lookup",
            )
            if candidate and candidate.get("confidence") == "high":
                return candidate

        ncbi = self._request_json(
            "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/",
            params={"format": "csl", "id": accession},
            source="ncbi",
        )
        doi = _normalize_doi(str((ncbi or {}).get("DOI") or ""))
        if not doi:
            return None
        message = self._crossref_works(doi=doi)
        if not message:
            return None
        return self._build_resolved_payload(
            citation=citation,
            message=message,
            source_api="ncbi_pubmed",
            resolution_method="accession_lookup",
            confidence="high",
            similarity=1.0,
        )

    def _resolve_by_search(self, citation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        first_author = _extract_first_author(citation.get("authors") or "")
        params = {
            "query.title": citation.get("title") or "",
            "rows": 5,
        }
        if first_author:
            params["query.author"] = first_author

        crossref = self._crossref_works(params=params)
        items = crossref.get("items") if crossref else []
        candidate = self._score_candidates(
            citation,
            items,
            source_api="crossref_title_author",
            resolution_method="title_author_search",
        )
        if candidate:
            return candidate

        broader_params: Dict[str, Any] = {"query": citation.get("title") or "", "rows": 5}
        year = _extract_year(citation.get("year"))
        if year:
            broader_params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
        broader = self._crossref_works(params=broader_params)
        broader_items = broader.get("items") if broader else []
        return self._score_candidates(
            citation,
            broader_items,
            source_api="crossref_broad",
            resolution_method="broader_search",
        )

    def _best_candidates_for_unresolved(self, citation: Dict[str, Any]) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"query": citation.get("title") or "", "rows": 3}
        fallback = self._crossref_works(params=params)
        items = fallback.get("items") if fallback else []
        return self._candidate_summaries(citation, items)[:3]

    def _candidate_summaries(self, citation: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summaries = []
        target_title = str(citation.get("title") or "").strip()
        target_year = _extract_year(citation.get("year"))
        for item in items or []:
            title = self._message_title(item)
            similarity = _title_similarity(target_title, title)
            year_match = target_year and _extract_crossref_year(item) == target_year
            if year_match:
                similarity = min(1.0, similarity + 0.02)
            summaries.append(
                {
                    "title": title,
                    "doi": _normalize_doi(str(item.get("DOI") or "")),
                    "similarity": round(similarity, 3),
                }
            )
        return sorted(summaries, key=lambda item: item.get("similarity", 0.0), reverse=True)

    def _score_candidates(
        self,
        citation: Dict[str, Any],
        items: List[Dict[str, Any]],
        *,
        source_api: str,
        resolution_method: str,
    ) -> Optional[Dict[str, Any]]:
        if not items:
            return None

        target_year = _extract_year(citation.get("year"))
        best_item: Optional[Dict[str, Any]] = None
        best_similarity = 0.0

        for item in items:
            title = self._message_title(item)
            similarity = _title_similarity(citation.get("title") or "", title)
            item_year = _extract_crossref_year(item)
            if target_year and item_year and item_year == target_year:
                similarity = min(1.0, similarity + 0.03)
            if similarity > best_similarity:
                best_similarity = similarity
                best_item = item

        if not best_item or best_similarity < 0.7:
            return None

        confidence = "low"
        year_match = target_year and _extract_crossref_year(best_item) == target_year
        if best_similarity > 0.9 and year_match:
            confidence = "high"

        return self._build_resolved_payload(
            citation=citation,
            message=best_item,
            source_api=source_api,
            resolution_method=resolution_method,
            confidence=confidence,
            similarity=best_similarity,
        )

    def _message_title(self, message: Dict[str, Any]) -> str:
        title = message.get("title")
        if isinstance(title, list):
            return str(title[0] or "").strip() if title else ""
        return str(title or "").strip()

    def _open_access_info(self, doi: str) -> Dict[str, Any]:
        if not self.options.get("check_open_access", True):
            return {"is_oa": None, "oa_status": "unknown", "pdf_url": "", "oa_source": ""}
        email = str(self.options.get("unpaywall_email") or "").strip()
        if not doi or not email:
            return {"is_oa": None, "oa_status": "unknown", "pdf_url": "", "oa_source": ""}
        payload = self._request_json(
            f"https://api.unpaywall.org/v2/{quote(doi, safe='')}",
            params={"email": email},
            source="unpaywall",
        )
        if not isinstance(payload, dict):
            return {"is_oa": None, "oa_status": "unknown", "pdf_url": "", "oa_source": ""}
        location = payload.get("best_oa_location") or {}
        pdf_url = ""
        if isinstance(location, dict):
            pdf_url = str(location.get("url_for_pdf") or location.get("url") or "").strip()
        return {
            "is_oa": payload.get("is_oa") if isinstance(payload.get("is_oa"), bool) else None,
            "oa_status": str(payload.get("oa_status") or "unknown").strip().lower() or "unknown",
            "pdf_url": pdf_url,
            "oa_source": "unpaywall",
        }

    def _journal_info(self, citation: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        journal_name = ""
        container = message.get("container-title")
        if isinstance(container, list) and container:
            journal_name = str(container[0] or "").strip()
        if not journal_name:
            journal_name = str(citation.get("journal") or "").strip()
        issn = _extract_issn(message)

        ranking = {
            "journal_title": journal_name,
            "journal_issn": issn,
            "journal_quartile": "",
            "journal_sjr": 0.0,
            "journal_rank_global": 0,
        }
        if self.options.get("enrich_sjr", True):
            ranking = classify_journal_authority(journal_name, f"{issn}\n{citation.get('journal') or ''}")
            if not ranking.get("journal_title") and journal_name:
                ranking["journal_title"] = journal_name
            if not ranking.get("journal_issn") and issn:
                ranking["journal_issn"] = issn

        return {
            "name": str(ranking.get("journal_title") or journal_name or "").strip(),
            "issn": str(ranking.get("journal_issn") or issn or "").strip(),
            "sjr_quartile": str(ranking.get("journal_quartile") or "").strip(),
            "sjr_score": float(ranking.get("journal_sjr") or 0.0),
            "sjr_rank": int(ranking.get("journal_rank_global") or 0),
        }

    def _credibility_tier(self, message: Dict[str, Any], resolved_url: str, journal_name: str) -> str:
        combined = " ".join(
            [
                resolved_url,
                str(message.get("publisher") or ""),
                str(message.get("subtype") or ""),
                str(message.get("type") or ""),
                journal_name,
            ]
        ).lower()
        if any(marker in combined for marker in _PREPRINT_MARKERS):
            return "pre-print"
        if str(message.get("type") or "").strip().lower() in {"journal-article", "proceedings-article"}:
            return "peer-reviewed"
        tier_value, tier_key, _, _ = classify_credibility_tier_with_reason(
            text=f"{resolved_url}\n{journal_name}\n{message.get('publisher') or ''}",
            source_type="Other",
            availability_status="available",
        )
        return tier_key if tier_value >= 0 else "unclassified"

    def _build_resolved_payload(
        self,
        *,
        citation: Dict[str, Any],
        message: Dict[str, Any],
        source_api: str,
        resolution_method: str,
        confidence: str,
        similarity: float,
    ) -> Dict[str, Any]:
        doi = _normalize_doi(str(message.get("DOI") or citation.get("doi") or ""))
        resolved_url = _pick_primary_url(message, doi)
        open_access = self._open_access_info(doi)
        journal = self._journal_info(citation, message)
        retraction = _retraction_info(message)

        return {
            "row_id": citation.get("row_id"),
            "input_title": str(citation.get("title") or "").strip(),
            "resolved_doi": doi,
            "resolved_url": resolved_url,
            "source_api": source_api,
            "confidence": confidence,
            "open_access": open_access,
            "journal": journal,
            "credibility_tier": self._credibility_tier(message, resolved_url, journal.get("name") or ""),
            "resolution_method": resolution_method,
            "publisher": str(message.get("publisher") or "").strip(),
            "type": str(message.get("type") or "").strip(),
            "matched_title": self._message_title(message),
            "title_similarity": round(similarity, 3),
            "retraction": retraction,
        }


def run_research_resolve(
    *,
    payload: Dict[str, Any],
    run_dir: Path,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    resolver = ResearchResolver(
        options=dict(payload.get("options") or {}),
        progress_cb=progress_cb,
        is_cancelled_cb=is_cancelled_cb,
    )
    output = resolver.resolve_all(list(payload.get("citations") or []))
    out_path = Path(run_dir) / "research_resolve_result.json"
    out_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    return output
