from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cortex_engine.research_resolve import _extract_year, _normalize_doi


_SYSTEMATIC_REVIEW_SIGNALS: Tuple[Tuple[str, int], ...] = (
    ("systematic review", 4),
    ("meta-analysis", 4),
    ("meta analysis", 4),
    ("prisma", 3),
    ("study selection", 2),
    ("search strategy", 2),
    ("included studies", 3),
    ("characteristics of included studies", 4),
    ("risk of bias", 2),
    ("eligibility criteria", 2),
    ("electronic databases", 2),
    ("record screening", 2),
)

_TABLE_HEADER_HINTS = (
    "study",
    "studies",
    "author",
    "authors",
    "reference",
    "citation",
    "trial",
    "intervention",
    "outcome",
    "design",
)

_CITATION_HEADER_PRIORITY: Tuple[Tuple[str, ...], ...] = (
    ("study", "study id", "included study", "included studies"),
    ("citation", "reference", "references"),
    ("author", "authors", "first author"),
    ("title", "paper title", "article title"),
)

_DEFAULT_DESIGN_QUERY = "RCT clinical trial randomised randomized"
_DEFAULT_OUTCOME_QUERY = "health-related quality of life HRQoL QoL patient-reported outcome"

_DESIGN_EXPANSIONS = {
    "rct": {"rct", "randomized", "randomised", "controlled trial", "randomized controlled trial", "clinical trial"},
    "randomized": {"randomized", "randomised", "random allocation", "randomly assigned"},
    "randomised": {"randomized", "randomised", "random allocation", "randomly assigned"},
    "clinical": {"clinical trial", "trial", "phase ii", "phase iii"},
    "trial": {"trial", "clinical trial", "controlled trial"},
}

_OUTCOME_EXPANSIONS = {
    "quality": {"quality of life", "health-related quality of life", "hrqol", "qol", "well-being"},
    "life": {"quality of life", "health-related quality of life", "hrqol", "qol"},
    "hrqol": {"health-related quality of life", "hrqol", "quality of life", "qol"},
    "qol": {"health-related quality of life", "quality of life", "hrqol", "qol"},
    "patient": {"patient-reported outcome", "patient reported outcome", "pro", "prom", "questionnaire"},
    "outcome": {"outcome", "patient-reported outcome", "patient reported outcome", "pro", "prom"},
}

_REFERENCE_ENTRY_RE = re.compile(r"(?m)^\s*(?:\[\d+\]|\d+[.)]|[A-Z][A-Za-z' -]+(?:,|\s+et al\.)).+?(?:19|20)\d{2}.*$")
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_NUMBERED_REFERENCE_START_RE = re.compile(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s*(.+?)\s*$")
_REFERENCE_POINTER_RE = re.compile(
    r"(?:\[(\d{1,3})\]|\(\s*(\d{1,3})\s*\)|\bref(?:erence)?(?:s)?\s*(\d{1,3})(?!\d)|\brefs?\.\s*(\d{1,3})(?!\d))",
    re.IGNORECASE,
)
_AUTHOR_YEAR_RE = re.compile(r"\b([A-Z][A-Za-z' -]+)(?:\s+et al\.)?[\s,;:/-]*(?:\(|\[)?((?:19|20)\d{2})(?:\)|\])?")
_NON_STUDY_ROW_LABELS = {
    "region",
    "study design",
    "study type",
    "study design study type",
    "hrqol outcomes",
    "qol outcomes",
    "source of utility values",
    "utility values in remission",
    "study author year",
    "time point",
    "follow up time",
    "follow up",
    "followup",
    "assessment measures",
    "scale",
    "treatment",
    "population",
}


def _normalize_text(text: str) -> str:
    base = unicodedata.normalize("NFKD", str(text or ""))
    ascii_text = base.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    return re.sub(r"\s+", " ", cleaned)


def _contains_phrase(haystack: str, phrase: str) -> bool:
    return _normalize_text(phrase) in _normalize_text(haystack)


def _tokenize_query(query: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"[,\n;/]+|\s{2,}", str(query or "").strip()) if part.strip()]
    if not parts and str(query or "").strip():
        parts = [str(query or "").strip()]
    return parts


def _expand_keywords(query: str, expansions: Dict[str, set[str]]) -> List[str]:
    keywords: List[str] = []
    seen: set[str] = set()
    for raw in _tokenize_query(query):
        normalized = _normalize_text(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            keywords.append(raw.strip())
        for token in normalized.split():
            for expanded in expansions.get(token, set()):
                norm_expanded = _normalize_text(expanded)
                if norm_expanded and norm_expanded not in seen:
                    seen.add(norm_expanded)
                    keywords.append(expanded)
    return keywords


def _find_matches(text: str, keywords: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for keyword in keywords:
        if _contains_phrase(text, keyword):
            matches.append(keyword)
    return sorted(dict.fromkeys(matches))


def _parse_markdown_tables(text: str) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    lines = str(text or "").splitlines()
    idx = 0
    while idx < len(lines):
        if "|" not in lines[idx]:
            idx += 1
            continue
        block: List[str] = []
        while idx < len(lines) and "|" in lines[idx]:
            block.append(lines[idx])
            idx += 1
        if len(block) < 2:
            continue
        rows = [_split_table_row(line) for line in block if _split_table_row(line)]
        if len(rows) < 2:
            continue
        header = rows[0]
        body_rows = rows[1:]
        if body_rows and _is_separator_row(body_rows[0]):
            body_rows = body_rows[1:]
        if not body_rows:
            continue
        tables.append({"header": header, "rows": body_rows, "raw_lines": block})
    return tables


def _split_table_row(line: str) -> List[str]:
    raw = str(line or "").strip()
    if "|" not in raw:
        return []
    parts = [part.strip() for part in raw.strip("|").split("|")]
    return parts if any(parts) else []


def _is_separator_row(row: Sequence[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{2,}:?", str(cell or "").strip()) for cell in row)


def _table_header_score(header: Sequence[str]) -> int:
    score = 0
    joined = " ".join(str(item or "") for item in header)
    for hint in _TABLE_HEADER_HINTS:
        if _contains_phrase(joined, hint):
            score += 1
    return score


def _pick_citation_cell(header: Sequence[str], row: Sequence[str]) -> str:
    normalized_headers = [_normalize_text(str(item or "")) for item in header]
    for aliases in _CITATION_HEADER_PRIORITY:
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            for idx, header_name in enumerate(normalized_headers):
                if header_name == alias_norm or alias_norm in header_name:
                    if idx < len(row) and str(row[idx] or "").strip():
                        return str(row[idx]).strip()
    if row:
        return str(row[0] or "").strip()
    return ""


def _extract_authors_and_year(blob: str) -> Tuple[str, str]:
    text = str(blob or "").strip()
    year = _extract_year(text)
    author_match = re.search(r"([A-Z][A-Za-z' -]+(?:\s+et al\.)?)", text)
    authors = author_match.group(1).strip() if author_match else ""
    return authors, year


def _extract_journal(blob: str) -> str:
    match = re.search(r"\.\s*([^.;]+?\b(?:journal|review|reports?|medicine|oncology|health)\b[^.;]*)", str(blob or ""), flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_title(blob: str, citation_cell: str, header: Sequence[str], row: Sequence[str]) -> str:
    normalized_headers = [_normalize_text(str(item or "")) for item in header]
    for idx, header_name in enumerate(normalized_headers):
        if idx >= len(row):
            continue
        value = str(row[idx] or "").strip()
        if not value:
            continue
        if "title" in header_name or "paper" in header_name or "article" in header_name:
            return value
    cleaned = str(citation_cell or "").strip()
    return cleaned


def _clean_reference_prefix(entry: str) -> str:
    text = str(entry or "").strip()
    text = re.sub(r"^\s*\[(\d{1,3})\]\s*", "", text)
    text = re.sub(r"^\s*(\d{1,3})[.)]\s*", "", text)
    return text.strip()


def _reference_title_from_entry(entry: str) -> str:
    cleaned = _clean_reference_prefix(entry)
    chunks = [chunk.strip(" .;:") for chunk in re.split(r"\.\s+", cleaned) if chunk.strip(" .;:")]
    if len(chunks) >= 2:
        for chunk in chunks[1:4]:
            if len(chunk.split()) >= 4:
                return chunk
        return chunks[1]
    return cleaned[:240]


def _reference_number_pointers(text: str) -> List[str]:
    pointers: List[str] = []
    for match in _REFERENCE_POINTER_RE.finditer(str(text or "")):
        number = next((group for group in match.groups() if group), "")
        if number:
            pointers.append(str(int(number)))
    return list(dict.fromkeys(pointers))


def _reference_match_key(text: str) -> str:
    return _normalize_text(text).replace(" et al", "")


def _parse_reference_entries(text: str) -> List[Dict[str, Any]]:
    section = _references_section(text)
    if not section:
        return []

    lines = [line.rstrip() for line in section.splitlines()]
    numbered_entries: List[Dict[str, Any]] = []
    current_number = ""
    current_lines: List[str] = []

    def _flush_current() -> None:
        if not current_lines:
            return
        entry_text = " ".join(part.strip() for part in current_lines if part.strip()).strip()
        if not entry_text:
            return
        cleaned = _clean_reference_prefix(entry_text)
        authors, year = _extract_authors_and_year(cleaned)
        numbered_entries.append(
            {
                "reference_number": current_number,
                "entry_text": entry_text,
                "cleaned_entry": cleaned,
                "authors": authors,
                "year": year,
                "doi": _normalize_doi(_DOI_RE.search(cleaned).group(0)) if _DOI_RE.search(cleaned) else "",
                "journal": _extract_journal(cleaned),
                "title": _reference_title_from_entry(cleaned),
                "match_key": _reference_match_key(cleaned),
            }
        )

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        start_match = _NUMBERED_REFERENCE_START_RE.match(line)
        if start_match:
            if current_lines:
                _flush_current()
                current_lines = []
            current_number = str(int(start_match.group(1) or start_match.group(2) or "0"))
            current_lines.append(start_match.group(3) or "")
            continue
        if current_lines:
            current_lines.append(line)

    if current_lines:
        _flush_current()

    if numbered_entries:
        return numbered_entries

    entries = [match.group(0).strip() for match in _REFERENCE_ENTRY_RE.finditer(section)]
    if not entries:
        chunks = re.split(r"\n\s*\n", section)
        entries = [chunk.strip() for chunk in chunks if _extract_year(chunk)]

    fallback_entries: List[Dict[str, Any]] = []
    for entry in entries:
        cleaned = _clean_reference_prefix(entry)
        authors, year = _extract_authors_and_year(cleaned)
        fallback_entries.append(
            {
                "reference_number": "",
                "entry_text": entry,
                "cleaned_entry": cleaned,
                "authors": authors,
                "year": year,
                "doi": _normalize_doi(_DOI_RE.search(cleaned).group(0)) if _DOI_RE.search(cleaned) else "",
                "journal": _extract_journal(cleaned),
                "title": _reference_title_from_entry(cleaned),
                "match_key": _reference_match_key(cleaned),
            }
        )
    return fallback_entries


def _score_review_document(text: str, title: str = "") -> Dict[str, Any]:
    haystack = "\n".join([str(title or ""), str(text or "")[:50000]])
    matched: List[str] = []
    score = 0
    for phrase, weight in _SYSTEMATIC_REVIEW_SIGNALS:
        if _contains_phrase(haystack, phrase):
            matched.append(phrase)
            score += weight
    confidence = "low"
    if score >= 10:
        confidence = "high"
    elif score >= 5:
        confidence = "medium"
    return {
        "is_systematic_review": score >= 5,
        "confidence": confidence,
        "score": score,
        "matched_signals": matched,
    }


def _references_section(text: str) -> str:
    match = re.search(r"(?is)\n(?:references|bibliography)\s*\n(.+)$", "\n" + str(text or ""))
    return match.group(1).strip() if match else ""


def _extract_reference_entries(text: str) -> List[str]:
    return [str(item.get("entry_text") or "").strip() for item in _parse_reference_entries(text) if str(item.get("entry_text") or "").strip()]


def _reference_match_from_row(citation_cell: str, joined: str, references: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    numbered = {str(item.get("reference_number") or "").strip(): item for item in references if str(item.get("reference_number") or "").strip()}
    for marker in _reference_number_pointers(" ".join([str(citation_cell or ""), str(joined or "")])):
        matched = numbered.get(marker)
        if matched:
            enriched = dict(matched)
            enriched["match_method"] = "reference_number"
            return enriched

    authors, year = _extract_authors_and_year(citation_cell or joined)
    author_key = _normalize_text(authors).split(" ")[0] if authors else ""
    title_hint = _normalize_text(citation_cell or joined)
    best_match: Optional[Dict[str, Any]] = None
    best_score = 0
    for ref in references:
        ref_key = str(ref.get("match_key") or "")
        score = 0
        if year and str(ref.get("year") or "") == year:
            score += 2
        if author_key and author_key in ref_key:
            score += 3
        if title_hint:
            title_tokens = [token for token in title_hint.split() if len(token) > 3][:5]
            score += sum(1 for token in title_tokens if token in ref_key)
        if score > best_score:
            best_score = score
            best_match = ref
    if best_match and best_score >= 5:
        enriched = dict(best_match)
        enriched["match_method"] = "author_year_fuzzy"
        return enriched
    return None


def _study_row_signal_score(citation_cell: str, joined: str, linked_reference: Optional[Dict[str, Any]]) -> int:
    combined = " ".join([str(citation_cell or "").strip(), str(joined or "").strip()]).strip()
    normalized = _normalize_text(citation_cell or joined)
    score = 0
    if _DOI_RE.search(combined):
        score += 4
    if _reference_number_pointers(combined):
        score += 2
    if _extract_year(combined):
        score += 2
    if _AUTHOR_YEAR_RE.search(combined):
        score += 3
    if re.search(r"\bet al\b", combined, flags=re.IGNORECASE):
        score += 2
    if linked_reference:
        score += 3
    if normalized in _NON_STUDY_ROW_LABELS:
        score -= 6
    if len(normalized.split()) <= 4 and not _extract_year(combined) and not _reference_number_pointers(combined):
        score -= 2
    return score


def _validate_linked_reference(citation_cell: str, joined: str, linked_reference: Optional[Dict[str, Any]]) -> Tuple[str, bool, str]:
    if not linked_reference:
        return "", False, ""

    ref_method = str(linked_reference.get("match_method") or "").strip()
    if not ref_method:
        return "", False, ""

    row_authors, row_year = _extract_authors_and_year(citation_cell or joined)
    ref_authors = str(linked_reference.get("authors") or "").strip()
    ref_year = str(linked_reference.get("year") or "").strip()
    row_author_key = _normalize_text(row_authors).split(" ")[0] if row_authors else ""
    ref_author_key = _normalize_text(ref_authors).split(" ")[0] if ref_authors else ""

    warnings: List[str] = []
    if row_year and ref_year and row_year != ref_year:
        warnings.append(f"table year {row_year} does not match reference year {ref_year}")
    if row_author_key and ref_author_key and row_author_key != ref_author_key:
        warnings.append(f"table author {row_authors} does not match reference author {ref_authors}")

    if warnings:
        return "mismatch", True, "; ".join(warnings)
    if ref_method == "reference_number":
        return "matched", False, ""
    return "fuzzy_match", False, ""


@dataclass
class ReviewMiningOptions:
    design_query: str = _DEFAULT_DESIGN_QUERY
    outcome_query: str = _DEFAULT_OUTCOME_QUERY
    require_all_criteria: bool = True
    include_reference_list_scan: bool = True


def extract_review_table_blocks(text: str, *, max_tables: int = 12) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for idx, table in enumerate(_parse_markdown_tables(text), start=1):
        if idx > max_tables:
            break
        raw_lines = [str(line or "") for line in (table.get("raw_lines") or [])]
        blocks.append(
            {
                "table_index": idx,
                "header": [str(item or "") for item in (table.get("header") or [])],
                "row_count": len(table.get("rows") or []),
                "markdown": "\n".join(raw_lines).strip(),
            }
        )
    return blocks


def extract_review_reference_section(text: str, *, max_chars: int = 24000) -> str:
    section = _references_section(text)
    return str(section or "").strip()[:max_chars]


def assess_review_documents(documents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    assessed_documents: List[Dict[str, Any]] = []
    review_count = 0
    for idx, item in enumerate(documents or [], start=1):
        source_name = str(item.get("source_name") or f"document_{idx}").strip() or f"document_{idx}"
        review_title = str(item.get("review_title") or source_name).strip() or source_name
        text = str(item.get("text") or "")
        review_assessment = _score_review_document(text=text, title=review_title)
        if review_assessment.get("is_systematic_review"):
            review_count += 1
        assessed_documents.append(
            {
                "doc_id": idx,
                "source_name": source_name,
                "review_title": review_title,
                "file_path": str(item.get("file_path") or ""),
                "text": text,
                "table_blocks": list(item.get("table_blocks") or []),
                "table_snapshots": list(item.get("table_snapshots") or []),
                "review_assessment": review_assessment,
                "systematic_review_likely": bool(review_assessment.get("is_systematic_review")),
                "confirm_review": bool(review_assessment.get("is_systematic_review")),
            }
        )

    return {
        "documents": assessed_documents,
        "stats": {
            "documents_total": len(assessed_documents),
            "systematic_review_likely": review_count,
            "not_systematic_review_likely": max(0, len(assessed_documents) - review_count),
        },
    }


def mine_review_documents(
    documents: Sequence[Dict[str, Any]],
    *,
    options: Optional[ReviewMiningOptions] = None,
    confirmed_doc_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    opts = options or ReviewMiningOptions()
    confirmed_ids = {int(item) for item in (confirmed_doc_ids or [])}

    source_documents = list(documents or [])
    if not source_documents:
        return {
            "documents": [],
            "per_review": [],
            "candidates": [],
            "stats": {
                "documents_total": 0,
                "documents_confirmed": 0,
                "reviews_mined": 0,
                "candidates_total": 0,
                "candidates_matching": 0,
                "table_candidates": 0,
                "reference_candidates": 0,
            },
        }

    assessed = assess_review_documents(source_documents)
    assessed_documents = list(assessed.get("documents") or [])
    if confirmed_ids:
        for item in assessed_documents:
            item["confirm_review"] = int(item.get("doc_id") or 0) in confirmed_ids

    all_candidates: List[Dict[str, Any]] = []
    per_review: List[Dict[str, Any]] = []
    next_row_id = 1

    for document in assessed_documents:
        if not bool(document.get("confirm_review")):
            continue
        text = str(document.get("text") or "")
        review_title = str(document.get("review_title") or document.get("source_name") or "").strip()
        source_name = str(document.get("source_name") or review_title or "").strip()
        mined = mine_review_studies_from_text(
            text,
            source_name=source_name,
            review_title=review_title,
            options=opts,
        )
        candidates = []
        for candidate in mined.get("candidates") or []:
            normalized = dict(candidate)
            normalized["row_id"] = next_row_id
            normalized["source_review"] = source_name
            normalized["source_review_title"] = review_title
            extras = dict(normalized.get("extra_fields") or {})
            extras["source_review"] = source_name
            extras["source_review_title"] = review_title
            extras["source_doc_id"] = str(document.get("doc_id") or "")
            normalized["extra_fields"] = extras
            candidates.append(normalized)
            all_candidates.append(normalized)
            next_row_id += 1

        review_stats = dict(mined.get("stats") or {})
        per_review.append(
            {
                "doc_id": int(document.get("doc_id") or 0),
                "source_name": source_name,
                "review_title": review_title,
                "review_assessment": dict(document.get("review_assessment") or {}),
                "confirmed": True,
                "stats": review_stats,
                "candidate_count": len(candidates),
                "matching_count": sum(1 for item in candidates if item.get("meets_criteria")),
                "preferred_candidates": candidates,
            }
        )

    return {
        "documents": assessed_documents,
        "per_review": per_review,
        "candidates": all_candidates,
        "stats": {
            "documents_total": len(assessed_documents),
            "documents_confirmed": sum(1 for item in assessed_documents if item.get("confirm_review")),
            "reviews_mined": len(per_review),
            "candidates_total": len(all_candidates),
            "candidates_matching": sum(1 for item in all_candidates if item.get("meets_criteria")),
            "table_candidates": sum(1 for item in all_candidates if item.get("source_section") == "table"),
            "reference_candidates": sum(1 for item in all_candidates if item.get("source_section") == "references"),
            "table_reference_links": sum(
                1 for item in all_candidates if item.get("source_section") == "table" and str(item.get("reference_match_method") or "").strip()
            ),
            "needs_review": sum(1 for item in all_candidates if item.get("needs_review")),
            "reference_mismatches": sum(1 for item in all_candidates if str(item.get("reference_validation") or "") == "mismatch"),
        },
    }


def mine_review_studies_from_text(
    text: str,
    *,
    source_name: str,
    review_title: str = "",
    options: Optional[ReviewMiningOptions] = None,
) -> Dict[str, Any]:
    opts = options or ReviewMiningOptions()
    design_keywords = _expand_keywords(opts.design_query, _DESIGN_EXPANSIONS)
    outcome_keywords = _expand_keywords(opts.outcome_query, _OUTCOME_EXPANSIONS)
    review_assessment = _score_review_document(text=text, title=review_title)
    reference_entries = _parse_reference_entries(text)

    candidates: List[Dict[str, Any]] = []
    row_id = 1

    for table in _parse_markdown_tables(text):
        header = table.get("header") or []
        rows = table.get("rows") or []
        table_header_score = _table_header_score(header)
        for row in rows:
            joined = " | ".join(str(item or "").strip() for item in row if str(item or "").strip())
            if not joined:
                continue
            design_matches = _find_matches(joined, design_keywords)
            outcome_matches = _find_matches(joined, outcome_keywords)
            meets = bool(design_matches) and bool(outcome_matches) if opts.require_all_criteria else bool(design_matches or outcome_matches)
            citation_cell = _pick_citation_cell(header, row)
            if not citation_cell:
                continue
            linked_reference = _reference_match_from_row(citation_cell, joined, reference_entries)
            row_signal_score = _study_row_signal_score(citation_cell, joined, linked_reference)
            if row_signal_score < 2:
                continue
            title = _extract_title(joined, citation_cell, header, row)
            authors, year = _extract_authors_and_year(citation_cell or joined)
            journal = _extract_journal(joined)
            doi_match = _DOI_RE.search(joined)
            reference_number = ""
            reference_match_method = ""
            reference_validation = ""
            needs_review = False
            review_warning = ""
            raw_citation = citation_cell
            if linked_reference:
                title = str(linked_reference.get("title") or title or "").strip() or title
                authors = str(linked_reference.get("authors") or authors or "").strip() or authors
                year = str(linked_reference.get("year") or year or "").strip() or year
                journal = str(linked_reference.get("journal") or journal or "").strip() or journal
                if str(linked_reference.get("doi") or "").strip():
                    doi_match = None
                raw_citation = str(linked_reference.get("entry_text") or citation_cell or "").strip()
                reference_number = str(linked_reference.get("reference_number") or "").strip()
                reference_match_method = str(linked_reference.get("match_method") or "").strip()
                reference_validation, needs_review, review_warning = _validate_linked_reference(citation_cell, joined, linked_reference)
            relevance_score = (len(design_matches) * 2) + (len(outcome_matches) * 2) + min(table_header_score, 3)
            if linked_reference:
                relevance_score += 2
            candidates.append(
                {
                    "row_id": row_id,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "doi": str(linked_reference.get("doi") or "").strip() if linked_reference and str(linked_reference.get("doi") or "").strip() else (_normalize_doi(doi_match.group(0)) if doi_match else ""),
                    "journal": journal,
                    "abstract": "",
                    "volume": "",
                    "issue": "",
                    "pages": "",
                    "accession": "",
                    "aim": "",
                    "notes": f"{source_name} | table row",
                    "source_section": "table",
                    "source_name": source_name,
                    "meets_criteria": meets,
                    "relevance_score": relevance_score,
                    "design_matches": design_matches,
                    "outcome_matches": outcome_matches,
                    "raw_citation": raw_citation,
                    "raw_excerpt": joined[:1000],
                    "reference_number": reference_number,
                    "reference_match_method": reference_match_method,
                    "reference_validation": reference_validation,
                    "needs_review": needs_review,
                    "review_warning": review_warning,
                    "extra_fields": {
                        "source_review": source_name,
                        "source_section": "table",
                        "design_matches": ", ".join(design_matches),
                        "outcome_matches": ", ".join(outcome_matches),
                        "reference_number": reference_number,
                        "reference_match_method": reference_match_method,
                        "reference_validation": reference_validation,
                        "needs_review": "yes" if needs_review else "",
                        "review_warning": review_warning,
                    },
                }
            )
            row_id += 1

    if opts.include_reference_list_scan:
        for ref_entry in reference_entries:
            entry = str(ref_entry.get("entry_text") or "").strip()
            design_matches = _find_matches(entry, design_keywords)
            outcome_matches = _find_matches(entry, outcome_keywords)
            meets = bool(design_matches) and bool(outcome_matches) if opts.require_all_criteria else bool(design_matches or outcome_matches)
            candidates.append(
                {
                    "row_id": row_id,
                    "title": str(ref_entry.get("title") or entry[:240]).strip(),
                    "authors": str(ref_entry.get("authors") or "").strip(),
                    "year": str(ref_entry.get("year") or "").strip(),
                    "doi": str(ref_entry.get("doi") or "").strip(),
                    "journal": str(ref_entry.get("journal") or "").strip(),
                    "abstract": "",
                    "volume": "",
                    "issue": "",
                    "pages": "",
                    "accession": "",
                    "aim": "",
                    "notes": f"{source_name} | reference list",
                    "source_section": "references",
                    "source_name": source_name,
                    "meets_criteria": meets,
                    "relevance_score": (len(design_matches) * 2) + (len(outcome_matches) * 2),
                    "design_matches": design_matches,
                    "outcome_matches": outcome_matches,
                    "raw_citation": entry[:1000],
                    "raw_excerpt": entry[:1000],
                    "reference_number": str(ref_entry.get("reference_number") or "").strip(),
                    "reference_match_method": "reference_list",
                    "reference_validation": "",
                    "needs_review": False,
                    "review_warning": "",
                    "extra_fields": {
                        "source_review": source_name,
                        "source_section": "references",
                        "design_matches": ", ".join(design_matches),
                        "outcome_matches": ", ".join(outcome_matches),
                        "reference_number": str(ref_entry.get("reference_number") or "").strip(),
                        "reference_match_method": "reference_list",
                        "reference_validation": "",
                        "needs_review": "",
                        "review_warning": "",
                    },
                }
            )
            row_id += 1

    deduped = _dedupe_candidates(candidates)
    deduped.sort(
        key=lambda item: (
            0 if item.get("meets_criteria") else 1,
            -int(item.get("relevance_score") or 0),
            str(item.get("year") or ""),
            str(item.get("title") or ""),
        )
    )

    stats = {
        "tables_detected": len(_parse_markdown_tables(text)),
        "reference_entries_detected": len(_extract_reference_entries(text)) if opts.include_reference_list_scan else 0,
        "candidates_total": len(deduped),
        "candidates_matching": sum(1 for item in deduped if item.get("meets_criteria")),
        "table_candidates": sum(1 for item in deduped if item.get("source_section") == "table"),
        "reference_candidates": sum(1 for item in deduped if item.get("source_section") == "references"),
        "table_reference_links": sum(1 for item in deduped if item.get("source_section") == "table" and str(item.get("reference_match_method") or "").strip()),
        "needs_review": sum(1 for item in deduped if item.get("needs_review")),
        "reference_mismatches": sum(1 for item in deduped if str(item.get("reference_validation") or "") == "mismatch"),
    }

    return {
        "review_assessment": review_assessment,
        "criteria": {
            "design_query": opts.design_query,
            "outcome_query": opts.outcome_query,
            "require_all_criteria": opts.require_all_criteria,
        },
        "stats": stats,
        "candidates": deduped,
    }


def _dedupe_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    next_row_id = 1
    for item in candidates:
        key = " | ".join(
            [
                _normalize_text(str(item.get("doi") or "")),
                _normalize_text(str(item.get("title") or "")),
                _normalize_text(str(item.get("authors") or "")),
                _normalize_text(str(item.get("year") or "")),
            ]
        )
        if not key.strip(" |"):
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(item)
        normalized["row_id"] = next_row_id
        deduped.append(normalized)
        next_row_id += 1
    return deduped
