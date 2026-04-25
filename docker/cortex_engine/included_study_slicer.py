from __future__ import annotations

import csv
import io
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import fitz

from cortex_engine.review_study_miner import _normalize_text, _parse_reference_entries


_TABLE_START_RE = re.compile(
    r"^\s*table\s+(\d{1,3}|[ivxlcdm]{1,8})\b([^\n]*)",
    re.IGNORECASE,
)
_NUMBERED_REFERENCE_LINE_RE = re.compile(r"^\[?\d{1,3}[\].)](?:\s|$)")
_NUMBERED_ROW_ENTRY_RE = re.compile(r"^\[?\d{1,3}[.\])]\s+[A-Z]")
_CONTINUED_RE = re.compile(r"\bcont(?:inued|'d|d)?\b", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*figure\s+(\d{1,3}|[ivxlcdm]{1,8})\s*[:.]\s*\S",
    re.IGNORECASE,
)
_COCHRANE_STUDY_SECTION_RE = re.compile(
    r"^\s*characteristics\s+of\s+(included|excluded)\s+studies\s*(?:\[.*\])?\s*$",
    re.IGNORECASE,
)
_COCHRANE_OTHER_CHARACTERISTICS_RE = re.compile(
    r"^\s*characteristics\s+of\s+(?:studies\s+awaiting\s+classification|ongoing\s+studies)\s*(?:\[.*\])?\s*$",
    re.IGNORECASE,
)
_COCHRANE_STUDY_LABEL_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z' \-]{1,80}\s+(?:19|20)\d{2}[a-z]?)\s*(?:\(?\s*continued\s*\)?)?\s*$",
    re.IGNORECASE,
)
_DATA_ANALYSES_HEADING_RE = re.compile(r"^\s*data\s+and\s+analyses\s*$", re.IGNORECASE)
_TERMINATOR_SECTION_HEADINGS = frozenset(
    {
        "discussion",
        "conclusion",
        "conclusions",
        "acknowledgement",
        "acknowledgements",
        "acknowledgment",
        "acknowledgments",
        "appendix",
        "supplementary",
        "supplementary material",
        "supporting information",
        "author contributions",
        "funding",
        "conflict of interest",
        "conflicts of interest",
    }
)
_ROMAN_VALIDATOR_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
_NARRATIVE_PRONOUNS = frozenset(
    {
        "We",
        "I",
        "They",
        "You",
        "It",
        "This",
        "That",
        "Our",
        "My",
        "Their",
        "He",
        "She",
        "Its",
        "Here",
        "There",
    }
)


def _page_text(doc: fitz.Document, page_index: int) -> str:
    try:
        return str(doc.load_page(page_index).get_text("text") or "")
    except Exception:
        return ""


def _all_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _top_lines(text: str, *, max_lines: int = 16) -> List[str]:
    return _all_lines(text)[:max_lines]


def _looks_like_reference_heading(text: str) -> bool:
    for line in _top_lines(text, max_lines=12):
        normalized = _normalize_text(line)
        compact = re.sub(r"[^a-z]+", "", line.lower())
        if normalized in {"references", "bibliography"} or compact in {"references", "bibliography"}:
            return True
        if normalized.startswith("references to studies "):
            return True
    return False


def _looks_like_reference_page(text: str, *, page_number: int, total_pages: int) -> bool:
    if _looks_like_reference_heading(text):
        return True
    if page_number < max(2, total_pages // 2):
        return False
    lines = _all_lines(text)
    for idx, line in enumerate(lines):
        normalized = _normalize_text(line)
        compact = re.sub(r"[^a-z]+", "", line.lower())
        if normalized not in {"references", "bibliography"} and compact not in {"references", "bibliography"}:
            continue
        numbered_after = sum(1 for probe in lines[idx + 1 : idx + 45] if _NUMBERED_REFERENCE_LINE_RE.match(probe))
        if numbered_after >= 2:
            return True
    numbered_top = sum(1 for line in lines[:40] if _NUMBERED_REFERENCE_LINE_RE.match(line))
    return numbered_top >= 3


def _is_valid_roman(raw: str) -> bool:
    if not raw:
        return False
    return bool(_ROMAN_VALIDATOR_RE.match(raw.upper()))


def _roman_to_int(roman: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for char in reversed(roman.upper()):
        val = values.get(char, 0)
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


def _parse_table_number(raw: str) -> int:
    s = str(raw or "").strip()
    if not s:
        return 0
    if s.isdigit():
        try:
            return int(s)
        except ValueError:
            return 0
    if _is_valid_roman(s):
        return _roman_to_int(s)
    return 0


def _is_valid_caption_tail(tail: str) -> bool:
    """Return True if the text after 'Table N' looks like a caption, not body text.

    Rejects:
    - Closing-bracket artefacts: ``Table 3). The prospective...`` (PDF column-break).
    - Narrative pronoun openings: ``Table I. We identified...``.
    - Empty / lowercase openings: ``Table 3 shows results``.
    Accepts any match of the continuation regex (``Cont.``, ``continued``).
    """
    raw = tail.lstrip()
    if not raw:
        return False
    if raw[0] in ")]}>":
        return False
    if _CONTINUED_RE.search(raw[:60]):
        return True
    stripped = raw.lstrip(" \t()[]{},.:;-'\"")
    if not stripped:
        return False
    first_word = stripped.split()[0].strip("(),.:;'\"")
    if not first_word:
        return False
    if not first_word[0].isupper():
        return False
    if first_word in _NARRATIVE_PRONOUNS:
        return False
    return True


def _looks_like_table_continuation(text: str, page_headings: List[Dict[str, Any]]) -> bool:
    """Return True if the page shows tabular signals worth merging into a slice.

    Accepts pages with a detected ``Table N`` heading (continued or not) OR
    pages whose top lines contain numbered row entries like ``4. Joly et al.``
    Rejects narrative-only pages (prose continuation after a short table).
    """
    if page_headings:
        return True
    for line in _top_lines(text, max_lines=15):
        if _NUMBERED_ROW_ENTRY_RE.match(line):
            return True
    return False


def _is_table_terminator_page(text: str) -> bool:
    """Return True if the top of the page clearly ends the prior table.

    Triggers on a Figure caption (``Figure 1: ...`` / ``Figure II. ...``) or a
    narrative section heading (``Discussion``, ``Appendix``, etc.) appearing
    near the top. Used during continuation probing to stop before swallowing
    figures or post-table body text.
    """
    for line in _top_lines(text, max_lines=10):
        if _FIGURE_CAPTION_RE.match(line):
            return True
        normalized = _normalize_text(line)
        if normalized in _TERMINATOR_SECTION_HEADINGS:
            return True
    return False


def _page_table_headings(text: str) -> List[Dict[str, Any]]:
    """Scan every line of a page and return all table-heading candidates."""
    headings: List[Dict[str, Any]] = []
    for line in _all_lines(text):
        match = _TABLE_START_RE.match(line)
        if not match:
            continue
        tail = str(match.group(2) or "")
        if not _is_valid_caption_tail(tail):
            continue
        number_raw = str(match.group(1) or "").strip()
        table_number = _parse_table_number(number_raw)
        if table_number <= 0:
            continue
        title = tail.strip(" -:.()")
        continued = bool(_CONTINUED_RE.search(title[:60]))
        headings.append(
            {
                "table_number": table_number,
                "table_number_raw": number_raw,
                "title": title,
                "continued": continued,
                "raw_line": line,
            }
        )
    return headings


def _cochrane_study_section_headings(text: str) -> List[Dict[str, Any]]:
    """Return RevMan/Cochrane characteristics-table section starts.

    Cochrane reviews often do not label the "Characteristics of included
    studies" and "Characteristics of excluded studies" blocks as numbered
    tables. Treat them as synthetic table slices so the downstream extractor
    sees the same one-table-per-PDF workflow as journal reviews.
    """
    headings: List[Dict[str, Any]] = []
    for line in _all_lines(text):
        cleaned = re.sub(r"\s+", " ", line.replace("\u00a0", " ")).strip()
        match = _COCHRANE_STUDY_SECTION_RE.match(cleaned)
        if not match:
            continue
        section_kind = str(match.group(1) or "").lower()
        if section_kind == "included":
            headings.append(
                {
                    "section": "included",
                    "kind": "included_studies",
                    "table_number": 901,
                    "title": "Characteristics of included studies",
                    "raw_line": line,
                }
            )
        else:
            headings.append(
                {
                    "section": "excluded",
                    "kind": "excluded_studies",
                    "table_number": 902,
                    "title": "Characteristics of excluded studies",
                    "raw_line": line,
                }
            )
    return headings


def _is_cochrane_section_terminator(text: str, section: str) -> bool:
    """Return True if a Cochrane characteristics section clearly ended."""
    for line in _top_lines(text, max_lines=18):
        cleaned = re.sub(r"\s+", " ", line.replace("\u00a0", " ")).strip()
        if _DATA_ANALYSES_HEADING_RE.match(cleaned):
            return True
        match = _COCHRANE_STUDY_SECTION_RE.match(cleaned)
        if match and str(match.group(1) or "").lower() != str(section or "").lower():
            return True
    return False


def _is_cochrane_post_analysis_table_page(text: str) -> bool:
    normalized = _normalize_text(text)
    if "additional tables" in normalized:
        return True
    top = " ".join(_normalize_text(line) for line in _top_lines(text, max_lines=24))
    return (
        "analysis" in top
        or "study or subgroup" in top
        or ("outcome or subgroup title" in top and "statistical method" in top)
    )


def _cochrane_section_ends_after_page(text: str, section: str) -> bool:
    """Return True when a later Cochrane section begins lower on this page."""
    for line in _all_lines(text):
        cleaned = re.sub(r"\s+", " ", line.replace("\u00a0", " ")).strip()
        if _DATA_ANALYSES_HEADING_RE.match(cleaned):
            return True
        if _COCHRANE_OTHER_CHARACTERISTICS_RE.match(cleaned):
            return True
        match = _COCHRANE_STUDY_SECTION_RE.match(cleaned)
        if match and str(match.group(1) or "").lower() != str(section or "").lower():
            return True
    return False


def _cochrane_study_label(line: str) -> tuple[str, bool]:
    cleaned = re.sub(r"\s+", " ", str(line or "").replace("\u00a0", " ")).strip()
    match = _COCHRANE_STUDY_LABEL_RE.match(cleaned)
    if not match:
        return "", False
    label = str(match.group(1) or "").strip()
    continued = "continued" in cleaned.lower()
    return label, continued


def _cochrane_included_study_starts(page_texts: List[str], pages: List[int]) -> List[Dict[str, Any]]:
    starts: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    within_section = False
    for page_number in pages:
        lines = _all_lines(page_texts[page_number - 1])
        for line in lines:
            cleaned = re.sub(r"\s+", " ", line.replace("\u00a0", " ")).strip()
            if _COCHRANE_STUDY_SECTION_RE.match(cleaned):
                within_section = True
                continue
            if page_number != pages[0]:
                within_section = True
            if not within_section:
                continue
            label, continued = _cochrane_study_label(line)
            if not label or continued:
                continue
            normalized_label = _normalize_text(label)
            if normalized_label in {
                "study characteristics",
                "methods",
                "participants",
                "interventions",
                "outcomes",
                "notes",
                "risk of bias",
            }:
                continue
            key = (normalized_label, page_number)
            if key in seen:
                continue
            seen.add(key)
            starts.append({"study_id": label, "page_number": page_number})
    return starts


def _extract_table_title(page_text: str, table_number: int) -> str:
    for line in _all_lines(page_text):
        match = _TABLE_START_RE.match(line)
        if not match:
            continue
        if _parse_table_number(str(match.group(1) or "")) != table_number:
            continue
        trailing = str(match.group(2) or "").strip(" -:.()")
        if trailing:
            return trailing
    return ""


def _classify_table_kind(title_text: str, body_text: str = "") -> str:
    normalized_title = _normalize_text(title_text)
    normalized_body = _normalize_text(body_text)
    if "picos criteria" in normalized_title:
        return "other"
    if any(marker in normalized_title for marker in ("hta report", "hta reports")):
        return "hta"
    if "economic studies" in normalized_title:
        return "economic"
    if any(
        marker in normalized_title
        for marker in (
            "included studies",
            "hrqol",
            "health state utility",
            "utility values",
            "study characteristics",
            "study design",
            "characteristics of included",
            "participants characteristics",
            "participant characteristics",
            "patient characteristics",
            "intervention characteristics",
            "trial characteristics",
            "risk of bias",
            "primary and secondary outcomes",
            "outcomes of the review",
            "correlates of",
        )
    ):
        return "included_studies"
    if any(marker in normalized_body for marker in ("hta report", "hta reports")):
        return "hta"
    if any(marker in normalized_body for marker in ("economic studies", "cost utility", "cost effectiveness")):
        return "economic"
    if any(marker in normalized_body for marker in ("included studies", "hrqol", "health state utility values")):
        return "included_studies"
    return "other"


def _write_pdf_slice(doc: fitz.Document, page_numbers: List[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as out_doc:
        for page_number in page_numbers:
            page_index = int(page_number) - 1
            if page_index < 0 or page_index >= len(doc):
                continue
            out_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
        out_doc.save(str(output_path))


def _bibliography_csv_bytes(entries: List[Dict[str, Any]]) -> bytes:
    if not entries:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "reference_number",
            "reference_section",
            "authors",
            "year",
            "title",
            "journal",
            "doi",
            "entry_text",
        ],
    )
    writer.writeheader()
    for entry in entries:
        writer.writerow(
            {
                "reference_number": str(entry.get("reference_number") or "").strip(),
                "reference_section": str(entry.get("reference_section") or "").strip(),
                "authors": str(entry.get("authors") or "").strip(),
                "year": str(entry.get("year") or "").strip(),
                "title": str(entry.get("title") or "").strip(),
                "journal": str(entry.get("journal") or "").strip(),
                "doi": str(entry.get("doi") or "").strip(),
                "entry_text": str(entry.get("entry_text") or "").strip(),
            }
        )
    return buf.getvalue().encode("utf-8")


def slice_review_pdf(pdf_path: str, *, work_dir: str = "") -> Dict[str, Any]:
    path = Path(str(pdf_path or "").strip())
    if path.suffix.lower() != ".pdf":
        raise ValueError("Included-study slicer requires a PDF input")
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    output_root = (
        Path(work_dir)
        if str(work_dir or "").strip()
        else Path(tempfile.mkdtemp(prefix="included_study_slices_"))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(path)) as doc:
        total_pages = int(len(doc) or 0)
        page_texts = [_page_text(doc, idx) for idx in range(total_pages)]

        bibliography_start = None
        for idx, text in enumerate(page_texts):
            if _looks_like_reference_page(text, page_number=idx + 1, total_pages=total_pages):
                bibliography_start = idx + 1
                break

        # Scan all pages (including post-bibliography) line-by-line. Supplementary
        # tables frequently appear after References in systematic reviews.
        page_headings_list: List[List[Dict[str, Any]]] = [
            _page_table_headings(text) for text in page_texts
        ]
        cochrane_section_headings: List[List[Dict[str, Any]]] = [
            _cochrane_study_section_headings(text) for text in page_texts
        ]
        cochrane_characteristics_present = any(cochrane_section_headings)
        data_analyses_start = None
        if cochrane_characteristics_present:
            for idx, text in enumerate(page_texts):
                if any(_DATA_ANALYSES_HEADING_RE.match(line) for line in _all_lines(text)):
                    data_analyses_start = idx + 1
                    break

        # Collect every non-continued heading as a candidate start, preserving
        # order (page, then within-page line order).
        starts: List[Dict[str, Any]] = []
        for idx, headings in enumerate(page_headings_list):
            page_number = idx + 1
            for heading in headings:
                if cochrane_characteristics_present and _is_cochrane_post_analysis_table_page(page_texts[idx]):
                    continue
                if data_analyses_start and page_number >= data_analyses_start:
                    continue
                if heading.get("continued"):
                    continue
                starts.append(
                    {
                        "page_number": page_number,
                        "table_number": int(heading.get("table_number") or 0),
                        "heading_title": str(heading.get("title") or "").strip(),
                    }
                )
            for heading in cochrane_section_headings[idx]:
                starts.append(
                    {
                        "page_number": page_number,
                        "table_number": int(heading.get("table_number") or 0),
                        "heading_title": str(heading.get("title") or "").strip(),
                        "forced_kind": str(heading.get("kind") or "").strip(),
                        "cochrane_section": str(heading.get("section") or "").strip(),
                    }
                )
        starts.sort(key=lambda item: (int(item.get("page_number") or 0), int(item.get("table_number") or 0)))

        # Cap bibliography coverage before the first post-References table start.
        bibliography_end = total_pages
        if bibliography_start:
            for start in starts:
                if start["page_number"] > bibliography_start:
                    bibliography_end = start["page_number"] - 1
                    break
        bibliography_pages = (
            list(range(bibliography_start, bibliography_end + 1))
            if bibliography_start
            else []
        )
        bibliography_text = (
            "\n\n".join(page_texts[(bibliography_start - 1) : bibliography_end])
            if bibliography_start
            else ""
        )
        bibliography_entries = _parse_reference_entries(bibliography_text) if bibliography_text else []

        bibliography_range = (
            set(range(bibliography_start, bibliography_end + 1))
            if bibliography_start
            else set()
        )

        table_slices: List[Dict[str, Any]] = []
        classified_slices: List[Dict[str, Any]] = []
        for pos, start in enumerate(starts):
            page_number = int(start["page_number"])
            table_number = int(start["table_number"])
            pages = [page_number]
            probe_page = page_number + 1
            forced_kind = str(start.get("forced_kind") or "").strip()
            cochrane_section = str(start.get("cochrane_section") or "").strip()
            next_start_page = (
                int(starts[pos + 1]["page_number"]) if pos + 1 < len(starts) else total_pages + 1
            )
            # Default-extend: many reviews flow table rows onto subsequent pages
            # with no "Table N continued" marker. Keep extending until we hit
            # the next table start, a bibliography page, a Figure caption, or
            # a narrative section heading.
            while probe_page <= total_pages:
                if probe_page >= next_start_page:
                    break
                if probe_page in bibliography_range:
                    break
                probe_text = page_texts[probe_page - 1]
                if cochrane_section:
                    if _is_cochrane_section_terminator(probe_text, cochrane_section):
                        break
                    pages.append(probe_page)
                    if _cochrane_section_ends_after_page(probe_text, cochrane_section):
                        break
                    probe_page += 1
                    continue
                if _is_table_terminator_page(probe_text):
                    break
                if not _looks_like_table_continuation(
                    probe_text, page_headings_list[probe_page - 1]
                ):
                    break
                pages.append(probe_page)
                probe_page += 1
            end_page = pages[-1]
            slice_text = "\n\n".join(page_texts[page_number - 1 : end_page])
            title = (
                str(start.get("heading_title") or "").strip()
                or _extract_table_title(page_texts[page_number - 1], table_number)
            )
            kind = forced_kind or _classify_table_kind(title, slice_text)
            classified_slices.append(
                {
                    "table_number": str(table_number),
                    "title": title,
                    "kind": kind,
                    "pages": list(pages),
                }
            )
            if kind == "other":
                continue
            if cochrane_section == "included":
                study_starts = _cochrane_included_study_starts(page_texts, pages)
                if study_starts:
                    for study_idx, study_start in enumerate(study_starts, start=1):
                        study_id = str(study_start.get("study_id") or "").strip()
                        start_page = int(study_start.get("page_number") or pages[0])
                        if study_idx < len(study_starts):
                            end_page = int(study_starts[study_idx].get("page_number") or start_page)
                        else:
                            end_page = pages[-1]
                        study_pages = list(range(start_page, max(start_page, end_page) + 1))
                        pdf_name = f"cochrane_included_{study_idx:03d}.pdf"
                        pdf_out = output_root / pdf_name
                        _write_pdf_slice(doc, study_pages, pdf_out)
                        table_slices.append(
                            {
                                "table_number": f"901.{study_idx}",
                                "table_title": f"Characteristics of included studies - {study_id}",
                                "kind": kind,
                                "page_numbers": study_pages,
                                "pdf_path": str(pdf_out),
                                "pdf_file_name": pdf_name,
                                "label": f"Cochrane included: {study_id}",
                                "cochrane_section": "included",
                                "cochrane_study_id": study_id,
                            }
                        )
                    continue
            pdf_name = f"table_{table_number}.pdf"
            pdf_out = output_root / pdf_name
            _write_pdf_slice(doc, pages, pdf_out)
            label = f"table {table_number}"
            if cochrane_section == "included":
                label = "Cochrane included studies"
            elif cochrane_section == "excluded":
                label = "Cochrane excluded studies"
            table_slices.append(
                {
                    "table_number": str(table_number),
                    "table_title": title,
                    "kind": kind,
                    "page_numbers": pages,
                    "pdf_path": str(pdf_out),
                    "pdf_file_name": pdf_name,
                    "label": label,
                    "cochrane_section": cochrane_section,
                }
            )

    bibliography_txt_path = output_root / "bibliography.txt"
    bibliography_txt_path.write_text(bibliography_text, encoding="utf-8")
    bibliography_csv_path = output_root / "bibliography.csv"
    bibliography_csv_bytes = _bibliography_csv_bytes(bibliography_entries)
    if bibliography_csv_bytes:
        bibliography_csv_path.write_bytes(bibliography_csv_bytes)

    diagnostics = {
        "total_pages": total_pages,
        "text_layer_chars": sum(len(t) for t in page_texts),
        "bibliography_start_page": bibliography_start,
        "headings_detected": [
            {
                "page": idx + 1,
                "table_number": int(h.get("table_number") or 0),
                "title": str(h.get("title") or ""),
                "continued": bool(h.get("continued")),
            }
            for idx, page_h in enumerate(page_headings_list)
            for h in page_h
        ],
        "cochrane_sections_detected": [
            {
                "page": idx + 1,
                "section": str(h.get("section") or ""),
                "table_number": int(h.get("table_number") or 0),
                "title": str(h.get("title") or ""),
                "kind": str(h.get("kind") or ""),
            }
            for idx, page_h in enumerate(cochrane_section_headings)
            for h in page_h
        ],
        "classified_slices": classified_slices,
    }

    return {
        "pdf_path": str(path),
        "work_dir": str(output_root),
        "table_slices": table_slices,
        "bibliography_pages": bibliography_pages,
        "bibliography_text": bibliography_text,
        "bibliography_entries": bibliography_entries,
        "bibliography_txt_path": str(bibliography_txt_path),
        "bibliography_csv_path": str(bibliography_csv_path) if bibliography_csv_bytes else "",
        "diagnostics": diagnostics,
    }
