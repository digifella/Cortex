from __future__ import annotations

import csv
import io
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import fitz

from cortex_engine.review_study_miner import _normalize_text, _parse_reference_entries


_TABLE_START_RE = re.compile(r"\btable\s+(\d{1,3})\b[:.\s-]*(.*)", re.IGNORECASE)


def _page_text(doc: fitz.Document, page_index: int) -> str:
    try:
        return str(doc.load_page(page_index).get_text("text") or "")
    except Exception:
        return ""


def _looks_like_reference_heading(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in lines[:12]:
        normalized = _normalize_text(line)
        if normalized in {"references", "bibliography"}:
            return True
    return False


def _extract_table_title(page_text: str, table_number: int) -> str:
    lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
    pattern = re.compile(rf"\btable\s+{int(table_number)}\b[:.\s-]*(.*)", re.IGNORECASE)
    for line in lines[:20]:
        match = pattern.search(line)
        if match:
            trailing = str(match.group(1) or "").strip(" -:.")
            if trailing:
                return trailing
    return ""


def _classify_table_kind(text: str) -> str:
    normalized = _normalize_text(text)
    if any(marker in normalized for marker in ("hta report", "hta reports", "nice", "cadth", "pbac")):
        return "hta"
    if any(marker in normalized for marker in ("economic studies", "cost utility", "cost effectiveness", "cua", "cea", "tto")):
        return "economic"
    if any(
        marker in normalized
        for marker in (
            "included studies",
            "hrqol",
            "quality of life",
            "health state utility",
            "utility values",
            "fact g",
            "eq 5d",
            "eortc",
            "sf 36",
        )
    ):
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

    output_root = Path(work_dir) if str(work_dir or "").strip() else Path(tempfile.mkdtemp(prefix="included_study_slices_"))
    output_root.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(path)) as doc:
        total_pages = int(len(doc) or 0)
        page_texts = [_page_text(doc, idx) for idx in range(total_pages)]

        bibliography_start = None
        for idx, text in enumerate(page_texts):
            if _looks_like_reference_heading(text):
                bibliography_start = idx + 1
                break
        bibliography_pages = list(range(bibliography_start, total_pages + 1)) if bibliography_start else []
        bibliography_text = "\n\n".join(page_texts[(bibliography_start - 1) :]) if bibliography_start else ""
        bibliography_entries = _parse_reference_entries(bibliography_text) if bibliography_text else []

        starts: List[Dict[str, Any]] = []
        for idx, text in enumerate(page_texts):
            page_number = idx + 1
            if bibliography_start and page_number >= bibliography_start:
                continue
            match = _TABLE_START_RE.search(text)
            if not match:
                continue
            starts.append({"page_number": page_number, "table_number": int(match.group(1) or 0)})

        table_slices: List[Dict[str, Any]] = []
        for pos, start in enumerate(starts):
            page_number = int(start["page_number"])
            table_number = int(start["table_number"])
            next_page = bibliography_start or (total_pages + 1)
            if pos + 1 < len(starts):
                next_page = min(next_page, int(starts[pos + 1]["page_number"]))
            end_page = max(page_number, next_page - 1)
            pages = list(range(page_number, end_page + 1))
            slice_text = "\n\n".join(page_texts[page_number - 1 : end_page])
            title = _extract_table_title(page_texts[page_number - 1], table_number)
            kind = _classify_table_kind(slice_text)
            pdf_name = f"table_{table_number}.pdf"
            pdf_out = output_root / pdf_name
            _write_pdf_slice(doc, pages, pdf_out)
            table_slices.append(
                {
                    "table_number": str(table_number),
                    "table_title": title,
                    "kind": kind,
                    "page_numbers": pages,
                    "pdf_path": str(pdf_out),
                    "pdf_file_name": pdf_name,
                    "label": f"table {table_number}",
                }
            )

    bibliography_txt_path = output_root / "bibliography.txt"
    bibliography_txt_path.write_text(bibliography_text, encoding="utf-8")
    bibliography_csv_path = output_root / "bibliography.csv"
    bibliography_csv_bytes = _bibliography_csv_bytes(bibliography_entries)
    if bibliography_csv_bytes:
        bibliography_csv_path.write_bytes(bibliography_csv_bytes)

    return {
        "pdf_path": str(path),
        "work_dir": str(output_root),
        "table_slices": table_slices,
        "bibliography_pages": bibliography_pages,
        "bibliography_text": bibliography_text,
        "bibliography_entries": bibliography_entries,
        "bibliography_txt_path": str(bibliography_txt_path),
        "bibliography_csv_path": str(bibliography_csv_path) if bibliography_csv_bytes else "",
    }
