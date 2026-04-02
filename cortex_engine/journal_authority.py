"""
Journal authority enrichment from SCImago rankings.

This module keeps journal ranking signals separate from credibility tiers.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SJR_INDEX_CACHE: Optional[Dict[str, Dict]] = None


def _normalize_title(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _normalize_issn(raw: str) -> List[str]:
    text = (raw or "").upper()
    candidates = set()
    for match in re.findall(r"\b\d{4}-?\d{3}[0-9X]\b", text):
        compact = match.replace("-", "")
        if len(compact) == 8:
            candidates.add(f"{compact[:4]}-{compact[4:]}")
    return sorted(candidates)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_sjr_json() -> Optional[Path]:
    root = _project_root()
    patterns = (
        "journal_quality_rankings_scimagojr_*.json",
        "scimagojr*.json",
    )
    for pattern in patterns:
        candidates = sorted(root.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def _find_sjr_workbook() -> Optional[Path]:
    candidates = sorted(_project_root().glob("scimagojr*.xlsx"))
    return candidates[0] if candidates else None


def _safe_int(raw: object, default: int = 10**9) -> int:
    try:
        return int(str(raw or "").strip())
    except Exception:
        return default


def _safe_float(raw: object, default: float = 0.0) -> float:
    try:
        cleaned = str(raw or "").strip().replace(",", ".")
        return float(cleaned)
    except Exception:
        return default


def _join_optional(value: object) -> str:
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _canonical_record(
    *,
    rank: object,
    sourceid: object,
    title: object,
    issn: object,
    publisher: object = "",
    sjr: object = "",
    quartile: object = "",
    h_index: object = "",
    categories: object = "",
    areas: object = "",
) -> Optional[Dict[str, object]]:
    title_text = str(title or "").strip()
    if not title_text:
        return None
    return {
        "rank": str(rank or "").strip(),
        "sourceid": str(sourceid or "").strip(),
        "title": title_text,
        "type": "",
        "issn": str(issn or "").strip(),
        "publisher": str(publisher or "").strip(),
        "sjr": str(sjr or "").strip(),
        "quartile": str(quartile or "").strip(),
        "h_index": str(h_index or "").strip(),
        "categories": _join_optional(categories),
        "areas": _join_optional(areas),
    }


def _parse_sjr_row(cells: List[str]) -> Optional[Dict[str, object]]:
    if not cells:
        return None
    line = ";".join(cells)
    try:
        fields = next(csv.reader([line], delimiter=";", quotechar='"', skipinitialspace=True))
    except Exception:
        return None
    if len(fields) < 10:
        return None

    quartile_re = re.compile(r"^Q[1-4]$", re.IGNORECASE)
    sjr_value = fields[8].strip() if len(fields) > 8 else ""
    quartile_idx = 9
    hindex_idx = 10
    if len(fields) > 10 and not quartile_re.match(fields[9].strip()) and quartile_re.match(fields[10].strip()):
        sjr_value = f"{fields[8].strip()}.{fields[9].strip()}"
        quartile_idx = 10
        hindex_idx = 11

    return _canonical_record(
        rank=fields[0].strip(),
        sourceid=fields[1].strip(),
        title=fields[2].strip(),
        issn=fields[4].strip(),
        publisher=fields[5].strip() if len(fields) > 5 else "",
        sjr=sjr_value,
        quartile=fields[quartile_idx].strip() if len(fields) > quartile_idx else "",
        h_index=fields[hindex_idx].strip() if len(fields) > hindex_idx else "",
    )


def _ingest_index_record(record: Dict[str, object], by_issn: Dict[str, Dict], by_title: Dict[str, Dict]) -> None:
    title_norm = _normalize_title(str(record.get("title", "") or ""))
    if not title_norm:
        return

    rank_value = _safe_int(record.get("rank", ""))
    record["rank_value"] = rank_value
    record["sjr_value"] = _safe_float(record.get("sjr", ""))

    current_by_title = by_title.get(title_norm)
    if current_by_title is None or rank_value < int(current_by_title.get("rank_value", 10**9)):
        by_title[title_norm] = record

    for issn in _normalize_issn(str(record.get("issn", "") or "")):
        current_by_issn = by_issn.get(issn)
        if current_by_issn is None or rank_value < int(current_by_issn.get("rank_value", 10**9)):
            by_issn[issn] = record


def _build_index_from_json(dataset_path: Path) -> Dict[str, Dict]:
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load SCImago JSON %s: %s", dataset_path.name, exc)
        return {"by_issn": {}, "by_title": {}}

    if not isinstance(raw, list):
        logger.warning("SCImago JSON %s did not contain a list payload", dataset_path.name)
        return {"by_issn": {}, "by_title": {}}

    by_issn: Dict[str, Dict] = {}
    by_title: Dict[str, Dict] = {}

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        record = _canonical_record(
            rank=entry.get("rank_global", entry.get("rank", "")),
            sourceid=entry.get("sourceid", ""),
            title=entry.get("title", ""),
            issn=entry.get("issn", entry.get("issn_combined", "")),
            publisher=entry.get("publisher", ""),
            sjr=entry.get("sjr", ""),
            quartile=entry.get(
                "best_quartile",
                entry.get("sjr_quartile", entry.get("quartile", entry.get("q", ""))),
            ),
            h_index=entry.get("h_index", entry.get("hindex", "")),
            categories=entry.get("categories", ""),
            areas=entry.get("areas", ""),
        )
        if not record:
            continue
        _ingest_index_record(record, by_issn, by_title)

    logger.info("Loaded SCImago JSON index: %s titles, %s ISSNs", len(by_title), len(by_issn))
    return {"by_issn": by_issn, "by_title": by_title}


def _build_index_from_workbook(workbook: Path) -> Dict[str, Dict]:
    try:
        import pandas as pd

        df = pd.read_excel(workbook, dtype=str)
    except Exception as exc:
        logger.warning("Could not load SCImago workbook %s: %s", workbook.name, exc)
        return {"by_issn": {}, "by_title": {}}

    by_issn: Dict[str, Dict] = {}
    by_title: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            if value is None:
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                cells.append(text)
        parsed = _parse_sjr_row(cells)
        if not parsed:
            continue
        _ingest_index_record(parsed, by_issn, by_title)

    logger.info("Loaded SCImago workbook index: %s titles, %s ISSNs", len(by_title), len(by_issn))
    return {"by_issn": by_issn, "by_title": by_title}


def _build_index() -> Dict[str, Dict]:
    json_path = _find_sjr_json()
    if json_path:
        return _build_index_from_json(json_path)

    workbook = _find_sjr_workbook()
    if workbook:
        return _build_index_from_workbook(workbook)

    logger.warning(
        "SCImago dataset not found (expected journal_quality_rankings_scimagojr_*.json, scimagojr*.json, or scimagojr*.xlsx in project root)"
    )
    return {"by_issn": {}, "by_title": {}}


def _get_index() -> Dict[str, Dict]:
    global _SJR_INDEX_CACHE
    if _SJR_INDEX_CACHE is None:
        _SJR_INDEX_CACHE = _build_index()
    return _SJR_INDEX_CACHE


def _extract_issn_candidates(text: str) -> List[str]:
    return _normalize_issn(text or "")


def _to_output(record: Dict, method: str, confidence: float) -> Dict[str, object]:
    quartile = str(record.get("quartile", "") or "")
    return {
        "journal_ranking_source": "scimagojr_2024",
        "journal_sourceid": str(record.get("sourceid", "") or ""),
        "journal_title": str(record.get("title", "") or ""),
        "journal_issn": str(record.get("issn", "") or ""),
        "journal_sjr": float(record.get("sjr_value", 0.0) or 0.0),
        "journal_quartile": quartile,
        "journal_rank_global": int(record.get("rank_value", 0) or 0),
        "journal_categories": str(record.get("categories", "") or ""),
        "journal_areas": str(record.get("areas", "") or ""),
        "journal_high_ranked": quartile.upper() == "Q1",
        "journal_match_method": method,
        "journal_match_confidence": round(confidence, 3),
    }


def classify_journal_authority(title: str, text: str) -> Dict[str, object]:
    """Classify journal authority from SCImago using ISSN/title matching."""
    default = {
        "journal_ranking_source": "scimagojr_2024",
        "journal_sourceid": "",
        "journal_title": "",
        "journal_issn": "",
        "journal_sjr": 0.0,
        "journal_quartile": "",
        "journal_rank_global": 0,
        "journal_categories": "",
        "journal_areas": "",
        "journal_high_ranked": False,
        "journal_match_method": "none",
        "journal_match_confidence": 0.0,
    }

    index = _get_index()
    by_issn = index.get("by_issn", {})
    by_title = index.get("by_title", {})
    if not by_issn and not by_title:
        return default

    combined_text = f"{title}\n{text or ''}"

    for issn in _extract_issn_candidates(combined_text):
        if issn in by_issn:
            return _to_output(by_issn[issn], method="issn_exact", confidence=1.0)

    norm_title = _normalize_title(title or "")
    if norm_title and norm_title in by_title:
        return _to_output(by_title[norm_title], method="title_exact", confidence=0.98)

    if norm_title:
        candidates = difflib.get_close_matches(norm_title, by_title.keys(), n=1, cutoff=0.93)
        if candidates:
            matched = candidates[0]
            ratio = difflib.SequenceMatcher(a=norm_title, b=matched).ratio()
            return _to_output(by_title[matched], method="title_fuzzy", confidence=float(ratio))

    return default
