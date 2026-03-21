from __future__ import annotations

import re
from typing import Any, Dict, List

from cortex_engine.stakeholder_signal_matcher import normalize_lookup

_TITLE_KEYWORDS = (
    "secretary",
    "minister",
    "chief",
    "head",
    "director",
    "manager",
    "lead",
    "officer",
    "executive",
    "president",
    "vice president",
    "vice-president",
    "vp",
    "dean",
    "provost",
    "principal",
    "chair",
    "partner",
    "consultant",
    "advisor",
    "adviser",
    "specialist",
    "transformation",
    "strategy",
    "operations",
    "technology",
    "digital",
    "clinical",
    "program",
    "programme",
)
_NON_PERSON_NAME_KEYWORDS = (
    "department",
    "office",
    "health",
    "wellbeing",
    "victoria",
    "hospitals",
    "hospital",
    "services",
    "service",
    "communications",
    "engagement",
    "governance",
    "research",
    "safety",
    "community",
    "public",
    "mental",
    "planning",
    "ehealth",
    "budget",
    "finance",
    "investment",
    "people",
    "operations",
    "legal",
    "regulation",
    "infrastructure",
    "aboriginal",
    "care",
    "system",
    "prevention",
    "family",
    "violence",
    "data",
    "analytics",
)
_ORG_CHART_NAME_RE = re.compile(r"^[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+){1,4}$")
_ORG_CHART_SPLIT_RE = re.compile(r"\s*[|/]\s*|\s{3,}")


def looks_like_org_chart_attachment(filename: str, text: str = "") -> bool:
    lowered_name = normalize_lookup(filename)
    lowered_text = normalize_lookup(text)
    markers = (
        ("org", "chart"),
        ("organisation", "chart"),
        ("organization", "chart"),
        ("management", "structure"),
        ("leadership", "team"),
        ("senior", "management"),
        ("governance", "structure"),
    )
    return any(all(token in lowered_name for token in pair) for pair in markers) or any(
        all(token in lowered_text for token in pair) for pair in markers
    )


def _clean_chart_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", str(raw_line or "").strip())
        if not line:
            continue
        if line.lower().startswith("attachment: "):
            continue
        lines.append(line)
    return lines


def _looks_like_person_name(line: str) -> bool:
    value = str(line or "").strip()
    if not value or len(value) > 60:
        return False
    if any(char.isdigit() for char in value):
        return False
    if "," in value or ":" in value or "@" in value:
        return False
    lowered = normalize_lookup(value)
    if any(token in lowered for token in _TITLE_KEYWORDS):
        return False
    if any(token in lowered for token in _NON_PERSON_NAME_KEYWORDS):
        return False
    if len(value.split()) < 2:
        return False
    return bool(_ORG_CHART_NAME_RE.match(value))


def _looks_like_role_title(line: str) -> bool:
    value = str(line or "").strip()
    if not value or len(value) > 120:
        return False
    lowered = normalize_lookup(value)
    if not lowered:
        return False
    if "http" in lowered or "www" in lowered or "@" in lowered:
        return False
    if lowered.startswith("office of "):
        return False
    if lowered.startswith("department of "):
        return False
    if _looks_like_person_name(value):
        return False
    if any(keyword in lowered for keyword in _TITLE_KEYWORDS):
        return True
    if any(
        lowered.startswith(prefix)
        for prefix in (
            "senior ",
            "general ",
            "associate ",
            "assistant ",
            "acting ",
            "interim ",
            "group ",
        )
    ):
        return True
    return False


def _normalize_role_title(line: str) -> str:
    cleaned = str(line or "").strip(" -|,;")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_org_chart_structured(
    attachment_texts: List[str],
    attachment_summaries: List[Dict[str, Any]],
    employer_hint: str = "",
) -> Dict[str, Any]:
    processed_texts: List[str] = []
    for block in attachment_texts or []:
        block_text = str(block or "").strip()
        if not block_text:
            continue
        if not looks_like_org_chart_attachment("", block_text):
            continue
        processed_texts.append(block_text)

    people: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for text in processed_texts:
        lines = _clean_chart_lines(text)
        for index, line in enumerate(lines):
            pairings = []
            candidates = [part.strip() for part in _ORG_CHART_SPLIT_RE.split(line) if part.strip()]
            if len(candidates) == 2:
                pairings.append((candidates[0], candidates[1]))
                pairings.append((candidates[1], candidates[0]))
            if index + 1 < len(lines):
                next_line = lines[index + 1]
                pairings.append((line, next_line))
                pairings.append((next_line, line))
            for name_line, title_line in pairings:
                if not _looks_like_person_name(name_line) or not _looks_like_role_title(title_line):
                    continue
                key = normalize_lookup(name_line)
                if key in seen:
                    continue
                seen.add(key)
                people.append(
                    {
                        "name": name_line.strip(),
                        "current_employer": str(employer_hint or "").strip(),
                        "current_role": _normalize_role_title(title_line),
                        "confidence": 0.72,
                        "evidence": f"Org chart OCR pair: {name_line.strip()} / {_normalize_role_title(title_line)}",
                        "extraction_method": "org_chart_heuristic",
                    }
                )
                break

    summary = ""
    if people:
        summary = f"Org chart heuristic recovered {len(people)} people with roles"
    return {
        "people": people,
        "organisations": [],
        "emails": [],
        "career_events": [],
        "summary": summary,
    }


def analyse_org_chart_attachments(attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
    processed = [item for item in attachments or [] if str(item.get("status") or "").strip().lower() == "processed"]
    chart_like = [
        item for item in processed
        if looks_like_org_chart_attachment(
            str(item.get("filename") or ""),
            str(item.get("excerpt") or ""),
        )
    ]
    return {
        "attachment_count": len(chart_like),
        "attachment_names": [str(item.get("filename") or "").strip() for item in chart_like],
        "processed_count": len(processed),
    }
