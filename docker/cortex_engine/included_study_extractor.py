from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cortex_engine.research_resolve import _extract_year, _normalize_doi
from cortex_engine.review_study_miner import _extract_authors_and_year


_ROOT = Path(__file__).resolve().parent.parent
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_DEFAULT_GEMINI_MODEL = os.environ.get("CORTEX_INCLUDED_STUDY_GEMINI_MODEL", "").strip() or "gemini-2.5-pro"
_DEFAULT_ANTHROPIC_MODEL = os.environ.get("CORTEX_INCLUDED_STUDY_ANTHROPIC_MODEL", "").strip() or "claude-sonnet-4-6"
_MAX_INLINE_PDF_BYTES = 22 * 1024 * 1024


def _worker_env_value(name: str) -> str:
    env_path = _ROOT / "worker" / "config.env"
    if not env_path.exists():
        return ""
    try:
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def get_gemini_api_key() -> str:
    direct = str(os.environ.get("GEMINI_API_KEY", "") or "").strip()
    if direct:
        return direct
    fallback = _worker_env_value("GEMINI_API_KEY")
    if fallback and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = fallback
    return fallback


def gemini_key_source() -> str:
    direct = str(os.environ.get("GEMINI_API_KEY", "") or "").strip()
    if direct:
        return "environment/.env"
    if _worker_env_value("GEMINI_API_KEY"):
        return "worker/config.env"
    return ""


def get_anthropic_api_key() -> str:
    direct = str(os.environ.get("ANTHROPIC_API_KEY", "") or "").strip()
    if direct:
        return direct
    fallback = _worker_env_value("ANTHROPIC_API_KEY")
    if fallback and "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = fallback
    return fallback


def anthropic_key_source() -> str:
    direct = str(os.environ.get("ANTHROPIC_API_KEY", "") or "").strip()
    if direct:
        return "environment/.env"
    if _worker_env_value("ANTHROPIC_API_KEY"):
        return "worker/config.env"
    return ""


def included_study_extractor_available(provider: str) -> bool:
    provider_name = str(provider or "").strip().lower()
    if provider_name == "anthropic":
        return bool(get_anthropic_api_key())
    return bool(get_gemini_api_key())


def parse_included_study_extraction_response(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    fenced = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(fenced)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _build_pdf_inline_data(pdf_path: str) -> tuple[str, int]:
    path = Path(str(pdf_path or "").strip())
    if path.suffix.lower() != ".pdf":
        raise ValueError("Included Study Extractor requires a PDF input")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    size_bytes = int(path.stat().st_size or 0)
    if size_bytes <= 0:
        raise ValueError(f"PDF file could not be read: {path}")
    if size_bytes > _MAX_INLINE_PDF_BYTES:
        raise ValueError(
            f"PDF is too large to inline safely ({round(size_bytes / (1024 * 1024), 1)} MB > {round(_MAX_INLINE_PDF_BYTES / (1024 * 1024), 1)} MB)"
        )
    return base64.b64encode(path.read_bytes()).decode("utf-8"), size_bytes


def _gemini_generate_content(model: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
        method="POST",
    )
    with urlopen(req, timeout=240) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _extract_gemini_text(response_json: Dict[str, Any]) -> str:
    parts: List[str] = []
    for candidate in list(response_json.get("candidates") or []):
        content = candidate.get("content") or {}
        for part in list(content.get("parts") or []):
            text = str(part.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _anthropic_client():
    import anthropic

    api_key = get_anthropic_api_key()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def _extract_reference_number(value: str) -> str:
    match = re.search(r"\[(\d{1,3})\]", str(value or ""))
    return str(match.group(1) or "").strip() if match else ""


def _normalize_citation_item(item: Dict[str, Any]) -> Dict[str, Any]:
    display = re.sub(r"\s+", " ", str(item.get("display") or item.get("citation") or "").strip())
    authors = re.sub(r"\s+", " ", str(item.get("authors") or "").strip())
    year = str(item.get("year") or "").strip() or _extract_year(display)
    reference_number = str(item.get("reference_number") or "").strip() or _extract_reference_number(display)
    resolved_title = re.sub(r"\s+", " ", str(item.get("resolved_title") or item.get("paper_title") or "").strip())
    resolved_authors = re.sub(r"\s+", " ", str(item.get("resolved_authors") or "").strip())
    resolved_year = str(item.get("resolved_year") or "").strip() or _extract_year(resolved_title)
    resolved_journal = re.sub(r"\s+", " ", str(item.get("resolved_journal") or item.get("journal") or "").strip())
    resolved_doi = _normalize_doi(str(item.get("resolved_doi") or item.get("doi") or "").strip())
    notes = re.sub(r"\s+", " ", str(item.get("notes") or "").strip())
    needs_review = bool(item.get("needs_review"))

    if not authors and display:
        inferred_authors, inferred_year = _extract_authors_and_year(display)
        authors = inferred_authors or authors
        year = year or inferred_year
    if not resolved_authors and resolved_title:
        resolved_authors = str(item.get("authors") or "").strip() or authors
    if not resolved_year:
        resolved_year = year

    return {
        "display": display,
        "authors": authors,
        "year": year,
        "reference_number": reference_number,
        "resolved_title": resolved_title,
        "resolved_authors": resolved_authors,
        "resolved_year": resolved_year,
        "resolved_journal": resolved_journal,
        "resolved_doi": resolved_doi,
        "notes": notes,
        "needs_review": needs_review,
    }


def _normalize_extracted_tables(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    for raw_table in list(payload.get("tables") or []):
        if not isinstance(raw_table, dict):
            continue
        table_number = str(raw_table.get("table_number") or raw_table.get("number") or "").strip()
        table_title = re.sub(r"\s+", " ", str(raw_table.get("table_title") or raw_table.get("title") or "").strip())
        grouping_basis = re.sub(r"\s+", " ", str(raw_table.get("grouping_basis") or "").strip())
        groups: List[Dict[str, Any]] = []
        for raw_group in list(raw_table.get("groups") or []):
            if not isinstance(raw_group, dict):
                continue
            group_label = re.sub(r"\s+", " ", str(raw_group.get("group_label") or "").strip())
            trial_label = re.sub(r"\s+", " ", str(raw_group.get("trial_label") or "").strip())
            notes = re.sub(r"\s+", " ", str(raw_group.get("notes") or "").strip())
            citations = [
                _normalize_citation_item(item)
                for item in list(raw_group.get("citations") or [])
                if isinstance(item, dict)
            ]
            citations = [item for item in citations if item.get("display") or item.get("resolved_title")]
            if not citations:
                continue
            groups.append(
                {
                    "group_label": group_label,
                    "trial_label": trial_label,
                    "notes": notes,
                    "citations": citations,
                }
            )
        if not groups:
            continue
        tables.append(
            {
                "table_number": table_number,
                "table_title": table_title,
                "grouping_basis": grouping_basis,
                "groups": groups,
            }
        )
    return tables


def _included_study_prompt(review_title: str = "") -> str:
    return (
        "You are extracting included-study tables from a systematic review PDF.\n\n"
        "Return only the tables that list included studies, health state utility studies, or HTA reports included in the review.\n"
        "Ignore eligibility tables, search strategy tables, risk-of-bias tables, and narrative text that does not list included studies.\n\n"
        "For each included-study table:\n"
        "1. Return the exact table number and table title.\n"
        "2. Explain the grouping basis briefly.\n"
        "3. Preserve the grouping structure the table uses conceptually.\n"
        "4. For each cited paper/report, return the short display label used in the table, including bibliography reference number when present.\n"
        "5. Resolve the actual bibliography entry when possible from the review bibliography, including the real paper title, authors, year, journal, and DOI.\n"
        "6. If multiple papers are grouped under the same trial, keep them grouped under that trial.\n"
        "7. If uncertain, set needs_review=true and explain briefly in notes.\n\n"
        f"Review title: {review_title}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "tables": [\n'
        "    {\n"
        '      "table_number": "",\n'
        '      "table_title": "",\n'
        '      "grouping_basis": "",\n'
        '      "groups": [\n'
        "        {\n"
        '          "group_label": "",\n'
        '          "trial_label": "",\n'
        '          "notes": "",\n'
        '          "citations": [\n'
        "            {\n"
        '              "display": "",\n'
        '              "authors": "",\n'
        '              "year": "",\n'
        '              "reference_number": "",\n'
        '              "resolved_title": "",\n'
        '              "resolved_authors": "",\n'
        '              "resolved_year": "",\n'
        '              "resolved_journal": "",\n'
        '              "resolved_doi": "",\n'
        '              "notes": "",\n'
        '              "needs_review": false\n'
        "            }\n"
        "          ]\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "warnings": []\n'
        "}\n"
    )


def run_included_study_extractor(
    *,
    pdf_path: str,
    provider: str = "gemini",
    model: str = "",
    review_title: str = "",
) -> Dict[str, Any]:
    pdf_b64, _size_bytes = _build_pdf_inline_data(pdf_path)
    provider_name = str(provider or "gemini").strip().lower()
    prompt = _included_study_prompt(review_title)

    if provider_name == "anthropic":
        client = _anthropic_client()
        model_name = str(model or _DEFAULT_ANTHROPIC_MODEL).strip() or _DEFAULT_ANTHROPIC_MODEL
        response = client.messages.create(
            model=model_name,
            max_tokens=4000,
            system="Extract grouped included-study tables from the PDF and return only valid JSON.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        parts: List[str] = []
        for block in response.content:
            text = str(getattr(block, "text", "") or "").strip()
            if text:
                parts.append(text)
        raw_response = "\n".join(parts).strip()
        parsed = parse_included_study_extraction_response(raw_response)
        return {
            "provider": "anthropic",
            "model": model_name,
            "tables": _normalize_extracted_tables(parsed),
            "warnings": [str(item).strip() for item in list(parsed.get("warnings") or []) if str(item).strip()],
            "raw_response": raw_response,
        }

    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model_name = str(model or _DEFAULT_GEMINI_MODEL).strip() or _DEFAULT_GEMINI_MODEL
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": pdf_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
        },
    }
    try:
        response_json = _gemini_generate_content(model_name, payload, api_key)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {body[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Gemini API connection failed: {exc}") from exc
    raw_response = _extract_gemini_text(response_json)
    parsed = parse_included_study_extraction_response(raw_response)
    return {
        "provider": "gemini",
        "model": model_name,
        "tables": _normalize_extracted_tables(parsed),
        "warnings": [str(item).strip() for item in list(parsed.get("warnings") or []) if str(item).strip()],
        "raw_response": raw_response,
    }
