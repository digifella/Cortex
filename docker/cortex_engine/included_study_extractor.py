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
_DEFAULT_GEMINI_MODEL = os.environ.get("CORTEX_INCLUDED_STUDY_GEMINI_MODEL", "").strip() or "gemini-2.5-flash"
_DEFAULT_ANTHROPIC_MODEL = os.environ.get("CORTEX_INCLUDED_STUDY_ANTHROPIC_MODEL", "").strip() or "claude-sonnet-4-6"
_MAX_INLINE_PDF_BYTES = 22 * 1024 * 1024
_COMMON_GEMINI_ACCESS_TEST_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
)
_DEFAULT_EXTRACTION_SCOPE = "all_trials"


class IncludedStudyExtractorAPIError(RuntimeError):
    def __init__(self, provider: str, status_code: int, message: str, *, body: str = ""):
        super().__init__(message)
        self.provider = str(provider or "").strip().lower()
        self.status_code = int(status_code or 0)
        self.body = str(body or "")


class IncludedStudyExtractorQuotaError(IncludedStudyExtractorAPIError):
    pass


def extract_retry_after_seconds(message: str) -> float | None:
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", str(message or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


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


def _parse_http_error_payload(body: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(str(body or "").strip())
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _gemini_http_error_message(status_code: int, body: str) -> str:
    parsed = _parse_http_error_payload(body)
    error_block = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
    detail = str(error_block.get("message") or "").strip()
    detail = re.sub(r"\s+", " ", detail)
    if status_code == 429:
        text = f"{detail} {body}".lower()
        if "quota exceeded" in text or "free_tier" in text or "rate limit" in text or "input_token_count" in text:
            return f"Gemini quota/rate limit exceeded: {detail or 'quota or rate limit reached for this PDF.'}"
    if detail:
        return f"Gemini API HTTP {status_code}: {detail}"
    snippet = re.sub(r"\s+", " ", str(body or "").strip())[:400]
    return f"Gemini API HTTP {status_code}: {snippet or 'request failed'}"


def _raise_gemini_http_error(status_code: int, body: str) -> None:
    message = _gemini_http_error_message(status_code, body)
    text = f"{message} {body}".lower()
    if int(status_code or 0) == 429 and (
        "quota exceeded" in text or "free_tier" in text or "rate limit" in text or "input_token_count" in text
    ):
        raise IncludedStudyExtractorQuotaError("gemini", status_code, message, body=body)
    raise IncludedStudyExtractorAPIError("gemini", status_code, message, body=body)


def _extract_gemini_text(response_json: Dict[str, Any]) -> str:
    parts: List[str] = []
    for candidate in list(response_json.get("candidates") or []):
        content = candidate.get("content") or {}
        for part in list(content.get("parts") or []):
            text = str(part.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def run_included_study_access_check(provider: str = "gemini", model: str = "") -> Dict[str, Any]:
    provider_name = str(provider or "gemini").strip().lower() or "gemini"
    if provider_name == "anthropic":
        client = _anthropic_client()
        model_name = str(model or _DEFAULT_ANTHROPIC_MODEL).strip() or _DEFAULT_ANTHROPIC_MODEL
        response = client.messages.create(
            model=model_name,
            max_tokens=32,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Reply with exactly ACCESS_OK"}]}],
        )
        parts: List[str] = []
        for block in response.content:
            text = str(getattr(block, "text", "") or "").strip()
            if text:
                parts.append(text)
        preview = "\n".join(parts).strip()
        return {
            "provider": "anthropic",
            "model": model_name,
            "ok": bool(preview),
            "preview": preview[:200],
        }

    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model_name = str(model or _DEFAULT_GEMINI_MODEL).strip() or _DEFAULT_GEMINI_MODEL
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Reply with exactly ACCESS_OK"},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 32,
        },
    }
    try:
        response_json = _gemini_generate_content(model_name, payload, api_key)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        _raise_gemini_http_error(int(exc.code or 0), body)
    except URLError as exc:
        raise RuntimeError(f"Gemini API connection failed: {exc}") from exc
    preview = _extract_gemini_text(response_json)
    return {
        "provider": "gemini",
        "model": model_name,
        "ok": bool(preview),
        "preview": preview[:200],
    }


def run_included_study_access_check_matrix(models: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_model in list(models or _COMMON_GEMINI_ACCESS_TEST_MODELS):
        model_name = str(raw_model or "").strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        try:
            result = run_included_study_access_check(provider="gemini", model=model_name)
            result["error"] = ""
        except Exception as exc:
            result = {
                "provider": "gemini",
                "model": model_name,
                "ok": False,
                "preview": "",
                "error": str(exc),
            }
        results.append(result)
    return results


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
    study_design = re.sub(r"\s+", " ", str(item.get("study_design") or "").strip())
    sample_size = re.sub(r"\s+", " ", str(item.get("sample_size") or "").strip())
    outcome_measure = re.sub(r"\s+", " ", str(item.get("outcome_measure") or "").strip())
    outcome_result = re.sub(r"\s+", " ", str(item.get("outcome_result") or "").strip())

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
        "study_design": study_design,
        "sample_size": sample_size,
        "outcome_measure": outcome_measure,
        "outcome_result": outcome_result,
    }


def _normalize_table_number(value: Any) -> str:
    raw = re.sub(r"\s+", " ", str(value or "").strip())
    if not raw:
        return ""
    match = re.search(r"(\d{1,3})", raw)
    return str(match.group(1) or "").strip() if match else raw


def _normalize_extracted_tables(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    for raw_table in list(payload.get("tables") or []):
        if not isinstance(raw_table, dict):
            continue
        table_number = _normalize_table_number(raw_table.get("table_number") or raw_table.get("number") or "")
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
                "included_study_count": re.sub(r"\s+", " ", str(raw_table.get("included_study_count") or "").strip()),
                "included_rct_count": re.sub(r"\s+", " ", str(raw_table.get("included_rct_count") or "").strip()),
                "groups": groups,
            }
        )
    return tables


def _finalize_included_study_result(
    *,
    provider: str,
    model: str,
    parsed: Dict[str, Any],
    raw_response: str,
    extra_warnings: Sequence[str] | None = None,
) -> Dict[str, Any]:
    tables = _normalize_extracted_tables(parsed)
    warnings = [str(item).strip() for item in list(parsed.get("warnings") or []) if str(item).strip()]
    warnings.extend(str(item).strip() for item in list(extra_warnings or []) if str(item).strip())
    raw_text = str(raw_response or "").strip()
    if not tables:
        if raw_text:
            warnings.insert(
                0,
                "The model call completed but Cortex could not parse any included-study tables from the response. Inspect Raw Model Output.",
            )
        else:
            warnings.insert(0, "The model call completed but returned no parsable output.")
    return {
        "provider": str(provider or "").strip(),
        "model": str(model or "").strip(),
        "tables": tables,
        "warnings": warnings,
        "raw_response": raw_text,
    }


def _scope_instruction(extraction_scope: str) -> str:
    scope_name = str(extraction_scope or _DEFAULT_EXTRACTION_SCOPE).strip().lower()
    if scope_name == "rct_or_clinical":
        return (
            "Only include randomized controlled trials, clinical trials, or explicitly trial-based study reports. "
            "Exclude observational-only, vignette-only, economic-only, or narrative-only studies unless they are directly tied to a trial group in the table."
        )
    return "Include all eligible included-study rows shown in the target included-study tables."


def _included_study_prompt(review_title: str = "", extraction_scope: str = _DEFAULT_EXTRACTION_SCOPE) -> str:
    return (
        "You are extracting included-study tables from a systematic review PDF.\n\n"
        "Return only the tables that list included studies, health state utility studies, or HTA reports included in the review.\n"
        "Ignore eligibility tables, search strategy tables, risk-of-bias tables, and narrative text that does not list included studies.\n\n"
        f"Study selection scope: {_scope_instruction(extraction_scope)}\n\n"
        "Return compact JSON only. Keep the output short.\n\n"
        "For each included-study table:\n"
        "1. Return the table number as digits only, for example `2`, not `Table 2`.\n"
        "2. Return the table title.\n"
        "3. Explain the grouping basis briefly.\n"
        "4. Preserve the grouping structure the table uses conceptually.\n"
        "5. For each cited paper/report, return only a short display label, short author label if known, year, and bibliography reference number when present.\n"
        "6. Prefer structured trial-detail fields when the table provides them: study design, sample size, outcome measure, and outcome result.\n"
        "7. Do not include full author lists, long paper titles, journals, or DOIs unless they are trivially obvious. Prefer leaving those fields blank.\n"
        "8. If multiple papers are grouped under the same trial, keep them grouped under that trial.\n"
        "9. If uncertain, set needs_review=true and explain briefly in notes.\n\n"
        f"Review title: {review_title}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "tables": [\n'
        "    {\n"
        '      "table_number": "",\n'
        '      "table_title": "",\n'
        '      "grouping_basis": "",\n'
        '      "included_study_count": "",\n'
        '      "included_rct_count": "",\n'
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
        '              "study_design": "",\n'
        '              "sample_size": "",\n'
        '              "outcome_measure": "",\n'
        '              "outcome_result": "",\n'
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


def _single_table_prompt(
    table_label: str = "",
    review_title: str = "",
    bibliography_text: str = "",
    extraction_scope: str = _DEFAULT_EXTRACTION_SCOPE,
) -> str:
    compact_bibliography = str(bibliography_text or "").strip()
    if len(compact_bibliography) > 18000:
        compact_bibliography = compact_bibliography[:18000].rstrip() + "\n...[truncated]"
    return (
        "You are extracting one included-study table from a systematic review PDF.\n\n"
        "Return compact JSON only.\n"
        "This PDF contains one table snippet, not the whole review.\n"
        f"Study selection scope: {_scope_instruction(extraction_scope)}\n"
        "Use the supplied bibliography text only to reconcile reference numbers and short author/year labels when possible.\n"
        "Keep the output short.\n"
        "Do not emit full author lists.\n"
        "Do not emit long paper titles or journal names unless a very short title is the only way to disambiguate the citation.\n"
        "Prefer leaving optional resolved fields blank rather than filling them with long text.\n"
        "Keep notes brief and high-level; do not restate every numeric table value.\n\n"
        f"Review title: {review_title}\n"
        f"Table label: {table_label}\n\n"
        "Bibliography text from the same review:\n"
        f"{compact_bibliography}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "tables": [\n'
        "    {\n"
        '      "table_number": "",\n'
        '      "table_title": "",\n'
        '      "grouping_basis": "",\n'
        '      "included_study_count": "",\n'
        '      "included_rct_count": "",\n'
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
        '              "study_design": "",\n'
        '              "sample_size": "",\n'
        '              "outcome_measure": "",\n'
        '              "outcome_result": "",\n'
        '              "notes": "",\n'
        '              "needs_review": false\n'
        "            }\n"
        "          ]\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "warnings": []\n'
        "}\n\n"
        "Important:\n"
        "- `display` should usually be like `Author 2021 [17]`.\n"
        "- `authors` should be short, e.g. `Patrick` or `Patrick et al.`.\n"
        "- `resolved_title` should be empty unless a short disambiguating title is trivial.\n"
        "- Leave `resolved_authors`, `resolved_year`, `resolved_journal`, and `resolved_doi` blank unless trivially obvious and short.\n"
        "- Consolidate repeated measure rows rather than repeating every metric.\n"
    )


def _gemini_response_warnings(response_json: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    for candidate in list(response_json.get("candidates") or []):
        finish_reason = str(candidate.get("finishReason") or "").strip()
        if finish_reason and finish_reason.upper() != "STOP":
            warnings.append(f"Gemini finish reason: {finish_reason}")
    prompt_feedback = response_json.get("promptFeedback") if isinstance(response_json.get("promptFeedback"), dict) else {}
    block_reason = str(prompt_feedback.get("blockReason") or "").strip()
    if block_reason:
        warnings.append(f"Gemini prompt feedback: {block_reason}")
    return warnings


def _anthropic_response_warnings(response: Any) -> List[str]:
    warnings: List[str] = []
    stop_reason = str(getattr(response, "stop_reason", "") or "").strip()
    if stop_reason and stop_reason.lower() not in {"end_turn", "stop_sequence"}:
        warnings.append(f"Anthropic stop reason: {stop_reason}")
    return warnings


def run_included_study_extractor(
    *,
    pdf_path: str,
    provider: str = "gemini",
    model: str = "",
    review_title: str = "",
    extraction_scope: str = _DEFAULT_EXTRACTION_SCOPE,
) -> Dict[str, Any]:
    pdf_b64, _size_bytes = _build_pdf_inline_data(pdf_path)
    provider_name = str(provider or "gemini").strip().lower()
    prompt = _included_study_prompt(review_title, extraction_scope)

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
        return _finalize_included_study_result(
            provider="anthropic",
            model=model_name,
            parsed=parsed,
            raw_response=raw_response,
            extra_warnings=_anthropic_response_warnings(response),
        )

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
        _raise_gemini_http_error(int(exc.code or 0), body)
    except URLError as exc:
        raise RuntimeError(f"Gemini API connection failed: {exc}") from exc
    raw_response = _extract_gemini_text(response_json)
    parsed = parse_included_study_extraction_response(raw_response)
    return _finalize_included_study_result(
        provider="gemini",
        model=model_name,
        parsed=parsed,
        raw_response=raw_response,
        extra_warnings=_gemini_response_warnings(response_json),
    )


def run_included_study_table_extractor(
    *,
    pdf_path: str,
    bibliography_text: str,
    provider: str = "gemini",
    model: str = "",
    review_title: str = "",
    table_label: str = "",
    extraction_scope: str = _DEFAULT_EXTRACTION_SCOPE,
) -> Dict[str, Any]:
    pdf_b64, _size_bytes = _build_pdf_inline_data(pdf_path)
    provider_name = str(provider or "gemini").strip().lower()
    prompt = _single_table_prompt(
        table_label=table_label,
        review_title=review_title,
        bibliography_text=bibliography_text,
        extraction_scope=extraction_scope,
    )

    if provider_name == "anthropic":
        client = _anthropic_client()
        model_name = str(model or _DEFAULT_ANTHROPIC_MODEL).strip() or _DEFAULT_ANTHROPIC_MODEL
        response = client.messages.create(
            model=model_name,
            max_tokens=5000,
            system="Extract one grouped included-study table from the PDF snippet and return only valid JSON.",
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
        return _finalize_included_study_result(
            provider="anthropic",
            model=model_name,
            parsed=parsed,
            raw_response=raw_response,
            extra_warnings=_anthropic_response_warnings(response),
        )

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
            "maxOutputTokens": 4096,
        },
    }
    try:
        response_json = _gemini_generate_content(model_name, payload, api_key)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        _raise_gemini_http_error(int(exc.code or 0), body)
    except URLError as exc:
        raise RuntimeError(f"Gemini API connection failed: {exc}") from exc
    raw_response = _extract_gemini_text(response_json)
    parsed = parse_included_study_extraction_response(raw_response)
    return _finalize_included_study_result(
        provider="gemini",
        model=model_name,
        parsed=parsed,
        raw_response=raw_response,
        extra_warnings=_gemini_response_warnings(response_json),
    )


def run_included_study_extractor_with_fallback(
    *,
    pdf_path: str,
    provider: str = "gemini",
    model: str = "",
    review_title: str = "",
    extraction_scope: str = _DEFAULT_EXTRACTION_SCOPE,
    fallback_provider: str = "",
    fallback_model: str = "",
) -> Dict[str, Any]:
    primary_provider = str(provider or "gemini").strip().lower() or "gemini"
    primary_model = str(model or "").strip()
    try:
        result = run_included_study_extractor(
            pdf_path=pdf_path,
            provider=primary_provider,
            model=primary_model,
            review_title=review_title,
            extraction_scope=extraction_scope,
        )
        result["requested_provider"] = primary_provider
        result["requested_model"] = primary_model or result.get("model") or ""
        return result
    except IncludedStudyExtractorQuotaError as exc:
        fallback_name = str(fallback_provider or "").strip().lower()
        if primary_provider != "gemini" or fallback_name != "anthropic" or not included_study_extractor_available("anthropic"):
            raise
        fallback_result = run_included_study_extractor(
            pdf_path=pdf_path,
            provider="anthropic",
            model=fallback_model,
            review_title=review_title,
            extraction_scope=extraction_scope,
        )
        warnings = [str(item).strip() for item in list(fallback_result.get("warnings") or []) if str(item).strip()]
        fallback_model_name = str(fallback_result.get("model") or fallback_model or _DEFAULT_ANTHROPIC_MODEL).strip()
        warnings.insert(
            0,
            f"Gemini hit a quota/rate limit and the extractor fell back to Anthropic `{fallback_model_name}`.",
        )
        fallback_result["warnings"] = warnings
        fallback_result["requested_provider"] = primary_provider
        fallback_result["requested_model"] = primary_model or _DEFAULT_GEMINI_MODEL
        fallback_result["fallback_reason"] = str(exc)
        return fallback_result
