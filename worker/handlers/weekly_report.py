"""
Weekly Intelligence Report Handler

Receives pre-gathered intel (submitted + optional web research) and calls
Ollama (Qwen) to synthesise a structured weekly intelligence report.
Falls back to Anthropic API if Ollama is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/chat"
_OLLAMA_MODEL = "qwen2.5:14b"
_OLLAMA_TIMEOUT = 600  # 10 minutes for large reports
_ANTHROPIC_FALLBACK_MODEL = "claude-haiku-4-5-20251001"
_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def _first_non_empty(*values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _resolve_report_geography(report_scope: dict, strategic_context: dict) -> str:
    explicit = _first_non_empty(
        *(report_scope.get(key, "") for key in ("geography", "geographic_scope", "country", "jurisdiction", "market")),
        *(strategic_context.get(key, "") for key in ("geography", "geographic_scope", "country", "jurisdiction", "market")),
    )
    if explicit:
        return explicit

    for profile in (report_scope.get("organisations", []) or []) + (report_scope.get("industry_profiles", []) or []):
        country = _first_non_empty(
            profile.get("country", ""),
            profile.get("address_country", ""),
            (profile.get("address") or {}).get("country", "") if isinstance(profile.get("address"), dict) else "",
        )
        if country:
            return country

    return "Australia"


def _looks_like_url(value: Any) -> bool:
    text = str(value or "").strip()
    return text.startswith("http://") or text.startswith("https://")


def _normalize_model_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if ":" in text else f"{text}:latest"


def _select_installed_ollama_model(candidates: list[str], installed_models: list[str]) -> str:
    installed_normalized = {_normalize_model_name(name) for name in installed_models if _normalize_model_name(name)}
    for candidate in candidates:
        normalized = _normalize_model_name(candidate)
        if normalized and normalized in installed_normalized:
            return normalized
    return _normalize_model_name(candidates[0] if candidates else "")


def _resolve_ollama_model() -> str:
    candidates = [
        str(os.environ.get("CORTEX_WEEKLY_OLLAMA_MODEL") or "").strip(),
        str(os.environ.get("CORTEX_WATCH_OLLAMA_MODEL") or "").strip(),
        str(os.environ.get("LOCAL_LLM_SYNTHESIS_MODEL") or "").strip(),
        _OLLAMA_MODEL,
        "mistral-small3.2:latest",
        "mistral:latest",
    ]
    candidates = [candidate for candidate in candidates if candidate]
    try:
        response = requests.get(_OLLAMA_TAGS_URL, timeout=15)
        if response.status_code == 200:
            data = response.json()
            installed_models = [str(item.get("name") or "").strip() for item in data.get("models") or []]
            selected = _select_installed_ollama_model(candidates, installed_models)
            if selected:
                return selected
    except Exception as exc:
        logger.warning("Unable to inspect installed Ollama models for weekly report: %s", exc)
    return _normalize_model_name(candidates[0] if candidates else _OLLAMA_MODEL)


def _submitter_label(item: dict) -> str:
    display_name = _first_non_empty(item.get("submitted_by_name"), item.get("from_name"))
    email = _first_non_empty(item.get("submitted_by"), item.get("from_email"))
    if display_name and email and display_name.lower() != email.lower():
        return f"{display_name} <{email}>"
    return display_name or email or "Unknown submitter"


def _is_submitter_identity_noise(entity_name: str, submitter: str, summary: str, reference_url: str) -> bool:
    entity_key = _normalize_reference_text(entity_name)
    submitter_key = _normalize_reference_text(submitter)
    if not entity_key or not submitter_key:
        return False
    if entity_key == submitter_key or entity_key in submitter_key or submitter_key in entity_key:
        return True
    if "longboardfella" in entity_key and "longboardfella" in submitter_key:
        return True
    if submitter_key and submitter_key in _normalize_reference_text(summary):
        return True
    if submitter_key and submitter_key in _normalize_reference_text(reference_url):
        return True
    return False


def _normalize_reference_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("@", " ").replace("/", " ").replace("|", " ").split())


def _prefer_subject_entity(item: dict) -> str:
    candidates = [
        item.get("primary_entity_name"),
        item.get("target_name"),
        item.get("organisation_name"),
        item.get("subject_org_hint"),
        item.get("stakeholder_name"),
        item.get("org_name"),
    ]
    submitter = _submitter_label(item)
    reference_url = _first_non_empty(
        item.get("url"),
        item.get("primary_url"),
        item.get("content") if _looks_like_url(item.get("content")) else "",
    )
    summary = _first_non_empty(
        item.get("text_note"),
        item.get("summary"),
        item.get("snippet"),
        item.get("content") if not _looks_like_url(item.get("content")) else "",
        item.get("raw_text"),
    )
    for candidate in candidates:
        cleaned = _first_non_empty(candidate)
        if cleaned and not _is_submitter_identity_noise(cleaned, submitter, summary, reference_url):
            return cleaned
    return _first_non_empty(*candidates, "Unspecified entity")


def _is_document_like_submission(item: dict) -> bool:
    combined = " ".join(
        str(item.get(key) or "")
        for key in ("source_type", "intel_type", "title", "headline", "subject")
    ).lower()
    return any(token in combined for token in ("org chart", "strategic", "annual report", "document", "pdf", "plan", "report"))


def _brief_provenance_label(item: dict) -> str:
    submitter = _submitter_label(item)
    date = _first_non_empty(item.get("intel_date"), item.get("submitted_at"), item.get("note_date"), item.get("received_at"))
    parts = []
    if submitter and submitter != "Unknown submitter":
        parts.append(f"Submitted by {submitter}")
    if date:
        parts.append(date)
    return "; ".join(parts)


def _format_submitted_intel_section(submitted_intel: list[dict]) -> str:
    if not submitted_intel:
        return "## Submitted Intelligence\nNo submitted intel found for the selected scope and date range."

    intel_lines = []
    for item in submitted_intel:
        entity_name = _prefer_subject_entity(item)
        title = _first_non_empty(
            item.get("title"),
            item.get("headline"),
            item.get("subject"),
            item.get("intel_type"),
            "Submitted intelligence",
        )
        date = _first_non_empty(item.get("intel_date"), item.get("submitted_at"), item.get("note_date"), item.get("received_at"))
        submitter = _submitter_label(item)
        source_type = _first_non_empty(item.get("source_type"), item.get("intel_type"), "submitted_intelligence")
        reference_url = _first_non_empty(
            item.get("url"),
            item.get("primary_url"),
            item.get("content") if _looks_like_url(item.get("content")) else "",
        )
        summary = _first_non_empty(
            item.get("text_note"),
            item.get("summary"),
            item.get("snippet"),
            item.get("content") if not _looks_like_url(item.get("content")) else "",
            item.get("raw_text"),
        )
        provenance = _brief_provenance_label(item)
        if _is_document_like_submission(item) and entity_name and entity_name != "Unspecified entity":
            headline = f"{entity_name}: {title}"
        else:
            headline = title
        line = f"- **{headline}**"
        details = []
        if entity_name and entity_name != "Unspecified entity":
            details.append(f"Entity: {entity_name}")
        if date:
            details.append(f"Date: {date}")
        if source_type:
            details.append(f"Type: {source_type}")
        if details:
            line += "\n  " + " | ".join(details)
        if provenance:
            line += f"\n  Provenance: {provenance}"
        if reference_url:
            line += f"\n  Reference: {reference_url}"
        if summary:
            line += f"\n  Summary: {summary[:1200]}"
        intel_lines.append(line)

    return f"## Submitted Intelligence ({len(submitted_intel)} items)\n" + "\n".join(intel_lines)


def _format_web_intel_section(web_intel: Any) -> str:
    if isinstance(web_intel, dict) and web_intel.get("deferred"):
        return (
            "## Internet-Sourced Intelligence\n"
            "Web research was deferred due to high target count. "
            "Please incorporate any publicly available recent news about the listed entities."
        )

    if not isinstance(web_intel, list) or not web_intel:
        return "## Internet-Sourced Intelligence\nNo internet-sourced research was provided for the selected scope and date range."

    web_lines = []
    for index, item in enumerate(web_intel):
        if not isinstance(item, dict):
            continue
        headline = _first_non_empty(
            item.get("headline"),
            item.get("title"),
            item.get("name"),
            item.get("type"),
            f"Web signal {index + 1}",
        )
        date = _first_non_empty(item.get("date"), item.get("published_at"), item.get("received_at"))
        source = _first_non_empty(item.get("source"), item.get("source_name"))
        reference_url = _first_non_empty(item.get("url"), item.get("primary_url"))
        summary = _first_non_empty(item.get("snippet"), item.get("summary"), item.get("content"), item.get("text"))

        line = f"- **{headline}**"
        details = []
        if date:
            details.append(f"Date: {date}")
        if source:
            details.append(f"Source: {source}")
        if reference_url:
            details.append(f"Reference: {reference_url}")
        if details:
            line += "\n  " + " | ".join(details)
        if summary:
            line += f"\n  Summary: {summary[:1500]}"
        web_lines.append(line)

    if not web_lines:
        return "## Internet-Sourced Intelligence\nNo internet-sourced research was provided for the selected scope and date range."

    return f"## Internet-Sourced Intelligence ({len(web_lines)} items)\n" + "\n".join(web_lines)


def _source_separation_instruction() -> str:
    return (
        "Keep submitted intelligence and internet-sourced intelligence as two clearly separate evidence streams. "
        "When both are present, include a dedicated 'Submitted Intelligence' section and a separate "
        "'Internet-Sourced Intelligence' section. "
        "For submitted intelligence, treat the submitter as provenance only unless the content itself is actually about that person or organisation. "
        "Do not create a stakeholder profile, highlight section, or strategic inference about a submitter solely because they emailed the material in. "
        "If the submitted item is a document such as an org chart or strategic plan, focus on the document subject/entity and use the submitter only as a brief provenance note. "
        "For internet-sourced intelligence, cite the source/publication and URL whenever provided. "
        "Do not present submitted intelligence as independently verified public reporting unless the same claim also appears in the provided web research."
    )


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    del input_path, job

    org_name = input_data.get("org_name", "")
    date_range = input_data.get("date_range", {})
    report_scope = input_data.get("report_scope", {})
    submitted_intel = input_data.get("submitted_intel", [])
    web_intel = input_data.get("web_intel", [])
    synthesis_instructions = input_data.get("synthesis_instructions", "")
    strategic_context = input_data.get("strategic_context", {})
    source_mode = input_data.get("source_mode", "submitted_only")
    geography = _resolve_report_geography(report_scope, strategic_context)
    geography_instruction = (
        f"The report's primary geography is {geography}. Prioritise that market heavily. "
        f"Do not treat foreign organisations, regulators, or sector conditions as representative unless they are "
        f"directly tied to {geography} operations or were explicitly requested in the provided data."
    )
    source_separation_instruction = _source_separation_instruction()

    if progress_cb:
        progress_cb(10, "Building synthesis prompt", "prepare")

    # Build the data payload for the LLM
    data_sections = []

    # Strategic context
    if strategic_context:
        ctx_parts = []
        if strategic_context.get("description"):
            ctx_parts.append(f"Description: {strategic_context['description']}")
        if strategic_context.get("industries"):
            ctx_parts.append(f"Industries: {strategic_context['industries']}")
        if strategic_context.get("key_themes"):
            ctx_parts.append(f"Key Themes: {strategic_context['key_themes']}")
        if strategic_context.get("strategic_objectives"):
            ctx_parts.append(f"Strategic Objectives: {strategic_context['strategic_objectives']}")
        if ctx_parts:
            data_sections.append("## Organisation Strategic Context\n" + "\n".join(ctx_parts))

    # Report scope summary
    industries = report_scope.get("industries", [])
    organisations = report_scope.get("organisations", [])
    stakeholders = report_scope.get("stakeholders", [])
    industry_profiles = report_scope.get("industry_profiles", [])

    scope_lines = []
    if industries:
        scope_lines.append(f"Industries: {', '.join(industries)}")
    if organisations:
        scope_lines.append(f"Organisations: {', '.join(o['name'] for o in organisations)}")
    if stakeholders:
        scope_lines.append(f"Stakeholders: {', '.join(s['name'] + ' (' + (s.get('role') or s.get('employer') or '') + ')' for s in stakeholders)}")
    if scope_lines:
        data_sections.append("## Report Scope\n" + "\n".join(scope_lines))
    data_sections.append("## Geographic Scope\n" + geography_instruction)

    # Organisation profiles
    if organisations:
        org_lines = []
        for o in organisations:
            parts = [f"- **{o['name']}**"]
            if o.get("industry"):
                parts.append(f"  Industry: {o['industry']}")
            if o.get("description"):
                parts.append(f"  Description: {o['description']}")
            org_lines.append("\n".join(parts))
        data_sections.append("## Organisation Profiles\n" + "\n".join(org_lines))

    # Stakeholder profiles
    if stakeholders:
        stk_lines = []
        for s in stakeholders:
            parts = [f"- **{s['name']}**"]
            if s.get("employer"):
                parts.append(f"  Employer: {s['employer']}")
            if s.get("role"):
                parts.append(f"  Role: {s['role']}")
            stk_lines.append("\n".join(parts))
        data_sections.append("## Stakeholder Profiles\n" + "\n".join(stk_lines))

    # Submitted intelligence
    data_sections.append(_format_submitted_intel_section(submitted_intel))

    # Web research results
    data_sections.append(_format_web_intel_section(web_intel))

    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before LLM synthesis")

    if progress_cb:
        progress_cb(30, "Calling LLM for synthesis", "synthesise")

    # Build prompts
    system_prompt = (
        "You are a senior intelligence analyst producing a weekly intelligence report in Markdown.\n"
        "Structure the report with clear headings and sections.\n"
        "Thread intelligence together — identify patterns, connections, and implications.\n"
        "Big picture first (sector/industry), then drill into organisation and individual detail.\n"
        "Be analytical, not just descriptive. Highlight what matters and why.\n"
        "If information is thin for a section, say so briefly rather than padding.\n"
        "Do not fabricate facts. Use only the provided data.\n"
        "Do not include confidence scores or metadata labels in the output.\n"
        f"{source_separation_instruction}\n"
        f"{geography_instruction}"
    )

    user_prompt = f"{synthesis_instructions}\n\n" + "\n\n".join(data_sections)

    # Try Ollama first, fall back to Anthropic
    report_text = None
    llm_provider = "ollama"
    llm_model = ""

    # Resolve preferred model
    preferred_model = _resolve_ollama_model()

    try:
        response = requests.post(
            _OLLAMA_URL,
            json={
                "model": preferred_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=_OLLAMA_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            report_text = (data.get("message") or {}).get("content", "")
            llm_model = preferred_model
            logger.info("Weekly report synthesised via Ollama (%s)", preferred_model)
        else:
            logger.warning("Ollama returned %d for weekly report synthesis", response.status_code)
    except Exception as e:
        logger.warning("Ollama unavailable for weekly report: %s", e)

    # Fallback to Anthropic
    if not report_text:
        if progress_cb:
            progress_cb(50, "Ollama unavailable, falling back to Claude API", "fallback")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            try:
                fallback_model = os.environ.get("CORTEX_WATCH_ANTHROPIC_MODEL") or _ANTHROPIC_FALLBACK_MODEL
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": fallback_model,
                        "max_tokens": 8192,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                    timeout=180,
                )
                if resp.status_code == 200:
                    resp_data = resp.json()
                    for block in resp_data.get("content", []):
                        if block.get("type") == "text":
                            report_text = block["text"]
                            break
                    llm_provider = "anthropic"
                    llm_model = fallback_model
                    logger.info("Weekly report synthesised via Anthropic fallback (%s)", fallback_model)
                else:
                    logger.error("Anthropic fallback failed with %d: %s", resp.status_code, resp.text[:500])
            except Exception as e:
                logger.error("Anthropic fallback error: %s", e)

    if not report_text:
        raise RuntimeError("Failed to synthesise weekly report — both Ollama and Anthropic unavailable")

    if progress_cb:
        progress_cb(90, "Writing report output", "output")

    # Write report to output file
    output_dir = Path(os.environ.get("QUEUE_OUTPUT_DIR", "/tmp"))
    job_id = input_data.get("_job_id", "unknown")
    output_filename = f"weekly_report_{org_name}_{date_range.get('start', '')}_{date_range.get('end', '')}.md"
    output_path = output_dir / output_filename
    output_path.write_text(report_text, encoding="utf-8")

    signal_count = len(submitted_intel) + (len(web_intel) if isinstance(web_intel, list) else 0)
    profiles_covered = len(organisations) + len(stakeholders) + len(industry_profiles)

    if progress_cb:
        progress_cb(100, f"Weekly report complete — {signal_count} items, {profiles_covered} profiles", "done")

    return {
        "output_data": {
            "status": "generated",
            "org_name": org_name,
            "signal_count": signal_count,
            "profiles_covered": profiles_covered,
            "output_path": str(output_path),
            "period_start": date_range.get("start", ""),
            "period_end": date_range.get("end", ""),
            "report_depth": "detailed",
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "source_mode": source_mode,
            "submitted_intel_count": len(submitted_intel),
            "web_intel_count": len(web_intel) if isinstance(web_intel, list) else 0,
            "submitter_count": len(
                {
                    _submitter_label(item)
                    for item in submitted_intel
                    if isinstance(item, dict) and _submitter_label(item) != "Unknown submitter"
                }
            ),
            "industries": industries,
        },
        "output_file": output_path,
    }
