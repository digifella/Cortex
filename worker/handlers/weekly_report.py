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
_ANTHROPIC_WEB_MODEL = os.environ.get("CORTEX_WEEKLY_ANTHROPIC_WEB_MODEL", "").strip() or "claude-sonnet-4-20250514"
_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
_DEFAULT_SUPPRESSED_SUBMITTER_ALIASES = (
    "paul cooper",
    "longboardfella",
    "longboardfella consulting",
)


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


def _suppressed_submitter_aliases() -> tuple[str, ...]:
    configured = str(os.environ.get("CORTEX_WEEKLY_SUPPRESSED_SUBMITTERS") or "").strip()
    aliases = list(_DEFAULT_SUPPRESSED_SUBMITTER_ALIASES)
    if configured:
        aliases.extend(part.strip().lower() for part in configured.split(",") if part.strip())
    deduped: list[str] = []
    for alias in aliases:
        cleaned = _normalize_reference_text(alias)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def _is_suppressed_submitter_identity(*values: Any) -> bool:
    aliases = _suppressed_submitter_aliases()
    for value in values:
        normalized = _normalize_reference_text(value)
        if not normalized:
            continue
        for alias in aliases:
            if alias and (alias in normalized or normalized in alias):
                return True
    return False


def _generic_submitter_label(display_name: str, email: str) -> str:
    combined = f"{display_name} {email}".lower()
    if "escient" in combined:
        return "Escient contributor"
    if "longboardfella" in combined:
        return "External contributor"
    return "Known contributor"


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
        "gemma4:26b",
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


def _internet_enriched_mode(source_mode: str, web_intel: Any) -> bool:
    if str(source_mode or "").strip().lower() == "submitted_and_internet":
        return True
    if isinstance(web_intel, list) and any(isinstance(item, dict) for item in web_intel):
        return True
    if isinstance(web_intel, dict) and web_intel.get("deferred"):
        return True
    return False


def _preferred_synthesis_provider(source_mode: str, web_intel: Any) -> str:
    return "anthropic" if _internet_enriched_mode(source_mode, web_intel) else "ollama"


def _count_web_intel_items(web_intel: Any) -> int:
    if not isinstance(web_intel, list):
        return 0
    return sum(1 for item in web_intel if isinstance(item, dict))


def _deferred_web_intel_notice(web_intel: Any) -> Optional[dict]:
    if not isinstance(web_intel, dict) or not web_intel.get("deferred"):
        return None
    target_count = int(web_intel.get("target_count") or 0)
    targets = [str(item).strip() for item in (web_intel.get("targets") or []) if str(item).strip()]
    summary = "Target-specific web research was deferred upstream due to the number of requested targets."
    if target_count:
        summary += f" Deferred target count: {target_count}."
    if targets:
        summary += f" Deferred targets included: {', '.join(targets[:8])}."
    return {
        "headline": "Deferred target web research",
        "source": "Cortex weekly report preflight",
        "summary": summary,
        "type": "deferred_web_research",
    }


def _merge_web_intel(existing_web_intel: Any, sector_sweep_items: list[dict]) -> Any:
    if not sector_sweep_items:
        return existing_web_intel
    merged: list[dict] = [dict(item) for item in sector_sweep_items if isinstance(item, dict)]
    if isinstance(existing_web_intel, list):
        merged.extend(dict(item) for item in existing_web_intel if isinstance(item, dict))
        return merged
    deferred_notice = _deferred_web_intel_notice(existing_web_intel)
    if deferred_notice:
        merged.append(deferred_notice)
    return merged


def _build_sector_sweep_prompt(
    org_name: str,
    geography: str,
    report_scope: dict,
    date_range: dict,
) -> str:
    industries = [str(item).strip() for item in (report_scope.get("industries") or []) if str(item).strip()]
    organisations = [str((item or {}).get("name") or "").strip() for item in (report_scope.get("organisations") or []) if str((item or {}).get("name") or "").strip()]
    stakeholders = [str((item or {}).get("name") or "").strip() for item in (report_scope.get("stakeholders") or []) if str((item or {}).get("name") or "").strip()]
    period_start = str(date_range.get("start") or "").strip()
    period_end = str(date_range.get("end") or "").strip()
    industry_label = ", ".join(industries) if industries else "the selected market"
    org_label = ", ".join(organisations[:8]) if organisations else org_name or "the selected organisations"
    stakeholder_label = ", ".join(stakeholders[:8]) if stakeholders else "the selected stakeholders"
    return (
        f"Run one final sector sweep for {org_name or 'the selected organisation scope'} in {geography} "
        f"covering {period_start} to {period_end}.\n\n"
        f"Focus industries: {industry_label}\n"
        f"Priority organisations: {org_label}\n"
        f"Priority stakeholders: {stakeholder_label}\n\n"
        "Find material public developments that should influence a weekly intelligence report even if they were not already captured in prior web research. "
        "Prioritise sector-wide moves, regulatory or policy developments, major partnerships, funding/investment shifts, infrastructure announcements, "
        "technology strategy signals, customer/service changes, and notable leadership or operating-model developments.\n\n"
        "Return a concise markdown briefing with short bullets. Each bullet should include: headline, date if known, source/publication, URL if available, "
        "and one-sentence why-it-matters. Stay tightly scoped to the requested geography and sector."
    )


def _run_anthropic_web_sector_sweep(
    org_name: str,
    geography: str,
    report_scope: dict,
    date_range: dict,
) -> list[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return []
    prompt = _build_sector_sweep_prompt(org_name, geography, report_scope, date_range)
    max_uses = 6 if report_scope.get("industries") else 4
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": _ANTHROPIC_WEB_MODEL,
                "max_tokens": 2400,
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}],
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        if resp.status_code != 200:
            logger.warning("Anthropic sector sweep failed with %d: %s", resp.status_code, resp.text[:500])
            return []
        data = resp.json()
    except Exception as exc:
        logger.warning("Anthropic sector sweep unavailable: %s", exc)
        return []

    text_blocks = [
        str(block.get("text") or "").strip()
        for block in data.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text") or "").strip()
    ]
    if not text_blocks:
        return []

    headline_bits = [str(item).strip() for item in (report_scope.get("industries") or []) if str(item).strip()]
    headline = f"{', '.join(headline_bits[:2])} sector sweep" if headline_bits else f"{geography} sector sweep"
    summary = "\n\n".join(text_blocks).strip()
    return [
        {
            "headline": headline,
            "source": "Anthropic web search",
            "summary": summary[:4000],
            "type": "sector_sweep",
            "date": str(date_range.get("end") or "").strip(),
        }
    ]


def _run_ollama_synthesis(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    preferred_model = _resolve_ollama_model()
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
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned {response.status_code} for weekly report synthesis")
    data = response.json()
    report_text = str((data.get("message") or {}).get("content") or "").strip()
    if not report_text:
        raise RuntimeError("Ollama returned an empty weekly report response")
    return report_text, preferred_model


def _run_anthropic_synthesis(system_prompt: str, user_prompt: str, model: str = "") -> tuple[str, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY for weekly report synthesis")
    selected_model = str(model or "").strip() or _ANTHROPIC_FALLBACK_MODEL
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": selected_model,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic returned {resp.status_code} for weekly report synthesis: {resp.text[:500]}")
    resp_data = resp.json()
    report_text = ""
    for block in resp_data.get("content", []):
        if block.get("type") == "text":
            report_text = str(block.get("text") or "").strip()
            if report_text:
                break
    if not report_text:
        raise RuntimeError("Anthropic returned an empty weekly report response")
    return report_text, selected_model


def _submitter_label(item: dict) -> str:
    display_name = _first_non_empty(item.get("submitted_by_name"), item.get("from_name"))
    email = _first_non_empty(item.get("submitted_by"), item.get("from_email"))
    if _is_suppressed_submitter_identity(display_name, email):
        return _generic_submitter_label(display_name, email)
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

    def _render_web_item(item: dict, index: int) -> str:
        if not isinstance(item, dict):
            return ""
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
        return line

    targeted_lines = []
    sector_sweep_lines = []
    for index, item in enumerate(web_intel):
        line = _render_web_item(item, index)
        if not line:
            continue
        bucket = sector_sweep_lines if str((item or {}).get("type") or "").strip().lower() == "sector_sweep" else targeted_lines
        bucket.append(line)

    if not targeted_lines and not sector_sweep_lines:
        return "## Internet-Sourced Intelligence\nNo internet-sourced research was provided for the selected scope and date range."

    blocks = [f"## Internet-Sourced Intelligence ({len(targeted_lines) + len(sector_sweep_lines)} items)"]
    if targeted_lines:
        if sector_sweep_lines:
            blocks.append("### Targeted Web Research")
        blocks.append("\n".join(targeted_lines))
    if sector_sweep_lines:
        blocks.append("### Final Sector Sweep")
        blocks.append("\n".join(sector_sweep_lines))
    return "\n".join(block for block in blocks if block).strip()


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


def _evidence_priority_instruction(submitted_intel: list[dict], web_intel: Any) -> str:
    if not submitted_intel:
        return (
            "Internet-sourced intelligence is the primary evidence stream for this run. "
            "Still keep sector sweep context separate from targeted web findings and avoid irrelevant public-sector filler."
        )
    instruction = (
        "Submitted intelligence is the primary evidence stream for this report. "
        "Build the Organisation Analysis and Stakeholder Highlights from submitted intelligence first, then use internet-sourced intelligence only to corroborate, update, or add missing sector context. "
        "Do not replace specific submitted signals with generic public web commentary. "
        "Do not say strategic directions are unclear when submitted strategic plans, annual reports, org charts, or explicit opportunity intel provide concrete organisation-level signals. "
        "Ignore broad public-sector or regulatory news unless it is directly relevant to the selected industries, named organisations, or Escient implications."
    )
    if isinstance(web_intel, list) and any(str((item or {}).get("type") or "").strip().lower() == "sector_sweep" for item in web_intel if isinstance(item, dict)):
        instruction += " Treat the final sector sweep as secondary enrichment, not the dominant narrative."
    return instruction


def _report_structure_instruction() -> str:
    return (
        "Always preserve these report sections in order: "
        "'SECTOR/INDUSTRY OVERVIEW', 'ORGANISATION ANALYSIS', 'STAKEHOLDER HIGHLIGHTS', "
        "'SUBMITTED INTELLIGENCE', 'INTERNET-SOURCED INTELLIGENCE', 'CROSS-CUTTING THEMES', and 'RECOMMENDED ACTIONS'. "
        "If submitted intelligence includes multiple organisations, give each organisation its own subsection before broader sector commentary."
    )


def _required_output_template() -> str:
    return (
        "Return Markdown using this exact structure and exact heading text:\n"
        "Weekly Intelligence Report - <scope>\n"
        "Period: <start-end>\n"
        "---\n"
        "SECTOR/INDUSTRY OVERVIEW\n"
        "<analysis>\n"
        "---\n"
        "ORGANISATION ANALYSIS\n"
        "<one subsection per organisation when submitted intelligence exists>\n"
        "---\n"
        "STAKEHOLDER HIGHLIGHTS\n"
        "<stakeholder, provenance, and relationship signals>\n"
        "---\n"
        "SUBMITTED INTELLIGENCE\n"
        "<submitted items, provenance, why it matters>\n"
        "---\n"
        "INTERNET-SOURCED INTELLIGENCE\n"
        "<targeted web research first, final sector sweep second>\n"
        "---\n"
        "CROSS-CUTTING THEMES\n"
        "<patterns>\n"
        "---\n"
        "RECOMMENDED ACTIONS\n"
        "<actions>\n"
    )


def _report_scope_names(report_scope: dict) -> list[str]:
    names: list[str] = []
    for bucket in ("organisations", "stakeholders", "industry_profiles"):
        for item in report_scope.get(bucket, []) or []:
            if not isinstance(item, dict):
                continue
            name = _first_non_empty(item.get("name"), item.get("canonical_name"))
            if name and name not in names:
                names.append(name)
    return names


def _submitted_entity_names(submitted_intel: list[dict]) -> list[str]:
    names: list[str] = []
    for item in submitted_intel or []:
        if not isinstance(item, dict):
            continue
        name = _prefer_subject_entity(item)
        if name and name != "Unspecified entity" and name not in names:
            names.append(name)
    return names


def _organisation_coverage_instruction(report_scope: dict, submitted_intel: list[dict]) -> str:
    submitted_names = _submitted_entity_names(submitted_intel)
    scope_names = _report_scope_names(report_scope)
    names = [name for name in submitted_names if name in scope_names] or submitted_names or scope_names
    if not names:
        return ""
    return (
        "In 'ORGANISATION ANALYSIS', include a separate subsection for each of these names where evidence exists: "
        + ", ".join(names[:12])
        + ". Do not collapse them into one generic sector paragraph."
    )


def _stakeholder_guidance_instruction(submitted_intel: list[dict]) -> str:
    if not submitted_intel:
        return (
            "In 'STAKEHOLDER HIGHLIGHTS', keep the section brief if genuine stakeholder intelligence is thin, "
            "but do not invent people or relationship claims."
        )
    return (
        "In 'STAKEHOLDER HIGHLIGHTS', include real submitter-originated relationship or account signals when they are analytically useful, "
        "such as repeated intelligence gathering by Escient or partner contributors, imminent opportunity intelligence, or evidence of coordinated account-mapping. "
        "Treat submitters as provenance and internal account-signal sources, not as external news subjects, unless the submitted material is actually about them. "
        "Do not surface suppressed submitter identities by name in stakeholder highlights; keep them generic."
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
    sector_sweep_items: list[dict] = []
    effective_web_intel: Any = web_intel
    geography_instruction = (
        f"The report's primary geography is {geography}. Prioritise that market heavily. "
        f"Do not treat foreign organisations, regulators, or sector conditions as representative unless they are "
        f"directly tied to {geography} operations or were explicitly requested in the provided data."
    )
    source_separation_instruction = _source_separation_instruction()

    if progress_cb:
        progress_cb(10, "Building synthesis prompt", "prepare")

    if _internet_enriched_mode(source_mode, web_intel):
        if progress_cb:
            progress_cb(20, "Running final internet sector sweep", "sector_sweep")
        sector_sweep_items = _run_anthropic_web_sector_sweep(org_name, geography, report_scope, date_range)
        effective_web_intel = _merge_web_intel(web_intel, sector_sweep_items)

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
    data_sections.append(_format_web_intel_section(effective_web_intel))

    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before LLM synthesis")

    if progress_cb:
        if _preferred_synthesis_provider(source_mode, effective_web_intel) == "anthropic":
            progress_cb(30, "Calling Claude for internet-enriched synthesis", "synthesise")
        else:
            progress_cb(30, "Calling Ollama for local synthesis", "synthesise")

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
        f"{_report_structure_instruction()}\n"
        f"{_evidence_priority_instruction(submitted_intel, effective_web_intel)}\n"
        f"{_organisation_coverage_instruction(report_scope, submitted_intel)}\n"
        f"{_stakeholder_guidance_instruction(submitted_intel)}\n"
        f"{geography_instruction}"
    )

    user_prompt = (
        f"{synthesis_instructions}\n\n"
        f"{_required_output_template()}\n\n"
        "Important output rules:\n"
        "- Do not omit any required section.\n"
        "- Keep 'SUBMITTED INTELLIGENCE' and 'INTERNET-SOURCED INTELLIGENCE' as explicit standalone sections.\n"
        "- Put targeted web research ahead of the final sector sweep inside the internet-sourced section.\n"
        "- If submitted intelligence is materially richer than web research, let submitted intelligence dominate the organisation and action sections.\n"
        "- Use provenance and account-intel signals in stakeholder highlights when supported by submitted intelligence.\n\n"
        + "\n\n".join(data_sections)
    )

    # Choose provider order based on whether the report is internet-enriched.
    report_text = ""
    llm_provider = ""
    llm_model = ""
    synth_provider = _preferred_synthesis_provider(source_mode, effective_web_intel)
    anthropic_model = os.environ.get("CORTEX_WATCH_ANTHROPIC_MODEL") or (
        _ANTHROPIC_WEB_MODEL if synth_provider == "anthropic" else _ANTHROPIC_FALLBACK_MODEL
    )
    synthesis_attempts = [
        ("anthropic", anthropic_model),
        ("ollama", ""),
    ] if synth_provider == "anthropic" else [
        ("ollama", ""),
        ("anthropic", anthropic_model),
    ]

    synthesis_errors: list[str] = []
    for provider_name, provider_model in synthesis_attempts:
        try:
            if provider_name == "anthropic":
                text, selected_model = _run_anthropic_synthesis(system_prompt, user_prompt, provider_model)
                report_text = text
                llm_provider = "anthropic"
                llm_model = selected_model
                logger.info("Weekly report synthesised via Anthropic (%s)", selected_model)
            else:
                text, selected_model = _run_ollama_synthesis(system_prompt, user_prompt)
                report_text = text
                llm_provider = "ollama"
                llm_model = selected_model
                logger.info("Weekly report synthesised via Ollama (%s)", selected_model)
            if report_text:
                break
        except Exception as exc:
            synthesis_errors.append(f"{provider_name}: {exc}")
            logger.warning("Weekly report %s synthesis failed: %s", provider_name, exc)
            if progress_cb and provider_name == synthesis_attempts[0][0]:
                next_provider = synthesis_attempts[1][0] if len(synthesis_attempts) > 1 else ""
                if next_provider == "anthropic":
                    progress_cb(50, "Local synthesis timed out or failed, falling back to Claude API", "fallback")
                elif next_provider == "ollama":
                    progress_cb(50, "Claude synthesis failed, falling back to Ollama", "fallback")

    if not report_text:
        raise RuntimeError(
            "Failed to synthesise weekly report — " + "; ".join(synthesis_errors or ["no synthesis provider succeeded"])
        )

    if progress_cb:
        progress_cb(90, "Writing report output", "output")

    # Write report to output file
    output_dir = Path(os.environ.get("QUEUE_OUTPUT_DIR", "/tmp"))
    job_id = input_data.get("_job_id", "unknown")
    output_filename = f"weekly_report_{org_name}_{date_range.get('start', '')}_{date_range.get('end', '')}.md"
    output_path = output_dir / output_filename
    output_path.write_text(report_text, encoding="utf-8")

    web_intel_count = _count_web_intel_items(effective_web_intel)
    signal_count = len(submitted_intel) + web_intel_count
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
            "web_intel_count": web_intel_count,
            "sector_sweep_count": len(sector_sweep_items),
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
