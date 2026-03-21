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
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/chat"
_OLLAMA_MODEL = "qwen2.5:14b"
_OLLAMA_TIMEOUT = 600  # 10 minutes for large reports
_ANTHROPIC_FALLBACK_MODEL = "claude-haiku-4-5-20251001"


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
    if submitted_intel:
        intel_lines = []
        for item in submitted_intel:
            name = item.get("stakeholder_name", "Unknown")
            title = item.get("title", "")
            content = item.get("content", "")
            note = item.get("text_note", "")
            date = item.get("intel_date", "")
            line = f"- **{name}** ({date}): {title}"
            if content:
                line += f"\n  URL: {content}"
            if note:
                line += f"\n  Notes: {note}"
            intel_lines.append(line)
        data_sections.append(f"## Submitted Intelligence ({len(submitted_intel)} items)\n" + "\n".join(intel_lines))
    else:
        data_sections.append("## Submitted Intelligence\nNo submitted intel found for the selected scope and date range.")

    # Web research results
    if web_intel and not isinstance(web_intel, dict):
        # web_intel is a list of {type, content} blocks from inline Sonnet search
        web_texts = [item.get("content", "") for item in web_intel if item.get("content")]
        if web_texts:
            data_sections.append("## Web Research Results\n" + "\n---\n".join(web_texts))
    elif isinstance(web_intel, dict) and web_intel.get("deferred"):
        data_sections.append(
            "## Web Research\nWeb research was deferred due to high target count. "
            "Please incorporate any publicly available recent news about the listed entities."
        )

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
        "Do not include confidence scores or metadata labels in the output."
    )

    user_prompt = f"{synthesis_instructions}\n\n" + "\n\n".join(data_sections)

    # Try Ollama first, fall back to Anthropic
    report_text = None
    llm_provider = "ollama"
    llm_model = ""

    # Resolve preferred model
    preferred_model = (
        os.environ.get("CORTEX_WEEKLY_OLLAMA_MODEL")
        or os.environ.get("CORTEX_WATCH_OLLAMA_MODEL")
        or _OLLAMA_MODEL
    ).strip()

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
            "industries": industries,
        },
        "output_file": output_path,
    }
