from __future__ import annotations

import hashlib
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cortex_engine.stakeholder_signal_matcher import match_signal_to_profiles, normalize_lookup
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore
from cortex_engine.textifier import DocumentTextifier
from cortex_engine.utils import convert_windows_to_wsl_path

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _anthropic_client():
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in config.env")
    return anthropic.Anthropic(api_key=api_key)


def _strip_html(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"(?is)<(script|style).*?>.*?</(script|style)>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_attachment_path(path_text: str) -> str:
    raw = str(path_text or "").strip()
    if not raw:
        return ""
    return raw if os.path.exists("/.dockerenv") else convert_windows_to_wsl_path(raw)


def _regex_emails(text: str) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    seen = set()
    for match in _EMAIL_RE.findall(text or ""):
        email = match.strip().lower()
        if email in seen:
            continue
        seen.add(email)
        found.append(
            {
                "email": email,
                "owner_hint": "",
                "confidence": 0.99,
                "evidence": f"Regex email extraction found {email}",
            }
        )
    return found


def _maybe_textifier() -> Optional[DocumentTextifier]:
    try:
        return DocumentTextifier(use_vision=True)
    except Exception as exc:
        logger.warning("intel_extract textifier unavailable: %s", exc)
        return None


def _ocr_image_text(path_obj: Path) -> str:
    if not path_obj.exists():
        return ""
    if not shutil.which("tesseract"):
        return ""

    candidates: List[str] = []
    commands = (
        ["tesseract", str(path_obj), "stdout", "--psm", "6"],
        ["tesseract", str(path_obj), "stdout", "--psm", "11"],
    )
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        if result.returncode != 0:
            continue
        text = re.sub(r"\s+\n", "\n", result.stdout or "").strip()
        if text:
            candidates.append(text)
    if not candidates:
        return ""
    candidates.sort(key=lambda item: len(re.findall(r"[A-Za-z0-9@._-]", item)), reverse=True)
    return candidates[0]


def _read_attachment_text(attachment: Dict[str, Any], textifier: Optional[DocumentTextifier]) -> Tuple[str, Dict[str, Any]]:
    filename = str(attachment.get("filename") or "").strip()
    stored_path = _normalize_attachment_path(attachment.get("stored_path") or "")
    mime_type = str(attachment.get("mime_type") or "").strip().lower()
    kind = str(attachment.get("kind") or "").strip().lower()
    summary = {
        "filename": filename,
        "stored_path": stored_path,
        "mime_type": mime_type,
        "kind": kind,
        "status": "skipped",
        "warning": "",
    }
    if not stored_path:
        summary["warning"] = "No stored_path provided"
        return "", summary
    path_obj = Path(stored_path)
    if not path_obj.exists():
        summary["warning"] = "Attachment path not accessible from Cortex worker"
        return "", summary

    suffix = path_obj.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".json", ".csv"}:
            text = path_obj.read_text(encoding="utf-8", errors="ignore")
        elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"} or kind == "image" or mime_type.startswith("image/"):
            ocr_text = _ocr_image_text(path_obj)
            vision_text = ""
            if textifier is not None:
                try:
                    vision_text = textifier.textify_image(str(path_obj))
                except Exception:
                    vision_text = ""
            if ocr_text and vision_text:
                text = f"{ocr_text}\n\n{vision_text}"
            else:
                text = ocr_text or vision_text
            if not text:
                summary["warning"] = "Vision/OCR textifier unavailable"
                return "", summary
        elif suffix in {".pdf", ".docx", ".pptx"}:
            if textifier is None:
                summary["warning"] = "Document textifier unavailable"
                return "", summary
            text = textifier.textify_file(str(path_obj))
        else:
            summary["warning"] = f"Unsupported attachment type: {suffix or mime_type or 'unknown'}"
            return "", summary
    except Exception as exc:
        summary["warning"] = f"Attachment extraction failed: {exc}"
        return "", summary

    text = str(text or "").strip()
    if not text:
        summary["warning"] = "No attachment text extracted"
        return "", summary
    summary["status"] = "processed"
    summary["excerpt"] = text[:300]
    return text, summary


def _extract_json_object(text: str) -> Dict[str, Any]:
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


def _call_haiku_extract(payload: Dict[str, Any], combined_text: str) -> Dict[str, Any]:
    client = _anthropic_client()
    system_prompt = (
        "You extract structured market-intelligence entities from emails, screenshots, and attachment OCR. "
        "Return only valid JSON. Be conservative. Do not invent people or organisations."
    )
    user_prompt = (
        "Extract relevant entities and career/contact details from the source below.\n\n"
        "Return JSON with this shape:\n"
        "{\n"
        '  "people": [{"name":"","current_employer":"","current_role":"","email":"","linkedin_url":"","confidence":0.0,"evidence":""}],\n'
        '  "organisations": [{"name":"","website_url":"","industry":"","parent_entity":"","email":"","confidence":0.0,"evidence":""}],\n'
        '  "emails": [{"email":"","owner_hint":"","confidence":0.0,"evidence":""}],\n'
        '  "career_events": [{"person_name":"","new_employer":"","new_role":"","event_type":"","confidence":0.0,"evidence":""}],\n'
        '  "summary": ""\n'
        "}\n"
        "Use empty arrays when absent. Keep evidence short.\n\n"
        f"Organisation scope: {payload.get('org_name', '')}\n"
        f"Subject: {payload.get('subject', '')}\n"
        f"Candidate hint: {payload.get('parsed_candidate_name', '')}\n"
        f"Employer hint: {payload.get('parsed_candidate_employer', '')}\n"
        f"Primary URL: {payload.get('primary_url', '')}\n\n"
        f"Source text:\n{combined_text[:16000]}"
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in response.content:
        text = getattr(block, "text", "")
        if text:
            parts.append(text)
    return _extract_json_object("\n".join(parts))


def _call_haiku_image_extract(payload: Dict[str, Any], attachment: Dict[str, Any]) -> Dict[str, Any]:
    stored_path = _normalize_attachment_path(attachment.get("stored_path") or "")
    if not stored_path:
        return {}
    path_obj = Path(stored_path)
    if not path_obj.exists():
        return {}

    mime_type = str(attachment.get("mime_type") or "").strip().lower()
    if not mime_type.startswith("image/"):
        suffix = path_obj.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/png")

    data = base64.b64encode(path_obj.read_bytes()).decode("utf-8")
    client = _anthropic_client()
    system_prompt = (
        "You extract visible stakeholder/contact information from screenshots and image attachments. "
        "Read the visible text carefully. Return only valid JSON. Do not invent names, organisations, or emails."
    )
    user_prompt = (
        "Read this screenshot or image attachment and extract any visible structured contact or stakeholder data.\n\n"
        "Focus on visible:\n"
        "- person names\n"
        "- organisations\n"
        "- email addresses\n"
        "- job titles\n"
        "- employer names\n"
        "- website or LinkedIn URLs\n\n"
        "If the image is a list, extract one item per visible row/card where possible.\n"
        "If text is not readable, return empty arrays and explain that in summary.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "people": [{"name":"","current_employer":"","current_role":"","email":"","linkedin_url":"","confidence":0.0,"evidence":""}],\n'
        '  "organisations": [{"name":"","website_url":"","industry":"","parent_entity":"","email":"","confidence":0.0,"evidence":""}],\n'
        '  "emails": [{"email":"","owner_hint":"","confidence":0.0,"evidence":""}],\n'
        '  "career_events": [{"person_name":"","new_employer":"","new_role":"","event_type":"","confidence":0.0,"evidence":""}],\n'
        '  "summary": ""\n'
        "}\n\n"
        f"Organisation scope: {payload.get('org_name', '')}\n"
        f"Email subject: {payload.get('subject', '')}\n"
        f"Attachment filename: {attachment.get('filename', '')}\n"
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1600,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data,
                        },
                    },
                ],
            }
        ],
    )
    parts = []
    for block in response.content:
        text = getattr(block, "text", "")
        if text:
            parts.append(text)
    return _extract_json_object("\n".join(parts))


def _normalized_confidence(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(0.0, min(1.0, parsed))


def _normalize_people(items: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("canonical_name") or "").strip()
        if not name:
            continue
        key = normalize_lookup(name)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "name": name,
                "target_type": "person",
                "current_employer": str(item.get("current_employer") or item.get("employer") or "").strip(),
                "current_role": str(item.get("current_role") or item.get("role") or item.get("title") or "").strip(),
                "email": str(item.get("email") or "").strip(),
                "linkedin_url": str(item.get("linkedin_url") or "").strip(),
                "confidence": _normalized_confidence(item.get("confidence"), 0.65),
                "evidence": str(item.get("evidence") or "").strip(),
            }
        )
    return normalized


def _normalize_organisations(items: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("canonical_name") or "").strip()
        if not name:
            continue
        key = normalize_lookup(name)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "name": name,
                "target_type": "organisation",
                "website_url": str(item.get("website_url") or "").strip(),
                "industry": str(item.get("industry") or "").strip(),
                "parent_entity": str(item.get("parent_entity") or "").strip(),
                "email": str(item.get("email") or "").strip(),
                "confidence": _normalized_confidence(item.get("confidence"), 0.65),
                "evidence": str(item.get("evidence") or item.get("notes") or "").strip(),
            }
        )
    return normalized


def _normalize_emails(model_items: Any, regex_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    regex_emails = {item["email"] for item in regex_items}
    for item in list(model_items or []) + regex_items:
        if not isinstance(item, dict):
            continue
        email = str(item.get("email") or "").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        merged.append(
            {
                "email": email,
                "owner_hint": str(item.get("owner_hint") or item.get("name") or "").strip(),
                "confidence": _normalized_confidence(item.get("confidence"), 0.99 if email in regex_emails else 0.7),
                "evidence": str(item.get("evidence") or "").strip(),
            }
        )
    return merged


def _normalize_career_events(items: Any) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        person_name = str(item.get("person_name") or item.get("name") or "").strip()
        if not person_name:
            continue
        events.append(
            {
                "person_name": person_name,
                "new_employer": str(item.get("new_employer") or item.get("current_employer") or "").strip(),
                "new_role": str(item.get("new_role") or item.get("current_role") or "").strip(),
                "event_type": str(item.get("event_type") or "").strip(),
                "confidence": _normalized_confidence(item.get("confidence"), 0.7),
                "evidence": str(item.get("evidence") or "").strip(),
            }
        )
    return events


def _merge_structured_results(*items: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "people": [],
        "organisations": [],
        "emails": [],
        "career_events": [],
        "summary": "",
    }
    summary_parts: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("people"), list):
            merged["people"].extend(item["people"])
        if isinstance(item.get("organisations"), list):
            merged["organisations"].extend(item["organisations"])
        if isinstance(item.get("emails"), list):
            merged["emails"].extend(item["emails"])
        if isinstance(item.get("career_events"), list):
            merged["career_events"].extend(item["career_events"])
        summary = str(item.get("summary") or "").strip()
        if summary:
            summary_parts.append(summary)
    if summary_parts:
        merged["summary"] = " ".join(summary_parts)
    return merged


def _candidate_signal(payload: Dict[str, Any], target_type: str, name: str, employer: str = "", note: str = "") -> Dict[str, Any]:
    base = " ".join(
        part for part in [
            payload.get("subject", ""),
            payload.get("raw_text", ""),
            note,
        ] if str(part or "").strip()
    )
    signal_id = "extract_" + hashlib.sha1(f"{payload.get('message_id','')}|{target_type}|{name}|{employer}".encode("utf-8")).hexdigest()[:16]
    return {
        "signal_id": signal_id,
        "org_name": payload.get("org_name", ""),
        "target_type": target_type,
        "subject": payload.get("subject", ""),
        "raw_text": base,
        "text_note": payload.get("text_note", ""),
        "primary_url": payload.get("primary_url", ""),
        "parsed_candidate_name": name,
        "parsed_candidate_employer": employer,
        "received_at": payload.get("received_at", ""),
    }


def _artifact_id(prefix: str, *parts: str) -> str:
    material = "|".join(str(part or "").strip() for part in parts)
    return f"{prefix}_{hashlib.sha1(material.encode('utf-8')).hexdigest()[:16]}"


def _build_field_artifacts(
    signal_id: str,
    org_name: str,
    match: Dict[str, Any],
    candidate: Dict[str, Any],
    fields: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    facts: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []
    evidence_excerpt = str(candidate.get("evidence") or "")[:500]
    profile_key = str(match.get("profile_key") or "")
    canonical_name = str(match.get("canonical_name") or "")
    target_type = str(match.get("target_type") or candidate.get("target_type") or "")

    for field_name in fields:
        observed_value = str(candidate.get(field_name) or "").strip()
        if not observed_value:
            continue
        current_value = str(match.get(field_name) or "").strip()
        confidence = _normalized_confidence(candidate.get("confidence"), 0.7)
        facts.append(
            {
                "fact_id": _artifact_id("fact", signal_id, profile_key, field_name, observed_value),
                "signal_id": signal_id,
                "org_name": org_name,
                "profile_key": profile_key,
                "canonical_name": canonical_name,
                "field_name": field_name,
                "observed_value": observed_value,
                "confidence": round(confidence, 4),
                "evidence_excerpt": evidence_excerpt,
            }
        )
        if normalize_lookup(observed_value) == normalize_lookup(current_value):
            continue
        suggestions.append(
            {
                "suggestion_id": _artifact_id("sug", profile_key, field_name, observed_value),
                "signal_id": signal_id,
                "org_name": org_name,
                "profile_key": profile_key,
                "target_type": target_type,
                "canonical_name": canonical_name,
                "field_name": field_name,
                "old_value": current_value,
                "proposed_value": observed_value,
                "confidence": round(confidence, 4),
                "evidence_excerpt": evidence_excerpt,
                "status": "pending",
            }
        )
    return facts, suggestions


def _build_entity_record(candidate: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "name": candidate.get("name", ""),
        "canonical_name": candidate.get("name", ""),
        "target_type": candidate.get("target_type", "person"),
        "current_employer": candidate.get("current_employer", ""),
        "current_role": candidate.get("current_role", ""),
        "email": candidate.get("email", ""),
        "linkedin_url": candidate.get("linkedin_url", ""),
        "website_url": candidate.get("website_url", ""),
        "industry": candidate.get("industry", ""),
        "parent_entity": candidate.get("parent_entity", ""),
        "confidence": _normalized_confidence(candidate.get("confidence"), 0.65),
        "evidence": candidate.get("evidence", ""),
    }
    return {
        key: value
        for key, value in record.items()
        if value is not None and value != "" and value != []
    }


def _build_markdown_summary(result: Dict[str, Any]) -> str:
    lines = [
        "# Intel Extraction Result",
        "",
        f"- Organisation: {result.get('org_name', '')}",
        f"- Intel ID: {result.get('intel_id', '')}",
        f"- Entity count: {len(result.get('entities') or [])}",
        f"- Email count: {len(result.get('emails') or [])}",
        f"- Attachment count: {len(result.get('attachments') or [])}",
        "",
    ]
    if result.get("summary"):
        lines.extend(["## Summary", "", result["summary"], ""])
    if result.get("entities"):
        lines.append("## Entities")
        lines.append("")
        for entity in result["entities"]:
            lines.append(f"- {entity.get('canonical_name') or entity.get('name')} ({entity.get('target_type', '')})")
        lines.append("")
    if result.get("emails"):
        lines.append("## Emails")
        lines.append("")
        for item in result["emails"]:
            lines.append(f"- {item.get('email')}")
        lines.append("")
    return "\n".join(lines)


def extract_intel(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    attachment_texts: List[str] = []
    attachment_summaries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    textifier = _maybe_textifier() if payload.get("attachments") else None
    image_structured_items: List[Dict[str, Any]] = []

    for attachment in payload.get("attachments") or []:
        text, summary = _read_attachment_text(attachment, textifier)
        attachment_summaries.append(summary)
        if text:
            attachment_texts.append(f"Attachment: {summary.get('filename') or 'unnamed'}\n{text}")
        elif summary.get("warning"):
            warnings.append(f"{summary.get('filename') or 'attachment'}: {summary['warning']}")
        try:
            if str(summary.get("kind") or "").lower() == "image" and summary.get("stored_path"):
                structured_image = _call_haiku_image_extract(payload, summary)
                if structured_image:
                    image_structured_items.append(structured_image)
        except Exception as exc:
            warnings.append(f"{summary.get('filename') or 'attachment'}: image extraction failed: {exc}")
            logger.warning("intel_extract image extraction failed for %s: %s", summary.get("filename", ""), exc)

    combined_text = "\n\n".join(
        part for part in [
            payload.get("subject", ""),
            payload.get("raw_text", ""),
            _strip_html(payload.get("html_text", "")),
            "\n\n".join(attachment_texts),
        ] if str(part or "").strip()
    ).strip()

    regex_emails = _regex_emails(combined_text)
    structured: Dict[str, Any] = {}
    if combined_text:
        try:
            structured = _call_haiku_extract(payload, combined_text)
        except Exception as exc:
            warnings.append(f"Haiku extraction failed: {exc}")
            logger.warning("intel_extract anthropic call failed: %s", exc)
    structured = _merge_structured_results(structured, *image_structured_items)

    people = _normalize_people(structured.get("people"))
    organisations = _normalize_organisations(structured.get("organisations"))
    emails = _normalize_emails(structured.get("emails"), regex_emails)
    career_events = _normalize_career_events(structured.get("career_events"))
    summary = str(structured.get("summary") or "").strip()

    profiles = StakeholderSignalStore().list_profiles(org_name=payload["org_name"])
    entities: List[Dict[str, Any]] = []
    matches_summary: List[Dict[str, Any]] = []
    observed_facts: List[Dict[str, Any]] = []
    update_suggestions: List[Dict[str, Any]] = []

    for candidate in people + organisations:
        target_type = str(candidate.get("target_type") or "person")
        candidate_name = str(candidate.get("name") or "").strip()
        employer = str(candidate.get("current_employer") or "").strip()
        signal = _candidate_signal(payload, target_type, candidate_name, employer, candidate.get("evidence", ""))
        matches = match_signal_to_profiles(signal, profiles, threshold=0.45)
        entities.append(_build_entity_record(candidate))
        matches_summary.append(
            {
                "candidate_name": candidate_name,
                "target_type": target_type,
                "matched": bool(matches),
                "matches": matches,
            }
        )
        if matches:
            field_list = ["email", "current_employer", "current_role", "linkedin_url"] if target_type == "person" else ["email", "website_url", "industry", "parent_entity"]
            facts, suggestions = _build_field_artifacts(
                signal_id=signal["signal_id"],
                org_name=payload["org_name"],
                match=matches[0],
                candidate=candidate,
                fields=field_list,
            )
            observed_facts.extend(facts)
            update_suggestions.extend(suggestions)

    result = {
        "status": "extracted",
        "org_name": payload["org_name"],
        "intel_id": payload.get("intel_id", ""),
        "message_id": payload.get("message_id", ""),
        "subject": payload.get("subject", ""),
        "target_type": payload.get("target_type", ""),
        "summary": summary,
        "entity_count": len(entities),
        "entities": entities,
        "contacts": entities,
        "extracted": entities,
        "emails": emails,
        "career_events": career_events,
        "matches": matches_summary,
        "observed_facts": observed_facts,
        "target_update_suggestions": update_suggestions,
        "attachments": attachment_summaries,
        "warnings": warnings,
    }

    output_path: Optional[Path] = None
    markdown = _build_markdown_summary(result)
    if markdown.strip():
        with tempfile.NamedTemporaryFile(mode="w", suffix="_intel_extract.md", delete=False, encoding="utf-8") as handle:
            handle.write(markdown)
            output_path = Path(handle.name)
    return result, output_path
