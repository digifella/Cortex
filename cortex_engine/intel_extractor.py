from __future__ import annotations

import hashlib
import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cortex_engine.org_chart_extractor import extract_org_chart_structured, looks_like_org_chart_attachment
from cortex_engine.stakeholder_signal_matcher import match_signal_to_profiles, normalize_lookup
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore
from cortex_engine.textifier import DocumentTextifier
from cortex_engine.utils import convert_windows_to_wsl_path

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_LINKEDIN_PERSON_VIEW_RE = re.compile(r"View\s+(.+?)(?:'s|\u2019s)\s+profile", re.IGNORECASE)
_LINKEDIN_COMPANY_VIEW_RE = re.compile(r"View\s+company:\s+(.+)", re.IGNORECASE)
_LINKEDIN_URL_RE = re.compile(r"https?://[^\s<>]+linkedin\.com/(?:in|company)/[^\s<>]+", re.IGNORECASE)
_CONTACT_CARD_MARKERS = ("linkedin", "twitter", "zoom", "acn", "ph", "phone", "mobile", "www.", "http", "https")
_EMAIL_WRAPPER_PREFIX_RE = re.compile(r"^\s*(?:from|sent|to|cc|bcc|subject|date)\s*:\s*", re.IGNORECASE)
_EMAIL_WRAPPER_NOISE_RE = re.compile(
    r"^\s*(?:-+\s*original message\s*-+|begin forwarded message:|forwarded message|external email|caution: external email)\s*$",
    re.IGNORECASE,
)
_MAIL_SUBJECT_PREFIX_RE = re.compile(r"^\s*((?:re|fw|fwd)\s*:\s*)+", re.IGNORECASE)
_EMAIL_TRIAGE_MODELS = (
    os.environ.get("CORTEX_INTEL_TRIAGE_MODEL", "").strip(),
    os.environ.get("LOCAL_LLM_SYNTHESIS_MODEL", "").strip(),
    "qwen2.5:72b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",
    "nemotron:70b-instruct-q4_K_M",
    "mistral-small3.2:latest",
    "qwen3.5:9b",
    "qwen3.5:9b-q8_0",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:14b",
)
_PRIVATE_SUBJECT_MARKERS = ("private", "sensitive", "confidential")
_PRIVATE_SUBJECT_RE = re.compile(r"(?i)\b(?:private|sensitive|confidential)\b")
_EXTERNAL_SUBJECT_RE = re.compile(r"(?i)^(?:open|public)\b")
_PUBLIC_COMPLEX_DOC_TYPES = {"annual_report", "strategic_plan", "org_chart"}
_ANTHROPIC_HAIKU_MODEL = os.environ.get("CORTEX_INTEL_ANTHROPIC_DEFAULT_MODEL", "").strip() or "claude-haiku-4-5-20251001"
_ANTHROPIC_DOCUMENT_MODEL = os.environ.get("CORTEX_INTEL_ANTHROPIC_DOCUMENT_MODEL", "").strip() or "claude-sonnet-4-6"
_LOCAL_INTEL_MODEL_CANDIDATES = (
    "qwen3:30b",
    "mistral-small3.2:latest",
    "nemotron-3-nano:latest",
    "qwen2.5:72b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",
    "nemotron:70b-instruct-q4_K_M",
    "mistral-small:latest",
    "qwen2.5:14b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q8_0",
    "llama3.2:3b-instruct-q8_0",
)


def _anthropic_client():
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in config.env")
    return anthropic.Anthropic(api_key=api_key)


def _ollama_client(timeout: float = 120):
    import ollama

    return ollama.Client(timeout=timeout)


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


def _is_weak_document_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    normalized = normalize_lookup(value)
    if not normalized:
        return True
    weak_markers = (
        "image could not be described",
        "image description timed out",
        "vision model unavailable",
        "logo icon omitted",
    )
    if any(marker in normalized for marker in weak_markers):
        return True
    return len(re.findall(r"[A-Za-z]", value)) < 80


def _should_force_pdf_ocr_fallback(filename: str, text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if not looks_like_org_chart_attachment(filename, value):
        return False

    normalized = normalize_lookup(value)
    generic_chart_markers = (
        "the image is an organizational chart",
        "the image is an organisational chart",
        "shows the structure with various departments and roles",
        "shows the structure with departments and roles",
        "page 1 image summary",
    )
    if any(marker in normalized for marker in generic_chart_markers):
        return True

    structured = extract_org_chart_structured(
        attachment_texts=[f"Attachment: {filename}\n{value}"],
        attachment_summaries=[{"filename": filename, "status": "processed"}],
        employer_hint="",
    )
    return not bool(structured.get("people"))


def _run_capture_text(command: List[str], timeout: int = 45) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return str(result.stdout or "").strip()


def _extract_pdf_text_local(path_obj: Path) -> str:
    parts: List[str] = []
    if shutil.which("pdftotext"):
        text = _run_capture_text(["pdftotext", str(path_obj), "-"])
        if text:
            parts.append(text)

    if shutil.which("pdftoppm") and shutil.which("tesseract"):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                prefix = Path(temp_dir) / "page"
                subprocess.run(
                    ["pdftoppm", "-f", "1", "-l", "3", "-png", str(path_obj), str(prefix)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                )
                ocr_blocks: List[str] = []
                for image_path in sorted(Path(temp_dir).glob("page-*.png")):
                    text = _ocr_image_text(image_path)
                    if text:
                        ocr_blocks.append(text)
                if ocr_blocks:
                    parts.append("\n\n".join(ocr_blocks))
        except Exception:
            pass

    merged = "\n\n".join(part.strip() for part in parts if str(part or "").strip()).strip()
    return merged


def _clean_subject_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    while True:
        updated = _MAIL_SUBJECT_PREFIX_RE.sub("", text).strip()
        if updated == text:
            break
        text = updated
    text = _PRIVATE_SUBJECT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|:")
    return text


def _looks_like_contact_card_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    marker_hits = sum(1 for marker in _CONTACT_CARD_MARKERS if marker in lowered)
    email_hits = len(_EMAIL_RE.findall(value))
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if marker_hits >= 3:
        return True
    if marker_hits >= 2 and email_hits >= 1:
        return True
    return marker_hits >= 1 and email_hits >= 1 and len(lines) <= 12 and len(value) <= 1200


def _should_suppress_email_wrapper_for_document(payload: Dict[str, Any], attachment_texts: List[str]) -> bool:
    attachments = payload.get("attachments") or []
    has_document = any(str(item.get("kind") or "").strip().lower() == "document" for item in attachments)
    if not has_document:
        return False
    attachment_text = "\n\n".join(str(item or "").strip() for item in attachment_texts if str(item or "").strip())
    if len(attachment_text) < 1500:
        return False
    wrapper_text = "\n\n".join(
        part for part in [
            str(payload.get("raw_text") or "").strip(),
            _strip_html(payload.get("html_text", "")),
        ]
        if part
    ).strip()
    return _looks_like_contact_card_text(wrapper_text)


def _clean_email_wrapper_text(text: str) -> str:
    lines = [str(line or "").rstrip() for line in str(text or "").splitlines()]
    cleaned: List[str] = []
    wrapper_run = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        normalized = normalize_lookup(line)
        if _EMAIL_WRAPPER_NOISE_RE.match(line):
            wrapper_run += 1
            continue
        if _EMAIL_WRAPPER_PREFIX_RE.match(line):
            wrapper_run += 1
            continue
        if normalized in {"fw", "fwd", "re", "fwd:", "fw:", "re:"}:
            wrapper_run += 1
            continue
        if wrapper_run >= 2 and len(line) < 4:
            continue
        cleaned.append(line)
        wrapper_run = 0
    return "\n".join(cleaned).strip()


def _triage_attachment_defaults(attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected: List[str] = []
    ignored: List[str] = []
    for item in attachments:
        filename = str(item.get("filename") or "").strip()
        lowered = normalize_lookup(filename)
        kind = str(item.get("kind") or "").strip().lower()
        if not filename:
            continue
        if kind == "image" and any(token in lowered for token in ("logo", "outlook", "signature", "icon", "header")):
            ignored.append(filename)
            continue
        selected.append(filename)
    return {
        "include_attachment_filenames": selected,
        "ignore_attachment_filenames": ignored,
    }


def _extract_chat_content(response: Any) -> str:
    if isinstance(response, dict):
        if isinstance(response.get("message"), dict):
            return str(response["message"].get("content") or "").strip()
        return str(response.get("response") or "").strip()
    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "").strip()
    return str(getattr(response, "response", "") or "").strip()


def _choose_email_triage_model(client: Any) -> str:
    try:
        listing = client.list()
    except Exception:
        return ""
    if isinstance(listing, dict):
        models = listing.get("models") or []
    else:
        models = getattr(listing, "models", []) or []
    names: List[str] = []
    for item in models:
        if isinstance(item, dict):
            name = str(item.get("model") or item.get("name") or "").strip()
        else:
            name = str(getattr(item, "model", "") or getattr(item, "name", "") or "").strip()
        if name:
            names.append(name)
    for candidate in _EMAIL_TRIAGE_MODELS:
        if candidate in names:
            return candidate
    return ""


def _default_email_triage(payload: Dict[str, Any]) -> Dict[str, Any]:
    attachments = list(payload.get("attachments") or [])
    attachment_defaults = _triage_attachment_defaults(attachments)
    wrapper_text = "\n\n".join(
        part for part in [
            str(payload.get("raw_text") or "").strip(),
            _strip_html(payload.get("html_text", "")),
        ]
        if part
    ).strip()
    actionable_body = _clean_email_wrapper_text(str(payload.get("raw_text") or "").strip())
    if _looks_like_contact_card_text(wrapper_text):
        actionable_body = ""
    return {
        "processing_mode": "attachments_only" if any(str(item.get("kind") or "").strip().lower() == "document" for item in attachments) else "body_plus_attachments",
        "clean_subject": _clean_subject_text(str(payload.get("subject") or "").strip()),
        "actionable_body_text": actionable_body,
        "wrapper_text": _clean_email_wrapper_text(wrapper_text) if actionable_body else wrapper_text,
        "signature_text": wrapper_text if _looks_like_contact_card_text(wrapper_text) else "",
        "include_attachment_filenames": attachment_defaults["include_attachment_filenames"],
        "ignore_attachment_filenames": attachment_defaults["ignore_attachment_filenames"],
        "confidence": 0.35,
        "used_model": "",
    }


def _triage_email_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    triage = _default_email_triage(payload)
    attachments = list(payload.get("attachments") or [])
    body_text = "\n\n".join(
        part for part in [
            str(payload.get("raw_text") or "").strip(),
            _strip_html(payload.get("html_text", "")),
        ]
        if part
    ).strip()
    if not body_text and not attachments:
        return triage
    if len(body_text) < 80 and len(attachments) <= 1:
        return triage
    try:
        import ollama

        client = ollama.Client(timeout=30)
        model = _choose_email_triage_model(client)
        if not model:
            return triage
        attachment_meta = [
            {
                "filename": str(item.get("filename") or "").strip(),
                "kind": str(item.get("kind") or "").strip(),
                "mime_type": str(item.get("mime_type") or "").strip(),
                "size_bytes": int(item.get("size_bytes") or 0) if str(item.get("size_bytes") or "").strip() else 0,
            }
            for item in attachments
        ]
        prompt = (
            "You are triaging a forwarded intelligence email before extraction.\n"
            "Separate actionable analyst content from forwarding wrappers and sender signatures.\n"
            "Return ONLY valid JSON with keys:\n"
            "{\"processing_mode\":\"body_only|attachments_only|body_plus_attachments\","
            "\"clean_subject\":\"string\","
            "\"actionable_body_text\":\"string\","
            "\"wrapper_text\":\"string\","
            "\"signature_text\":\"string\","
            "\"include_attachment_filenames\":[\"...\"],"
            "\"ignore_attachment_filenames\":[\"...\"],"
            "\"confidence\":0.0}\n"
            "Rules:\n"
            "- If the email body is mostly forwarding junk or a signature block and there are substantive document attachments, choose attachments_only.\n"
            "- Prefer dropping sender signatures and contact cards unless they are clearly the intended intelligence.\n"
            "- Keep multiple attachments if they all look relevant.\n"
            "- Preserve only substantive analyst note text in actionable_body_text.\n"
            "- clean_subject must remove Re:/Fwd: prefixes and transport junk.\n\n"
            f"Subject: {payload.get('subject', '')}\n\n"
            f"Body:\n{body_text[:6000]}\n\n"
            f"Attachments JSON:\n{json.dumps(attachment_meta, ensure_ascii=True)}"
        )
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 800},
        )
        parsed = _extract_json_object(_extract_chat_content(response))
        if not parsed:
            return triage
        mode = str(parsed.get("processing_mode") or "").strip()
        if mode in {"body_only", "attachments_only", "body_plus_attachments"}:
            triage["processing_mode"] = mode
        triage["clean_subject"] = _clean_subject_text(str(parsed.get("clean_subject") or triage["clean_subject"] or ""))
        triage["actionable_body_text"] = str(parsed.get("actionable_body_text") or "").strip()
        triage["wrapper_text"] = str(parsed.get("wrapper_text") or "").strip()
        triage["signature_text"] = str(parsed.get("signature_text") or "").strip()
        known_names = {str(item.get("filename") or "").strip() for item in attachments if str(item.get("filename") or "").strip()}
        selected = [str(item).strip() for item in parsed.get("include_attachment_filenames") or [] if str(item).strip() in known_names]
        ignored = [str(item).strip() for item in parsed.get("ignore_attachment_filenames") or [] if str(item).strip() in known_names]
        if selected:
            triage["include_attachment_filenames"] = selected
        triage["ignore_attachment_filenames"] = ignored
        try:
            triage["confidence"] = float(parsed.get("confidence") or triage["confidence"])
        except Exception:
            pass
        triage["used_model"] = model
        return triage
    except Exception as exc:
        logger.debug("email triage unavailable, falling back to rules: %s", exc)
        return triage


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
                text = ""
            else:
                text = textifier.textify_file(str(path_obj))
            if suffix == ".pdf" and (
                _is_weak_document_text(text)
                or _should_force_pdf_ocr_fallback(filename, text)
            ):
                fallback_text = _extract_pdf_text_local(path_obj)
                if fallback_text:
                    text = fallback_text
            if not text:
                summary["warning"] = "No document text extracted"
                return "", summary
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


def _subject_requests_local_only(subject: str) -> bool:
    lowered = normalize_lookup(subject)
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in _PRIVATE_SUBJECT_MARKERS)


def _subject_requests_external_llm(subject: str) -> bool:
    text = str(subject or "").strip()
    if not text:
        return False
    while True:
        updated = _MAIL_SUBJECT_PREFIX_RE.sub("", text).strip()
        if updated == text:
            break
        text = updated
    return bool(_EXTERNAL_SUBJECT_RE.search(text))


def _infer_policy_doc_type(payload: Dict[str, Any], attachments: List[Dict[str, Any]]) -> str:
    subject = _clean_subject_text(str(payload.get("subject") or ""))
    subject_lower = normalize_lookup(subject)
    names = " ".join(str(item.get("filename") or "").strip() for item in attachments)
    names_lower = normalize_lookup(names)
    combined = " ".join(part for part in [subject_lower, names_lower] if part).strip()
    if not combined:
        return ""
    if any(token in combined for token in ("org chart", "organisation chart", "organizational chart")):
        return "org_chart"
    if "annual report" in combined:
        return "annual_report"
    if any(
        token in combined
        for token in (
            "strategic plan",
            "corporate strategy",
            "corporate plan",
            "strategy 20",
            "strategy202",
        )
    ):
        return "strategic_plan"
    return ""


def _preferred_local_extract_models(requested_model: str = "") -> List[str]:
    candidates: List[str] = []
    for candidate in (
        requested_model,
        os.environ.get("CORTEX_INTEL_LOCAL_MODEL", ""),
        os.environ.get("INTEL_LOCAL_MODEL", ""),
        *_LOCAL_INTEL_MODEL_CANDIDATES,
        os.environ.get("LOCAL_LLM_SYNTHESIS_MODEL", ""),
    ):
        name = str(candidate or "").strip()
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def _select_local_extract_model(requested_model: str = "") -> str:
    candidates = _preferred_local_extract_models(requested_model)
    try:
        client = _ollama_client(timeout=30)
        listing = client.list()
        models = listing.get("models") if isinstance(listing, dict) else getattr(listing, "models", []) or []
        installed: set[str] = set()
        for item in models:
            if isinstance(item, dict):
                name = str(item.get("model") or item.get("name") or "").strip()
            else:
                name = str(getattr(item, "model", "") or getattr(item, "name", "") or "").strip()
            if name:
                installed.add(name)
        for candidate in candidates:
            if candidate in installed:
                return candidate
    except Exception:
        pass
    return candidates[0] if candidates else "qwen2.5:14b-instruct-q4_K_M"


def _choose_llm_policy(payload: Dict[str, Any], attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
    subject = str(payload.get("original_subject") or payload.get("subject") or "").strip()
    doc_type = _infer_policy_doc_type(payload, attachments)
    local_only = _subject_requests_local_only(subject)
    external_ok = _subject_requests_external_llm(subject)
    if doc_type in _PUBLIC_COMPLEX_DOC_TYPES and external_ok and not local_only:
        return {
            "provider": "anthropic",
            "model": _ANTHROPIC_DOCUMENT_MODEL,
            "doc_type": doc_type,
            "local_only": False,
            "reason": f"subject_marked_open_{doc_type}",
        }
    return {
        "provider": "ollama",
        "model": _select_local_extract_model(),
        "doc_type": doc_type,
        "local_only": local_only,
        "reason": "subject_marked_private" if local_only else "default_local_intel",
    }


def _build_structured_extract_prompts(payload: Dict[str, Any], combined_text: str) -> Tuple[str, str]:
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
    return system_prompt, user_prompt


def _call_anthropic_extract(payload: Dict[str, Any], combined_text: str, model: str) -> Tuple[Dict[str, Any], str]:
    client = _anthropic_client()
    system_prompt, user_prompt = _build_structured_extract_prompts(payload, combined_text)
    response = client.messages.create(
        model=model,
        max_tokens=1800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in response.content:
        text = getattr(block, "text", "")
        if text:
            parts.append(text)
    return _extract_json_object("\n".join(parts)), model


def _call_haiku_extract(payload: Dict[str, Any], combined_text: str) -> Dict[str, Any]:
    parsed, _model = _call_anthropic_extract(payload, combined_text, _ANTHROPIC_HAIKU_MODEL)
    return parsed


def _call_local_extract(payload: Dict[str, Any], combined_text: str, model: str = "") -> Tuple[Dict[str, Any], str]:
    system_prompt, user_prompt = _build_structured_extract_prompts(payload, combined_text)
    doc_type = str(_infer_policy_doc_type(payload, list(payload.get("attachments") or [])) or "").strip().lower()
    timeout_seconds = float(
        os.environ.get(
            "CORTEX_INTEL_LOCAL_DOCUMENT_TIMEOUT_SECONDS" if doc_type in _PUBLIC_COMPLEX_DOC_TYPES else "CORTEX_INTEL_LOCAL_TIMEOUT_SECONDS",
            "240" if doc_type in _PUBLIC_COMPLEX_DOC_TYPES else "120",
        )
        or ("240" if doc_type in _PUBLIC_COMPLEX_DOC_TYPES else "120")
    )
    last_error: Optional[Exception] = None
    for candidate in _preferred_local_extract_models(model):
        try:
            client = _ollama_client(timeout=timeout_seconds)
            response = client.chat(
                model=candidate,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.1, "num_predict": 2200},
            )
            parsed = _extract_json_object(_extract_chat_content(response))
            if parsed:
                return parsed, candidate
        except Exception as exc:
            last_error = exc
            logger.warning("intel_extract local model %s failed: %s", candidate, exc)
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No local extraction model available")


def _prepare_anthropic_image_bytes(path_obj: Path, mime_type: str) -> Tuple[bytes, str]:
    original_bytes = path_obj.read_bytes()
    normalized_mime = str(mime_type or "").strip().lower() or "image/png"
    max_edge = int(float(os.getenv("CORTEX_ANTHROPIC_IMAGE_MAX_EDGE_PX", "4096") or 4096))
    max_input_bytes = int(
        float(os.getenv("CORTEX_ANTHROPIC_IMAGE_MAX_INPUT_BYTES", str(5 * 1024 * 1024)) or (5 * 1024 * 1024))
    )
    if not original_bytes or max_edge <= 0:
        return original_bytes, normalized_mime

    try:
        from PIL import Image
    except Exception:
        return original_bytes, normalized_mime

    try:
        with Image.open(io.BytesIO(original_bytes)) as img:
            width, height = img.size
            if max(width, height) <= max_edge and len(original_bytes) <= max_input_bytes:
                return original_bytes, normalized_mime

            working = img.copy()
            if max(width, height) > max_edge:
                working.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

            preserve_png = normalized_mime == "image/png" or path_obj.suffix.lower() == ".png"
            out = io.BytesIO()
            prepared_mime = normalized_mime

            if preserve_png:
                if working.mode not in {"RGB", "RGBA", "L", "P"}:
                    working = working.convert("RGBA")
                working.save(out, format="PNG", optimize=True)
                prepared_mime = "image/png"
            else:
                if working.mode != "RGB":
                    working = working.convert("RGB")
                working.save(out, format="JPEG", quality=88, optimize=True)
                prepared_mime = "image/jpeg"

            prepared_bytes = out.getvalue()
            if len(prepared_bytes) > max_input_bytes:
                jpeg_out = io.BytesIO()
                jpeg_ready = working if working.mode == "RGB" else working.convert("RGB")
                jpeg_ready.save(jpeg_out, format="JPEG", quality=82, optimize=True)
                jpeg_bytes = jpeg_out.getvalue()
                if jpeg_bytes:
                    prepared_bytes = jpeg_bytes
                    prepared_mime = "image/jpeg"

            if prepared_bytes:
                logger.info(
                    "Prepared Anthropic image from %sx%s (%s KB) to %sx%s (%s KB, %s)",
                    width,
                    height,
                    max(1, len(original_bytes) // 1024),
                    working.size[0],
                    working.size[1],
                    max(1, len(prepared_bytes) // 1024),
                    prepared_mime,
                )
                return prepared_bytes, prepared_mime
    except Exception as exc:
        logger.debug("Could not preprocess Anthropic image %s: %s", path_obj, exc)

    return original_bytes, normalized_mime


def _call_anthropic_image_extract(payload: Dict[str, Any], attachment: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], str]:
    stored_path = _normalize_attachment_path(attachment.get("stored_path") or "")
    if not stored_path:
        return {}, model
    path_obj = Path(stored_path)
    if not path_obj.exists():
        return {}, model

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

    prepared_bytes, prepared_mime_type = _prepare_anthropic_image_bytes(path_obj, mime_type)
    data = base64.b64encode(prepared_bytes).decode("utf-8")
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
        model=model,
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
                            "media_type": prepared_mime_type,
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
    return _extract_json_object("\n".join(parts)), model


def _call_haiku_image_extract(payload: Dict[str, Any], attachment: Dict[str, Any]) -> Dict[str, Any]:
    parsed, _model = _call_anthropic_image_extract(payload, attachment, _ANTHROPIC_HAIKU_MODEL)
    return parsed


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
                "extraction_method": str(item.get("extraction_method") or "").strip(),
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
                "extraction_method": str(item.get("extraction_method") or "").strip(),
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


def _assign_org_chart_employers(
    people: List[Dict[str, Any]],
    organisations: List[Dict[str, Any]],
) -> None:
    org_names = [str(item.get("name") or "").strip() for item in organisations or [] if str(item.get("name") or "").strip()]
    if len(org_names) != 1:
        return
    employer = org_names[0]
    for item in people:
        if str(item.get("extraction_method") or "").strip() != "org_chart_heuristic":
            continue
        if str(item.get("current_employer") or "").strip():
            continue
        item["current_employer"] = employer


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


def _clean_feed_lines(text: str) -> List[str]:
    cleaned: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", str(raw_line or "").strip())
        if not line:
            continue
        cleaned.append(line)
    return cleaned


def _is_noise_line(line: str) -> bool:
    normalized = normalize_lookup(line)
    if not normalized:
        return True
    if normalized in {
        "feed post",
        "sort by top",
        "sort by",
        "like",
        "comment",
        "repost",
        "send",
        "promoted",
    }:
        return True
    if re.match(r"^\d+\s*(comments?|reactions?|reposts?)$", normalized):
        return True
    if re.match(r"^\d+\s*[smhdwymo]+\s*$", normalized):
        return True
    if normalized.startswith("view company"):
        return False
    if normalized.startswith("view ") and normalized.endswith(" profile"):
        return False
    return False


def _normalize_linkedin_role_line(line: str, name: str = "") -> str:
    cleaned = str(line or "")
    if name:
        cleaned = re.sub(re.escape(name), " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("•", " ").replace("·", " ")
    cleaned = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" |-")


def _normalize_linkedin_entity_name(value: str) -> str:
    cleaned = str(value or "")
    cleaned = re.sub(r"<https?://[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("[", " ").replace("]", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" |,-")


def _linkedin_block_signal_type(lines: List[str], index: int) -> str:
    nearby = " ".join(lines[max(0, index - 3): index + 12]).lower()
    if "commented on this" in nearby or re.search(r"\bcommented\b", nearby):
        return "commented"
    if re.search(r"\bliked this\b|\breacted\b|\bfollow(?:ed|s)?\b|\bpromoted\b", nearby):
        return "passive"
    return "authored"


def _looks_like_authored_content(lines: List[str], index: int, entity_name: str) -> bool:
    substantive = 0
    for look_ahead in lines[index + 1:index + 12]:
        normalized = normalize_lookup(look_ahead)
        if not normalized:
            continue
        if normalize_lookup(entity_name) and normalized == normalize_lookup(entity_name):
            continue
        if _is_noise_line(look_ahead):
            continue
        if _LINKEDIN_URL_RE.search(look_ahead):
            continue
        if "followers" in normalized or "connections" in normalized:
            continue
        if re.match(r"^\d+\s*/\s*\d+$", normalized):
            continue
        if len(normalized) < 25:
            continue
        substantive += 1
        if substantive >= 1:
            return True
    return False


def _extract_linkedin_feed_structured(text: str) -> Dict[str, Any]:
    lines = _clean_feed_lines(text)
    people: List[Dict[str, Any]] = []
    organisations: List[Dict[str, Any]] = []
    seen_people: set[str] = set()
    seen_orgs: set[str] = set()

    for index, line in enumerate(lines):
        person_match = _LINKEDIN_PERSON_VIEW_RE.search(line)
        company_match = _LINKEDIN_COMPANY_VIEW_RE.search(line)
        if person_match:
            name = _normalize_linkedin_entity_name(person_match.group(1))
            signal_type = _linkedin_block_signal_type(lines, index)
            if signal_type == "passive":
                continue
            if signal_type != "commented" and not _looks_like_authored_content(lines, index, name):
                continue
            key = normalize_lookup(name)
            if not name or key in seen_people:
                continue
            seen_people.add(key)
            linkedin_url = ""
            role_parts: List[str] = []
            for look_ahead in lines[index + 1:index + 8]:
                if _LINKEDIN_URL_RE.search(look_ahead) and "/in/" in look_ahead:
                    linkedin_url = _LINKEDIN_URL_RE.search(look_ahead).group(0)
                    continue
                if normalize_lookup(look_ahead) == key:
                    continue
                if _is_noise_line(look_ahead):
                    if role_parts:
                        break
                    continue
                if look_ahead in {"•", "-", "·"}:
                    continue
                cleaned_role = _normalize_linkedin_role_line(look_ahead, name=name)
                if not cleaned_role:
                    continue
                role_parts.append(cleaned_role)
                if len(role_parts) >= 2:
                    break
            role_text = " | ".join(role_parts[:2]).strip(" |")
            people.append(
                {
                    "name": name,
                    "current_role": role_text,
                    "linkedin_url": linkedin_url,
                    "confidence": 0.72 if role_text or linkedin_url else 0.58,
                    "evidence": f"LinkedIn feed block: {line[:140]}",
                    "extraction_method": "linkedin_feed_heuristic",
                    "signal_kind": signal_type,
                }
            )
            continue

        if company_match:
            name = _normalize_linkedin_entity_name(company_match.group(1))
            signal_type = _linkedin_block_signal_type(lines, index)
            if signal_type == "passive":
                continue
            if signal_type != "commented" and not _looks_like_authored_content(lines, index, name):
                continue
            key = normalize_lookup(name)
            if not name or key in seen_orgs:
                continue
            seen_orgs.add(key)
            company_url = ""
            descriptor_parts: List[str] = []
            for look_ahead in lines[index + 1:index + 7]:
                if _LINKEDIN_URL_RE.search(look_ahead) and "/company/" in look_ahead:
                    company_url = _LINKEDIN_URL_RE.search(look_ahead).group(0)
                    continue
                if normalize_lookup(look_ahead) == key:
                    continue
                if _is_noise_line(look_ahead):
                    if descriptor_parts:
                        break
                    continue
                descriptor_parts.append(look_ahead)
                if len(descriptor_parts) >= 2:
                    break
            organisations.append(
                {
                    "name": name,
                    "website_url": company_url,
                    "industry": descriptor_parts[0] if descriptor_parts else "",
                    "confidence": 0.68 if company_url or descriptor_parts else 0.56,
                    "evidence": f"LinkedIn company block: {line[:140]}",
                    "extraction_method": "linkedin_feed_heuristic",
                    "signal_kind": signal_type,
                }
            )

    summary_parts: List[str] = []
    if people:
        summary_parts.append(f"Heuristic LinkedIn parsing recovered {len(people)} people")
    if organisations:
        summary_parts.append(f"{len(organisations)} organisations")
    return {
        "people": people,
        "organisations": organisations,
        "emails": [],
        "career_events": [],
        "summary": ". ".join(summary_parts),
    }


def _candidate_signal(
    payload: Dict[str, Any],
    target_type: str,
    name: str,
    employer: str = "",
    note: str = "",
    primary_url: str = "",
) -> Dict[str, Any]:
    base = " ".join(
        part for part in [
            payload.get("subject", ""),
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
        "primary_url": primary_url or payload.get("primary_url", ""),
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
    triage = _triage_email_payload(payload)
    selected_names = {str(item).strip() for item in triage.get("include_attachment_filenames") or [] if str(item).strip()}
    ignored_names = {str(item).strip() for item in triage.get("ignore_attachment_filenames") or [] if str(item).strip()}
    attachments = list(payload.get("attachments") or [])
    if selected_names:
        attachments = [item for item in attachments if str(item.get("filename") or "").strip() in selected_names]
    if ignored_names:
        attachments = [item for item in attachments if str(item.get("filename") or "").strip() not in ignored_names]
    llm_policy = _choose_llm_policy(payload, attachments)
    attachment_texts: List[str] = []
    attachment_summaries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    textifier = _maybe_textifier() if attachments else None
    image_structured_items: List[Dict[str, Any]] = []

    for attachment in attachments:
        text, summary = _read_attachment_text(attachment, textifier)
        attachment_summaries.append(summary)
        if text:
            attachment_texts.append(f"Attachment: {summary.get('filename') or 'unnamed'}\n{text}")
        elif summary.get("warning"):
            warnings.append(f"{summary.get('filename') or 'attachment'}: {summary['warning']}")
        try:
            if (
                llm_policy.get("provider") == "anthropic"
                and str(summary.get("kind") or "").lower() == "image"
                and summary.get("stored_path")
            ):
                structured_image, actual_model = _call_anthropic_image_extract(payload, summary, str(llm_policy.get("model") or _ANTHROPIC_DOCUMENT_MODEL))
                if structured_image:
                    image_structured_items.append(structured_image)
                llm_policy["actual_model"] = actual_model
        except Exception as exc:
            warnings.append(f"{summary.get('filename') or 'attachment'}: image extraction failed: {exc}")
            logger.warning("intel_extract image extraction failed for %s: %s", summary.get("filename", ""), exc)

    email_body_text = str(triage.get("actionable_body_text") or "").strip()
    html_body_text = ""
    if not email_body_text:
        email_body_text = _clean_email_wrapper_text(str(payload.get("raw_text") or "").strip())
        html_body_text = _clean_email_wrapper_text(_strip_html(payload.get("html_text", "")))
    if str(triage.get("processing_mode") or "").strip() == "attachments_only" or _should_suppress_email_wrapper_for_document({**payload, "attachments": attachments}, attachment_texts):
        email_body_text = ""
        html_body_text = ""
    elif str(triage.get("processing_mode") or "").strip() == "body_only":
        attachment_texts = []

    combined_text = "\n\n".join(
        part for part in [
            triage.get("clean_subject") or payload.get("subject", ""),
            email_body_text,
            html_body_text,
            "\n\n".join(attachment_texts),
        ] if str(part or "").strip()
    ).strip()

    regex_emails = _regex_emails(combined_text)
    structured: Dict[str, Any] = {}
    if combined_text:
        try:
            if llm_policy.get("provider") == "anthropic":
                structured, actual_model = _call_anthropic_extract(
                    payload,
                    combined_text,
                    str(llm_policy.get("model") or _ANTHROPIC_DOCUMENT_MODEL),
                )
                llm_policy["actual_model"] = actual_model
            else:
                structured, actual_model = _call_local_extract(
                    payload,
                    combined_text,
                    str(llm_policy.get("model") or ""),
                )
                llm_policy["actual_model"] = actual_model
        except Exception as exc:
            if llm_policy.get("provider") == "anthropic":
                warnings.append(f"Anthropic extraction failed: {exc}")
                logger.warning("intel_extract anthropic call failed: %s", exc)
                try:
                    structured, actual_model = _call_local_extract(payload, combined_text)
                    llm_policy["fallback_provider"] = "ollama"
                    llm_policy["fallback_model"] = actual_model
                    llm_policy["actual_model"] = actual_model
                except Exception as fallback_exc:
                    warnings.append(f"Local fallback extraction failed: {fallback_exc}")
                    logger.warning("intel_extract local fallback failed: %s", fallback_exc)
            else:
                warnings.append(f"Local extraction failed: {exc}")
                logger.warning("intel_extract local ollama call failed: %s", exc)
    heuristic_structured = _extract_linkedin_feed_structured(combined_text)
    org_chart_structured = extract_org_chart_structured(
        attachment_texts=attachment_texts,
        attachment_summaries=attachment_summaries,
        employer_hint=str(payload.get("parsed_candidate_employer") or "").strip(),
    )
    structured = _merge_structured_results(structured, heuristic_structured, org_chart_structured, *image_structured_items)

    people = _normalize_people(structured.get("people"))
    organisations = _normalize_organisations(structured.get("organisations"))
    _assign_org_chart_employers(people, organisations)
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
        is_heuristic_feed = str(candidate.get("extraction_method") or "").strip() == "linkedin_feed_heuristic"
        signal = _candidate_signal(
            payload,
            target_type,
            candidate_name,
            employer,
            candidate.get("evidence", ""),
            candidate.get("linkedin_url") or candidate.get("website_url") or "",
        )
        matches = match_signal_to_profiles(signal, profiles, threshold=0.75 if is_heuristic_feed else 0.45)
        entities.append(_build_entity_record(candidate))
        matches_summary.append(
            {
                "candidate_name": candidate_name,
                "target_type": target_type,
                "matched": bool(matches),
                "matches": matches,
            }
        )
        if matches and not is_heuristic_feed:
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
        "subject": triage.get("clean_subject") or payload.get("subject", ""),
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
        "email_triage": triage,
        "llm_policy": llm_policy,
    }

    output_path: Optional[Path] = None
    markdown = _build_markdown_summary(result)
    if markdown.strip():
        with tempfile.NamedTemporaryFile(mode="w", suffix="_intel_extract.md", delete=False, encoding="utf-8") as handle:
            handle.write(markdown)
            output_path = Path(handle.name)
    return result, output_path
