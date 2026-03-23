from __future__ import annotations

import email
import hashlib
import imaplib
import json
import logging
import os
import re
import smtplib
import ssl
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parsedate_to_datetime, parseaddr
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from cortex_engine.config_manager import ConfigManager
from cortex_engine.email_handlers import (
    CsvProfileImportError,
    CsvProfileImportProcessor,
)
from cortex_engine.industry_classifier import classify_entity_industry
from cortex_engine.intel_extractor import extract_intel
from cortex_engine.intel_note_classifier import classify_mailbox_message
from cortex_engine.intel_note_processor import IntelNoteProcessor
from cortex_engine.org_chart_extractor import looks_like_org_chart_attachment
from cortex_engine.strategic_doc_analyser import clean_strategic_role_label
from cortex_engine.stakeholder_signal_matcher import normalize_lookup
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore, orgs_compatible
from cortex_engine.utils import convert_windows_to_wsl_path

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[^@<>\s]+@[^@<>\s]+\.[^@<>\s]+")
_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_GENERIC_DOC_NAME_RE = re.compile(
    r"^(?:screenshot|screen shot|img|image|photo|picture|scan|attachment|document|file|outlook)(?:[\s_-]+\d.*)?$",
    re.IGNORECASE,
)
_MAIL_SUBJECT_PREFIX_RE = re.compile(r"^\s*((?:re|fw|fwd)\s*:\s*)+", re.IGNORECASE)
_SUBJECT_ENTITY_OVERRIDE_RE = re.compile(
    r"(?i)(?:^|[\s\[\]()|;])(?:entity|org|organisation|organization)\s*:\s*([^|\];,\n]+)"
)
_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*(?:to|[-–])\s*(20\d{2})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_MANAGED_NOTE_SECTIONS = {
    "entities",
    "people",
    "contacts",
    "organisations",
    "organizations",
    "emails",
    "strategic insights",
    "key stakeholders",
    "performance snapshot",
    "performance indicators",
    "kpi focus areas",
}
_KPI_FOCUS_STOPWORDS = {
    "board",
    "college",
    "committee",
    "council",
    "education",
    "financials",
    "finance",
    "governance",
    "implement",
    "operations",
    "people",
    "performance",
    "review",
    "streamline",
    "strategy",
    "support",
    "training",
}
_SUBJECT_ORG_HINT_STOPWORDS = {
    "annual",
    "attachment",
    "call",
    "chart",
    "document",
    "email",
    "fyi",
    "image",
    "industry",
    "intro",
    "introduction",
    "meeting",
    "note",
    "notes",
    "org",
    "organisation",
    "organization",
    "plan",
    "report",
    "screenshot",
    "screen",
    "shot",
    "strategic",
    "strategy",
    "update",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_db_root() -> Path:
    config = ConfigManager().get_config()
    raw_db_path = str(config.get("ai_database_path") or "").strip()
    if not raw_db_path:
        raise RuntimeError("ai_database_path is not configured; Cortex intel mailbox cannot initialize")
    safe_db_path = raw_db_path if os.path.exists("/.dockerenv") else convert_windows_to_wsl_path(raw_db_path)
    root = Path(safe_db_path) / "intel_mailbox"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _decode_header_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(make_header(decode_header(text))).strip()
    except Exception:
        return text


def _sanitize_filename(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("._")
    return cleaned or fallback


def _derive_message_key(message_id: str, raw_bytes: bytes) -> str:
    basis = (message_id or "").strip().lower() or hashlib.sha1(raw_bytes).hexdigest()
    return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:16]


def _extract_emails_from_text(text: str) -> List[str]:
    found = {match.group(0).strip(".,;:()<>[]{}") for match in _EMAIL_RE.finditer(text or "")}
    return sorted(item for item in found if item)


def _extract_urls_from_text(text: str) -> List[str]:
    found = {match.group(0).strip(".,;:()<>[]{}") for match in _URL_RE.finditer(text or "")}
    return sorted(item for item in found if item)


def _looks_like_generic_document_name(value: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip(" -_"))
    if not cleaned:
        return True
    if _GENERIC_DOC_NAME_RE.match(cleaned):
        return True
    alpha_tokens = [token.lower() for token in re.split(r"[\s_-]+", cleaned) if re.search(r"[A-Za-z]", token)]
    if not alpha_tokens:
        return True
    generic_tokens = {
        "attachment",
        "compressed",
        "copy",
        "direction",
        "doc",
        "document",
        "draft",
        "file",
        "final",
        "image",
        "img",
        "outlook",
        "pdf",
        "picture",
        "plan",
        "report",
        "scan",
        "screenshot",
        "screen",
        "shot",
        "strategic",
        "v",
        "version",
    }
    if all(token in generic_tokens for token in alpha_tokens):
        return True
    return False


def _normalized_mail_subject(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    while True:
        updated = re.sub(r"^\s*subject\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        if updated == text:
            break
        text = updated
    while True:
        updated = _MAIL_SUBJECT_PREFIX_RE.sub("", text).strip()
        if updated == text:
            break
        text = updated
    text = re.sub(r"\s+", " ", text).strip(" -|:")
    return text


def _extract_subject_entity_override(value: str) -> str:
    text = str(value or "")
    match = _SUBJECT_ENTITY_OVERRIDE_RE.search(text)
    if not match:
        return ""
    return _clean_display_label(match.group(1))


def _strip_subject_entity_override(value: str) -> str:
    text = _SUBJECT_ENTITY_OVERRIDE_RE.sub(" ", str(value or ""))
    text = re.sub(r"\s*[|;,]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|:;,")
    return text


def _clean_display_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text.strip(" ,.;:")


def _clean_display_role(role: str, employer: str = "") -> str:
    return clean_strategic_role_label(_clean_display_label(role), employer)


def _extract_document_year_label(*values: str) -> str:
    for value in values:
        text = str(value or "")
        match = _YEAR_RANGE_RE.search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
    for value in values:
        text = str(value or "")
        match = _YEAR_RE.search(text)
        if match:
            return match.group(1)
    return ""


def _looks_like_useful_subject_title(value: str) -> bool:
    cleaned = _clean_display_label(value)
    if not cleaned:
        return False
    lowered = normalize_lookup(cleaned)
    if lowered in {"fwd", "fw", "re"}:
        return False
    if _looks_like_generic_document_name(cleaned):
        return False
    return True


def _subject_org_hint(value: str) -> str:
    text = _clean_display_label(value)
    if not text or not _looks_like_useful_subject_title(text):
        return ""
    candidate = re.sub(
        r"\b(?:org(?:anisation|anization)?\s+chart|strategic\s+plan|strategic\s+direction|annual\s+report|industry\s+report|sector\s+report|report|plan|strategy|direction|roadmap)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\b20\d{2}(?:\s*[-–to]+\s*20\d{2})?\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -|:;,")
    if not candidate:
        return ""
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", candidate) if word]
    if not words or len(words) > 6:
        return ""
    lowered_words = [normalize_lookup(word) for word in words]
    if any(word in _SUBJECT_ORG_HINT_STOPWORDS for word in lowered_words):
        return ""
    return candidate


def _document_label(doc_type: str, context_text: str) -> str:
    lowered = normalize_lookup(context_text)
    if doc_type == "annual_report":
        return "Annual Report"
    if doc_type == "industry_report":
        return "Industry Report"
    if "strategic direction" in lowered:
        return "Strategic Direction"
    if "roadmap" in lowered:
        return "Roadmap"
    if "statement of strategic priorities" in lowered:
        return "Statement of Strategic Priorities"
    return "Strategic Plan"


def _looks_like_high_signal_kpi_focus(value: str) -> bool:
    text = _clean_display_label(value)
    lowered = normalize_lookup(text)
    if not lowered:
        return False
    if any(char.isdigit() for char in text):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) < 2 or len(words) > 8:
        return False
    normalized_words = [normalize_lookup(word) for word in words]
    content_words = [word for word in normalized_words if word not in {"and", "of", "the", "to", "for", "in"}]
    if len(content_words) < 2:
        return False
    if all(word in _KPI_FOCUS_STOPWORDS for word in content_words):
        return False
    return True


def _strip_managed_note_sections(markdown_text: str) -> str:
    kept_lines: List[str] = []
    skipping = False
    for raw_line in str(markdown_text or "").splitlines():
        line = str(raw_line or "")
        if line.startswith("## "):
            heading_key = normalize_lookup(line[3:])
            skipping = heading_key in _MANAGED_NOTE_SECTIONS
            if skipping:
                continue
        if not skipping:
            kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _kind_for_mime(mime_type: str) -> str:
    mime = str(mime_type or "").lower()
    if mime.startswith("image/"):
        return "image"
    if mime in {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/plain",
        "text/html",
        "text/csv",
    }:
        return "document"
    return "other"


@dataclass
class IntelMailboxConfig:
    host: str
    port: int
    username: str
    password: str
    folder: str
    org_name: str
    poll_limit: int
    search_criteria: str
    allowed_senders: tuple[str, ...]
    source_system: str
    callback_url: str
    note_callback_url: str
    callback_secret: str
    callback_timeout: int
    profile_import_url: str
    profile_import_timeout: int
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_use_ssl: bool
    reply_from: str
    mark_seen_on_success: bool

    @classmethod
    def from_env(cls, env: Dict[str, str]) -> "IntelMailboxConfig":
        def get(name: str, default: str = "") -> str:
            return str(os.environ.get(name, env.get(name, default)) or "").strip()

        allowed = tuple(
            item.strip().lower()
            for item in get("INTEL_ALLOWED_SENDERS", "").split(",")
            if item.strip()
        )
        callback_url = get("INTEL_RESULTS_POST_URL")
        note_callback_url = get("INTEL_NOTE_POST_URL")
        if not note_callback_url and callback_url.endswith("/admin/queue_worker_api.php?action=import_cortex_extract"):
            note_callback_url = callback_url.replace(
                "/admin/queue_worker_api.php?action=import_cortex_extract",
                "/lab/market_radar_api.php?action=ingest_intel_note",
            )
        return cls(
            host=get("INTEL_IMAP_HOST", "imap.gmail.com"),
            port=int(get("INTEL_IMAP_PORT", "993") or "993"),
            username=get("INTEL_IMAP_USERNAME"),
            password=get("INTEL_IMAP_PASSWORD"),
            folder=get("INTEL_IMAP_FOLDER", "INBOX"),
            org_name=get("INTEL_IMAP_ORG_NAME", "Longboardfella"),
            poll_limit=max(1, int(get("INTEL_IMAP_POLL_LIMIT", "10") or "10")),
            search_criteria=get("INTEL_IMAP_SEARCH", "UNSEEN"),
            allowed_senders=allowed,
            source_system=get("INTEL_SOURCE_SYSTEM", "cortex_mailbox"),
            callback_url=callback_url,
            note_callback_url=note_callback_url,
            callback_secret=get("INTEL_RESULTS_POST_SECRET"),
            callback_timeout=max(5, int(get("INTEL_RESULTS_POST_TIMEOUT", "30") or "30")),
            profile_import_url=get("INTEL_PROFILE_IMPORT_URL"),
            profile_import_timeout=max(5, int(get("INTEL_PROFILE_IMPORT_TIMEOUT", "45") or "45")),
            smtp_host=get("INTEL_SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(get("INTEL_SMTP_PORT", "465") or "465"),
            smtp_username=get("INTEL_SMTP_USERNAME", get("INTEL_IMAP_USERNAME")),
            smtp_password=get("INTEL_SMTP_PASSWORD", get("INTEL_IMAP_PASSWORD")),
            smtp_use_ssl=get("INTEL_SMTP_USE_SSL", "1").lower() in {"1", "true", "yes", "on"},
            reply_from=get("INTEL_REPLY_FROM", get("INTEL_IMAP_USERNAME")),
            mark_seen_on_success=get("INTEL_IMAP_MARK_SEEN_ON_SUCCESS", "0").lower() in {"1", "true", "yes", "on"},
        )

    def validate(self) -> None:
        if not self.host:
            raise RuntimeError("INTEL_IMAP_HOST is required")
        if not self.username:
            raise RuntimeError("INTEL_IMAP_USERNAME is required")
        if not self.password:
            raise RuntimeError("INTEL_IMAP_PASSWORD is required")


class IntelMailboxStore:
    def __init__(self, base_path: Optional[Path] = None):
        self.root = Path(base_path) if base_path is not None else _safe_db_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.messages_dir = self.root / "messages"
        self.outbox_dir = self.root / "outbox"
        self.results_dir = self.root / "results"
        self.messages_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state(self._initial_state())

    @staticmethod
    def _initial_state() -> Dict[str, Any]:
        return {"updated_at": _utc_now_iso(), "messages": []}

    def _read_state(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._initial_state()
            payload.setdefault("updated_at", _utc_now_iso())
            payload.setdefault("messages", [])
            return payload
        except Exception:
            return self._initial_state()

    def _write_state(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = _utc_now_iso()
        fd, tmp_path = tempfile.mkstemp(prefix="intel_mailbox_", suffix=".json", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=True, indent=2, sort_keys=True)
            os.replace(tmp_path, self.state_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def list_messages(self) -> List[Dict[str, Any]]:
        return list(self._read_state().get("messages") or [])

    def has_processed_message(self, message_id: str) -> bool:
        message_id = str(message_id or "").strip()
        if not message_id:
            return False
        for entry in self._read_state().get("messages") or []:
            if str(entry.get("message_id") or "").strip() == message_id and entry.get("status") == "processed":
                return True
        return False

    def persist_message(self, metadata: Dict[str, Any], raw_bytes: bytes, attachments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        message_id = str(metadata.get("message_id") or "").strip()
        message_key = _derive_message_key(message_id, raw_bytes)
        message_dir = self.messages_dir / message_key
        attachments_dir = message_dir / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)

        raw_path = message_dir / "message.eml"
        raw_path.write_bytes(raw_bytes)

        stored_attachments: List[Dict[str, Any]] = []
        used_names: set[str] = set()
        for index, item in enumerate(attachments):
            base_name = _sanitize_filename(item.get("filename") or "", f"attachment_{index + 1}")
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            filename = base_name
            counter = 2
            while filename in used_names:
                filename = f"{stem}_{counter}{suffix}"
                counter += 1
            used_names.add(filename)
            stored_path = attachments_dir / filename
            stored_path.write_bytes(item.get("content") or b"")
            stored_attachments.append(
                {
                    "filename": filename,
                    "mime_type": str(item.get("mime_type") or ""),
                    "stored_path": str(stored_path),
                    "kind": str(item.get("kind") or _kind_for_mime(item.get("mime_type") or "")),
                    "size_bytes": stored_path.stat().st_size,
                    "content_id": str(item.get("content_id") or ""),
                }
            )

        record = {
            "message_key": message_key,
            "message_id": message_id,
            "from_email": str(metadata.get("from_email") or ""),
            "from_name": str(metadata.get("from_name") or ""),
            "subject": str(metadata.get("subject") or ""),
            "received_at": str(metadata.get("received_at") or ""),
            "raw_path": str(raw_path),
            "attachments": stored_attachments,
            "status": "persisted",
            "updated_at": _utc_now_iso(),
        }
        state = self._read_state()
        messages = [item for item in state.get("messages") or [] if item.get("message_key") != message_key]
        messages.append(record)
        state["messages"] = sorted(messages, key=lambda item: item.get("received_at", ""), reverse=True)
        self._write_state(state)
        return record

    def record_processed(self, message_key: str, trace_id: str, result_payload: Dict[str, Any], delivery: Dict[str, Any]) -> Dict[str, Any]:
        result_path = self.results_dir / f"{message_key}.json"
        result_path.write_text(json.dumps(result_payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        state = self._read_state()
        messages = list(state.get("messages") or [])
        for item in messages:
            if item.get("message_key") != message_key:
                continue
            item["status"] = "processed"
            item["trace_id"] = trace_id
            item["result_path"] = str(result_path)
            item["delivery"] = dict(delivery or {})
            item["updated_at"] = _utc_now_iso()
            break
        state["messages"] = messages
        self._write_state(state)
        return {"result_path": str(result_path)}

    def record_failure(self, message_key: str, error_message: str) -> None:
        state = self._read_state()
        for item in state.get("messages") or []:
            if item.get("message_key") != message_key:
                continue
            item["status"] = "failed"
            item["error"] = str(error_message or "")[:2000]
            item["updated_at"] = _utc_now_iso()
            break
        self._write_state(state)

    def write_outbox_payload(self, message_key: str, payload: Dict[str, Any]) -> Path:
        outbox_path = self.outbox_dir / f"{message_key}.json"
        outbox_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        return outbox_path


def parse_email_bytes(raw_bytes: bytes) -> Dict[str, Any]:
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = _decode_header_value(message.get("Subject"))
    from_name, from_email = parseaddr(message.get("From", ""))
    from_name = _decode_header_value(from_name)
    from_email = str(from_email or "").strip().lower()
    message_id = str(message.get("Message-ID") or "").strip()
    if not message_id:
        message_id = f"<sha1-{hashlib.sha1(raw_bytes).hexdigest()[:24]}@cortex.local>"
    date_header = str(message.get("Date") or "").strip()
    try:
        received_at = parsedate_to_datetime(date_header).astimezone(timezone.utc).replace(microsecond=0).isoformat()
    except Exception:
        received_at = _utc_now_iso()

    text_parts: List[str] = []
    html_parts: List[str] = []
    attachments: List[Dict[str, Any]] = []

    part_index = 0
    for part in message.walk():
        if part.is_multipart():
            continue
        content_type = str(part.get_content_type() or "").lower()
        disposition = str(part.get_content_disposition() or "").lower()
        filename = _decode_header_value(part.get_filename())
        payload = part.get_payload(decode=True) or b""

        if disposition in {"attachment", "inline"} or filename:
            part_index += 1
            inferred_name = filename or f"part_{part_index}"
            attachments.append(
                {
                    "filename": inferred_name,
                    "mime_type": content_type,
                    "kind": _kind_for_mime(content_type),
                    "content": payload,
                    "content_id": str(part.get("Content-ID") or "").strip("<> "),
                }
            )
            continue

        charset = part.get_content_charset() or "utf-8"
        try:
            decoded = payload.decode(charset, errors="replace")
        except Exception:
            decoded = payload.decode("utf-8", errors="replace")
        if content_type == "text/plain":
            text_parts.append(decoded)
        elif content_type == "text/html":
            html_parts.append(decoded)

    raw_text = "\n\n".join(part.strip() for part in text_parts if part.strip())
    html_text = "\n\n".join(part.strip() for part in html_parts if part.strip())
    all_email_sources = [subject, raw_text, html_text, from_email]
    all_email_sources.extend(item.get("filename") or "" for item in attachments)
    extracted_emails = _extract_emails_from_text("\n".join(all_email_sources))

    return {
        "message_id": message_id,
        "subject": subject,
        "from_name": from_name,
        "from_email": from_email,
        "received_at": received_at,
        "raw_text": raw_text,
        "html_text": html_text,
        "attachments": attachments,
        "extracted_emails": extracted_emails,
    }


class IntelMailboxResultClient:
    def __init__(self, store: IntelMailboxStore, callback_url: str = "", callback_secret: str = "", timeout: int = 30):
        self.store = store
        self.callback_url = str(callback_url or "").strip()
        self.callback_secret = str(callback_secret or "").strip()
        self.timeout = max(5, int(timeout or 30))

    def deliver(
        self,
        message_key: str,
        payload: Dict[str, Any],
        delivery_payload: Optional[Dict[str, Any]] = None,
        callback_url_override: str = "",
    ) -> Dict[str, Any]:
        effective_url = str(callback_url_override or self.callback_url).strip()
        if not effective_url:
            outbox_path = self.store.write_outbox_payload(message_key, payload)
            return {"status": "outbox", "path": str(outbox_path)}

        headers = {"Content-Type": "application/json"}
        if self.callback_secret:
            headers["X-Queue-Key"] = self.callback_secret
        response = requests.post(effective_url, headers=headers, json=delivery_payload or payload, timeout=self.timeout)
        response.raise_for_status()
        delivery: Dict[str, Any] = {"status": "posted", "http_status": response.status_code}
        try:
            body = response.json()
            if isinstance(body, dict):
                delivery["response"] = body
                if body.get("error"):
                    raise RuntimeError(f"Website import error: {body['error']}")
        except Exception:
            text = (response.text or "").strip()
            if text:
                delivery["response_text"] = text[:500]
            raise
        return delivery


class IntelMailboxReplyClient:
    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 465,
        username: str = "",
        password: str = "",
        use_ssl: bool = True,
        reply_from: str = "",
    ):
        self.smtp_host = str(smtp_host or "").strip()
        self.smtp_port = int(smtp_port or 465)
        self.username = str(username or "").strip()
        self.password = str(password or "").strip()
        self.use_ssl = bool(use_ssl)
        self.reply_from = str(reply_from or username or "").strip()

    def enabled(self) -> bool:
        return bool(self.smtp_host and self.reply_from and self.username and self.password)

    def send(self, to_email: str, subject: str, body: str, in_reply_to: str = "") -> Dict[str, Any]:
        recipient = str(to_email or "").strip().lower()
        if not recipient:
            return {"status": "skipped", "reason": "no_recipient"}
        if not self.enabled():
            return {"status": "disabled"}

        message = EmailMessage()
        message["Subject"] = str(subject or "").strip()
        message["From"] = self.reply_from
        message["To"] = recipient
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = in_reply_to
        message.set_content(str(body or "").strip() + "\n")

        if self.use_ssl:
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30, context=ssl.create_default_context()) as client:
                client.login(self.username, self.password)
                client.send_message(message)
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as client:
                client.starttls(context=ssl.create_default_context())
                client.login(self.username, self.password)
                client.send_message(message)
        return {"status": "sent"}


class IntelMailboxPoller:
    def __init__(
        self,
        config: IntelMailboxConfig,
        store: Optional[IntelMailboxStore] = None,
        extractor: Optional[Callable[[Dict[str, Any]], tuple[Dict[str, Any], Optional[Path]]]] = None,
        result_client: Optional[IntelMailboxResultClient] = None,
        reply_client: Optional[IntelMailboxReplyClient] = None,
        csv_importer: Optional[CsvProfileImportProcessor] = None,
        imap_factory: Optional[Callable[..., Any]] = None,
        signal_store: Optional[StakeholderSignalStore] = None,
    ):
        self.config = config
        self.store = store or IntelMailboxStore()
        self.extractor = extractor or extract_intel
        self.result_client = result_client or IntelMailboxResultClient(
            self.store,
            callback_url=config.callback_url,
            callback_secret=config.callback_secret,
            timeout=config.callback_timeout,
        )
        self.reply_client = reply_client or IntelMailboxReplyClient(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            username=config.smtp_username,
            password=config.smtp_password,
            use_ssl=config.smtp_use_ssl,
            reply_from=config.reply_from,
        )
        self.csv_importer = csv_importer or CsvProfileImportProcessor.from_config(
            explicit_url=config.profile_import_url,
            callback_url=config.callback_url,
            queue_server_url=str(os.environ.get("QUEUE_SERVER_URL") or ""),
            queue_secret=config.callback_secret or str(os.environ.get("QUEUE_SECRET_KEY") or ""),
            timeout=config.profile_import_timeout,
        )
        self.imap_factory = imap_factory or imaplib.IMAP4_SSL
        self.signal_store = signal_store or StakeholderSignalStore()
        self.note_processor = IntelNoteProcessor(self.extractor)

    def _allowed_sender(self, email_address: str) -> bool:
        sender = str(email_address or "").strip().lower()
        if not self.config.allowed_senders:
            return True
        return sender in self.config.allowed_senders

    def _known_org_scopes(self) -> List[str]:
        names = {str(self.config.org_name or "").strip()}
        try:
            profiles = self.signal_store.list_profiles(org_name="")
        except Exception:
            profiles = []
        for profile in profiles:
            org_name = str(profile.get("org_name") or "").strip()
            if org_name:
                names.add(org_name)
        try:
            state = self.signal_store.get_state()
        except Exception:
            state = {}
        for key, context in dict(state.get("org_contexts") or {}).items():
            del key
            org_name = str((context or {}).get("org_name") or "").strip()
            if org_name:
                names.add(org_name)
        return sorted(name for name in names if name)

    def _match_known_org_scope(self, requested_org_name: str) -> str:
        requested = str(requested_org_name or "").strip()
        if not requested:
            return ""
        wanted = normalize_lookup(requested)
        exact_matches = [
            candidate
            for candidate in self._known_org_scopes()
            if normalize_lookup(candidate) == wanted
        ]
        if exact_matches:
            return exact_matches[0]
        compatible_matches = [
            candidate
            for candidate in self._known_org_scopes()
            if orgs_compatible(candidate, requested)
        ]
        if compatible_matches:
            compatible_matches.sort(key=lambda item: (len(normalize_lookup(item)), item))
            return compatible_matches[0]
        return ""

    def _resolve_message_routing(self, message: Dict[str, Any], persisted: Dict[str, Any]) -> Dict[str, Any]:
        raw_subject = str(message.get("subject") or "").strip()
        requested_org_name = _extract_subject_entity_override(raw_subject)
        subject_without_override = _strip_subject_entity_override(raw_subject)
        clean_subject = _normalized_mail_subject(subject_without_override or raw_subject)
        matched_org_name = self._match_known_org_scope(requested_org_name)
        effective_org_name = matched_org_name or self.config.org_name
        attachments = list(persisted.get("attachments") or [])
        has_document_attachment = any(
            str(item.get("kind") or "").strip().lower() == "document"
            for item in attachments
        )
        has_org_chart_image_attachment = any(
            str(item.get("kind") or "").strip().lower() == "image"
            and looks_like_org_chart_attachment(
                str(item.get("filename") or "").strip(),
                clean_subject,
            )
            for item in attachments
        )
        subject_org_hint = _subject_org_hint(clean_subject) if (has_document_attachment or has_org_chart_image_attachment) else ""
        status = "default"
        if requested_org_name and matched_org_name:
            status = "matched_override"
        elif requested_org_name:
            status = "unmatched_override"
        return {
            "default_org_name": self.config.org_name,
            "requested_org_name": requested_org_name,
            "matched_org_name": matched_org_name,
            "effective_org_name": effective_org_name,
            "status": status,
            "clean_subject": clean_subject,
            "subject_org_hint": subject_org_hint,
            "has_document_attachment": has_document_attachment,
            "has_org_chart_image_attachment": has_org_chart_image_attachment,
        }

    def _build_extract_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        trace_id: str,
        routing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        routing = routing or {}
        tags = ["email_intake"]
        if any(item.get("kind") == "image" for item in persisted.get("attachments") or []):
            tags.append("image_attachment")
        if any(item.get("kind") == "document" for item in persisted.get("attachments") or []):
            tags.append("document_attachment")
        clean_subject = str(routing.get("clean_subject") or _normalized_mail_subject(message.get("subject", ""))).strip()
        subject_org_hint = str(routing.get("subject_org_hint") or "").strip()
        effective_org_name = str(routing.get("effective_org_name") or self.config.org_name).strip()
        return {
            "org_name": effective_org_name,
            "source_system": self.config.source_system,
            "trace_id": trace_id,
            "signal_type": "email_intel",
            "submitted_by": message.get("from_email", ""),
            "message_id": message.get("message_id", ""),
            "received_at": message.get("received_at", ""),
            "subject": clean_subject,
            "raw_text": message.get("raw_text", ""),
            "html_text": message.get("html_text", ""),
            "primary_url": "",
            "text_note": "",
            "parsed_candidate_name": subject_org_hint,
            "parsed_candidate_employer": subject_org_hint,
            "target_type": "",
            "attachments": list(persisted.get("attachments") or []),
            "tags": tags,
            "mailbox_routing": dict(routing),
        }

    def _subscriber_strategic_profile(self, scope_org_name: str = "") -> Dict[str, Any]:
        org_name = str(scope_org_name or self.config.org_name).strip()
        return dict(self.signal_store.get_org_context(org_name).get("org_strategic_profile") or {})

    def _lookup_org_profile(self, org_name: str, scope_org_name: str = "") -> Dict[str, Any]:
        wanted = normalize_lookup(org_name)
        if not wanted:
            return {}
        effective_scope = str(scope_org_name or self.config.org_name).strip()
        for profile in self.signal_store.list_profiles(org_name=effective_scope):
            if str(profile.get("target_type") or "").strip().lower() != "organisation":
                continue
            if normalize_lookup(profile.get("canonical_name") or "") == wanted:
                return profile
            for alias in profile.get("aliases") or []:
                if normalize_lookup(alias) == wanted:
                    return profile
        return {}

    def _infer_industry_name(self, message: Dict[str, Any], entity: Dict[str, Any], scope_org_name: str = "") -> str:
        explicit = str(entity.get("industry") or "").strip()
        if explicit:
            return explicit

        org_name = str(entity.get("canonical_name") or entity.get("name") or entity.get("current_employer") or "").strip()
        org_profile = self._lookup_org_profile(org_name, scope_org_name=scope_org_name)
        affiliations = org_profile.get("industry_affiliations") or []
        if affiliations:
            first = affiliations[0]
            name = str(first.get("industry_name") or "").strip()
            if name:
                return name
        existing = str(org_profile.get("industry") or "").strip()
        if existing:
            return existing

        return classify_entity_industry(
            entity=entity,
            message=message,
            strategic_profile=self._subscriber_strategic_profile(scope_org_name),
            org_profile_lookup=lambda item: self._lookup_org_profile(item, scope_org_name=scope_org_name),
        )

    def _extract_note_urls(self, message: Dict[str, Any], markdown_text: str) -> List[Dict[str, str]]:
        combined_urls = _extract_urls_from_text(
            "\n".join(
                [
                    str(message.get("raw_text") or ""),
                    str(message.get("html_text") or ""),
                    str(markdown_text or ""),
                ]
            )
        )
        urls: List[Dict[str, str]] = []
        seen: set[str] = set()
        for url in combined_urls:
            if url in seen:
                continue
            seen.add(url)
            urls.append(
                {
                    "url": url,
                    "url_type": "reference",
                    "title": "",
                    "description": "",
                }
            )
        return urls

    @staticmethod
    def _choose_primary_entity(output_data: Dict[str, Any]) -> Dict[str, str]:
        for bucket_name, target_type in (("people", "person"), ("organisations", "organisation")):
            bucket = output_data.get(bucket_name) or []
            if not bucket:
                continue
            item = bucket[0]
            return {
                "target_type": target_type,
                "name": str(item.get("canonical_name") or item.get("name") or "").strip(),
                "employer": str(item.get("current_employer") or item.get("employer") or "").strip(),
            }
        return {"target_type": "", "name": "", "employer": ""}

    @staticmethod
    def _choose_subject_primary_entity(subject_org_hint: str, output_data: Dict[str, Any]) -> Dict[str, str]:
        wanted = normalize_lookup(subject_org_hint)
        if not wanted:
            return {"target_type": "", "name": "", "employer": ""}

        for item in output_data.get("entities") or []:
            if str(item.get("target_type") or "").strip().lower() != "organisation":
                continue
            name = str(item.get("canonical_name") or item.get("name") or "").strip()
            if name and normalize_lookup(name) == wanted:
                return {"target_type": "organisation", "name": name, "employer": ""}
            if name and orgs_compatible(name, subject_org_hint):
                return {"target_type": "organisation", "name": subject_org_hint, "employer": ""}

        for item in output_data.get("organisations") or []:
            name = str(item.get("canonical_name") or item.get("name") or "").strip()
            if name and normalize_lookup(name) == wanted:
                return {"target_type": "organisation", "name": name, "employer": ""}
            if name and orgs_compatible(name, subject_org_hint):
                return {"target_type": "organisation", "name": subject_org_hint, "employer": ""}

        for item in output_data.get("people") or []:
            employer = str(item.get("current_employer") or item.get("employer") or "").strip()
            if employer and normalize_lookup(employer) == wanted:
                return {"target_type": "organisation", "name": employer, "employer": ""}
            if employer and orgs_compatible(employer, subject_org_hint):
                return {"target_type": "organisation", "name": subject_org_hint, "employer": ""}

        return {"target_type": "organisation", "name": subject_org_hint, "employer": ""}

    def _choose_document_primary_entity(
        self,
        output_data: Dict[str, Any],
        strategic_doc: Dict[str, Any],
    ) -> Dict[str, str]:
        strategic_org_name = str(strategic_doc.get("org_name") or "").strip()
        strategic_org_key = normalize_lookup(strategic_org_name)
        summary_text = " ".join(
            [
                str(output_data.get("summary") or ""),
                str(strategic_doc.get("strategic_summary") or ""),
                " ".join(str(item.get("excerpt") or "") for item in (output_data.get("attachments") or [])),
            ]
        )
        haystack = normalize_lookup(summary_text)

        candidates: List[tuple[int, Dict[str, str]]] = []
        for entity in output_data.get("entities") or []:
            if str(entity.get("target_type") or "").strip().lower() != "organisation":
                continue
            name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
            if not name or _looks_like_generic_document_name(name):
                continue
            name_key = normalize_lookup(name)
            score = 0
            if strategic_org_key and name_key == strategic_org_key:
                score += 10
            if haystack and name_key and name_key in haystack:
                score += 4
            evidence = normalize_lookup(str(entity.get("evidence") or ""))
            if any(token in evidence for token in ("strategic", "annual report", "authored", "foreword", "chief executive officer", "president", "organizational chart", "organisational chart")):
                score += 2
            if normalize_lookup(name) == normalize_lookup(self.config.org_name):
                score -= 3
            candidates.append(
                (
                    score,
                    {
                        "target_type": "organisation",
                        "name": name,
                        "employer": "",
                    },
                )
            )

        if candidates:
            best_score, best = max(candidates, key=lambda item: item[0])
            if best_score > 0:
                return best

        if strategic_org_name and not _looks_like_generic_document_name(strategic_org_name):
            return {
                "target_type": "organisation",
                "name": strategic_org_name,
                "employer": "",
            }
        return {"target_type": "", "name": "", "employer": ""}

    @staticmethod
    def _infer_note_source_type(message: Dict[str, Any], output_data: Dict[str, Any], message_kind: str = "") -> str:
        subject = str(message.get("subject") or "").lower()
        attachment_names = " ".join(str(item.get("filename") or "").lower() for item in output_data.get("attachments") or [])
        strategic_doc = dict((output_data.get("processing_meta") or {}).get("strategic_doc") or {})
        doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
        text = " ".join(
            [
                subject,
                str(message.get("raw_text") or "").lower(),
                str(output_data.get("summary") or "").lower(),
                attachment_names,
                doc_type,
            ]
        )
        if doc_type in {"strategic_plan", "annual_report", "industry_report"}:
            return doc_type
        if "org chart" in text or "organisational chart" in text or "organization chart" in text:
            return "org_chart"
        if "annual report" in text:
            return "annual_report"
        if any(token in text for token in ("strategic plan", "strategic direction")) or "strategy" in subject:
            return "strategic_plan"
        if "industry report" in text or "sector report" in text:
            return "industry_report"
        if message_kind == "document_analysis":
            return "general"
        if any(token in text for token in ("meeting", "met with", "call with", "spoke with", "introduction", "intro")):
            return "meeting_note"
        return "general"

    @staticmethod
    def _infer_reference_type(message: Dict[str, Any], entity: Dict[str, Any]) -> str:
        context = " ".join(
            [
                str(entity.get("evidence") or ""),
                str(message.get("subject") or ""),
                str(message.get("raw_text") or ""),
            ]
        ).lower()
        if "introduc" in context:
            return "intro"
        if any(token in context for token in ("decision maker", "decision-maker", "head of", "chief", "executive")):
            return "decision_maker"
        if any(token in context for token in ("met", "meeting", "spoke with", "call with")):
            return "meeting"
        if any(token in context for token in ("lead", "leading", "owner")):
            return "lead"
        if str(entity.get("target_type") or "").strip().lower() == "industry":
            return "affiliation"
        return "mention"

    @staticmethod
    def _infer_signal_type(message: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        text = " ".join(
            [
                str(message.get("subject") or ""),
                str(message.get("raw_text") or ""),
                str(output_data.get("summary") or ""),
            ]
        ).lower()
        if "intro" in text or "introduc" in text:
            return "intro_offered"
        if any(token in text for token in ("strategic", "transformation", "roadmap", "priority")):
            return "strategic_intent"
        if any(token in text for token in ("restructure", "reorganisation", "reorg")):
            return "org_restructure"
        if any(token in text for token in ("appointed", "joins", "joined", "new role")):
            return "leadership_change"
        return "strategic_intent"

    @staticmethod
    def _confidence_label(value: Any) -> str:
        try:
            score = float(value)
        except Exception:
            score = 0.5
        if score >= 0.85:
            return "confirmed"
        if score >= 0.6:
            return "probable"
        return "speculative"

    @staticmethod
    def _enrich_note_markdown(
        markdown_text: str,
        strategic_doc: Dict[str, Any],
        note_summary: str = "",
    ) -> str:
        doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
        doc_types_with_curated_notes = {"strategic_plan", "annual_report", "industry_report"}
        base = _strip_managed_note_sections(markdown_text)
        strategic_signals = [
            item
            for item in strategic_doc.get("strategic_signals") or []
            if str(item.get("headline") or "").strip()
        ]
        stakeholders = [
            item
            for item in strategic_doc.get("key_stakeholders") or []
            if str(item.get("name") or "").strip()
        ]
        performance_indicators = [
            item
            for item in strategic_doc.get("performance_indicators") or []
            if str(item.get("label") or "").strip()
        ]
        kpi_focuses = [str(item).strip() for item in strategic_doc.get("kpi_focuses") or [] if str(item).strip()]
        kpi_focuses = [item for item in kpi_focuses if _looks_like_high_signal_kpi_focus(item)]

        sections: List[str] = []
        if doc_type in doc_types_with_curated_notes and note_summary.strip():
            sections.append("## Summary\n\n" + note_summary.strip())
        if strategic_signals:
            lines = ["## Strategic Insights", ""]
            for item in strategic_signals[:5]:
                headline = _clean_display_label(str(item.get("headline") or "").strip())
                snippet = _clean_display_label(str(item.get("snippet") or item.get("evidence") or "").strip())
                line = f"- {headline}"
                if snippet:
                    line += f": {snippet}"
                lines.append(line)
            sections.append("\n".join(lines))
        if stakeholders:
            lines = ["## Key Stakeholders", ""]
            for item in stakeholders[:6]:
                name = _clean_display_label(str(item.get("name") or "").strip())
                employer = _clean_display_label(str(item.get("current_employer") or "").strip())
                role = _clean_display_role(str(item.get("current_role") or "").strip(), employer)
                parts = [name]
                if role:
                    parts.append(role)
                if employer:
                    parts.append(employer)
                lines.append("- " + " | ".join(parts))
            sections.append("\n".join(lines))
        if performance_indicators:
            heading = "## Performance Snapshot" if doc_type == "annual_report" else "## Performance Indicators"
            lines = [heading, ""]
            for item in performance_indicators[:6]:
                label = _clean_display_label(str(item.get("label") or "").strip())
                value = _clean_display_label(str(item.get("value") or "").strip())
                evidence = _clean_display_label(str(item.get("evidence") or "").strip())
                line = f"- {label}"
                if value:
                    line += f": {value}"
                if evidence:
                    line += f" | {evidence}"
                lines.append(line)
            sections.append("\n".join(lines))
        if kpi_focuses:
            lines = ["## KPI Focus Areas", ""]
            for item in kpi_focuses[:5]:
                lines.append(f"- {_clean_display_label(item)}")
            sections.append("\n".join(lines))

        if not sections:
            return base
        if doc_type in doc_types_with_curated_notes:
            return "\n\n".join(sections).strip()
        if not base:
            return "\n\n".join(sections)
        return base + "\n\n" + "\n\n".join(sections)

    @staticmethod
    def _build_performance_signals(
        strategic_doc: Dict[str, Any],
        strategic_summary: str,
        note_summary: str,
        raw_text: str,
    ) -> List[Dict[str, str]]:
        indicators = [
            item
            for item in strategic_doc.get("performance_indicators") or []
            if str(item.get("label") or "").strip()
        ]
        signals: List[Dict[str, str]] = []
        for item in indicators[:4]:
            snippet = str(item.get("evidence") or item.get("value") or strategic_summary or note_summary or raw_text).strip()[:320]
            signals.append(
                {
                    "headline": str(item.get("label") or "Performance indicator").strip(),
                    "snippet": snippet,
                    "signal_type": "performance_snapshot",
                    "urgency": "medium",
                    "actionable": True,
                    "suggested_action": "Review this organisation-level performance signal for trend and positioning implications",
                }
            )
        return signals

    @staticmethod
    def _compose_note_summary(note_summary: str, strategic_summary: str, doc_type: str) -> str:
        base = str(note_summary or "").strip()
        strategic = str(strategic_summary or "").strip()
        if doc_type in {"strategic_plan", "annual_report", "industry_report"}:
            return base or strategic
        if not base:
            return strategic
        if strategic and normalize_lookup(strategic) not in normalize_lookup(base):
            return " ".join([base, strategic]).strip()
        return base

    @staticmethod
    def _derive_note_title(
        clean_subject: str,
        strategic_doc: Dict[str, Any],
        output_data: Dict[str, Any],
        primary: Dict[str, str],
    ) -> str:
        if _looks_like_useful_subject_title(clean_subject):
            return clean_subject

        doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
        org_name = str(strategic_doc.get("org_name") or primary.get("name") or "").strip()
        attachments = list(output_data.get("attachments") or [])
        attachment_names = [str(item.get("filename") or "").strip() for item in attachments if str(item.get("filename") or "").strip()]
        attachment_excerpts = [str(item.get("excerpt") or "").strip() for item in attachments if str(item.get("excerpt") or "").strip()]
        title_context = " ".join([clean_subject, *attachment_names, *attachment_excerpts])

        if doc_type in {"strategic_plan", "annual_report", "industry_report"} and org_name:
            label = _document_label(doc_type, title_context)
            year_label = _extract_document_year_label(clean_subject, *attachment_names, *attachment_excerpts)
            parts = [org_name, label]
            if year_label:
                parts.append(year_label)
            return _clean_display_label(" ".join(parts))

        return str(clean_subject or org_name or primary.get("name") or "").strip()

    def _build_ingest_note_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        output_data: Dict[str, Any],
        signal: Dict[str, Any],
        markdown_text: str,
        message_kind: str,
        scope_org_name: str = "",
        routing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        routing = dict(routing or {})
        processing_meta = dict(output_data.get("processing_meta") or {})
        strategic_doc = dict(processing_meta.get("strategic_doc") or {})
        strategic_org_name = str(strategic_doc.get("org_name") or "").strip()
        email_triage = dict(processing_meta.get("email_triage") or {})
        clean_subject = str(
            routing.get("clean_subject")
            or email_triage.get("clean_subject")
            or _normalized_mail_subject(message.get("subject", ""))
        ).strip()
        subject_org_hint = str(routing.get("subject_org_hint") or "").strip()
        primary = self._choose_primary_entity(output_data)
        can_use_document_subject_hint = message_kind in {"document_analysis", "org_chart"} or bool(routing.get("has_org_chart_image_attachment"))
        subject_primary = self._choose_subject_primary_entity(subject_org_hint, output_data) if can_use_document_subject_hint else {"target_type": "", "name": "", "employer": ""}
        document_primary = self._choose_document_primary_entity(output_data, strategic_doc) if can_use_document_subject_hint else {"target_type": "", "name": "", "employer": ""}
        if subject_primary.get("name") and document_primary.get("name"):
            subject_key = normalize_lookup(subject_primary.get("name") or "")
            document_key = normalize_lookup(document_primary.get("name") or "")
            if (
                subject_key and document_key
                and subject_key != document_key
                and (subject_key in document_key or orgs_compatible(subject_primary.get("name") or "", document_primary.get("name") or ""))
            ):
                document_primary = {"target_type": "", "name": "", "employer": ""}
        if document_primary.get("name"):
            primary = document_primary
        elif subject_primary.get("name") and (
            not primary.get("name")
            or primary.get("target_type") != "organisation"
            or orgs_compatible(primary.get("name") or "", subject_primary.get("name") or "")
        ):
            primary = subject_primary
        elif strategic_org_name and not _looks_like_generic_document_name(strategic_org_name) and (
            not primary.get("name")
            or (
                can_use_document_subject_hint and primary.get("target_type") != "organisation"
            )
            or (
                primary.get("target_type") == "person"
                and normalize_lookup(primary.get("employer") or "") == normalize_lookup(scope_org_name)
            )
        ):
            primary = {
                "target_type": "organisation",
                "name": strategic_org_name,
                "employer": "",
            }
        entities = list(output_data.get("entities") or [])
        primary_key = normalize_lookup(primary.get("name", ""))
        referenced_entities: List[Dict[str, Any]] = []
        seen_refs: set[tuple[str, str]] = set()

        for entity in entities:
            name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
            target_type = str(entity.get("target_type") or "").strip().lower() or "person"
            if not name:
                continue
            dedupe = (normalize_lookup(name), target_type)
            if dedupe in seen_refs or (target_type == primary.get("target_type") and normalize_lookup(name) == primary_key):
                continue
            seen_refs.add(dedupe)
            current_employer = _clean_display_label(str(entity.get("current_employer") or "").strip())
            current_role = _clean_display_role(str(entity.get("current_role") or "").strip(), current_employer)
            referenced_entities.append(
                {
                    "name": _clean_display_label(name),
                    "target_type": target_type,
                    "current_employer": current_employer,
                    "current_role": current_role,
                    "reference_type": self._infer_reference_type(message, entity),
                    "confidence": self._confidence_label(entity.get("confidence")),
                    "context": str(entity.get("evidence") or "").strip()[:240],
                }
            )
            industry_name = self._infer_industry_name(message, entity, scope_org_name=scope_org_name)
            if target_type == "organisation" and industry_name:
                industry_key = (normalize_lookup(industry_name), "industry")
                if industry_key not in seen_refs:
                    seen_refs.add(industry_key)
                    referenced_entities.append(
                        {
                            "name": industry_name,
                            "target_type": "industry",
                            "reference_type": "affiliation",
                            "confidence": self._confidence_label(entity.get("confidence")),
                        "context": f"{name} is associated with the {industry_name} sector",
                    }
                )

        primary_employer = str(primary.get("employer") or "").strip()
        if strategic_org_name and not _looks_like_generic_document_name(strategic_org_name):
            strategic_org_key = (normalize_lookup(strategic_org_name), "organisation")
            if strategic_org_key not in seen_refs and normalize_lookup(strategic_org_name) != primary_key:
                seen_refs.add(strategic_org_key)
                referenced_entities.append(
                    {
                        "name": strategic_org_name,
                        "target_type": "organisation",
                        "reference_type": "mention",
                        "confidence": "confirmed",
                        "context": "Document title/content indicates this organisation is the subject of the strategic document",
                    }
                )
        if primary.get("target_type") == "person" and primary_employer:
            employer_key = (normalize_lookup(primary_employer), "organisation")
            if employer_key not in seen_refs:
                seen_refs.add(employer_key)
                referenced_entities.append(
                    {
                        "name": primary_employer,
                        "target_type": "organisation",
                        "reference_type": "meeting",
                        "confidence": "confirmed",
                        "context": f"Primary contact is affiliated with {primary_employer}",
                    }
                )

        matched_names = []
        suggested_names = []
        for match in output_data.get("matches") or []:
            candidate_name = str(match.get("candidate_name") or "").strip()
            if candidate_name and match.get("matched"):
                matched_names.append(candidate_name)
            elif candidate_name:
                suggested_names.append(candidate_name)

        note_date = ""
        received_at = str(message.get("received_at") or "").strip()
        if received_at:
            note_date = received_at[:10]

        note_summary = str(output_data.get("summary") or "").strip()
        strategic_summary = str(strategic_doc.get("strategic_summary") or "").strip()
        strategic_signals = list(strategic_doc.get("strategic_signals") or [])
        doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
        note_title = self._derive_note_title(clean_subject, strategic_doc, output_data, primary)
        note_summary = self._compose_note_summary(note_summary, strategic_summary, doc_type)
        note_content = self._enrich_note_markdown(markdown_text, strategic_doc, note_summary)
        attachments_processed = [
            str(item.get("filename") or "").strip()
            for item in output_data.get("attachments") or []
            if str(item.get("status") or "").strip().lower() == "processed"
            and str(item.get("filename") or "").strip()
        ]
        signals = [
            {
                "headline": str(note_title or strategic_org_name or "Mailbox note").strip(),
                "snippet": (note_summary or str(message.get("raw_text") or "").strip())[:320],
                "signal_type": self._infer_signal_type(message, output_data),
                "urgency": "medium",
                "actionable": True if output_data.get("target_update_suggestions") or strategic_signals or strategic_doc.get("themes") or strategic_doc.get("initiatives") else False,
                "suggested_action": "Review referenced entities and follow up on any introductions or strategic leads",
            }
        ]
        for item in strategic_signals[:5]:
            signals.append(
                {
                    "headline": str(item.get("headline") or "Strategic signal").strip(),
                    "snippet": str(item.get("snippet") or strategic_summary or note_summary or str(message.get("raw_text") or "").strip())[:320],
                    "signal_type": "strategic_intent",
                    "urgency": "medium",
                    "actionable": True,
                    "suggested_action": "Review whether this strategic signal creates an engagement, policy, or positioning opportunity",
                }
            )
        if not strategic_signals:
            for theme in (strategic_doc.get("themes") or [])[:3]:
                signals.append(
                    {
                        "headline": f"Strategic theme: {theme}",
                        "snippet": (strategic_summary or note_summary or str(message.get("raw_text") or "").strip())[:320],
                        "signal_type": "strategic_intent",
                        "urgency": "medium",
                        "actionable": True,
                        "suggested_action": "Review whether this strategic theme creates an engagement or positioning opportunity",
                    }
                )
        signals.extend(
            self._build_performance_signals(
                strategic_doc,
                strategic_summary,
                note_summary,
                str(message.get("raw_text") or "").strip(),
            )
        )
        relationship_paths = self.signal_store.find_relationship_paths(
            org_name=scope_org_name,
            target_names=[
                primary.get("name", ""),
                *[str(item.get("name") or "").strip() for item in referenced_entities if str(item.get("target_type") or "").strip().lower() == "person"],
            ],
            max_hops=4,
            limit=5,
        )

        return {
            "action": "ingest_intel_note",
            "secret": self.config.callback_secret,
            "org_name": scope_org_name,
            "mailbox_routing": {
                "default_org_name": str(routing.get("default_org_name") or self.config.org_name).strip(),
                "requested_org_name": str(routing.get("requested_org_name") or "").strip(),
                "matched_org_name": str(routing.get("matched_org_name") or "").strip(),
                "effective_org_name": scope_org_name,
                "status": str(routing.get("status") or "default").strip(),
                "subject_org_hint": subject_org_hint,
            },
            "note": {
                "source_type": self._infer_note_source_type(message, output_data, message_kind),
                "title": note_title,
                "content": note_content,
                "original_text": str(message.get("raw_text") or message.get("html_text") or "").strip(),
                "submitted_by": str(message.get("from_email") or "").strip(),
                "note_date": note_date,
                "attachments_processed": attachments_processed,
                "attachment_fingerprints": self._build_attachment_fingerprints(persisted, output_data),
            },
            "primary_entity": {
                "name": primary.get("name", ""),
                "target_type": primary.get("target_type", ""),
                "current_employer": primary_employer,
                "current_role": "",
                "linkedin_url": "",
                "tags": ["existing-relationship"] if matched_names else [],
            },
            "referenced_entities": referenced_entities,
            "urls": self._extract_note_urls(message, note_content),
            "signals": signals,
            "graph_enrichment": {
                "existing_connections": sorted(set(matched_names)),
                "new_profiles_suggested": sorted(set(suggested_names)),
                "relationship_paths": relationship_paths,
            },
        }

    def _build_attachment_fingerprints(
        self,
        persisted: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> List[str]:
        processed_names = {
            str(item.get("filename") or "").strip()
            for item in output_data.get("attachments") or []
            if str(item.get("status") or "").strip().lower() == "processed"
            and str(item.get("filename") or "").strip()
        }
        fingerprints: List[str] = []
        for attachment in persisted.get("attachments") or []:
            filename = str(attachment.get("filename") or "").strip()
            if processed_names and filename not in processed_names:
                continue
            stored_path = str(attachment.get("stored_path") or "").strip()
            digest = ""
            if stored_path and Path(stored_path).exists():
                try:
                    digest = hashlib.sha1(Path(stored_path).read_bytes()).hexdigest()
                except Exception:
                    digest = ""
            if not digest:
                fallback = "|".join(
                    [
                        filename,
                        str(attachment.get("mime_type") or "").strip(),
                        str(attachment.get("size_bytes") or "").strip(),
                    ]
                ).strip()
                if fallback:
                    digest = hashlib.sha1(fallback.encode("utf-8", "ignore")).hexdigest()
            if digest:
                fingerprints.append(f"{filename}:{digest}")
        return sorted(set(fingerprints))

    def _build_duplicate_delivery(
        self,
        duplicate: Dict[str, Any],
        note_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        intel_id = str(duplicate.get("existing_intel_id") or "").strip()
        if not intel_id:
            intel_id = str(((duplicate.get("delivery") or {}).get("response") or {}).get("intel_id") or "").strip()
        return {
            "status": "duplicate_local",
            "duplicate_of_trace_id": str(duplicate.get("trace_id") or "").strip(),
            "similarity": float(duplicate.get("similarity") or 0.0),
            "response": {
                "status": "duplicate",
                "intel_id": intel_id,
                "primary_entity": ((note_payload.get("primary_entity") or {}).get("name") or "").strip(),
            },
        }

    def _ingest_signal(
        self,
        message: Dict[str, Any],
        output_data: Dict[str, Any],
        trace_id: str,
        scope_org_name: str = "",
        clean_subject: str = "",
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        primary = self._choose_primary_entity(output_data)
        payload = {
            "org_name": scope_org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "signal_type": "email_intel",
            "submitted_by": message.get("from_email", ""),
            "message_id": message.get("message_id", ""),
            "received_at": message.get("received_at", ""),
            "subject": clean_subject or message.get("subject", ""),
            "raw_text": message.get("raw_text", ""),
            "primary_url": "",
            "text_note": "",
            "parsed_candidate_name": primary["name"],
            "parsed_candidate_employer": primary["employer"],
            "target_type": primary["target_type"],
            "notification_kind": "mailbox_auto_extract",
            "tags": ["email_intake"],
        }
        return self.signal_store.ingest_signal(payload)

    def _build_result_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        trace_id: str,
        output_data: Dict[str, Any],
        signal: Dict[str, Any],
        scope_org_name: str = "",
        routing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        routing = dict(routing or {})
        output_entities = list(output_data.get("entities") or [])
        if not output_entities:
            for candidate in output_data.get("people") or []:
                output_entities.append(
                    {
                        "name": str(candidate.get("canonical_name") or candidate.get("name") or "").strip(),
                        "canonical_name": str(candidate.get("canonical_name") or candidate.get("name") or "").strip(),
                        "target_type": "person",
                        "current_employer": str(candidate.get("current_employer") or candidate.get("employer") or "").strip(),
                        "current_role": str(candidate.get("current_role") or candidate.get("title") or "").strip(),
                        "email": str(candidate.get("email") or "").strip(),
                        "linkedin_url": str(candidate.get("linkedin_url") or "").strip(),
                        "confidence": candidate.get("confidence"),
                        "evidence": candidate.get("evidence", ""),
                    }
                )
            for candidate in output_data.get("organisations") or []:
                output_entities.append(
                    {
                        "name": str(candidate.get("canonical_name") or candidate.get("name") or "").strip(),
                        "canonical_name": str(candidate.get("canonical_name") or candidate.get("name") or "").strip(),
                        "target_type": "organisation",
                        "website_url": str(candidate.get("website_url") or "").strip(),
                        "industry": str(candidate.get("industry") or "").strip(),
                        "parent_entity": str(candidate.get("parent_entity") or "").strip(),
                        "confidence": candidate.get("confidence"),
                        "evidence": candidate.get("evidence", ""),
                    }
                )
        output_suggestions = output_data.get("target_update_suggestions") or []
        output_warnings = output_data.get("warnings") or []
        return {
            "result_type": "intel_extract_result",
            "org_name": scope_org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "message_id": message.get("message_id", ""),
            "intel_id": f"mail_{persisted.get('message_key', '')}",
            "raw_summary": str(output_data.get("summary") or ""),
            "entities": output_entities,
            "target_update_suggestions": output_suggestions,
            "suggested_targets": [],
            "warnings": output_warnings,
            "mailbox_message": {
                "subject": message.get("subject", ""),
                "from_email": message.get("from_email", ""),
                "from_name": message.get("from_name", ""),
                "received_at": message.get("received_at", ""),
                "raw_path": persisted.get("raw_path", ""),
                "attachments": persisted.get("attachments") or [],
            },
            "signal": {
                "signal_id": signal.get("signal_id", ""),
                "matched_profile_keys": signal.get("matched_profile_keys") or [],
                "needs_review": bool(signal.get("needs_review")),
            },
            "mailbox_routing": {
                "default_org_name": str(routing.get("default_org_name") or self.config.org_name).strip(),
                "requested_org_name": str(routing.get("requested_org_name") or "").strip(),
                "matched_org_name": str(routing.get("matched_org_name") or "").strip(),
                "effective_org_name": scope_org_name,
                "status": str(routing.get("status") or "default").strip(),
                "subject_org_hint": str(routing.get("subject_org_hint") or "").strip(),
            },
            "output_data": output_data,
        }

    def _send_reply(
        self,
        message: Dict[str, Any],
        subject: str,
        body: str,
    ) -> Dict[str, Any]:
        try:
            return self.reply_client.send(
                to_email=message.get("from_email", ""),
                subject=subject,
                body=body,
                in_reply_to=message.get("message_id", ""),
            )
        except Exception as exc:
            logger.warning("Intel mailbox reply send failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

    def _build_csv_result_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        trace_id: str,
        csv_result: Dict[str, Any],
        reply_delivery: Dict[str, Any],
        scope_org_name: str = "",
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        return {
            "result_type": "csv_profile_import_result",
            "org_name": scope_org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "message_id": message.get("message_id", ""),
            "mailbox_message": {
                "subject": message.get("subject", ""),
                "from_email": message.get("from_email", ""),
                "from_name": message.get("from_name", ""),
                "received_at": message.get("received_at", ""),
                "raw_path": persisted.get("raw_path", ""),
                "attachments": persisted.get("attachments") or [],
            },
            "csv_import": {
                "filename": csv_result.get("filename", ""),
                "row_count": int(csv_result.get("row_count") or 0),
                "dry_run": bool(csv_result.get("dry_run")),
                "created": int(csv_result.get("created") or 0),
                "updated": int(csv_result.get("updated") or 0),
                "skipped": int(csv_result.get("skipped") or 0),
                "errors": list(csv_result.get("errors") or []),
                "api_result": dict(csv_result.get("api_result") or {}),
            },
            "reply": {
                "subject": csv_result.get("reply_subject", ""),
                "body": csv_result.get("reply_body", ""),
                "delivery": reply_delivery,
            },
        }

    def _build_csv_failure_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        trace_id: str,
        error_message: str,
        reply_subject: str,
        reply_body: str,
        reply_delivery: Dict[str, Any],
        scope_org_name: str = "",
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        return {
            "result_type": "csv_profile_import_result",
            "org_name": scope_org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "message_id": message.get("message_id", ""),
            "mailbox_message": {
                "subject": message.get("subject", ""),
                "from_email": message.get("from_email", ""),
                "from_name": message.get("from_name", ""),
                "received_at": message.get("received_at", ""),
                "raw_path": persisted.get("raw_path", ""),
                "attachments": persisted.get("attachments") or [],
            },
            "csv_import": {
                "status": "failed",
                "error": error_message,
            },
            "reply": {
                "subject": reply_subject,
                "body": reply_body,
                "delivery": reply_delivery,
            },
        }

    def _process_message(self, raw_bytes: bytes) -> Optional[Dict[str, Any]]:
        message = parse_email_bytes(raw_bytes)
        if not self._allowed_sender(message.get("from_email", "")):
            logger.info("Skipping intel mailbox message from unapproved sender: %s", message.get("from_email", ""))
            return None

        persisted = self.store.persist_message(message, raw_bytes, message.get("attachments") or [])
        message_id = str(message.get("message_id") or "").strip()
        if message_id and self.store.has_processed_message(message_id):
            return None

        trace_seed = f"{message_id}|{message.get('subject','')}|{message.get('received_at','')}"
        trace_id = f"trace-{hashlib.sha1(trace_seed.encode('utf-8', 'ignore')).hexdigest()[:32]}"

        message_kind = classify_mailbox_message(message, persisted)
        routing = self._resolve_message_routing(message, persisted)
        effective_org_name = str(routing.get("effective_org_name") or self.config.org_name).strip()
        clean_subject = str(routing.get("clean_subject") or _normalized_mail_subject(message.get("subject", ""))).strip()

        if message_kind == "csv_profile_import":
            try:
                csv_result = self.csv_importer.process_message(message, persisted, effective_org_name)
                reply_delivery = self._send_reply(message, csv_result["reply_subject"], csv_result["reply_body"])
                result_payload = self._build_csv_result_payload(
                    message,
                    persisted,
                    trace_id,
                    csv_result,
                    reply_delivery,
                    scope_org_name=effective_org_name,
                )
                result_payload["mailbox_routing"] = {
                    "default_org_name": self.config.org_name,
                    "requested_org_name": str(routing.get("requested_org_name") or "").strip(),
                    "matched_org_name": str(routing.get("matched_org_name") or "").strip(),
                    "effective_org_name": effective_org_name,
                    "status": str(routing.get("status") or "default").strip(),
                    "subject_org_hint": str(routing.get("subject_org_hint") or "").strip(),
                }
                delivery = {
                    "status": "processed",
                    "created": int(csv_result.get("created") or 0),
                    "updated": int(csv_result.get("updated") or 0),
                    "skipped": int(csv_result.get("skipped") or 0),
                    "dry_run": bool(csv_result.get("dry_run")),
                    "reply": reply_delivery,
                }
                self.store.record_processed(persisted["message_key"], trace_id, result_payload, delivery)
                return {
                    "message_id": message.get("message_id", ""),
                    "trace_id": trace_id,
                    "signal_id": "",
                    "delivery": delivery,
                    "entity_count": 0,
                    "entity_names": [],
                    "update_suggestion_count": 0,
                    "warning_count": len(csv_result.get("errors") or []),
                    "warnings": list(csv_result.get("errors") or []),
                }
            except CsvProfileImportError as exc:
                reply_subject = self.csv_importer._reply_subject(message.get("subject", ""), dry_run="dry run" in str(message.get("subject", "")).lower(), ok=False)
                reply_body = str(exc)
                reply_delivery = self._send_reply(message, reply_subject, reply_body)
                result_payload = self._build_csv_failure_payload(
                    message,
                    persisted,
                    trace_id,
                    str(exc),
                    reply_subject,
                    reply_body,
                    reply_delivery,
                    scope_org_name=effective_org_name,
                )
                result_payload["mailbox_routing"] = {
                    "default_org_name": self.config.org_name,
                    "requested_org_name": str(routing.get("requested_org_name") or "").strip(),
                    "matched_org_name": str(routing.get("matched_org_name") or "").strip(),
                    "effective_org_name": effective_org_name,
                    "status": str(routing.get("status") or "default").strip(),
                    "subject_org_hint": str(routing.get("subject_org_hint") or "").strip(),
                }
                delivery = {"status": "failed", "error": str(exc), "reply": reply_delivery}
                self.store.record_processed(persisted["message_key"], trace_id, result_payload, delivery)
                return {
                    "message_id": message.get("message_id", ""),
                    "trace_id": trace_id,
                    "signal_id": "",
                    "delivery": delivery,
                    "entity_count": 0,
                    "entity_names": [],
                    "update_suggestion_count": 0,
                    "warning_count": 1,
                    "warnings": [str(exc)],
                }

        payload = self._build_extract_payload(message, persisted, trace_id, routing=routing)
        output_data, _output_file, processing_meta = self.note_processor.process(payload, message_kind)
        output_data["processing_meta"] = processing_meta
        markdown_text = ""
        if _output_file and Path(_output_file).exists():
            markdown_text = Path(_output_file).read_text(encoding="utf-8", errors="ignore")
        if message_kind == "intel_extract":
            signal = self._ingest_signal(message, output_data, trace_id, scope_org_name=effective_org_name, clean_subject=clean_subject)
            result_payload = self._build_result_payload(
                message,
                persisted,
                trace_id,
                output_data,
                signal,
                scope_org_name=effective_org_name,
                routing=routing,
            )
            delivery = self.result_client.deliver(persisted["message_key"], result_payload)
        else:
            delivery_payload = self._build_ingest_note_payload(
                message,
                persisted,
                output_data,
                {},
                markdown_text,
                message_kind,
                scope_org_name=effective_org_name,
                routing=routing,
            )
            signal = self._ingest_signal(message, output_data, trace_id, scope_org_name=effective_org_name, clean_subject=clean_subject)
            result_payload = self._build_result_payload(
                message,
                persisted,
                trace_id,
                output_data,
                signal,
                scope_org_name=effective_org_name,
                routing=routing,
            )
            result_payload["website_payload"] = {**delivery_payload, "secret": "[redacted]"}
            delivery = self.result_client.deliver(
                persisted["message_key"],
                result_payload,
                delivery_payload=delivery_payload,
                callback_url_override=self.config.note_callback_url or self.config.callback_url,
            )
            response = dict(delivery.get("response") or {})
            if response.get("intel_id"):
                reconciliation = self.signal_store.reconcile_intel_note_delivery(
                    org_name=effective_org_name,
                    trace_id=trace_id,
                    payload=delivery_payload,
                    response=response,
                )
                result_payload["graph_reconciliation"] = reconciliation
        self.store.record_processed(persisted["message_key"], trace_id, result_payload, delivery)
        entity_names = [
            str(item.get("canonical_name") or item.get("name") or "").strip()
            for item in (result_payload.get("entities") or [])
            if str(item.get("canonical_name") or item.get("name") or "").strip()
        ]
        return {
            "message_id": message.get("message_id", ""),
            "trace_id": trace_id,
            "signal_id": signal.get("signal_id", ""),
            "delivery": delivery,
            "entity_count": int(output_data.get("entity_count") or 0),
            "entity_names": entity_names,
            "update_suggestion_count": len(output_data.get("target_update_suggestions") or []),
            "warning_count": len(output_data.get("warnings") or []),
            "warnings": list(output_data.get("warnings") or []),
        }

    def poll_once(self) -> Dict[str, Any]:
        self.config.validate()
        processed = 0
        skipped = 0
        failures = 0
        results: List[Dict[str, Any]] = []

        client = self.imap_factory(self.config.host, self.config.port)
        try:
            client.login(self.config.username, self.config.password)
            client.select(self.config.folder)
            status, data = client.search(None, self.config.search_criteria)
            if status != "OK":
                raise RuntimeError(f"IMAP search failed: {status}")
            message_ids = [item for item in (data[0] or b"").split() if item][: self.config.poll_limit]

            for imap_id in message_ids:
                status, parts = client.fetch(imap_id, "(BODY.PEEK[])")
                if status != "OK":
                    failures += 1
                    continue
                raw_bytes = b""
                for part in parts or []:
                    if isinstance(part, tuple) and len(part) > 1 and isinstance(part[1], (bytes, bytearray)):
                        raw_bytes = bytes(part[1])
                        break
                if not raw_bytes:
                    failures += 1
                    continue
                parsed = parse_email_bytes(raw_bytes)
                if parsed.get("message_id") and self.store.has_processed_message(parsed["message_id"]):
                    skipped += 1
                    continue
                try:
                    result = self._process_message(raw_bytes)
                    if result:
                        processed += 1
                        results.append(result)
                        if self.config.mark_seen_on_success:
                            try:
                                client.store(imap_id, "+FLAGS", "\\Seen")
                            except Exception:
                                logger.warning("Failed to mark IMAP message %s as seen", imap_id)
                    else:
                        skipped += 1
                except Exception as exc:
                    failures += 1
                    logger.exception("Intel mailbox message processing failed")
                    try:
                        message_id = parsed.get("message_id") or ""
                        message_key = _derive_message_key(message_id, raw_bytes)
                        self.store.record_failure(message_key, str(exc))
                    except Exception:
                        pass
        finally:
            try:
                client.logout()
            except Exception:
                pass

        return {
            "processed": processed,
            "skipped": skipped,
            "failures": failures,
            "results": results,
        }
