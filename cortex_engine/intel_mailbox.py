from __future__ import annotations

import email
import hashlib
import imaplib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parsedate_to_datetime, parseaddr
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from cortex_engine.config_manager import ConfigManager
from cortex_engine.intel_extractor import extract_intel
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore
from cortex_engine.utils import convert_windows_to_wsl_path

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[^@<>\s]+@[^@<>\s]+\.[^@<>\s]+")


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
    callback_secret: str
    callback_timeout: int
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
            callback_url=get("INTEL_RESULTS_POST_URL"),
            callback_secret=get("INTEL_RESULTS_POST_SECRET"),
            callback_timeout=max(5, int(get("INTEL_RESULTS_POST_TIMEOUT", "30") or "30")),
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

    def deliver(self, message_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.callback_url:
            outbox_path = self.store.write_outbox_payload(message_key, payload)
            return {"status": "outbox", "path": str(outbox_path)}

        headers = {"Content-Type": "application/json"}
        if self.callback_secret:
            headers["X-Queue-Key"] = self.callback_secret
        response = requests.post(self.callback_url, headers=headers, json=payload, timeout=self.timeout)
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


class IntelMailboxPoller:
    def __init__(
        self,
        config: IntelMailboxConfig,
        store: Optional[IntelMailboxStore] = None,
        extractor: Optional[Callable[[Dict[str, Any]], tuple[Dict[str, Any], Optional[Path]]]] = None,
        result_client: Optional[IntelMailboxResultClient] = None,
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
        self.imap_factory = imap_factory or imaplib.IMAP4_SSL
        self.signal_store = signal_store or StakeholderSignalStore()

    def _allowed_sender(self, email_address: str) -> bool:
        sender = str(email_address or "").strip().lower()
        if not self.config.allowed_senders:
            return True
        return sender in self.config.allowed_senders

    def _build_extract_payload(self, message: Dict[str, Any], persisted: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        tags = ["email_intake"]
        if any(item.get("kind") == "image" for item in persisted.get("attachments") or []):
            tags.append("image_attachment")
        if any(item.get("kind") == "document" for item in persisted.get("attachments") or []):
            tags.append("document_attachment")
        return {
            "org_name": self.config.org_name,
            "source_system": self.config.source_system,
            "trace_id": trace_id,
            "signal_type": "email_intel",
            "submitted_by": message.get("from_email", ""),
            "message_id": message.get("message_id", ""),
            "received_at": message.get("received_at", ""),
            "subject": message.get("subject", ""),
            "raw_text": message.get("raw_text", ""),
            "html_text": message.get("html_text", ""),
            "primary_url": "",
            "text_note": "",
            "parsed_candidate_name": "",
            "parsed_candidate_employer": "",
            "target_type": "",
            "attachments": list(persisted.get("attachments") or []),
            "tags": tags,
        }

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

    def _ingest_signal(self, message: Dict[str, Any], output_data: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        primary = self._choose_primary_entity(output_data)
        payload = {
            "org_name": self.config.org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "signal_type": "email_intel",
            "submitted_by": message.get("from_email", ""),
            "message_id": message.get("message_id", ""),
            "received_at": message.get("received_at", ""),
            "subject": message.get("subject", ""),
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
    ) -> Dict[str, Any]:
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
            "org_name": self.config.org_name,
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
            "output_data": output_data,
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
        payload = self._build_extract_payload(message, persisted, trace_id)
        output_data, _output_file = self.extractor(payload)
        signal = self._ingest_signal(message, output_data, trace_id)
        result_payload = self._build_result_payload(message, persisted, trace_id, output_data, signal)
        delivery = self.result_client.deliver(persisted["message_key"], result_payload)
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
