from __future__ import annotations

import json
import logging
import signal
import threading
import html
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortex_engine.notes_mailbox import NOTES_MAILBOX_IDENTITY, classify_notes_mailbox_route


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass
class NotesMailboxConfig:
    source_system: str
    poll_interval: int
    transport_mode: str
    mailbox_identity: str
    public_outbox_dir: str
    private_outbox_dir: str
    public_vault_dir: str
    private_vault_dir: str
    write_vault_markdown: bool
    graph_tenant_id: str
    graph_client_id: str
    graph_client_secret: str
    graph_mailbox: str
    graph_page_size: int
    graph_timeout: int

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "NotesMailboxConfig":
        def get(name: str, default: str = "") -> str:
            return str(env.get(name, default) or "").strip()

        def get_bool(name: str, default: str = "1") -> bool:
            return get(name, default).lower() not in {"0", "false", "no", "off"}

        return cls(
            source_system=get("NOTES_SOURCE_SYSTEM", "notes_mailbox"),
            poll_interval=max(15, int(get("NOTES_POLL_INTERVAL", "60") or "60")),
            transport_mode=get("NOTES_TRANSPORT_MODE", "manual").lower() or "manual",
            mailbox_identity=get("NOTES_MAILBOX_IDENTITY", NOTES_MAILBOX_IDENTITY),
            public_outbox_dir=get("NOTES_PUBLIC_OUTBOX_DIR", str(ROOT / "tmp" / "notes_public_outbox")),
            private_outbox_dir=get("NOTES_PRIVATE_OUTBOX_DIR", str(ROOT / "tmp" / "notes_private_outbox")),
            public_vault_dir=get("NOTES_PUBLIC_VAULT_DIR", "/mnt/c/Users/paul/Documents/AI-Vault/Inbox"),
            private_vault_dir=get("NOTES_PRIVATE_VAULT_DIR", "/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/notes"),
            write_vault_markdown=get_bool("NOTES_WRITE_VAULT_MARKDOWN", "1"),
            graph_tenant_id=get("NOTES_GRAPH_TENANT_ID"),
            graph_client_id=get("NOTES_GRAPH_CLIENT_ID"),
            graph_client_secret=get("NOTES_GRAPH_CLIENT_SECRET"),
            graph_mailbox=get("NOTES_GRAPH_MAILBOX", get("NOTES_MAILBOX_IDENTITY", NOTES_MAILBOX_IDENTITY)),
            graph_page_size=max(1, min(25, int(get("NOTES_GRAPH_PAGE_SIZE", "10") or "10"))),
            graph_timeout=max(5, int(get("NOTES_GRAPH_TIMEOUT", "30") or "30")),
        )


class NotesMailboxProcessor:
    def __init__(self, config: NotesMailboxConfig):
        self.config = config
        self.public_dir = Path(config.public_outbox_dir)
        self.private_dir = Path(config.private_outbox_dir)
        self.public_vault_dir = Path(config.public_vault_dir)
        self.private_vault_dir = Path(config.private_vault_dir)
        self.state_path = self.public_dir.parent / "notes_mailbox_state.json"
        self.public_dir.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if config.write_vault_markdown:
            self.public_vault_dir.mkdir(parents=True, exist_ok=True)
            self.private_vault_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"processed_graph_ids": []}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("processed_graph_ids", [])
                return data
        except Exception:
            pass
        return {"processed_graph_ids": []}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def has_processed_graph_id(self, graph_id: str) -> bool:
        return bool(graph_id and graph_id in set(self.state.get("processed_graph_ids") or []))

    def record_processed_graph_id(self, graph_id: str) -> None:
        if not graph_id:
            return
        ids = list(self.state.get("processed_graph_ids") or [])
        if graph_id not in ids:
            ids.append(graph_id)
        self.state["processed_graph_ids"] = ids[-5000:]
        self._save_state()

    def _safe_subject(self, subject: str, default: str = "note") -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(subject or default))[:80].strip("_")
        return safe or default

    def _title_from_message(self, message: dict[str, str], route: str) -> str:
        subject = str(message.get("subject") or "").strip()
        title = re.sub(r"^\s*(?:private|vault|note|notes|memo)\s*:\s*", "", subject, flags=re.IGNORECASE).strip()
        if not title:
            title = "Private note" if route == "private_vault" else "Inbox note"
        return title

    def _markdown_path(self, target_dir: Path, title: str, message: dict[str, str]) -> Path:
        received_at = str(message.get("received_at") or "").strip()
        date = received_at[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", received_at) else datetime.now(timezone.utc).date().isoformat()
        slug = self._safe_subject(title.replace(" ", "-").replace("/", "-"), "note")
        stem = f"{date}-{slug}"
        path = target_dir / f"{stem}.md"
        if not path.exists():
            return path
        fingerprint = hashlib.sha1(
            (str(message.get("message_id") or "") + str(message.get("graph_message_id") or "") + str(message.get("text_body") or "")).encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()[:8]
        return target_dir / f"{stem}-{fingerprint}.md"

    def _render_markdown(self, *, title: str, route: str, reason: str, message: dict[str, str]) -> str:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        source_type = "private-email-note" if route == "private_vault" else "email-note"
        tags = ["private", "email-note"] if route == "private_vault" else ["inbox", "email-note"]
        body = str(message.get("text_body") or "").strip() or "(empty note)"
        subject = str(message.get("subject") or "").replace('"', "'")
        from_email = str(message.get("from_email") or "").replace('"', "'")
        message_id = str(message.get("message_id") or "").replace('"', "'")
        return "\n".join(
            [
                "---",
                f'title: "{title.replace(chr(34), chr(39))}"',
                f"created: {now}",
                f"source: {self.config.source_system}",
                f"source_type: {source_type}",
                f"route: {route}",
                f"route_reason: {reason}",
                f"from_email: \"{from_email}\"",
                f"mailbox_identity: {self.config.mailbox_identity}",
                f"original_subject: \"{subject}\"",
                f"message_id: \"{message_id}\"",
                f"tags: [{', '.join(tags)}]",
                f"wiki-ready: {'false' if route == 'private_vault' else 'true'}",
                "---",
                "",
                f"# {title}",
                "",
                body,
                "",
            ]
        )

    def _write_vault_markdown(self, route: str, reason: str, message: dict[str, str]) -> str:
        target_dir = self.private_vault_dir if route == "private_vault" else self.public_vault_dir
        title = self._title_from_message(message, route)
        output_path = self._markdown_path(target_dir, title, message)
        output_path.write_text(self._render_markdown(title=title, route=route, reason=reason, message=message), encoding="utf-8")
        return str(output_path)

    def process_message(self, message: dict[str, str]) -> dict[str, str]:
        route = classify_notes_mailbox_route(message.get("subject", ""), message.get("text_body", ""))
        if route["route"] in {"unsupported_market_intel", "rejected_lab_result_error"}:
            return {
                "status": "rejected",
                "route": route["route"],
                "reason": route["reason"],
                "mailbox_identity": self.config.mailbox_identity,
            }

        target_dir = self.private_dir if route["route"] == "private_vault" else self.public_dir
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_subject = self._safe_subject(message.get("subject", "note"))
        output_path = target_dir / f"{timestamp}_{safe_subject}.json"
        vault_path = self._write_vault_markdown(route["route"], route["reason"], message) if self.config.write_vault_markdown else ""
        payload = {
            "mailbox_identity": self.config.mailbox_identity,
            "source_system": self.config.source_system,
            "received_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "route": route["route"],
            "reason": route["reason"],
            "vault_path": vault_path,
            "message": message,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return {
            "status": "outbox_written",
            "route": route["route"],
            "reason": route["reason"],
            "path": str(output_path),
            "vault_path": vault_path,
            "mailbox_identity": self.config.mailbox_identity,
        }


def _strip_html(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</(script|style)>", " ", str(value or ""))
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


class NotesGraphClient:
    def __init__(self, config: NotesMailboxConfig):
        self.config = config
        self._access_token = ""
        self._token_expires_at = 0.0

    def _token(self) -> str:
        import time

        if self._access_token and time.time() < self._token_expires_at - 120:
            return self._access_token
        url = f"https://login.microsoftonline.com/{self.config.graph_tenant_id}/oauth2/v2.0/token"
        response = requests.post(
            url,
            data={
                "client_id": self.config.graph_client_id,
                "client_secret": self.config.graph_client_secret,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            },
            timeout=self.config.graph_timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                payload = response.json()
                detail = f"{payload.get('error', '')}: {payload.get('error_description', '')}"
            except Exception:
                detail = (response.text or "").strip()[:500]
            raise RuntimeError(f"Microsoft Graph token request failed ({response.status_code}): {detail}") from exc
        payload = response.json()
        self._access_token = str(payload.get("access_token") or "")
        self._token_expires_at = time.time() + int(payload.get("expires_in") or 3600)
        if not self._access_token:
            raise RuntimeError("Microsoft Graph token response did not include access_token")
        return self._access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token()}", "Accept": "application/json"}

    def list_unread_messages(self) -> list[dict]:
        url = f"https://graph.microsoft.com/v1.0/users/{self.config.graph_mailbox}/mailFolders/inbox/messages"
        response = requests.get(
            url,
            headers=self._headers(),
            params={
                "$filter": "isRead eq false",
                "$orderby": "receivedDateTime asc",
                "$top": str(self.config.graph_page_size),
                "$select": "id,subject,from,toRecipients,receivedDateTime,body,internetMessageId,hasAttachments",
            },
            timeout=self.config.graph_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("value") or [])

    def mark_read(self, message_id: str) -> None:
        url = f"https://graph.microsoft.com/v1.0/users/{self.config.graph_mailbox}/messages/{message_id}"
        response = requests.patch(
            url,
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"isRead": True},
            timeout=self.config.graph_timeout,
        )
        response.raise_for_status()


def _email_address(value: dict) -> tuple[str, str]:
    addr = dict((value or {}).get("emailAddress") or {})
    return str(addr.get("name") or "").strip(), str(addr.get("address") or "").strip()


def _normalize_graph_message(item: dict) -> dict[str, str]:
    from_name, from_email = _email_address(dict(item.get("from") or {}))
    body = dict(item.get("body") or {})
    body_content = str(body.get("content") or "")
    content_type = str(body.get("contentType") or "").strip().lower()
    text_body = _strip_html(body_content) if content_type == "html" else body_content.strip()
    html_body = body_content if content_type == "html" else ""
    recipients = list(item.get("toRecipients") or [])
    to_name, to_email = _email_address(dict(recipients[0] if recipients else {}))
    return {
        "message_id": str(item.get("internetMessageId") or item.get("id") or "").strip(),
        "graph_message_id": str(item.get("id") or "").strip(),
        "from_name": from_name,
        "from_email": from_email,
        "to_name": to_name,
        "to_email": to_email,
        "subject": str(item.get("subject") or "").strip(),
        "received_at": str(item.get("receivedDateTime") or "").strip(),
        "text_body": text_body,
        "html_body": html_body,
    }


def _read_notes_config() -> tuple[NotesMailboxConfig, str]:
    env_path = ROOT / "worker" / "config.env"
    file_vars = load_env_file(env_path)
    config = NotesMailboxConfig.from_env(file_vars)
    log_level = str(file_vars.get("LOG_LEVEL", "INFO") or "INFO").upper()
    return config, log_level


def main() -> int:
    config, log_level = _read_notes_config()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    logging.info(
        "Notes mailbox worker started: transport=%s mailbox=%s poll=%ss public_outbox=%s private_outbox=%s",
        config.transport_mode,
        config.mailbox_identity,
        config.poll_interval,
        config.public_outbox_dir,
        config.private_outbox_dir,
    )

    stop_event = threading.Event()

    def _stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if config.transport_mode != "manual":
        logging.warning(
            "Notes mailbox transport mode '%s' selected.",
            config.transport_mode,
        )
    graph_client = NotesGraphClient(config) if config.transport_mode == "graph" else None
    processor = NotesMailboxProcessor(config)

    while not stop_event.wait(config.poll_interval):
        if graph_client is None:
            logging.debug("Notes mailbox worker heartbeat: transport=%s mailbox=%s", config.transport_mode, config.mailbox_identity)
            continue
        try:
            processed = 0
            rejected = 0
            for item in graph_client.list_unread_messages():
                message = _normalize_graph_message(item)
                graph_id = str(message.get("graph_message_id") or "").strip()
                if processor.has_processed_graph_id(graph_id):
                    logging.debug("Notes mailbox skipping already processed graph message id=%s subject=%s", graph_id, message.get("subject", ""))
                    continue
                result = processor.process_message(message)
                processor.record_processed_graph_id(graph_id)
                if result["status"] == "rejected":
                    rejected += 1
                    logging.warning(
                        "Notes mailbox rejected message subject=%s route=%s reason=%s",
                        message.get("subject", ""),
                        result.get("route", ""),
                        result.get("reason", ""),
                    )
                else:
                    processed += 1
                    logging.info(
                        "Notes mailbox processed subject=%s route=%s path=%s vault_path=%s",
                        message.get("subject", ""),
                        result.get("route", ""),
                        result.get("path", ""),
                        result.get("vault_path", ""),
                    )
                if graph_id:
                    try:
                        graph_client.mark_read(graph_id)
                    except Exception as exc:
                        logging.warning(
                            "Notes mailbox left message unread in Graph id=%s; local dedupe recorded it. "
                            "This is expected with least-privilege Mail.Read. detail=%s",
                            graph_id,
                            exc,
                        )
            if processed or rejected:
                logging.info("Notes mailbox poll complete: processed=%s rejected=%s", processed, rejected)
        except Exception:
            logging.exception("Notes mailbox poll failed")

    logging.info("Notes mailbox worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
