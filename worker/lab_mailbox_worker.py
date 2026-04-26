from __future__ import annotations

import argparse
import html
import json
import logging
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


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
class LabMailboxConfig:
    transport_mode: str
    poll_interval: int
    graph_tenant_id: str
    graph_client_id: str
    graph_client_secret: str
    graph_mailbox: str
    graph_page_size: int
    graph_timeout: int
    webhook_url: str
    webhook_secret: str
    suppress_replies: bool
    state_path: Path

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "LabMailboxConfig":
        def get(name: str, default: str = "") -> str:
            return str(env.get(name, default) or "").strip()

        def get_bool(name: str, default: str = "0") -> bool:
            return get(name, default).lower() in {"1", "true", "yes", "on"}

        return cls(
            transport_mode=get("LAB_TRANSPORT_MODE", "manual").lower() or "manual",
            poll_interval=max(15, int(get("LAB_POLL_INTERVAL", "60") or "60")),
            graph_tenant_id=get("LAB_GRAPH_TENANT_ID"),
            graph_client_id=get("LAB_GRAPH_CLIENT_ID"),
            graph_client_secret=get("LAB_GRAPH_CLIENT_SECRET"),
            graph_mailbox=get("LAB_GRAPH_MAILBOX", "lab@longboardfella.com.au"),
            graph_page_size=max(1, min(25, int(get("LAB_GRAPH_PAGE_SIZE", "10") or "10"))),
            graph_timeout=max(5, int(get("LAB_GRAPH_TIMEOUT", "30") or "30")),
            webhook_url=get("LAB_WEBHOOK_URL", "https://longboardfella.com.au/admin/email_job_webhook.php"),
            webhook_secret=get("LAB_WEBHOOK_SECRET"),
            suppress_replies=get_bool("LAB_SUPPRESS_REPLIES", "0"),
            state_path=Path(get("LAB_STATE_FILE", str(ROOT / "tmp" / "lab_mailbox_state.json"))),
        )

    def validate(self) -> None:
        missing = [
            name
            for name, value in {
                "LAB_GRAPH_TENANT_ID": self.graph_tenant_id,
                "LAB_GRAPH_CLIENT_ID": self.graph_client_id,
                "LAB_GRAPH_CLIENT_SECRET": self.graph_client_secret,
                "LAB_GRAPH_MAILBOX": self.graph_mailbox,
                "LAB_WEBHOOK_URL": self.webhook_url,
                "LAB_WEBHOOK_SECRET": self.webhook_secret,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError("Missing Lab mailbox config keys: " + ", ".join(missing))


def strip_html(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</(script|style)>", " ", str(value or ""))
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"(?i)</div\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class LabState:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"processed_graph_ids": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("processed_graph_ids", [])
                return data
        except Exception:
            pass
        return {"processed_graph_ids": []}

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def has_processed(self, graph_id: str) -> bool:
        return bool(graph_id and graph_id in set(self.data.get("processed_graph_ids") or []))

    def record_processed(self, graph_id: str) -> None:
        if not graph_id:
            return
        ids = list(self.data.get("processed_graph_ids") or [])
        if graph_id not in ids:
            ids.append(graph_id)
        self.data["processed_graph_ids"] = ids[-5000:]
        self.save()


class LabGraphClient:
    def __init__(self, config: LabMailboxConfig):
        self.config = config
        self._access_token = ""
        self._token_expires_at = 0.0

    def _token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - 120:
            return self._access_token
        response = requests.post(
            f"https://login.microsoftonline.com/{self.config.graph_tenant_id}/oauth2/v2.0/token",
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
            detail = response.text[:500]
            try:
                payload = response.json()
                detail = f"{payload.get('error', '')}: {payload.get('error_description', '')}"
            except Exception:
                pass
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
        response = requests.get(
            f"https://graph.microsoft.com/v1.0/users/{self.config.graph_mailbox}/mailFolders/inbox/messages",
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
        return list(response.json().get("value") or [])

    def mark_read(self, message_id: str) -> None:
        response = requests.patch(
            f"https://graph.microsoft.com/v1.0/users/{self.config.graph_mailbox}/messages/{message_id}",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"isRead": True},
            timeout=self.config.graph_timeout,
        )
        response.raise_for_status()

    def send_reply(self, to_email: str, subject: str, body: str) -> None:
        if not to_email:
            return
        response = requests.post(
            f"https://graph.microsoft.com/v1.0/users/{self.config.graph_mailbox}/sendMail",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={
                "message": {
                    "subject": subject,
                    "body": {"contentType": "Text", "content": body},
                    "toRecipients": [{"emailAddress": {"address": to_email}}],
                },
                "saveToSentItems": True,
            },
            timeout=self.config.graph_timeout,
        )
        response.raise_for_status()


def email_address(value: dict) -> tuple[str, str]:
    addr = dict((value or {}).get("emailAddress") or {})
    return str(addr.get("name") or "").strip(), str(addr.get("address") or "").strip()


def normalize_graph_message(item: dict) -> dict:
    from_name, from_email = email_address(dict(item.get("from") or {}))
    recipients = list(item.get("toRecipients") or [])
    to_name, to_email = email_address(dict(recipients[0] if recipients else {}))
    body = dict(item.get("body") or {})
    content = str(body.get("content") or "")
    content_type = str(body.get("contentType") or "").strip().lower()
    text_body = strip_html(content) if content_type == "html" else content.strip()
    html_body = content if content_type == "html" else ""
    return {
        "message_id": str(item.get("internetMessageId") or item.get("id") or "").strip(),
        "graph_message_id": str(item.get("id") or "").strip(),
        "from_name": from_name,
        "from_email": from_email,
        "to_name": to_name,
        "to_email": to_email,
        "subject": str(item.get("subject") or "").strip(),
        "received_at": str(item.get("receivedDateTime") or datetime.now(timezone.utc).isoformat()).strip(),
        "text_body": text_body,
        "html_body": html_body,
        "attachments": [],
    }


class LabWebhookClient:
    def __init__(self, config: LabMailboxConfig):
        self.config = config

    def post_message(self, message: dict, *, dry_run: bool = False) -> dict:
        response = requests.post(
            self.config.webhook_url,
            headers={
                "Content-Type": "application/json",
                "X-Longboardfella-Webhook-Secret": self.config.webhook_secret,
            },
            params={"dry_run": "1"} if dry_run else {},
            json=message,
            timeout=self.config.graph_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Lab webhook returned ok=false: {payload}")
        return payload


def poll_once(config: LabMailboxConfig, *, dry_run: bool = False) -> tuple[int, int]:
    graph = LabGraphClient(config)
    webhook = LabWebhookClient(config)
    state = LabState(config.state_path)
    processed = 0
    skipped = 0

    for item in graph.list_unread_messages():
        message = normalize_graph_message(item)
        graph_id = str(message.get("graph_message_id") or "").strip()
        if state.has_processed(graph_id):
            skipped += 1
            logging.debug("Skipping already processed graph id=%s subject=%s", graph_id, message.get("subject", ""))
            continue

        result = webhook.post_message(message, dry_run=dry_run)
        reply = result.get("reply") if isinstance(result.get("reply"), dict) else None
        if reply and not reply.get("suppressed") and not config.suppress_replies and not dry_run:
            graph.send_reply(
                str(reply.get("to") or message.get("from_email") or "").strip(),
                str(reply.get("subject") or f"Re: {message.get('subject', '')}").strip(),
                str(reply.get("body") or "").strip(),
            )

        if not dry_run and graph_id:
            graph.mark_read(graph_id)
            state.record_processed(graph_id)
        processed += 1
        logging.info(
            "Processed Lab email subject=%s outcome=%s job_ids=%s dry_run=%s",
            message.get("subject", ""),
            result.get("outcome", ""),
            result.get("job_ids", []),
            dry_run,
        )

    return processed, skipped


def read_config() -> tuple[LabMailboxConfig, str]:
    env = load_env_file(ROOT / "worker" / "config.env")
    config = LabMailboxConfig.from_env(env)
    return config, str(env.get("LOG_LEVEL", "INFO") or "INFO").upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Lab mailbox via Graph and post emails to website job webhook.")
    parser.add_argument("--once", action="store_true", help="Poll once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Post to webhook dry_run=1 and do not mark messages read or send replies.")
    args = parser.parse_args()

    config, log_level = read_config()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    config.validate()
    if config.transport_mode != "graph":
        logging.warning("LAB_TRANSPORT_MODE=%s; worker will idle until set to graph.", config.transport_mode)

    stop_event = threading.Event()

    def stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    logging.info("Lab mailbox worker started: mailbox=%s webhook=%s once=%s dry_run=%s", config.graph_mailbox, config.webhook_url, args.once, args.dry_run)

    while True:
        if config.transport_mode == "graph":
            try:
                processed, skipped = poll_once(config, dry_run=args.dry_run)
                logging.info("Lab mailbox poll complete: processed=%s skipped=%s", processed, skipped)
            except Exception:
                logging.exception("Lab mailbox poll failed")
                if args.once:
                    return 1
        if args.once:
            return 0
        if stop_event.wait(config.poll_interval):
            logging.info("Lab mailbox worker stopped")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
