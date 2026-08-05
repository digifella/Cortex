#!/usr/bin/env python3
"""Narrow outbound-only document delivery service for the Hermes SP4 agent."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mimetypes
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ALLOWED_EXTENSIONS = {
    ".csv", ".docx", ".jpg", ".jpeg", ".md", ".pdf", ".png",
    ".pptx", ".txt", ".xlsx", ".zip",
}
MAX_SUBJECT = 180
MAX_BODY = 4000


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _safe_filename(value: str) -> str:
    name = str(value or "").strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("filename must be a plain basename")
    if len(name) > 160 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ ()-]*", name):
        raise ValueError("filename contains unsafe characters")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("file type is not allowed")
    return name


class DeliveryConfig:
    def __init__(self) -> None:
        self.tenant_id = _required_env("HERMES_DELIVERY_TENANT_ID")
        self.client_id = _required_env("HERMES_DELIVERY_CLIENT_ID")
        self.client_secret = _required_env("HERMES_DELIVERY_CLIENT_SECRET")
        self.sender = _required_env("HERMES_DELIVERY_SENDER")
        self.recipient = _required_env("HERMES_DELIVERY_RECIPIENT")
        self.api_token = _required_env("HERMES_DELIVERY_API_TOKEN")
        self.host = os.environ.get("HERMES_DELIVERY_HOST", "127.0.0.1").strip()
        self.port = int(os.environ.get("HERMES_DELIVERY_PORT", "7341"))
        self.max_bytes = int(os.environ.get("HERMES_DELIVERY_MAX_BYTES", str(20 * 1024 * 1024)))
        self.rate_per_hour = int(os.environ.get("HERMES_DELIVERY_RATE_PER_HOUR", "6"))
        self.audit_log = Path(
            os.environ.get("HERMES_DELIVERY_AUDIT_LOG", "/tmp/hermes-delivery-audit.jsonl")
        )


class GraphMailer:
    def __init__(self, config: DeliveryConfig):
        self.config = config

    def _access_token(self) -> str:
        body = urllib.parse.urlencode({
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        }).encode()
        url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"
        request = urllib.request.Request(url, data=body, method="POST")
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)["access_token"]

    def send(self, *, filename: str, content: bytes, subject: str, body: str) -> None:
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        payload = {
            "message": {
                "subject": subject,
                "body": {"contentType": "Text", "content": body},
                "toRecipients": [{"emailAddress": {"address": self.config.recipient}}],
                "attachments": [{
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": filename,
                    "contentType": mime_type,
                    "contentBytes": base64.b64encode(content).decode("ascii"),
                }],
            },
            "saveToSentItems": True,
        }
        sender = urllib.parse.quote(self.config.sender, safe="@")
        request = urllib.request.Request(
            f"https://graph.microsoft.com/v1.0/users/{sender}/sendMail",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._access_token()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 202:
                raise RuntimeError(f"Graph sendMail returned HTTP {response.status}")


class DeliveryRuntime:
    def __init__(self, config: DeliveryConfig, mailer: GraphMailer | None = None):
        self.config = config
        self.mailer = mailer or GraphMailer(config)
        self._recent: deque[float] = deque()
        self._lock = threading.Lock()

    def _take_rate_slot(self) -> None:
        now = time.time()
        with self._lock:
            while self._recent and self._recent[0] < now - 3600:
                self._recent.popleft()
            if len(self._recent) >= self.config.rate_per_hour:
                raise PermissionError("hourly delivery limit reached")
            self._recent.append(now)

    def deliver(self, payload: dict[str, Any]) -> dict[str, Any]:
        if any(key in payload for key in ("to", "cc", "bcc", "recipient", "reply_to")):
            raise ValueError("recipient fields are forbidden; delivery target is fixed")
        filename = _safe_filename(payload.get("filename", ""))
        subject = str(payload.get("subject") or f"Hermes document: {filename}").strip()
        body = str(payload.get("body") or "Document delivered by your Hermes agent.").strip()
        if not subject or len(subject) > MAX_SUBJECT or len(body) > MAX_BODY:
            raise ValueError("subject or body exceeds the allowed length")
        try:
            content = base64.b64decode(str(payload.get("content_b64", "")), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("content_b64 is invalid") from exc
        if not content or len(content) > self.config.max_bytes:
            raise ValueError("attachment is empty or exceeds the size limit")
        self._take_rate_slot()
        self.mailer.send(filename=filename, content=content, subject=subject, body=body)
        event = {
            "time": int(time.time()),
            "recipient": self.config.recipient,
            "filename": filename,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        self.config.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with self.config.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return {"ok": True, "filename": filename, "bytes": len(content)}


class DeliveryHandler(BaseHTTPRequestHandler):
    runtime: DeliveryRuntime

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._json(200, {"ok": True}) if self.path == "/health" else self._json(404, {"ok": False})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/deliver":
            self._json(404, {"ok": False})
            return
        supplied = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied, self.runtime.config.api_token):
            self._json(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.runtime.config.max_bytes * 2:
                raise ValueError("request size is invalid")
            payload = json.loads(self.rfile.read(length))
            self._json(200, self.runtime.deliver(payload))
        except PermissionError as exc:
            self._json(429, {"ok": False, "error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
        except (urllib.error.URLError, RuntimeError) as exc:
            self._json(502, {"ok": False, "error": "mail provider rejected delivery"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    config = DeliveryConfig()
    DeliveryHandler.runtime = DeliveryRuntime(config)
    server = ThreadingHTTPServer((config.host, config.port), DeliveryHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
