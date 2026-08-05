#!/usr/bin/env python3
"""Interactively configure Hermes delivery without exposing secrets in chat/history."""

from __future__ import annotations

import getpass
import os
import secrets
import tempfile
from pathlib import Path


CONFIG = Path(__file__).with_name("config.env")


def prompt(label: str, *, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    value = (getpass.getpass if secret else input)(f"{label}{suffix}: ").strip()
    return value or default


def update_env(original: str, values: dict[str, str]) -> str:
    remaining = dict(values)
    output: list[str] = []
    for line in original.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in remaining:
                output.append(f"{key}={remaining.pop(key)}")
                continue
        output.append(line)
    if remaining:
        output.extend(["", "# Hermes outbound-only document delivery"])
        output.extend(f"{key}={value}" for key, value in remaining.items())
    return "\n".join(output) + "\n"


def main() -> None:
    tenant_id = prompt("Microsoft tenant ID")
    client_id = prompt("Hermes Delivery application client ID")
    client_secret = prompt("Hermes Delivery client secret", secret=True)
    recipient = prompt("Fixed destination email address")
    sender = prompt(
        "Sender shared mailbox",
        default="hermes-delivery@longboardfella.com.au",
    )
    if not all((tenant_id, client_id, client_secret, recipient, sender)):
        raise SystemExit("all values are required; nothing was changed")
    if recipient.casefold() == sender.casefold():
        raise SystemExit(
            "fixed destination must be the owner's inbox, not the sender shared mailbox; "
            "nothing was changed"
        )

    values = {
        "HERMES_DELIVERY_TENANT_ID": tenant_id,
        "HERMES_DELIVERY_CLIENT_ID": client_id,
        "HERMES_DELIVERY_CLIENT_SECRET": client_secret,
        "HERMES_DELIVERY_SENDER": sender,
        "HERMES_DELIVERY_RECIPIENT": recipient,
        "HERMES_DELIVERY_API_TOKEN": secrets.token_urlsafe(48),
        "HERMES_DELIVERY_HOST": "100.118.92.17",
        "HERMES_DELIVERY_PORT": "7341",
        "HERMES_DELIVERY_MAX_BYTES": str(20 * 1024 * 1024),
        "HERMES_DELIVERY_RATE_PER_HOUR": "6",
        "HERMES_DELIVERY_AUDIT_LOG": "/home/longboardfella/vault-rag-db/hermes-delivery-audit.jsonl",
    }
    original = CONFIG.read_text() if CONFIG.exists() else ""
    updated = update_env(original, values)
    fd, temporary = tempfile.mkstemp(prefix="config.env.", dir=CONFIG.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, CONFIG)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Configured {sender} -> {recipient}; secret and API token were not displayed.")


if __name__ == "__main__":
    main()
