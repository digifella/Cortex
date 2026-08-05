"""Fail-closed approval policy for Hermes document email delivery."""

from __future__ import annotations

import hashlib
import re


DELIVERY = re.compile(r"^\s*/home/paul/\.local/bin/kb-email-me(?:\s|$)")
SENSITIVE_MARKERS = (
    ".hermes-delivery-client.env",
    "100.118.92.17:7341",
    "hermes_delivery_api_token",
)


def _pre_tool_call(tool_name="", args=None, tool_call_id="", **_kwargs):
    if tool_name not in {"terminal", "terminal_tool"}:
        return None
    command = str((args or {}).get("command", ""))
    lowered = command.lower()
    if DELIVERY.match(command):
        # A call-specific key makes even an "always" choice apply only to this
        # invocation; the next delivery must prompt again.
        fallback = hashlib.sha256(command.encode("utf-8")).hexdigest()[:20]
        call_key = tool_call_id or fallback
        display = command.strip()
        if len(display) > 500:
            display = display[:497] + "..."
        return {
            "action": "approve",
            "message": "Approve this one email to the configured fixed recipient?\n" + display,
            "rule_key": "email-delivery:" + call_key,
        }
    if any(marker in lowered for marker in SENSITIVE_MARKERS):
        return {
            "action": "block",
            "message": "BLOCKED: use kb-email-me; direct credential or broker access is prohibited",
        }
    return None


def register(ctx):
    ctx.register_hook("pre_tool_call", _pre_tool_call)
