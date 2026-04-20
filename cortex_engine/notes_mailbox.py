from __future__ import annotations

import re
from typing import Dict

NOTES_MAILBOX_IDENTITY = "notes@longboardfella.com.au"
_PRIVATE_SUBJECT_RE = re.compile(r"(?i)\b(?:private|sensitive|confidential)\b")
_PRIVATE_PREFIX_RE = re.compile(r"^\s*(?:private|vault)\s*:", re.IGNORECASE)
_PUBLIC_PREFIX_RE = re.compile(r"^\s*(?:note|notes|memo)\s*:", re.IGNORECASE)
_INTEL_PREFIX_RE = re.compile(r"^\s*intel\s*:", re.IGNORECASE)
_INTEL_ROUTING_RE = re.compile(r"(?i)(?:^|[\s\[\]()|;>])(?:entity|org|organisation|organization)\s*:")
_CSV_IMPORT_RE = re.compile(r"(?i)\bprofiles?\b.*\b(?:csv|import)\b")


def classify_notes_mailbox_route(subject: str, body_text: str = "") -> Dict[str, str]:
    clean_subject = str(subject or "").strip()
    subject_lower = clean_subject.lower()
    body_lower = str(body_text or "").strip().lower()

    if _INTEL_PREFIX_RE.search(clean_subject) or _INTEL_ROUTING_RE.search(clean_subject) or _CSV_IMPORT_RE.search(subject_lower):
        return {"route": "unsupported_market_intel", "reason": "market_intel_shape"}

    if _PRIVATE_PREFIX_RE.search(clean_subject) or _PRIVATE_SUBJECT_RE.search(clean_subject):
        return {"route": "private_vault", "reason": "subject_marked_private"}

    if _PUBLIC_PREFIX_RE.search(clean_subject):
        return {"route": "public_stash", "reason": "explicit_public_note"}

    if any(token in body_lower for token in ("entity:", "organisation:", "organization:", "org:")):
        return {"route": "unsupported_market_intel", "reason": "body_market_intel_shape"}

    return {"route": "public_stash", "reason": "default_public_note"}
