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
_LAB_RESULT_RE = re.compile(r"(?i)\b(?:youtube summariser job|the lab\s*[·-]\s*longboardfella|source_type:\s*youtube_summary)\b")
_GENERATED_CONTENT_ERROR_RE = re.compile(
    r"(?i)\[(?:error generating(?:\s+\w+)?|error|failed to generate|could not generate)[^\]]*\]"
)


def classify_notes_mailbox_route(subject: str, body_text: str = "") -> Dict[str, str]:
    clean_subject = str(subject or "").strip()
    subject_lower = clean_subject.lower()
    body_text = str(body_text or "")
    body_lower = body_text.strip().lower()

    if _LAB_RESULT_RE.search(body_text) and _GENERATED_CONTENT_ERROR_RE.search(body_text):
        return {"route": "rejected_lab_result_error", "reason": "lab_result_contains_generation_error"}

    if _INTEL_PREFIX_RE.search(clean_subject) or _INTEL_ROUTING_RE.search(clean_subject) or _CSV_IMPORT_RE.search(subject_lower):
        return {"route": "unsupported_market_intel", "reason": "market_intel_shape"}

    if _PRIVATE_PREFIX_RE.search(clean_subject) or _PRIVATE_SUBJECT_RE.search(clean_subject):
        return {"route": "private_vault", "reason": "subject_marked_private"}

    if _PUBLIC_PREFIX_RE.search(clean_subject):
        return {"route": "public_stash", "reason": "explicit_public_note"}

    if any(token in body_lower for token in ("entity:", "organisation:", "organization:", "org:")):
        return {"route": "unsupported_market_intel", "reason": "body_market_intel_shape"}

    return {"route": "public_stash", "reason": "default_public_note"}
