from __future__ import annotations

from typing import Any, Dict

from cortex_engine.email_handlers import subject_looks_like_csv_profile_import


def classify_mailbox_message(message: Dict[str, Any], persisted: Dict[str, Any]) -> str:
    subject = str(message.get("subject") or "").strip().lower()
    raw_text = str(message.get("raw_text") or "").strip().lower()
    attachments = list(persisted.get("attachments") or [])
    attachment_mimes = {str(item.get("mime_type") or "").strip().lower() for item in attachments}

    if subject_looks_like_csv_profile_import(message.get("subject", "")):
        return "csv_profile_import"
    if subject.startswith("intel:") or "[intel]" in subject:
        return "intel_extract"
    if any(
        mime in {
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        for mime in attachment_mimes
    ):
        return "document_analysis"
    if subject.startswith("note:") or any(
        token in raw_text for token in ("meeting with", "met with", "call with", "spoke with", "note:", "notes:")
    ):
        return "intel_note"
    return "intel_note"
