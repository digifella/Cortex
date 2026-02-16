"""Shared contract helpers for website <-> Cortex handoff."""

from __future__ import annotations

from typing import Any, Dict, Optional
import uuid

HANDOFF_CONTRACT_VERSION = "2026-02-15.v1"

SUPPORTED_JOB_TYPES = ["pdf_anonymise", "pdf_textify", "url_ingest"]

SUPPORTED_ANONYMIZER_OPTIONS = [
    "redact_people",
    "redact_organizations",
    "redact_projects",
    "redact_locations",
    "redact_emails",
    "redact_phones",
    "redact_urls",
    "redact_headers_footers",
    "redact_personal_pronouns",
    "redact_company_names",
    "custom_company_names",
    "preserve_source_formatting",
]


def ensure_trace_id(trace_id: Optional[str] = None) -> str:
    value = (trace_id or "").strip()
    if value:
        return value
    return f"trace-{uuid.uuid4()}"


def normalize_handoff_metadata(
    job: Optional[Dict[str, Any]] = None,
    input_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    job = job or {}
    input_data = input_data or {}

    trace_id = ensure_trace_id(
        input_data.get("trace_id")
        or job.get("trace_id")
        or job.get("trace")
    )
    idempotency_key = (
        str(input_data.get("idempotency_key") or job.get("idempotency_key") or "").strip()
    )
    source_system = str(
        input_data.get("source_system") or job.get("source_system") or "website"
    ).strip()

    return {
        "contract_version": HANDOFF_CONTRACT_VERSION,
        "trace_id": trace_id,
        "idempotency_key": idempotency_key,
        "source_system": source_system,
    }
