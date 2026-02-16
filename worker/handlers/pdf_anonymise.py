from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from cortex_engine.anonymizer import AnonymizationMapping, AnonymizationOptions, DocumentAnonymizer


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    PDF anonymiser handler.

    Returns:
        {
            "output_data": dict,
            "output_file": Path | None
        }
    """
    if input_path is None:
        raise ValueError("pdf_anonymise requires an input file")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    confidence_threshold = float((input_data or {}).get("confidence_threshold", 0.3))
    options_payload = dict((input_data or {}).get("anonymization_options") or {})
    # Backward-compatible flat keys accepted at top-level input_data.
    for key in (
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
    ):
        if key in (input_data or {}) and key not in options_payload:
            options_payload[key] = (input_data or {}).get(key)
    options = AnonymizationOptions.from_input(options_payload)
    if progress_cb:
        progress_cb(15, "Starting anonymization", "anonymize_start")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before anonymization started")

    anonymizer = DocumentAnonymizer()
    mapping = AnonymizationMapping()
    if progress_cb:
        progress_cb(35, "Applying entity anonymization", "anonymize_processing")
    output_path, final_mapping = anonymizer.anonymize_single_file(
        input_path=input_path,
        output_path=None,
        mapping=mapping,
        confidence_threshold=confidence_threshold,
        options=options,
    )
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled after anonymization")
    if progress_cb:
        progress_cb(90, "Preparing output metadata", "anonymize_finalize")

    output_data = {
        "summary": "Processed via cortex_engine.anonymizer.DocumentAnonymizer",
        "entities_replaced": len(final_mapping.mappings),
        "confidence_threshold": confidence_threshold,
        "anonymization_options": {
            "redact_people": options.redact_people,
            "redact_organizations": options.redact_organizations,
            "redact_projects": options.redact_projects,
            "redact_locations": options.redact_locations,
            "redact_personal_pronouns": options.redact_personal_pronouns,
            "redact_company_names": options.redact_company_names,
            "custom_company_names_count": len(options.custom_company_names),
            "preserve_source_formatting": options.preserve_source_formatting,
        },
    }
    if progress_cb:
        progress_cb(100, "Anonymization complete", "done")
    return {"output_data": output_data, "output_file": Path(output_path)}
