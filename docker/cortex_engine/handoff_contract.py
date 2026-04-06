"""Shared contract helpers for website <-> Cortex handoff."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import uuid

HANDOFF_CONTRACT_VERSION = "2026-02-15.v1"
DEFAULT_TENANT_ID = "default"
DEFAULT_PROJECT_ID = "default"

SUPPORTED_JOB_TYPES = [
    "pdf_anonymise",
    "pdf_textify",
    "included_study_extract",
    "url_ingest",
    "research_resolve",
    "org_profile_refresh",
    "cortex_sync",
    "intel_extract",
    "stakeholder_profile_sync",
    "signal_ingest",
    "signal_digest",
    "stakeholder_graph_view",
    "org_context_sync",
]

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

SUPPORTED_TEXTIFY_STRATEGIES = {"docling", "qwen30b", "hybrid"}
SUPPORTED_CLEANUP_PROVIDERS = {"ollama", "lmstudio"}
SUPPORTED_INCLUDED_STUDY_PROVIDERS = {"anthropic", "gemini"}
SUPPORTED_INCLUDED_STUDY_SCOPES = {"all_trials", "rct_or_clinical"}
SUPPORTED_INCLUDED_STUDY_OUTPUT_DETAILS = {"reference_map", "detailed_fields"}
SUPPORTED_INCLUDED_STUDY_DOWNLOAD_FORMATS = {"json", "xlsx", "csv"}
SUPPORTED_TEXTIFY_OPTION_KEYS = {
    "use_vision",
    "pdf_strategy",
    "cleanup_provider",
    "cleanup_model",
    "docling_timeout_seconds",
    "image_description_timeout_seconds",
    "image_enrich_max_seconds",
}


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
    tenant_id = str(
        input_data.get("tenant_id") or job.get("tenant_id") or DEFAULT_TENANT_ID
    ).strip() or DEFAULT_TENANT_ID
    project_id = str(
        input_data.get("project_id") or job.get("project_id") or DEFAULT_PROJECT_ID
    ).strip() or DEFAULT_PROJECT_ID

    return {
        "contract_version": HANDOFF_CONTRACT_VERSION,
        "trace_id": trace_id,
        "idempotency_key": idempotency_key,
        "source_system": source_system,
        "tenant_id": tenant_id,
        "project_id": project_id,
    }


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_positive_int(value: Any, default: int, name: str) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {name}: {value!r}") from e
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _coerce_positive_float(value: Any, default: float, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except Exception as e:
        raise ValueError(f"Invalid {name}: {value!r}") from e
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _normalize_affiliation_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"current", "former", "board", "advisory", "consultant", "affiliate"}:
        return text
    return "current"


def _normalize_affiliation_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"confirmed", "probable", "speculative"}:
        return text
    return "confirmed"


def _normalize_affiliations(value: Any) -> list[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("stakeholder_profile_sync affiliations must be an array when provided")

    normalized: list[Dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"stakeholder_profile_sync affiliations[{idx}] must be an object")
        org_name_text = str(item.get("org_name_text") or item.get("org_name") or "").strip()
        if not org_name_text:
            continue
        normalized.append(
            {
                "org_name_text": org_name_text,
                "role": str(item.get("role") or "").strip(),
                "affiliation_type": _normalize_affiliation_type(item.get("affiliation_type") or item.get("type")),
                "confidence": _normalize_affiliation_confidence(item.get("confidence")),
                "is_primary": 1 if _coerce_bool(item.get("is_primary"), False) else 0,
                "start_date": str(item.get("start_date") or "").strip(),
                "end_date": str(item.get("end_date") or "").strip(),
                "source": str(item.get("source") or "").strip(),
            }
        )

    if normalized and not any(item["is_primary"] for item in normalized):
        normalized[0]["is_primary"] = 1

    primary_seen = False
    for item in normalized:
        if item["is_primary"] and not primary_seen:
            primary_seen = True
            continue
        if item["is_primary"]:
            item["is_primary"] = 0
    return normalized


def _normalize_string_array(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array when provided")
    seen: set[str] = set()
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


def _normalize_linkedin_connections(value: Any, field_name: str) -> list[Dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array when provided")
    normalized: list[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{idx}] must be an object")
        member = str(item.get("member") or item.get("email") or "").strip()
        degree = str(item.get("degree") or "").strip()
        if not member:
            continue
        dedupe_key = (member.lower(), degree.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append({"member": member, "degree": degree})
    return normalized


def _normalize_industry_affiliations(value: Any, field_name: str) -> list[Dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array when provided")
    normalized: list[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{idx}] must be an object")
        industry_profile_key = str(
            item.get("industry_profile_key") or item.get("profile_key") or item.get("industry_id") or ""
        ).strip()
        industry_name = str(
            item.get("industry_name") or item.get("canonical_name") or item.get("name") or ""
        ).strip()
        if not industry_profile_key and not industry_name:
            continue
        role = str(item.get("role") or "").strip()
        affiliation_type = str(item.get("affiliation_type") or item.get("type") or "active").strip().lower() or "active"
        dedupe_key = (industry_profile_key.lower(), industry_name.lower(), role.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(
            {
                "industry_profile_key": industry_profile_key,
                "industry_name": industry_name,
                "role": role,
                "affiliation_type": affiliation_type,
                "source": str(item.get("source") or "").strip(),
            }
        )
    return normalized


def _normalize_org_strategic_profile(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object when provided")

    industries_value = value.get("industries")
    industries: list[str] = []
    if industries_value is not None:
        if not isinstance(industries_value, list):
            raise ValueError(f"{field_name}.industries must be an array when provided")
        seen: set[str] = set()
        for idx, item in enumerate(industries_value):
            if isinstance(item, dict):
                name = str(
                    item.get("industry_name")
                    or item.get("canonical_name")
                    or item.get("name")
                    or item.get("label")
                    or ""
                ).strip()
            else:
                name = str(item or "").strip()
            if not name:
                continue
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            industries.append(name)

    return {
        "description": str(value.get("description") or "").strip(),
        "industries": industries,
        "priority_industries": _normalize_string_array(value.get("priority_industries"), f"{field_name}.priority_industries"),
        "key_themes": _normalize_string_array(value.get("key_themes"), f"{field_name}.key_themes"),
        "strategic_objectives": _normalize_string_array(value.get("strategic_objectives"), f"{field_name}.strategic_objectives"),
        "low_relevance_themes": _normalize_string_array(value.get("low_relevance_themes"), f"{field_name}.low_relevance_themes"),
        "updated_at": str(value.get("updated_at") or "").strip(),
    }


def _normalize_watch_signals(value: Any, field_name: str) -> list[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array when provided")
    normalized: list[Dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{idx}] must be an object")
        normalized.append(
            {
                "target_name": str(item.get("target_name") or item.get("name") or item.get("target") or "").strip(),
                "target_type": str(item.get("target_type") or item.get("type") or "person").strip().lower() or "person",
                "current_employer": str(item.get("current_employer") or item.get("employer") or "").strip(),
                "headline": str(item.get("headline") or item.get("subject") or "").strip(),
                "url": str(item.get("url") or item.get("primary_url") or item.get("source_url") or "").strip(),
                "date": str(item.get("date") or item.get("received_at") or "").strip(),
                "snippet": str(item.get("snippet") or item.get("summary") or item.get("text_note") or item.get("raw_text") or "").strip(),
                "source_name": str(item.get("source_name") or item.get("publication") or "").strip(),
                "source_type": str(item.get("source_type") or "").strip().lower(),
                "confidence_hint": str(item.get("confidence_hint") or "").strip().lower(),
                "tags": [str(tag).strip() for tag in item.get("tags") or [] if str(tag).strip()],
                "source_org_name": str(item.get("source_org_name") or item.get("org_name") or "").strip(),
                "visible_to_orgs": _normalize_string_array(item.get("visible_to_orgs"), f"{field_name}[{idx}] visible_to_orgs"),
                "shared_with_orgs": _normalize_string_array(item.get("shared_with_orgs"), f"{field_name}[{idx}] shared_with_orgs"),
                "scope_profile_key": str(item.get("scope_profile_key") or item.get("industry_profile_key") or "").strip(),
                "child_profile_keys": [str(val).strip() for val in item.get("child_profile_keys") or [] if str(val).strip()],
                "child_org_names": [str(val).strip() for val in item.get("child_org_names") or [] if str(val).strip()],
            }
        )
    return normalized


def normalize_textify_options(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    options = dict(options or {})
    normalized: Dict[str, Any] = {
        "use_vision": _coerce_bool(options.get("use_vision"), True),
        "pdf_strategy": "hybrid",
    }

    strategy = str(options.get("pdf_strategy") or "hybrid").strip().lower()
    if strategy not in SUPPORTED_TEXTIFY_STRATEGIES:
        raise ValueError(
            f"Invalid pdf_strategy: {strategy!r}. Expected one of {sorted(SUPPORTED_TEXTIFY_STRATEGIES)}"
        )
    normalized["pdf_strategy"] = strategy

    provider = str(options.get("cleanup_provider") or "").strip().lower()
    if provider:
        if provider not in SUPPORTED_CLEANUP_PROVIDERS:
            raise ValueError(
                f"Invalid cleanup_provider: {provider!r}. Expected one of {sorted(SUPPORTED_CLEANUP_PROVIDERS)}"
            )
        normalized["cleanup_provider"] = provider

    model = str(options.get("cleanup_model") or "").strip()
    if model:
        normalized["cleanup_model"] = model

    normalized["docling_timeout_seconds"] = _coerce_positive_float(
        options.get("docling_timeout_seconds"), 240.0, "docling_timeout_seconds"
    )
    normalized["image_description_timeout_seconds"] = _coerce_positive_float(
        options.get("image_description_timeout_seconds"), 20.0, "image_description_timeout_seconds"
    )
    normalized["image_enrich_max_seconds"] = _coerce_positive_float(
        options.get("image_enrich_max_seconds"), 120.0, "image_enrich_max_seconds"
    )
    return normalized


def validate_pdf_textify_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})
    raw_options = payload.get("textify_options")
    if raw_options is None:
        raw_options = {}
    if not isinstance(raw_options, dict):
        raise ValueError("pdf_textify input_data.textify_options must be an object")
    payload["textify_options"] = normalize_textify_options(raw_options)
    return payload


def validate_included_study_extract_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})

    provider = str(payload.get("provider") or "anthropic").strip().lower() or "anthropic"
    if provider not in SUPPORTED_INCLUDED_STUDY_PROVIDERS:
        raise ValueError(
            f"Invalid provider: {provider!r}. Expected one of {sorted(SUPPORTED_INCLUDED_STUDY_PROVIDERS)}"
        )
    payload["provider"] = provider

    payload["model"] = str(
        payload.get("model") or ("claude-sonnet-4-6" if provider == "anthropic" else "gemini-2.5-flash")
    ).strip()

    fallback_provider = str(payload.get("fallback_provider") or "").strip().lower()
    if fallback_provider:
        if fallback_provider not in SUPPORTED_INCLUDED_STUDY_PROVIDERS:
            raise ValueError(
                f"Invalid fallback_provider: {fallback_provider!r}. Expected one of {sorted(SUPPORTED_INCLUDED_STUDY_PROVIDERS)}"
            )
        payload["fallback_provider"] = fallback_provider
        payload["fallback_model"] = str(payload.get("fallback_model") or "").strip()
    else:
        payload["fallback_provider"] = ""
        payload["fallback_model"] = ""

    extraction_scope = str(payload.get("extraction_scope") or "").strip().lower()
    if not extraction_scope:
        raise ValueError("included_study_extract requires input_data.extraction_scope")
    if extraction_scope not in SUPPORTED_INCLUDED_STUDY_SCOPES:
        raise ValueError(
            f"Invalid extraction_scope: {extraction_scope!r}. Expected one of {sorted(SUPPORTED_INCLUDED_STUDY_SCOPES)}"
        )
    payload["extraction_scope"] = extraction_scope

    output_detail = str(payload.get("output_detail") or "reference_map").strip().lower() or "reference_map"
    if output_detail not in SUPPORTED_INCLUDED_STUDY_OUTPUT_DETAILS:
        raise ValueError(
            f"Invalid output_detail: {output_detail!r}. Expected one of {sorted(SUPPORTED_INCLUDED_STUDY_OUTPUT_DETAILS)}"
        )
    payload["output_detail"] = output_detail

    payload["review_title"] = str(payload.get("review_title") or "").strip()
    payload["include_low_value_tables"] = _coerce_bool(payload.get("include_low_value_tables"), False)

    download_formats = payload.get("download_formats")
    if download_formats is None:
        normalized_formats = ["json", "xlsx"]
    else:
        if not isinstance(download_formats, list):
            raise ValueError("included_study_extract input_data.download_formats must be an array")
        normalized_formats = []
        seen: set[str] = set()
        for idx, item in enumerate(download_formats):
            fmt = str(item or "").strip().lower()
            if not fmt:
                continue
            if fmt not in SUPPORTED_INCLUDED_STUDY_DOWNLOAD_FORMATS:
                raise ValueError(
                    f"Invalid download_formats[{idx}]: {fmt!r}. Expected one of {sorted(SUPPORTED_INCLUDED_STUDY_DOWNLOAD_FORMATS)}"
                )
            if fmt in seen:
                continue
            seen.add(fmt)
            normalized_formats.append(fmt)
        if not normalized_formats:
            normalized_formats = ["json", "xlsx"]
    payload["download_formats"] = normalized_formats

    resolver_defaults = payload.get("resolver_defaults")
    if resolver_defaults is None:
        resolver_defaults = {}
    if not isinstance(resolver_defaults, dict):
        raise ValueError("included_study_extract input_data.resolver_defaults must be an object")
    payload["resolver_defaults"] = {
        "check_open_access": _coerce_bool(resolver_defaults.get("check_open_access"), True),
        "enrich_sjr": _coerce_bool(resolver_defaults.get("enrich_sjr"), True),
        "unpaywall_email": str(resolver_defaults.get("unpaywall_email") or "").strip(),
    }

    return payload


def validate_url_ingest_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})

    urls = payload.get("urls")
    if urls is not None:
        if not isinstance(urls, list):
            raise ValueError("url_ingest input_data.urls must be an array of URLs")
        payload["urls"] = [str(u).strip() for u in urls if str(u).strip()]
    else:
        payload["urls"] = []

    url_text = str(payload.get("url_text") or payload.get("url_list") or "").strip()
    if not payload["urls"] and not url_text:
        raise ValueError("url_ingest requires input_data.urls (array) or url_text/url_list")

    ingest_options = payload.get("ingest_options") or {}
    if not isinstance(ingest_options, dict):
        raise ValueError("url_ingest input_data.ingest_options must be an object")
    normalized_ingest = dict(ingest_options)
    normalized_ingest["convert_to_md"] = _coerce_bool(normalized_ingest.get("convert_to_md"), False)
    normalized_ingest["use_vision"] = _coerce_bool(normalized_ingest.get("use_vision"), False)
    normalized_ingest["capture_web_md_on_no_pdf"] = _coerce_bool(
        normalized_ingest.get("capture_web_md_on_no_pdf"), True
    )
    payload["ingest_options"] = normalized_ingest

    timeout_source = payload.get("timeout_seconds", normalized_ingest.get("timeout_seconds"))
    payload["timeout_seconds"] = _coerce_positive_int(timeout_source, 25, "timeout_seconds")

    top_level_textify = {k: payload[k] for k in SUPPORTED_TEXTIFY_OPTION_KEYS if k in payload}
    payload["textify_options"] = normalize_textify_options(top_level_textify)
    return payload


def validate_research_resolve_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})

    citations = payload.get("citations")
    if not isinstance(citations, list) or not citations:
        raise ValueError("research_resolve requires input_data.citations (non-empty array)")

    normalized_citations = []
    for idx, item in enumerate(citations):
        if not isinstance(item, dict):
            raise ValueError(f"research_resolve citations[{idx}] must be an object")

        title = str(item.get("title") or "").strip()
        if not title:
            raise ValueError(f"research_resolve citations[{idx}].title is required")

        extra_fields = item.get("extra_fields")
        if extra_fields is None:
            extra_fields = {}
        if not isinstance(extra_fields, dict):
            raise ValueError(f"research_resolve citations[{idx}].extra_fields must be an object")

        raw_row_id = item.get("row_id")
        row_id: Any
        if raw_row_id is None or raw_row_id == "":
            row_id = idx + 1
        else:
            try:
                row_id = int(raw_row_id)
            except Exception:
                row_id = str(raw_row_id).strip() or (idx + 1)

        normalized_citations.append(
            {
                "row_id": row_id,
                "title": title,
                "authors": str(item.get("authors") or "").strip(),
                "year": str(item.get("year") or "").strip(),
                "doi": str(item.get("doi") or "").strip(),
                "journal": str(item.get("journal") or "").strip(),
                "abstract": str(item.get("abstract") or "").strip(),
                "volume": str(item.get("volume") or "").strip(),
                "issue": str(item.get("issue") or "").strip(),
                "pages": str(item.get("pages") or "").strip(),
                "accession": str(item.get("accession") or "").strip(),
                "aim": str(item.get("aim") or "").strip(),
                "notes": str(item.get("notes") or "").strip(),
                "extra_fields": dict(extra_fields),
            }
        )

    options = payload.get("options")
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("research_resolve input_data.options must be an object")

    payload["citations"] = normalized_citations
    payload["options"] = {
        "check_open_access": _coerce_bool(options.get("check_open_access"), True),
        "enrich_sjr": _coerce_bool(options.get("enrich_sjr"), True),
        "unpaywall_email": str(options.get("unpaywall_email") or "").strip(),
    }
    return payload


def validate_org_profile_refresh_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})
    payload["profile_id"] = str(payload.get("profile_id") or "").strip()
    payload["org_name"] = str(payload.get("org_name") or "").strip()
    if not payload["org_name"]:
        raise ValueError("org_profile_refresh requires org_name")

    payload["target_org_name"] = str(payload.get("target_org_name") or "").strip()
    if not payload["target_org_name"]:
        raise ValueError("org_profile_refresh requires target_org_name")

    current_profile_snapshot = payload.get("current_profile_snapshot")
    if current_profile_snapshot is None:
        current_profile_snapshot = {}
    if not isinstance(current_profile_snapshot, dict):
        raise ValueError("org_profile_refresh current_profile_snapshot must be an object")
    payload["current_profile_snapshot"] = dict(current_profile_snapshot)

    payload["discovery_mode"] = str(payload.get("discovery_mode") or "official_sources_first").strip().lower() or "official_sources_first"
    if payload["discovery_mode"] not in {"official_sources_first", "official_only"}:
        raise ValueError(f"Invalid discovery_mode: {payload['discovery_mode']!r}")

    payload["requested_docs"] = _normalize_string_array(
        payload.get("requested_docs") or ["annual_report", "strategic_plan", "org_chart"],
        "org_profile_refresh requested_docs",
    )
    allowed_docs = {"annual_report", "strategic_plan", "org_chart", "about_page"}
    invalid_docs = [item for item in payload["requested_docs"] if item not in allowed_docs]
    if invalid_docs:
        raise ValueError(f"Invalid requested_docs: {invalid_docs!r}")

    payload["max_sources"] = min(12, _coerce_positive_int(payload.get("max_sources"), 6, "max_sources"))
    payload["timeout_seconds"] = _coerce_positive_int(payload.get("timeout_seconds"), 25, "timeout_seconds")
    payload["use_vision"] = _coerce_bool(payload.get("use_vision"), True)
    payload["website_url"] = str(
        payload.get("website_url")
        or payload["current_profile_snapshot"].get("website_url")
        or payload["current_profile_snapshot"].get("url")
        or ""
    ).strip()
    return payload


def validate_cortex_sync_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})

    # Accept either 'manifest' (new: ZIP-based) or 'file_paths' (legacy: direct paths)
    manifest = payload.get("manifest")
    file_paths = payload.get("file_paths", [])
    if file_paths is None:
        file_paths = []
    if not isinstance(file_paths, list):
        raise ValueError("cortex_sync 'file_paths' must be a list when provided")
    normalized_paths = [str(p).strip() for p in file_paths if str(p).strip()]
    payload["file_paths"] = normalized_paths

    if manifest is not None:
        if not isinstance(manifest, list):
            raise ValueError("cortex_sync 'manifest' must be a list when provided")
        normalized_manifest = []
        for idx, entry in enumerate(manifest):
            if not isinstance(entry, dict):
                raise ValueError(f"cortex_sync manifest[{idx}] must be an object")
            zip_path = str(entry.get("zip_path") or "").strip()
            if not zip_path:
                raise ValueError(f"cortex_sync manifest[{idx}].zip_path is required")
            normalized_entry = dict(entry)
            normalized_entry["zip_path"] = zip_path
            normalized_manifest.append(normalized_entry)
        payload["manifest"] = normalized_manifest

    collection_name = str(payload.get("collection_name") or "").strip()
    payload["collection_name"] = collection_name or "default"

    payload["topic"] = str(payload.get("topic") or "").strip()
    payload["fresh"] = _coerce_bool(payload.get("fresh"), False)
    return payload


def validate_stakeholder_profile_sync_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})
    payload["org_name"] = str(payload.get("org_name") or "").strip()
    if not payload["org_name"]:
        raise ValueError("stakeholder_profile_sync requires org_name")

    profiles = payload.get("profiles")
    if profiles is None:
        profiles = []
    if not isinstance(profiles, list):
        raise ValueError("stakeholder_profile_sync profiles must be an array when provided")

    normalized_profiles = []
    payload["org_alumni"] = _normalize_string_array(payload.get("org_alumni"), "stakeholder_profile_sync org_alumni")
    payload["org_strategic_profile"] = _normalize_org_strategic_profile(
        payload.get("org_strategic_profile"),
        "stakeholder_profile_sync org_strategic_profile",
    )
    if not profiles and not payload["org_alumni"] and not payload["org_strategic_profile"]:
        raise ValueError("stakeholder_profile_sync requires a non-empty profiles array or organisation context fields")
    for idx, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise ValueError(f"stakeholder_profile_sync profiles[{idx}] must be an object")
        canonical_name = str(profile.get("canonical_name") or profile.get("name") or "").strip()
        if not canonical_name:
            raise ValueError(f"stakeholder_profile_sync profiles[{idx}] requires canonical_name or name")
        normalized = dict(profile)
        normalized["canonical_name"] = canonical_name
        normalized["target_type"] = str(profile.get("target_type") or "person").strip().lower() or "person"
        normalized["external_profile_id"] = str(
            profile.get("external_profile_id") or profile.get("website_profile_id") or profile.get("id") or ""
        ).strip()
        normalized["email"] = str(profile.get("email") or "").strip()
        normalized["industry"] = str(profile.get("industry") or "").strip()
        normalized["function"] = str(profile.get("function") or "").strip()
        normalized["status"] = str(profile.get("status") or "active").strip().lower() or "active"
        normalized["last_verified_at"] = str(profile.get("last_verified_at") or "").strip()
        normalized["watch_status"] = str(profile.get("watch_status") or "off").strip().lower() or "off"
        normalized["website_url"] = str(profile.get("website_url") or "").strip()
        normalized["description"] = str(profile.get("description") or "").strip()
        normalized["key_themes"] = _normalize_string_array(
            profile.get("key_themes") if profile.get("key_themes") is not None else profile.get("key_themes_json"),
            f"stakeholder_profile_sync profiles[{idx}] key_themes",
        )
        normalized["regulatory_context"] = str(profile.get("regulatory_context") or "").strip()
        normalized["market_size"] = str(profile.get("market_size") or "").strip()
        normalized["acn_abn"] = str(profile.get("acn_abn") or "").strip()
        normalized["phone"] = str(profile.get("phone") or "").strip()
        normalized["parent_entity"] = str(profile.get("parent_entity") or "").strip()
        address = profile.get("address") or {}
        normalized["address"] = dict(address) if isinstance(address, dict) else {}
        normalized["aliases"] = [str(item).strip() for item in profile.get("aliases") or [] if str(item).strip()]
        normalized["known_employers"] = [str(item).strip() for item in profile.get("known_employers") or [] if str(item).strip()]
        normalized["current_employer"] = str(profile.get("current_employer") or "").strip()
        normalized["current_role"] = str(profile.get("current_role") or "").strip()
        normalized["affiliations"] = _normalize_affiliations(profile.get("affiliations"))
        normalized["industry_affiliations"] = _normalize_industry_affiliations(
            profile.get("industry_affiliations"),
            f"stakeholder_profile_sync profiles[{idx}] industry_affiliations",
        )
        normalized["tags"] = [str(item).strip() for item in profile.get("tags") or [] if str(item).strip()]
        normalized["alumni"] = _normalize_string_array(profile.get("alumni"), f"stakeholder_profile_sync profiles[{idx}] alumni")
        normalized["linkedin_connections"] = _normalize_linkedin_connections(
            profile.get("linkedin_connections"),
            f"stakeholder_profile_sync profiles[{idx}] linkedin_connections",
        )
        normalized_profiles.append(normalized)

    payload["profiles"] = normalized_profiles
    return payload


def validate_intel_extract_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})
    payload["org_name"] = str(payload.get("org_name") or "").strip()
    if not payload["org_name"]:
        raise ValueError("intel_extract requires org_name")

    for key in (
        "subject",
        "raw_text",
        "html_text",
        "primary_url",
        "text_note",
        "parsed_candidate_name",
        "parsed_candidate_employer",
        "target_type",
        "message_id",
        "received_at",
        "submitted_by",
        "signal_type",
        "intel_id",
    ):
        payload[key] = str(payload.get(key) or "").strip()

    payload["target_type"] = payload["target_type"].lower() or "person"
    payload["signal_type"] = payload["signal_type"] or "email_intel"

    attachments = payload.get("attachments") or []
    if not isinstance(attachments, list):
        raise ValueError("intel_extract attachments must be an array when provided")
    normalized_attachments = []
    for idx, item in enumerate(attachments):
        if not isinstance(item, dict):
            raise ValueError(f"intel_extract attachments[{idx}] must be an object")
        normalized_attachments.append(
            {
                "filename": str(item.get("filename") or "").strip(),
                "mime_type": str(item.get("mime_type") or "").strip(),
                "stored_path": str(item.get("stored_path") or "").strip(),
                "kind": str(item.get("kind") or "").strip().lower(),
            }
        )
    payload["attachments"] = normalized_attachments

    tags = payload.get("tags") or []
    if not isinstance(tags, list):
        raise ValueError("intel_extract tags must be an array when provided")
    payload["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]

    if not (payload["subject"] or payload["raw_text"] or payload["html_text"] or payload["attachments"]):
        raise ValueError("intel_extract requires subject, raw_text, html_text, or attachments")
    return payload


def validate_csv_profile_import_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})
    payload["org_name"] = str(payload.get("org_name") or "").strip()
    if not payload["org_name"]:
        raise ValueError("csv_profile_import requires org_name")
    payload["on_behalf_of"] = str(payload.get("on_behalf_of") or "").strip().lower()
    if not payload["on_behalf_of"]:
        raise ValueError("csv_profile_import requires on_behalf_of")
    payload["dry_run"] = _coerce_bool(payload.get("dry_run"), False)

    rows = payload.get("rows") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("csv_profile_import requires rows")
    if len(rows) > 500:
        raise ValueError("csv_profile_import supports at most 500 rows")

    normalized_rows = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"csv_profile_import rows[{idx}] must be an object")
        canonical_name = str(row.get("canonical_name") or "").strip()
        if not canonical_name:
            raise ValueError(f"csv_profile_import rows[{idx}] canonical_name is required")
        normalized = {str(key).strip(): str(value).strip() for key, value in row.items() if str(key).strip()}
        normalized["canonical_name"] = canonical_name
        normalized["target_type"] = (
            str(normalized.get("target_type") or "person").strip().lower() or "person"
        )
        if normalized["target_type"] not in {"person", "organisation"}:
            normalized["target_type"] = "person"
        if "watch" in normalized:
            normalized["watch"] = "yes" if _coerce_bool(normalized["watch"], False) else "no"
        if "status" in normalized:
            status = str(normalized["status"]).strip().lower()
            normalized["status"] = status if status in {"active", "archived"} else "active"
        normalized_rows.append(normalized)
    payload["rows"] = normalized_rows
    return payload


def validate_signal_ingest_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})
    payload["org_name"] = str(payload.get("org_name") or "").strip()
    if not payload["org_name"]:
        raise ValueError("signal_ingest requires org_name")

    payload["source_system"] = str(payload.get("source_system") or payload.get("source") or "market_radar").strip()
    payload["source_job"] = str(payload.get("source_job") or "").strip()
    payload["submitted_by"] = str(payload.get("submitted_by") or "").strip()
    payload["source_org_name"] = str(payload.get("source_org_name") or "").strip()
    payload["visible_to_orgs"] = _normalize_string_array(payload.get("visible_to_orgs"), "signal_ingest visible_to_orgs")
    payload["shared_with_orgs"] = _normalize_string_array(payload.get("shared_with_orgs"), "signal_ingest shared_with_orgs")
    payload["scope_profile_key"] = str(payload.get("scope_profile_key") or payload.get("industry_profile_key") or "").strip()
    payload["child_profile_keys"] = [str(item).strip() for item in payload.get("child_profile_keys") or [] if str(item).strip()]
    payload["child_org_names"] = [str(item).strip() for item in payload.get("child_org_names") or [] if str(item).strip()]
    payload["key_themes"] = _normalize_string_array(payload.get("key_themes"), "signal_ingest key_themes")
    payload["regulatory_context"] = str(payload.get("regulatory_context") or "").strip()
    payload["market_size"] = str(payload.get("market_size") or "").strip()

    payload["subject"] = str(payload.get("subject") or "").strip()
    payload["raw_text"] = str(payload.get("raw_text") or payload.get("body") or "").strip()
    payload["signals"] = _normalize_watch_signals(payload.get("signals"), "signal_ingest signals")
    if not payload["subject"] and not payload["raw_text"] and not payload["signals"]:
        raise ValueError("signal_ingest requires subject or raw_text")

    payload["target_type"] = str(payload.get("target_type") or "person").strip().lower() or "person"
    payload["primary_url"] = str(payload.get("primary_url") or payload.get("content") or "").strip()
    payload["text_note"] = str(payload.get("text_note") or "").strip()
    payload["parsed_candidate_name"] = str(
        payload.get("parsed_candidate_name") or payload.get("stakeholder_name") or ""
    ).strip()
    payload["parsed_candidate_employer"] = str(
        payload.get("parsed_candidate_employer") or payload.get("stakeholder_employer") or ""
    ).strip()
    payload["tags"] = [str(item).strip() for item in payload.get("tags") or [] if str(item).strip()]
    return payload


def validate_signal_digest_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})
    payload["org_name"] = str(payload.get("org_name") or "").strip()
    if not payload["org_name"]:
        raise ValueError("signal_digest requires org_name")

    payload["since_ts"] = str(payload.get("since_ts") or "").strip()
    payload["max_items"] = _coerce_positive_int(payload.get("max_items"), 25, "max_items")
    payload["include_needs_review"] = _coerce_bool(payload.get("include_needs_review"), True)
    payload["matched_only"] = _coerce_bool(payload.get("matched_only"), True)
    payload["llm_synthesis"] = _coerce_bool(payload.get("llm_synthesis"), False)
    payload["llm_provider"] = str(payload.get("llm_provider") or "ollama").strip().lower() or "ollama"
    payload["llm_model"] = str(payload.get("llm_model") or "").strip()
    payload["profile_keys"] = [str(item).strip() for item in payload.get("profile_keys") or [] if str(item).strip()]
    payload["priority_profile_keys"] = [
        str(item).strip() for item in payload.get("priority_profile_keys") or [] if str(item).strip()
    ]
    payload["scope_type"] = str(payload.get("scope_type") or "org").strip().lower() or "org"
    if payload["scope_type"] not in {"org", "industry"}:
        raise ValueError(f"Invalid scope_type: {payload['scope_type']!r}")
    payload["scope_profile_key"] = str(payload.get("scope_profile_key") or payload.get("industry_profile_key") or "").strip()
    payload["child_profile_keys"] = [str(item).strip() for item in payload.get("child_profile_keys") or [] if str(item).strip()]
    payload["child_org_names"] = [str(item).strip() for item in payload.get("child_org_names") or [] if str(item).strip()]
    payload["key_themes"] = _normalize_string_array(payload.get("key_themes"), "signal_digest key_themes")
    payload["regulatory_context"] = str(payload.get("regulatory_context") or "").strip()
    payload["market_size"] = str(payload.get("market_size") or "").strip()
    payload["shared_with_orgs"] = _normalize_string_array(payload.get("shared_with_orgs"), "signal_digest shared_with_orgs")
    payload["member_alumni"] = _normalize_string_array(payload.get("member_alumni"), "signal_digest member_alumni")
    payload["org_alumni"] = _normalize_string_array(payload.get("org_alumni"), "signal_digest org_alumni")
    payload["digest_tier"] = str(payload.get("digest_tier") or "standard").strip().lower() or "standard"
    if payload["digest_tier"] not in {"priority", "standard"}:
        raise ValueError(f"Invalid digest_tier: {payload['digest_tier']!r}")
    payload["report_depth"] = str(payload.get("report_depth") or "detailed").strip().lower() or "detailed"
    if payload["report_depth"] not in {"summary", "detailed", "strategic"}:
        raise ValueError(f"Invalid report_depth: {payload['report_depth']!r}")
    payload["deep_analysis"] = _coerce_bool(payload.get("deep_analysis"), False)
    return payload


def validate_stakeholder_graph_view_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})
    payload["org_name"] = str(payload.get("org_name") or "").strip()
    if not payload["org_name"]:
        raise ValueError("stakeholder_graph_view requires org_name")

    payload["view_mode"] = str(payload.get("view_mode") or "watch_network").strip().lower() or "watch_network"
    if payload["view_mode"] not in {"watch_network", "ego", "org_focus", "warm_intro", "alumni_cluster", "cross_target", "industry_network"}:
        raise ValueError(f"Invalid view_mode: {payload['view_mode']!r}")

    payload["profile_keys"] = [str(item).strip() for item in payload.get("profile_keys") or [] if str(item).strip()]
    payload["focus_profile_key"] = str(payload.get("focus_profile_key") or payload.get("industry_profile_key") or "").strip()
    payload["focus_org_name"] = str(payload.get("focus_org_name") or "").strip()
    payload["child_profile_keys"] = [str(item).strip() for item in payload.get("child_profile_keys") or [] if str(item).strip()]
    payload["since_ts"] = str(payload.get("since_ts") or "").strip()
    payload["max_hops"] = min(4, _coerce_positive_int(payload.get("max_hops"), 2, "max_hops"))
    payload["max_nodes"] = min(250, _coerce_positive_int(payload.get("max_nodes"), 100, "max_nodes"))
    payload["max_edges"] = min(600, _coerce_positive_int(payload.get("max_edges"), 200, "max_edges"))
    payload["top_k_paths"] = min(20, _coerce_positive_int(payload.get("top_k_paths"), 5, "top_k_paths"))
    payload["include_signals"] = _coerce_bool(payload.get("include_signals"), True)
    payload["include_sources"] = _coerce_bool(payload.get("include_sources"), False)
    payload["include_lab_members"] = _coerce_bool(payload.get("include_lab_members"), True)
    payload["include_alumni"] = _coerce_bool(payload.get("include_alumni"), True)
    payload["include_unwatched_bridges"] = _coerce_bool(payload.get("include_unwatched_bridges"), False)
    payload["min_edge_weight"] = float(payload.get("min_edge_weight") or 0.2)
    payload["min_confidence"] = float(payload.get("min_confidence") or 5.0)
    payload["layout_hint"] = str(payload.get("layout_hint") or "force").strip().lower() or "force"
    if payload["layout_hint"] not in {"force", "concentric", "cose", "breadthfirst"}:
        raise ValueError(f"Invalid layout_hint: {payload['layout_hint']!r}")

    allowed_edge_types = {
        "works_at",
        "affiliated_with",
        "alumni_of",
        "linkedin_connection",
        "tracked_by",
        "mentions_profile",
        "mentions_organization",
        "published_by",
        "co_mentioned",
        "org_alumni_context",
        "belongs_to_industry",
    }
    raw_edge_types = payload.get("edge_types") or list(allowed_edge_types)
    if not isinstance(raw_edge_types, list):
        raise ValueError("stakeholder_graph_view edge_types must be an array when provided")
    edge_types = [str(item).strip() for item in raw_edge_types if str(item).strip()]
    invalid_edge_types = [item for item in edge_types if item not in allowed_edge_types]
    if invalid_edge_types:
        raise ValueError(f"Invalid edge_types: {invalid_edge_types!r}")
    payload["edge_types"] = edge_types or list(allowed_edge_types)

    if payload["view_mode"] == "ego" and not payload["focus_profile_key"]:
        raise ValueError("stakeholder_graph_view ego view requires focus_profile_key")
    if payload["view_mode"] == "org_focus" and not payload["focus_org_name"]:
        raise ValueError("stakeholder_graph_view org_focus view requires focus_org_name")
    if payload["view_mode"] == "industry_network" and not (payload["focus_profile_key"] or payload["profile_keys"] or payload["child_profile_keys"]):
        raise ValueError("stakeholder_graph_view industry_network view requires focus_profile_key, profile_keys, or child_profile_keys")
    return payload
