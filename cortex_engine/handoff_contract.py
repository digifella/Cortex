"""Shared contract helpers for website <-> Cortex handoff."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import uuid

HANDOFF_CONTRACT_VERSION = "2026-02-15.v1"
DEFAULT_TENANT_ID = "default"
DEFAULT_PROJECT_ID = "default"

SUPPORTED_JOB_TYPES = ["pdf_anonymise", "pdf_textify", "url_ingest", "portal_ingest", "cortex_sync"]

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


def validate_portal_ingest_input(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(input_data or {})

    portal_document_id = payload.get("portal_document_id")
    if portal_document_id is None or str(portal_document_id).strip() == "":
        payload["portal_document_id"] = ""
    else:
        payload["portal_document_id"] = str(portal_document_id).strip()

    project_id = payload.get("project_id")
    payload["project_id"] = (
        str(project_id).strip() if project_id is not None and str(project_id).strip() else DEFAULT_PROJECT_ID
    )

    tenant_id = payload.get("tenant_id")
    payload["tenant_id"] = (
        str(tenant_id).strip() if tenant_id is not None and str(tenant_id).strip() else DEFAULT_TENANT_ID
    )

    payload["chunk_target_chars"] = _coerce_positive_int(
        payload.get("chunk_target_chars"), 2000, "chunk_target_chars"
    )
    payload["chunk_min_chars"] = _coerce_positive_int(
        payload.get("chunk_min_chars"), 200, "chunk_min_chars"
    )
    payload["max_chunks"] = _coerce_positive_int(
        payload.get("max_chunks"), 250, "max_chunks"
    )
    if payload["chunk_min_chars"] > payload["chunk_target_chars"]:
        raise ValueError("chunk_min_chars must be <= chunk_target_chars")

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
    if not collection_name:
        raise ValueError("cortex_sync requires 'collection_name'")
    payload["collection_name"] = collection_name

    payload["topic"] = str(payload.get("topic") or "").strip()
    payload["fresh"] = _coerce_bool(payload.get("fresh"), False)
    return payload
