from __future__ import annotations

import email
import hashlib
import imaplib
import json
import logging
import os
import re
import smtplib
import ssl
import tempfile
from urllib.parse import urlparse
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parsedate_to_datetime, parseaddr
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from cortex_engine.config_manager import ConfigManager
from cortex_engine.document_registry import (
    STRICT_DOC_TYPES,
    build_content_fingerprint,
    build_document_meta,
    derive_period_label,
)
from cortex_engine.email_handlers import (
    CsvProfileImportError,
    CsvProfileImportProcessor,
)
from cortex_engine.industry_classifier import classify_entity_industry
from cortex_engine.intel_extractor import extract_intel
from cortex_engine.intel_note_classifier import classify_mailbox_message
from cortex_engine.intel_note_processor import IntelNoteProcessor
from cortex_engine.org_chart_extractor import looks_like_org_chart_attachment
from cortex_engine.strategic_doc_analyser import clean_strategic_role_label
from cortex_engine.stakeholder_signal_matcher import normalize_lookup
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore, orgs_compatible
from cortex_engine.utils import convert_windows_to_wsl_path

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[^@<>\s]+@[^@<>\s]+\.[^@<>\s]+")
_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_DIRTY_STRATEGY_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[a-z0-9’”)])\s+(?=(?:We|Our|With|Committed|Building|Creating|Protecting|Optimising|Optimizing|Delivering|Driving|Minimising|Minimizing|Reducing|Support|Care|Digital|Partnerships)\b)"
)
_GENERIC_DOC_NAME_RE = re.compile(
    r"^(?:screenshot|screen shot|img|image|photo|picture|scan|attachment|document|file|outlook)(?:[\s_-]+\d.*)?$",
    re.IGNORECASE,
)
_MAIL_SUBJECT_PREFIX_RE = re.compile(r"^\s*((?:re|fw|fwd)\s*:\s*)+", re.IGNORECASE)
_SUBJECT_ENTITY_OVERRIDE_RE = re.compile(
    r"(?i)(?:^|[\s\[\]()|;>])(?:entity|org|organisation|organization)\s*:\s*([^|\];>,\n]+)"
)
_SUBJECT_DEPTH_OVERRIDE_RE = re.compile(
    r"(?i)(?:^|[\s\[\]()|;>])depth\s*:\s*(brief|default|detailed)\b"
)
_SUBJECT_FORCE_OVERRIDE_RE = re.compile(
    r"(?i)(?:^|[\s\[\]()|;>])(?:force|dedupe)\s*:\s*(?:yes|true|on|off|skip|force)\b"
)
_SUBJECT_PRIVACY_MARKER_RE = re.compile(r"(?i)\b(?:private|sensitive|confidential)\b")
_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*(?:to|[-–])\s*(20\d{2})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_MANAGED_NOTE_SECTIONS = {
    "entities",
    "people",
    "contacts",
    "organisations",
    "organizations",
    "emails",
    "strategic insights",
    "key stakeholders",
    "performance snapshot",
    "performance indicators",
    "kpi focus areas",
}
_KPI_FOCUS_STOPWORDS = {
    "board",
    "college",
    "committee",
    "council",
    "education",
    "financials",
    "finance",
    "governance",
    "implement",
    "operations",
    "people",
    "performance",
    "review",
    "streamline",
    "strategy",
    "support",
    "training",
}
_SUBJECT_ORG_HINT_STOPWORDS = {
    "annual",
    "attachment",
    "call",
    "chart",
    "document",
    "email",
    "fyi",
    "image",
    "industry",
    "intro",
    "introduction",
    "meeting",
    "note",
    "notes",
    "org",
    "organisation",
    "organization",
    "plan",
    "report",
    "screenshot",
    "screen",
    "shot",
    "strategic",
    "strategy",
    "update",
}
_ORG_ACRONYM_STOPWORDS = {
    "the",
    "of",
    "and",
    "for",
    "to",
    "in",
    "on",
    "at",
    "pty",
    "ltd",
    "limited",
    "group",
    "company",
    "corporation",
    "corp",
    "inc",
    "llc",
}
_GENERIC_DOMAIN_LABELS = {
    "www",
    "mail",
    "email",
    "app",
    "portal",
    "api",
    "com",
    "net",
    "org",
    "gov",
    "edu",
    "co",
    "au",
    "nz",
    "uk",
}
_ANNUAL_REPORT_STAKEHOLDER_STOPWORDS = {
    "and information",
    "committee",
    "evaluation panel",
    "independent auditor",
    "integrated water management",
    "organisational chart",
    "organizational chart",
    "responsible ministers",
    "risk management",
    "governance",
    "performance",
    "review",
}
_LOW_SIGNAL_EMPLOYER_MARKERS = (
    "i am pleased to present",
    "this report",
    "annual report",
    "document appears to relate",
)
_LOW_SIGNAL_ROLE_MARKERS = (
    "board committees",
    "lead indicators",
    "board of directors",
    "appointed a director",
    "natural resource management",
    "through leading",
    "the board is responsible",
    "consideration of",
    "independent auditor",
    "responsible ministers",
)
_MAIL_SIGNATURE_ROLE_MARKERS = (
    "chief",
    "director",
    "manager",
    "officer",
    "lead",
    "partner",
    "consultant",
    "adviser",
    "advisor",
)
_MAIL_SIGNATURE_CONTACT_MARKERS = (
    "linkedin",
    "http",
    "www.",
    "@",
    "ph:",
    "phone",
    "mobile",
)
_MAIL_DISCLAIMER_MARKERS = (
    "i acknowledge the aboriginal and torres strait islander peoples",
    "i acknowledge the traditional owners",
    "encourages flexible working",
    "i do not expect you will read",
    "please consider the environment",
    "this email and any attachments",
)
_FIT_STOPWORDS = {
    "and",
    "the",
    "for",
    "to",
    "of",
    "in",
    "on",
    "with",
    "a",
    "an",
    "by",
    "or",
    "our",
    "their",
    "your",
}
_FIT_TOKEN_EQUIVALENTS = {
    "it": {"it", "technology", "tech", "digital"},
    "technology": {"technology", "tech", "digital", "it"},
    "tech": {"technology", "tech", "digital", "it"},
    "digital": {"digital", "technology", "tech", "it", "data", "ai"},
    "strategy": {"strategy", "strategic", "roadmap", "planning", "plan"},
    "strategic": {"strategy", "strategic", "roadmap", "planning", "plan"},
    "roadmap": {"roadmap", "strategy", "strategic", "planning", "plan"},
    "planning": {"planning", "plan", "strategy", "strategic", "roadmap"},
    "plan": {"plan", "planning", "strategy", "strategic", "roadmap"},
    "enablement": {"enablement", "transformation", "uplift", "capability"},
    "transformation": {"transformation", "change", "enablement", "uplift"},
    "change": {"change", "transformation", "enablement"},
    "leadership": {"leadership", "change", "capability", "performance", "uplift"},
    "performance": {"performance", "capability", "enablement", "execution", "uplift"},
    "capability": {"capability", "performance", "enablement", "uplift", "change"},
    "execution": {"execution", "performance", "delivery", "discipline"},
    "customer": {"customer", "consumer", "member", "client"},
    "experience": {"experience", "service", "engagement"},
    "water": {"water", "utility", "utilities"},
    "utility": {"utility", "utilities", "water"},
    "utilities": {"utility", "utilities", "water"},
    "compliance": {"compliance", "regulatory", "regulation", "governance", "risk"},
    "regulatory": {"regulatory", "regulation", "compliance", "governance", "risk"},
    "risk": {"risk", "compliance", "governance", "regulatory"},
    "governance": {"governance", "risk", "compliance", "regulatory"},
    "care": {"care", "healthcare", "health"},
    "health": {"health", "healthcare", "care"},
    "healthcare": {"healthcare", "health", "care"},
}
_TRUSTED_SELF_RELAY_MAILBOX = "intel.longboardfella@gmail.com"
_TRUSTED_SELF_RELAY_SUBMITTER = "paul@longboardfella.com.au"
_CORTEX_PROCESSED_REPLY_RE = re.compile(r"(?im)^\s*Cortex processed your submission for\b")
_YOUTUBE_SUMMARISER_SUBJECT_RE = re.compile(r"youtube\s+summariser", re.IGNORECASE)
_MAILBOX_FORWARDING_CONFIRMATION_MARKERS = (
    "has requested to automatically forward mail to your email address",
    "please click the link below to confirm the request",
    "cannot automatically forward messages to your email address unless you confirm",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_db_root() -> Path:
    config = ConfigManager().get_config()
    raw_db_path = str(config.get("ai_database_path") or "").strip()
    if not raw_db_path:
        raise RuntimeError("ai_database_path is not configured; Cortex intel mailbox cannot initialize")
    safe_db_path = raw_db_path if os.path.exists("/.dockerenv") else convert_windows_to_wsl_path(raw_db_path)
    root = Path(safe_db_path) / "intel_mailbox"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _decode_header_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(make_header(decode_header(text))).strip()
    except Exception:
        return text


def _sanitize_filename(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("._")
    return cleaned or fallback


def _derive_message_key(message_id: str, raw_bytes: bytes) -> str:
    basis = (message_id or "").strip().lower() or hashlib.sha1(raw_bytes).hexdigest()
    return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:16]


def _extract_emails_from_text(text: str) -> List[str]:
    found = {match.group(0).strip(".,;:()<>[]{}") for match in _EMAIL_RE.finditer(text or "")}
    return sorted(item for item in found if item)


def _extract_urls_from_text(text: str) -> List[str]:
    found = {match.group(0).strip(".,;:()<>[]{}") for match in _URL_RE.finditer(text or "")}
    return sorted(item for item in found if item)


def _combined_mailbox_text(message: Dict[str, Any]) -> str:
    return "\n\n".join(
        str(message.get(key) or "")
        for key in ("raw_text", "html_text")
        if str(message.get(key) or "").strip()
    )


def _looks_like_generic_document_name(value: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip(" -_"))
    if not cleaned:
        return True
    if _GENERIC_DOC_NAME_RE.match(cleaned):
        return True
    alpha_tokens = [token.lower() for token in re.split(r"[\s_-]+", cleaned) if re.search(r"[A-Za-z]", token)]
    if not alpha_tokens:
        return True
    generic_tokens = {
        "attachment",
        "compressed",
        "copy",
        "direction",
        "doc",
        "document",
        "draft",
        "file",
        "final",
        "image",
        "img",
        "outlook",
        "pdf",
        "picture",
        "plan",
        "report",
        "scan",
        "screenshot",
        "screen",
        "shot",
        "strategic",
        "v",
        "version",
    }
    if all(token in generic_tokens for token in alpha_tokens):
        return True
    return False


def _normalized_mail_subject(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    while True:
        updated = re.sub(r"^\s*subject\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        if updated == text:
            break
        text = updated
    while True:
        updated = _MAIL_SUBJECT_PREFIX_RE.sub("", text).strip()
        if updated == text:
            break
        text = updated
    text = re.sub(r"\s+", " ", text).strip(" -|:")
    return text


def _extract_subject_entity_override(value: str) -> str:
    text = str(value or "")
    match = _SUBJECT_ENTITY_OVERRIDE_RE.search(text)
    if not match:
        return ""
    return _clean_display_label(match.group(1))


def _strip_subject_entity_override(value: str) -> str:
    text = _SUBJECT_ENTITY_OVERRIDE_RE.sub(" ", str(value or ""))
    text = re.sub(r"\s*[|;>,]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|:;,")
    return text


def _extract_subject_depth_override(value: str) -> str:
    text = str(value or "")
    match = _SUBJECT_DEPTH_OVERRIDE_RE.search(text)
    if not match:
        return ""
    return str(match.group(1) or "").strip().lower()


def _strip_subject_depth_override(value: str) -> str:
    text = _SUBJECT_DEPTH_OVERRIDE_RE.sub(" ", str(value or ""))
    text = re.sub(r"\s*[|;>,]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|:;,")
    return text


def _extract_subject_force_override(value: str) -> bool:
    return bool(_SUBJECT_FORCE_OVERRIDE_RE.search(str(value or "")))


def _strip_subject_force_override(value: str) -> str:
    text = _SUBJECT_FORCE_OVERRIDE_RE.sub(" ", str(value or ""))
    text = re.sub(r"\s*[|;>,]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|:;,")
    return text


def _strip_subject_privacy_markers(value: str) -> str:
    text = _SUBJECT_PRIVACY_MARKER_RE.sub(" ", str(value or ""))
    text = re.sub(r"\s*[|;>,]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|:;,")
    return text


def _clean_display_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text.lstrip("•*-· ").strip(" ,.;:")


def _clean_display_role(role: str, employer: str = "") -> str:
    cleaned = _clean_display_label(role).lstrip("| ").strip()
    return clean_strategic_role_label(cleaned, employer)


def _looks_like_person_label(value: str) -> bool:
    text = _clean_display_label(value)
    if not text:
        return False
    if any(char.isdigit() for char in text):
        return False
    if "@" in text or "http" in normalize_lookup(text):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) < 2 or len(words) > 5:
        return False
    lowered = normalize_lookup(text)
    if any(marker in lowered for marker in _ORG_CHART_FUNCTION_MARKERS):
        return False
    uppercase_words = sum(1 for word in words if word[:1].isupper())
    return uppercase_words >= 2


def _org_initialism(value: str) -> str:
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", _clean_display_label(value)) if word]
    initials = [
        normalize_lookup(word[:1])
        for word in words
        if normalize_lookup(word) not in _ORG_ACRONYM_STOPWORDS
    ]
    return "".join(initials).upper()


def _looks_like_org_acronym(value: str) -> bool:
    cleaned = _clean_display_label(value)
    return bool(re.fullmatch(r"[A-Z]{2,6}", cleaned))


def _org_hint_match_type(subject_org_hint: str, candidate_name: str) -> str:
    hint = _clean_display_label(subject_org_hint)
    candidate = _clean_display_label(candidate_name)
    if not hint or not candidate:
        return ""
    if normalize_lookup(hint) == normalize_lookup(candidate):
        return "exact"
    if orgs_compatible(candidate, hint):
        return "compatible"
    if _looks_like_org_acronym(hint) and _org_initialism(candidate) == hint.upper():
        return "acronym"
    hint_tokens = {normalize_lookup(token) for token in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", hint) if token}
    candidate_tokens = {normalize_lookup(token) for token in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", candidate) if token}
    if len(hint_tokens) >= 2 and hint_tokens.issubset(candidate_tokens):
        return "compatible"
    return ""


def _clean_org_chart_function_label(value: str, org_name: str = "") -> str:
    text = _clean_display_label(value)
    text = re.sub(r"\s*\((?:person|organisation|organization)\)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:")
    if not text:
        return ""
    if org_name and orgs_compatible(text, org_name):
        return ""
    return text


def _extract_email_domain(value: str) -> str:
    email_value = str(value or "").strip().lower()
    if "@" not in email_value:
        return ""
    return email_value.rsplit("@", 1)[-1].strip(". ")


def _domain_labels(value: str) -> List[str]:
    domain = str(value or "").strip().lower()
    if not domain:
        return []
    labels = []
    for part in domain.split("."):
        clean = re.sub(r"[^a-z0-9-]+", "", part)
        if not clean or clean in _GENERIC_DOMAIN_LABELS:
            continue
        labels.append(clean)
    return labels


def _extract_url_domain(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return str(parsed.netloc or parsed.path or "").strip().lower()


_ORG_CHART_FUNCTION_MARKERS = (
    "operations",
    "planning",
    "delivery",
    "environment",
    "people",
    "corporate",
    "customer",
    "customers",
    "community",
    "strategy",
    "infrastructure",
    "digital",
    "business",
    "asset",
    "assets",
    "services",
    "finance",
    "technology",
    "governance",
    "risk",
    "legal",
    "procurement",
    "safety",
    "culture",
    "sustainable",
    "water",
    "wastewater",
)
_ORG_CHART_FUNCTION_ROLE_PREFIXES = (
    "general manager",
    "executive manager",
    "group manager",
    "chief",
    "head of",
    "director",
    "manager",
    "office of",
)
_ORG_CHART_FUNCTION_STOPWORDS = (
    "attachment",
    "subject",
    "from:",
    "to:",
    "sent:",
    "linkedin",
    "twitter",
    "zoom",
    "ph:",
    "email",
    "contact",
    "org chart",
    "organisational chart",
    "organization chart",
    "leadership team",
    "no email addresses",
    "no career transitions",
    "sender",
    "director of",
    "longboardfella",
)


def _extract_document_year_label(*values: str) -> str:
    for value in values:
        text = str(value or "")
        match = _YEAR_RANGE_RE.search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
    for value in values:
        text = str(value or "")
        match = re.search(r"(?<!\d)(20\d{2})\s*[/\-_]\s*(\d{2,4})(?!\d)", text)
        if not match:
            continue
        left = match.group(1)
        right = match.group(2)
        if len(right) == 2:
            return f"{left}-{right}"
        if right[:2] == left[:2]:
            return f"{left}-{right[-2:]}"
        return f"{left}-{right}"
    for value in values:
        text = str(value or "")
        match = _YEAR_RE.search(text)
        if match:
            return match.group(1)
    for value in values:
        text = str(value or "")
        match = re.search(r"(20\d{2})", text)
        if match:
            return match.group(1)
    return ""


def _looks_like_useful_subject_title(value: str) -> bool:
    cleaned = _clean_display_label(value)
    if not cleaned:
        return False
    lowered = normalize_lookup(cleaned)
    if lowered in {"fwd", "fw", "re"}:
        return False
    if _looks_like_generic_document_name(cleaned):
        return False
    return True


def _subject_org_hint(value: str) -> str:
    text = _clean_display_label(value)
    if not text or not _looks_like_useful_subject_title(text):
        return ""
    text = _strip_subject_privacy_markers(text)
    candidate = re.sub(
        r"\b(?:org(?:anisation|anization)?\s+chart|strategic\s+plan|strategic\s+direction|annual\s+report|industry\s+report|sector\s+report|report|plan|strategy|direction|roadmap)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\b20\d{2}(?:\s*[-–to]+\s*20\d{2})?\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^\s*(?:for|re)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -|:;,")
    if not candidate:
        return ""
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", candidate) if word]
    if not words or len(words) > 6:
        return ""
    lowered_words = [normalize_lookup(word) for word in words]
    if any(word in _SUBJECT_ORG_HINT_STOPWORDS for word in lowered_words):
        return ""
    return candidate


def _strip_scope_prefix_from_subject_hint(subject_org_hint: str, scope_org_name: str) -> str:
    hint = _clean_display_label(subject_org_hint)
    scope = _clean_display_label(scope_org_name)
    if not hint or not scope:
        return hint
    hint_key = normalize_lookup(hint)
    scope_key = normalize_lookup(scope)
    if not hint_key.startswith(scope_key + " "):
        return hint
    remainder = hint[len(scope):].strip(" -|:;,")
    if len(re.findall(r"[A-Za-z][A-Za-z'’.\-]+", remainder)) < 2:
        return hint
    return remainder


def _document_label(doc_type: str, context_text: str) -> str:
    lowered = normalize_lookup(context_text)
    if doc_type == "annual_report":
        return "Annual Report"
    if doc_type == "industry_report":
        return "Industry Report"
    if "strategic direction" in lowered:
        return "Strategic Direction"
    if "roadmap" in lowered:
        return "Roadmap"
    if "statement of strategic priorities" in lowered:
        return "Statement of Strategic Priorities"
    return "Strategic Plan"


def _looks_like_high_signal_kpi_focus(value: str) -> bool:
    text = _clean_display_label(value)
    lowered = normalize_lookup(text)
    if not lowered:
        return False
    if any(char.isdigit() for char in text):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) < 2 or len(words) > 8:
        return False
    normalized_words = [normalize_lookup(word) for word in words]
    content_words = [word for word in normalized_words if word not in {"and", "of", "the", "to", "for", "in"}]
    if len(content_words) < 2:
        return False
    if all(word in _KPI_FOCUS_STOPWORDS for word in content_words):
        return False
    return True


def _looks_like_named_stakeholder(value: str) -> bool:
    text = _clean_display_label(value)
    lowered = normalize_lookup(text)
    if not lowered:
        return False
    if any(marker in lowered for marker in _ANNUAL_REPORT_STAKEHOLDER_STOPWORDS):
        return False
    if any(token in lowered for token in ("corporation", "pty ltd", "proprietary limited", "group accessible", "declaration")):
        return False
    if any(char.isdigit() for char in text):
        return False
    parts = [part for part in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if part]
    return 2 <= len(parts) <= 6


def _should_keep_note_stakeholder(item: Dict[str, Any], org_name: str = "") -> bool:
    name = _clean_display_label(str(item.get("name") or "").strip())
    raw_role = str(item.get("current_role") or "").strip()
    role = _clean_display_role(raw_role, str(item.get("current_employer") or "").strip())
    employer = _clean_display_label(str(item.get("current_employer") or "").strip())
    if not _looks_like_named_stakeholder(name):
        return False
    if role:
        lowered_role = normalize_lookup(role)
        if "|" in raw_role and normalize_lookup(role) != "deputy chair":
            return False
        if any(marker in lowered_role for marker in ("declaration", "accounting officer", "finance and accounting officer")):
            return False
        if any(marker in lowered_role for marker in _LOW_SIGNAL_ROLE_MARKERS):
            return False
        role_words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", role) if word]
        if len(role_words) > 5:
            return False
    if org_name and orgs_compatible(name, org_name):
        return False
    if employer and "group accessible" in normalize_lookup(employer):
        return False
    return True


def _compact_note_evidence(value: str, limit: int = 160) -> str:
    text = _clean_display_label(value)
    if not text:
        return ""
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _looks_like_low_signal_org_label(value: str) -> bool:
    text = _clean_display_label(value)
    lowered = normalize_lookup(text)
    if not lowered:
        return True
    if _looks_like_generic_document_name(text):
        return True
    if any(marker in lowered for marker in ("annual report", "strategic plan", "strategy", "roadmap", "direction")) and re.search(r"20\d{2}", text):
        return True
    if re.search(r"\b(?:plan|report|strategy|roadmap|direction)\b", lowered) and _org_initialism(text):
        return True
    if any(marker in lowered for marker in _LOW_SIGNAL_EMPLOYER_MARKERS):
        return True
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    return len(words) > 8


def _normalize_document_summary_org_label(note_summary: str, subject_org_hint: str, doc_type: str) -> str:
    text = str(note_summary or "").strip()
    subject = _clean_display_label(subject_org_hint)
    if doc_type not in {"annual_report", "strategic_plan"} or not text or not subject:
        return text
    first_sentence = re.match(
        r"^\s*(?:Document from|Strategic planning document from) [^.]+?\.\s*",
        text,
        flags=re.IGNORECASE,
    )
    if first_sentence:
        remainder = text[first_sentence.end():].lstrip()
        replacement = f"{'Strategic planning document' if doc_type == 'strategic_plan' else 'Document'} from {subject}."
        return f"{replacement} {remainder}".strip() if remainder else replacement
    return text


def _display_submitter_name(message: Dict[str, Any]) -> str:
    name = _clean_display_label(str(message.get("from_name") or "").strip())
    if name:
        return name
    email_address = str(message.get("from_email") or "").strip().lower()
    if not email_address or "@" not in email_address:
        return ""
    local = email_address.split("@", 1)[0]
    local = re.sub(r"[._-]+", " ", local).strip()
    words = [word.capitalize() for word in local.split() if word]
    return " ".join(words[:4]).strip()


def _compact_mailbox_provenance(message: Dict[str, Any]) -> str:
    name = _display_submitter_name(message)
    email_address = str(message.get("from_email") or "").strip().lower()
    received_at = str(message.get("received_at") or "").strip()
    parts = []
    if name and email_address:
        parts.append(f"{name} <{email_address}>")
    elif email_address:
        parts.append(email_address)
    elif name:
        parts.append(name)
    if received_at:
        parts.append(received_at)
    return " | ".join(parts).strip()


def _looks_like_signature_start(lines: Sequence[str], index: int) -> bool:
    line = _clean_display_label(lines[index] if index < len(lines) else "")
    if not line:
        return False
    lowered = normalize_lookup(line)
    if any(marker in lowered for marker in _MAIL_DISCLAIMER_MARKERS):
        return True
    if lowered in {"regards", "kind regards", "best regards", "thanks", "thank you"}:
        return True
    if re.fullmatch(r"[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){1,4}(?:\s+\([^)]+\))?", line):
        window = " ".join(normalize_lookup(_clean_display_label(item)) for item in lines[index : index + 5] if _clean_display_label(item))
        if any(marker in window for marker in _MAIL_SIGNATURE_ROLE_MARKERS) and any(marker in window for marker in _MAIL_SIGNATURE_CONTACT_MARKERS):
            return True
    return False


def _trim_mailbox_body_text(value: str) -> str:
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in raw_lines]
    cut_index: Optional[int] = None
    for idx, line in enumerate(lines):
        lowered = normalize_lookup(line)
        if any(marker in lowered for marker in _MAIL_DISCLAIMER_MARKERS):
            cut_index = idx
            break
        if _looks_like_signature_start(lines, idx):
            cut_index = idx
            break
        if line in {"--", "—", "–"}:
            cut_index = idx
            break
    kept = lines[:cut_index] if cut_index is not None else lines
    text = "\n".join(kept).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        text = _clean_display_label(value)
        key = normalize_lookup(text)
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _fit_phrase_token_set(value: str, *, expand: bool = True) -> set[str]:
    normalized = normalize_lookup(value)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token and token not in _FIT_STOPWORDS and len(token) > 1
    ]
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        if expand:
            expanded.update(_FIT_TOKEN_EQUIVALENTS.get(token, {token}))
    return expanded


def _clean_note_signal_snippet(value: str, *, doc_type: str = "") -> str:
    text = _clean_display_label(value)
    if not text:
        return ""
    if doc_type == "annual_report":
        text = re.sub(r"\.{4,}\s*\d+\b", " ", text)
        text = re.sub(r"(?:\s*[.][. ]*){3,}", " ", text)
        text = re.sub(r"\bpage\s+\d+\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:table of contents|contents)\b", " ", text, flags=re.IGNORECASE)
    if doc_type == "strategic_plan":
        lowered = normalize_lookup(text)
        if any(
            marker in lowered
            for marker in (
                "our ambitions for",
                "intent statements and measures",
                "measured by",
                "safety maturity model",
                "output measure",
                "have your say",
            )
        ):
            return ""
        if re.search(r"\btrifr\b|hazard identifications|training completion|<\s*\d|>\s*\d|%\s+safety", text, flags=re.IGNORECASE):
            return ""
        if len(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,}", text)) >= 3 and "." not in text:
            return ""
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    return text


def _looks_like_dirty_strategy_snippet(value: str) -> bool:
    text = _clean_display_label(value)
    lowered = normalize_lookup(text)
    if not text:
        return True
    if re.match(r"^[a-z]\s+[a-z]", text):
        return True
    if any(
        marker in lowered
        for marker in (
            "safety maturity model",
            "hazard identifications",
            "safety training completion",
            "have your say",
            "trifr",
            "output measure",
            "intent statements and measures",
        )
    ):
        return True
    if sum(1 for _ in _DIRTY_STRATEGY_CLAUSE_SPLIT_RE.finditer(text)) >= 2:
        return True
    if len(re.findall(r"[<>]\s*\d", text)) >= 2:
        return True
    return False


def _compact_strategy_snippet(
    value: str,
    *,
    local_only: bool = False,
    limit: int = 220,
    allow_two_sentences: bool = False,
) -> str:
    had_truncation_marker = str(value or "").strip().endswith("...")
    text = _clean_note_signal_snippet(value, doc_type="strategic_plan")
    if not text:
        return ""
    text = re.sub(r"(?<=[A-Za-z])\.\s+(?=[a-z])", " ", text)
    if "." in text:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        text = " ".join(sentences[:2] if allow_two_sentences else sentences[:1]).strip()
    elif _looks_like_dirty_strategy_snippet(text):
        chunks = [chunk.strip(" ,.;:-") for chunk in _DIRTY_STRATEGY_CLAUSE_SPLIT_RE.split(text) if chunk.strip(" ,.;:-")]
        if chunks:
            text = " ".join(chunks[:2] if allow_two_sentences else chunks[:1])
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    if had_truncation_marker and not text.endswith("..."):
        text = text.rstrip(" ,.;:-") + "..."
    if local_only and _looks_like_dirty_strategy_snippet(text):
        return ""
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _polish_strategy_sentence(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" ,.;:-"))
    if not text:
        return ""
    replacements = (
        (r"\bI n terms\b", "In terms"),
        (r"\bselfdetermination\b", "self-determination"),
        (r"\blongterm\b", "long-term"),
        (r"\bnoncompliances\b", "non-compliances"),
        (r"\brealtime\b", "real-time"),
        (r"\btraditional Owners\b", "Traditional Owners"),
        (r"\biota\b", "Iota"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\btrade waste commitment\s*\(2023-?28\)\b", "trade waste standards", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwater insights gained\b", "water. Insights gained", text)
    text = re.sub(r"\bcustomers continues to add value\b", "customers. Continues to add value", text)
    if text[:1].islower():
        text = text[:1].upper() + text[1:]
    return text.strip(" ,.;:-")


def _looks_like_low_quality_detail_sentence(value: str) -> bool:
    text = _polish_strategy_sentence(_clean_display_label(value))
    lowered = normalize_lookup(text)
    words = [normalize_lookup(word) for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text)]
    if len(words) < 5:
        return True
    if text[:1].islower():
        return True
    if words[0] in {"and", "to", "in", "of", "by", "information", "innovation", "customers", "owners", "water", "commitment", "standards"}:
        return True
    if words[-1] in {"and", "the", "for", "of", "our", "to", "with", "by", "having"}:
        return True
    if any(marker in lowered for marker in ("i ota", "i n terms", "-e.g", "trade waste commitment")):
        return True
    if " having customers have " in f" {lowered} ":
        return True
    return False


def _render_detail_point_text(point: Dict[str, Any], *, local_only: bool = False) -> str:
    raw_snippet = str(point.get("snippet") or "").strip()
    if not raw_snippet:
        return ""
    candidate = _compact_strategy_snippet(raw_snippet, local_only=local_only, limit=520, allow_two_sentences=True)
    candidate = _polish_strategy_sentence(candidate)
    if candidate and not _looks_like_low_quality_detail_sentence(candidate):
        return candidate
    raw_heading = str(point.get("raw_heading") or point.get("heading") or "").strip()
    if raw_heading:
        combined = f"{raw_heading} {raw_snippet}".strip()
        repaired = _compact_strategy_snippet(combined, local_only=local_only, limit=520, allow_two_sentences=True)
        repaired = _polish_strategy_sentence(repaired)
        if repaired and not _looks_like_low_quality_detail_sentence(repaired):
            return repaired
    return ""


def _should_keep_rendered_strategic_signal(headline: str, *, doc_type: str = "") -> bool:
    text = _clean_display_label(headline)
    if not text:
        return False
    lowered = normalize_lookup(text)
    if lowered in {"official"}:
        return False
    if doc_type == "strategic_plan":
        if lowered.startswith(("we ", "we re ", "were ", "our people can ", "reflects the diversity")):
            return False
        if text.endswith(("That", "With", "And")):
            return False
    return True


def _strip_managed_note_sections(markdown_text: str) -> str:
    kept_lines: List[str] = []
    skipping = False
    for raw_line in str(markdown_text or "").splitlines():
        line = str(raw_line or "")
        if line.startswith("## "):
            heading_key = normalize_lookup(line[3:])
            skipping = heading_key in _MANAGED_NOTE_SECTIONS
            if skipping:
                continue
        if not skipping:
            kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _kind_for_mime(mime_type: str) -> str:
    mime = str(mime_type or "").lower()
    if mime.startswith("image/"):
        return "image"
    if mime in {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/plain",
        "text/html",
        "text/csv",
    }:
        return "document"
    return "other"


@dataclass
class IntelMailboxConfig:
    host: str
    port: int
    username: str
    password: str
    folder: str
    org_name: str
    poll_limit: int
    search_criteria: str
    allowed_senders: tuple[str, ...]
    source_system: str
    callback_url: str
    note_callback_url: str
    callback_secret: str
    callback_timeout: int
    profile_import_url: str
    profile_import_timeout: int
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_use_ssl: bool
    reply_from: str
    mark_seen_on_success: bool

    @classmethod
    def from_env(cls, env: Dict[str, str]) -> "IntelMailboxConfig":
        def get(name: str, default: str = "") -> str:
            return str(os.environ.get(name, env.get(name, default)) or "").strip()

        allowed = tuple(
            item.strip().lower()
            for item in get("INTEL_ALLOWED_SENDERS", "").split(",")
            if item.strip()
        )
        callback_url = get("INTEL_RESULTS_POST_URL")
        note_callback_url = get("INTEL_NOTE_POST_URL")
        if not note_callback_url and callback_url.endswith("/admin/queue_worker_api.php?action=import_cortex_extract"):
            note_callback_url = callback_url.replace(
                "/admin/queue_worker_api.php?action=import_cortex_extract",
                "/lab/market_radar_api.php?action=ingest_intel_note",
            )
        return cls(
            host=get("INTEL_IMAP_HOST", "imap.gmail.com"),
            port=int(get("INTEL_IMAP_PORT", "993") or "993"),
            username=get("INTEL_IMAP_USERNAME"),
            password=get("INTEL_IMAP_PASSWORD"),
            folder=get("INTEL_IMAP_FOLDER", "INBOX"),
            org_name=get("INTEL_IMAP_ORG_NAME", "Longboardfella"),
            poll_limit=max(1, int(get("INTEL_IMAP_POLL_LIMIT", "10") or "10")),
            search_criteria=get("INTEL_IMAP_SEARCH", "UNSEEN"),
            allowed_senders=allowed,
            source_system=get("INTEL_SOURCE_SYSTEM", "cortex_mailbox"),
            callback_url=callback_url,
            note_callback_url=note_callback_url,
            callback_secret=get("INTEL_RESULTS_POST_SECRET"),
            callback_timeout=max(5, int(get("INTEL_RESULTS_POST_TIMEOUT", "30") or "30")),
            profile_import_url=get("INTEL_PROFILE_IMPORT_URL"),
            profile_import_timeout=max(5, int(get("INTEL_PROFILE_IMPORT_TIMEOUT", "45") or "45")),
            smtp_host=get("INTEL_SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(get("INTEL_SMTP_PORT", "465") or "465"),
            smtp_username=get("INTEL_SMTP_USERNAME", get("INTEL_IMAP_USERNAME")),
            smtp_password=get("INTEL_SMTP_PASSWORD", get("INTEL_IMAP_PASSWORD")),
            smtp_use_ssl=get("INTEL_SMTP_USE_SSL", "1").lower() in {"1", "true", "yes", "on"},
            reply_from=get("INTEL_REPLY_FROM", get("INTEL_IMAP_USERNAME")),
            mark_seen_on_success=get("INTEL_IMAP_MARK_SEEN_ON_SUCCESS", "0").lower() in {"1", "true", "yes", "on"},
        )

    def validate(self) -> None:
        if not self.host:
            raise RuntimeError("INTEL_IMAP_HOST is required")
        if not self.username:
            raise RuntimeError("INTEL_IMAP_USERNAME is required")
        if not self.password:
            raise RuntimeError("INTEL_IMAP_PASSWORD is required")


class IntelMailboxStore:
    def __init__(self, base_path: Optional[Path] = None):
        self.root = Path(base_path) if base_path is not None else _safe_db_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.messages_dir = self.root / "messages"
        self.outbox_dir = self.root / "outbox"
        self.results_dir = self.root / "results"
        self.messages_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state(self._initial_state())

    @staticmethod
    def _initial_state() -> Dict[str, Any]:
        return {"updated_at": _utc_now_iso(), "messages": []}

    def _read_state(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._initial_state()
            payload.setdefault("updated_at", _utc_now_iso())
            payload.setdefault("messages", [])
            return payload
        except Exception:
            return self._initial_state()

    def _write_state(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = _utc_now_iso()
        fd, tmp_path = tempfile.mkstemp(prefix="intel_mailbox_", suffix=".json", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=True, indent=2, sort_keys=True)
            os.replace(tmp_path, self.state_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def list_messages(self) -> List[Dict[str, Any]]:
        return list(self._read_state().get("messages") or [])

    def has_processed_message(self, message_id: str) -> bool:
        message_id = str(message_id or "").strip()
        if not message_id:
            return False
        for entry in self._read_state().get("messages") or []:
            if str(entry.get("message_id") or "").strip() == message_id and entry.get("status") == "processed":
                return True
        return False

    def persist_message(self, metadata: Dict[str, Any], raw_bytes: bytes, attachments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        message_id = str(metadata.get("message_id") or "").strip()
        message_key = _derive_message_key(message_id, raw_bytes)
        message_dir = self.messages_dir / message_key
        attachments_dir = message_dir / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)

        raw_path = message_dir / "message.eml"
        raw_path.write_bytes(raw_bytes)

        stored_attachments: List[Dict[str, Any]] = []
        used_names: set[str] = set()
        for index, item in enumerate(attachments):
            base_name = _sanitize_filename(item.get("filename") or "", f"attachment_{index + 1}")
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            filename = base_name
            counter = 2
            while filename in used_names:
                filename = f"{stem}_{counter}{suffix}"
                counter += 1
            used_names.add(filename)
            stored_path = attachments_dir / filename
            stored_path.write_bytes(item.get("content") or b"")
            stored_attachments.append(
                {
                    "filename": filename,
                    "mime_type": str(item.get("mime_type") or ""),
                    "stored_path": str(stored_path),
                    "kind": str(item.get("kind") or _kind_for_mime(item.get("mime_type") or "")),
                    "size_bytes": stored_path.stat().st_size,
                    "content_id": str(item.get("content_id") or ""),
                }
            )

        record = {
            "message_key": message_key,
            "message_id": message_id,
            "from_email": str(metadata.get("from_email") or ""),
            "from_name": str(metadata.get("from_name") or ""),
            "subject": str(metadata.get("subject") or ""),
            "received_at": str(metadata.get("received_at") or ""),
            "raw_path": str(raw_path),
            "attachments": stored_attachments,
            "status": "persisted",
            "updated_at": _utc_now_iso(),
        }
        state = self._read_state()
        messages = [item for item in state.get("messages") or [] if item.get("message_key") != message_key]
        messages.append(record)
        state["messages"] = sorted(messages, key=lambda item: item.get("received_at", ""), reverse=True)
        self._write_state(state)
        return record

    def record_processed(self, message_key: str, trace_id: str, result_payload: Dict[str, Any], delivery: Dict[str, Any]) -> Dict[str, Any]:
        result_path = self.results_dir / f"{message_key}.json"
        result_path.write_text(json.dumps(result_payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        state = self._read_state()
        messages = list(state.get("messages") or [])
        for item in messages:
            if item.get("message_key") != message_key:
                continue
            item["status"] = "processed"
            item["trace_id"] = trace_id
            item["result_path"] = str(result_path)
            item["delivery"] = dict(delivery or {})
            item["updated_at"] = _utc_now_iso()
            break
        state["messages"] = messages
        self._write_state(state)
        return {"result_path": str(result_path)}

    def record_failure(self, message_key: str, error_message: str) -> None:
        state = self._read_state()
        for item in state.get("messages") or []:
            if item.get("message_key") != message_key:
                continue
            item["status"] = "failed"
            item["error"] = str(error_message or "")[:2000]
            item["updated_at"] = _utc_now_iso()
            break
        self._write_state(state)

    def write_outbox_payload(self, message_key: str, payload: Dict[str, Any]) -> Path:
        outbox_path = self.outbox_dir / f"{message_key}.json"
        outbox_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        return outbox_path


def parse_email_bytes(raw_bytes: bytes) -> Dict[str, Any]:
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = _decode_header_value(message.get("Subject"))
    from_name, from_email = parseaddr(message.get("From", ""))
    to_name, to_email = parseaddr(message.get("To", ""))
    from_name = _decode_header_value(from_name)
    to_name = _decode_header_value(to_name)
    from_email = str(from_email or "").strip().lower()
    to_email = str(to_email or "").strip().lower()
    message_id = str(message.get("Message-ID") or "").strip()
    if not message_id:
        message_id = f"<sha1-{hashlib.sha1(raw_bytes).hexdigest()[:24]}@cortex.local>"
    date_header = str(message.get("Date") or "").strip()
    try:
        received_at = parsedate_to_datetime(date_header).astimezone(timezone.utc).replace(microsecond=0).isoformat()
    except Exception:
        received_at = _utc_now_iso()

    text_parts: List[str] = []
    html_parts: List[str] = []
    attachments: List[Dict[str, Any]] = []

    part_index = 0
    for part in message.walk():
        if part.is_multipart():
            continue
        content_type = str(part.get_content_type() or "").lower()
        disposition = str(part.get_content_disposition() or "").lower()
        filename = _decode_header_value(part.get_filename())
        payload = part.get_payload(decode=True) or b""

        if disposition in {"attachment", "inline"} or filename:
            part_index += 1
            inferred_name = filename or f"part_{part_index}"
            attachments.append(
                {
                    "filename": inferred_name,
                    "mime_type": content_type,
                    "kind": _kind_for_mime(content_type),
                    "content": payload,
                    "content_id": str(part.get("Content-ID") or "").strip("<> "),
                }
            )
            continue

        charset = part.get_content_charset() or "utf-8"
        try:
            decoded = payload.decode(charset, errors="replace")
        except Exception:
            decoded = payload.decode("utf-8", errors="replace")
        if content_type == "text/plain":
            text_parts.append(decoded)
        elif content_type == "text/html":
            html_parts.append(decoded)

    raw_text = "\n\n".join(part.strip() for part in text_parts if part.strip())
    html_text = "\n\n".join(part.strip() for part in html_parts if part.strip())
    all_email_sources = [subject, raw_text, html_text, from_email]
    all_email_sources.extend(item.get("filename") or "" for item in attachments)
    extracted_emails = _extract_emails_from_text("\n".join(all_email_sources))

    return {
        "message_id": message_id,
        "subject": subject,
        "from_name": from_name,
        "from_email": from_email,
        "to_name": to_name,
        "to_email": to_email,
        "received_at": received_at,
        "raw_text": raw_text,
        "html_text": html_text,
        "attachments": attachments,
        "extracted_emails": extracted_emails,
        "auto_submitted": str(message.get("Auto-Submitted") or "").strip().lower(),
        "x_cortex_mailbox_reply": str(message.get("X-Cortex-Mailbox-Reply") or "").strip().lower(),
    }


class IntelMailboxResultClient:
    def __init__(self, store: IntelMailboxStore, callback_url: str = "", callback_secret: str = "", timeout: int = 30):
        self.store = store
        self.callback_url = str(callback_url or "").strip()
        self.callback_secret = str(callback_secret or "").strip()
        self.timeout = max(5, int(timeout or 30))

    def deliver(
        self,
        message_key: str,
        payload: Dict[str, Any],
        delivery_payload: Optional[Dict[str, Any]] = None,
        callback_url_override: str = "",
    ) -> Dict[str, Any]:
        effective_url = str(callback_url_override or self.callback_url).strip()
        if not effective_url:
            outbox_path = self.store.write_outbox_payload(message_key, payload)
            return {"status": "outbox", "path": str(outbox_path)}

        headers = {"Content-Type": "application/json"}
        if self.callback_secret:
            headers["X-Queue-Key"] = self.callback_secret
        response = requests.post(effective_url, headers=headers, json=delivery_payload or payload, timeout=self.timeout)
        response.raise_for_status()
        delivery: Dict[str, Any] = {"status": "posted", "http_status": response.status_code}
        try:
            body = response.json()
            if isinstance(body, dict):
                delivery["response"] = body
                if body.get("error"):
                    raise RuntimeError(f"Website import error: {body['error']}")
        except Exception:
            text = (response.text or "").strip()
            if text:
                delivery["response_text"] = text[:500]
            raise
        return delivery


class IntelMailboxReplyClient:
    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 465,
        username: str = "",
        password: str = "",
        use_ssl: bool = True,
        reply_from: str = "",
    ):
        self.smtp_host = str(smtp_host or "").strip()
        self.smtp_port = int(smtp_port or 465)
        self.username = str(username or "").strip()
        self.password = str(password or "").strip()
        self.use_ssl = bool(use_ssl)
        self.reply_from = str(reply_from or username or "").strip()

    def enabled(self) -> bool:
        return bool(self.smtp_host and self.reply_from and self.username and self.password)

    def send(self, to_email: str, subject: str, body: str, in_reply_to: str = "") -> Dict[str, Any]:
        recipient = str(to_email or "").strip().lower()
        if not recipient:
            return {"status": "skipped", "reason": "no_recipient"}
        if not self.enabled():
            return {"status": "disabled"}

        message = EmailMessage()
        message["Subject"] = str(subject or "").strip()
        message["From"] = self.reply_from
        message["To"] = recipient
        message["Auto-Submitted"] = "auto-replied"
        message["X-Auto-Response-Suppress"] = "All"
        message["X-Cortex-Mailbox-Reply"] = "1"
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = in_reply_to
        message.set_content(str(body or "").strip() + "\n")

        if self.use_ssl:
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30, context=ssl.create_default_context()) as client:
                client.login(self.username, self.password)
                client.send_message(message)
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as client:
                client.starttls(context=ssl.create_default_context())
                client.login(self.username, self.password)
                client.send_message(message)
        return {"status": "sent"}


class IntelMailboxPoller:
    def __init__(
        self,
        config: IntelMailboxConfig,
        store: Optional[IntelMailboxStore] = None,
        extractor: Optional[Callable[[Dict[str, Any]], tuple[Dict[str, Any], Optional[Path]]]] = None,
        result_client: Optional[IntelMailboxResultClient] = None,
        reply_client: Optional[IntelMailboxReplyClient] = None,
        csv_importer: Optional[CsvProfileImportProcessor] = None,
        imap_factory: Optional[Callable[..., Any]] = None,
        signal_store: Optional[StakeholderSignalStore] = None,
    ):
        self.config = config
        self.store = store or IntelMailboxStore()
        self.extractor = extractor or extract_intel
        self.result_client = result_client or IntelMailboxResultClient(
            self.store,
            callback_url=config.callback_url,
            callback_secret=config.callback_secret,
            timeout=config.callback_timeout,
        )
        self.reply_client = reply_client or IntelMailboxReplyClient(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            username=config.smtp_username,
            password=config.smtp_password,
            use_ssl=config.smtp_use_ssl,
            reply_from=config.reply_from,
        )
        self.csv_importer = csv_importer or CsvProfileImportProcessor.from_config(
            explicit_url=config.profile_import_url,
            callback_url=config.callback_url,
            queue_server_url=str(os.environ.get("QUEUE_SERVER_URL") or ""),
            queue_secret=config.callback_secret or str(os.environ.get("QUEUE_SECRET_KEY") or ""),
            timeout=config.profile_import_timeout,
        )
        self.imap_factory = imap_factory or imaplib.IMAP4_SSL
        self.signal_store = signal_store or StakeholderSignalStore()
        self.note_processor = IntelNoteProcessor(self.extractor)

    @staticmethod
    def _looks_truncated_render_line(line: str) -> bool:
        text = str(line or "").strip()
        if not text:
            return False
        if text.endswith("..."):
            return True
        if ": " not in text:
            return False
        tail = text.split(": ", 1)[1].strip()
        if not tail:
            return False
        tail_words = [normalize_lookup(word) for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", tail)]
        if not tail_words:
            return False
        if tail_words[-1] in {"and", "the", "for", "of", "our", "to", "with", "by", "support"}:
            return True
        if len(text) > 120 and sum(1 for _ in re.finditer(r"(?<=[a-z0-9’”)])\s+(?=[A-Z][a-z])", tail)) >= 1:
            return True
        return False

    @classmethod
    def _heuristic_note_sanity_cleanup(cls, markdown_text: str) -> str:
        cleaned_lines: List[str] = []
        for raw_line in str(markdown_text or "").splitlines():
            line = str(raw_line or "").rstrip()
            stripped = line.strip()
            if stripped.startswith("- ") and ": " in stripped:
                prefix, tail = stripped.split(": ", 1)
                tail = re.sub(r"\s+", " ", tail).strip()
                if cls._looks_truncated_render_line(stripped):
                    tail = _compact_strategy_snippet(tail, local_only=False, limit=220, allow_two_sentences=False) or tail
                if tail.endswith("..."):
                    tail = re.sub(r"\s+\S*\.\.\.$", "", tail).rstrip(" ,.;:-")
                while True:
                    words = [word for word in tail.split() if word]
                    if not words or normalize_lookup(words[-1]) not in {"and", "the", "for", "of", "our", "to", "with", "by", "support"}:
                        break
                    words = words[:-1]
                    tail = " ".join(words).rstrip(" ,.;:-")
                if not tail or _looks_like_dirty_strategy_snippet(tail):
                    cleaned_lines.append(prefix)
                    continue
                cleaned_lines.append(f"{prefix}: {tail}")
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @classmethod
    def _note_needs_sanity_pass(cls, markdown_text: str) -> bool:
        text = str(markdown_text or "").strip()
        if not text:
            return False
        if "..." in text:
            return True
        return any(cls._looks_truncated_render_line(line) for line in text.splitlines())

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        if isinstance(response, dict):
            message = response.get("message")
            if isinstance(message, dict):
                return str(message.get("content") or "").strip()
            return str(response.get("response") or "").strip()
        message = getattr(response, "message", None)
        if message is not None:
            return str(getattr(message, "content", "") or "").strip()
        return str(getattr(response, "response", "") or "").strip()

    def _run_markdown_sanity_llm(self, markdown_text: str, llm_policy: Dict[str, Any]) -> str:
        model = str(
            llm_policy.get("actual_model")
            or llm_policy.get("model")
            or os.environ.get("LOCAL_LLM_SYNTHESIS_MODEL")
            or "qwen2.5:14b-instruct-q4_K_M"
        ).strip()
        if not model:
            return ""
        try:
            import ollama

            client = ollama.Client(timeout=45)
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are editing a markdown intelligence note for readability only. "
                            "Preserve structure, do not add facts, and never leave truncated or garbled sentences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rewrite this markdown note for readability only.\n"
                            "Rules:\n"
                            "- Keep the same headings and bullet structure.\n"
                            "- Do not add facts, claims, or recommendations.\n"
                            "- Fix broken English, OCR fragments, and incomplete bullet lines.\n"
                            "- Never leave ellipses, hanging clauses, or bullets ending mid-sentence.\n"
                            "- If a broken fragment cannot be repaired safely, remove only the broken tail and keep the bullet clean.\n"
                            "- Prefer a shorter clean bullet over a longer garbled one.\n"
                            "- Return markdown only.\n\n"
                            f"{markdown_text[:12000]}"
                        ),
                    },
                ],
                options={"temperature": 0.1, "num_predict": 2200},
            )
            return self._extract_chat_text(response)
        except Exception as exc:
            logger.debug("Intel mailbox markdown sanity pass unavailable: %s", exc)
            return ""

    def _finalize_note_markdown(self, markdown_text: str, llm_policy: Optional[Dict[str, Any]] = None) -> str:
        original = str(markdown_text or "").strip()
        policy = dict(llm_policy or {})
        if not self._note_needs_sanity_pass(original):
            return self._heuristic_note_sanity_cleanup(original)
        rewritten = self._run_markdown_sanity_llm(original, policy)
        if rewritten:
            rewritten = self._heuristic_note_sanity_cleanup(rewritten)
            if len(rewritten) >= max(120, len(original) // 2):
                return rewritten
        return self._heuristic_note_sanity_cleanup(original)

    def _allowed_sender(self, email_address: str) -> bool:
        sender = str(email_address or "").strip().lower()
        if sender == _TRUSTED_SELF_RELAY_MAILBOX:
            return True
        if not self.config.allowed_senders:
            return True
        return sender in self.config.allowed_senders

    @staticmethod
    def _effective_submitter_email(message: Dict[str, Any]) -> str:
        sender = str(message.get("from_email") or "").strip().lower()
        recipient = str(message.get("to_email") or "").strip().lower()
        if sender == _TRUSTED_SELF_RELAY_MAILBOX and recipient in {"", _TRUSTED_SELF_RELAY_MAILBOX}:
            return _TRUSTED_SELF_RELAY_SUBMITTER
        return sender

    def _reply_recipient(self, message: Dict[str, Any]) -> str:
        recipient = str(message.get("from_email") or "").strip().lower()
        mailbox_addresses = {
            str(self.config.username or "").strip().lower(),
            str(self.config.reply_from or "").strip().lower(),
            _TRUSTED_SELF_RELAY_MAILBOX,
        }
        mailbox_addresses = {item for item in mailbox_addresses if item}
        if recipient in mailbox_addresses:
            submitter = self._effective_submitter_email(message)
            if submitter and submitter not in mailbox_addresses:
                return submitter
            return ""
        return recipient

    @staticmethod
    def _mailbox_suppression_reason(message: Dict[str, Any]) -> str:
        subject = str(message.get("subject") or "").strip().lower()
        body = _combined_mailbox_text(message)
        body_lower = re.sub(r"\s+", " ", body.lower()).strip()
        if _YOUTUBE_SUMMARISER_SUBJECT_RE.search(subject):
            return "youtube_summariser"
        if str(message.get("x_cortex_mailbox_reply") or "").strip().lower() in {"1", "true", "yes"}:
            return "cortex_processed_reply"
        if (
            _CORTEX_PROCESSED_REPLY_RE.search(body)
            and re.search(r"(?im)^\s*Depth\s*:", body)
            and re.search(r"(?im)^\s*Title\s*:", body)
        ):
            return "cortex_processed_reply"
        if "forwarding confirmation" in subject and all(
            marker in body_lower for marker in _MAILBOX_FORWARDING_CONFIRMATION_MARKERS[:2]
        ):
            return "mailbox_forwarding_confirmation"
        if all(marker in body_lower for marker in _MAILBOX_FORWARDING_CONFIRMATION_MARKERS):
            return "mailbox_forwarding_confirmation"
        return ""

    @staticmethod
    def _mark_imap_seen(client: Any, imap_id: Any) -> None:
        try:
            client.store(imap_id, "+FLAGS", "\\Seen")
        except Exception:
            logger.warning("Failed to mark IMAP message %s as seen", imap_id)

    def _known_org_scopes(self) -> List[str]:
        names = {str(self.config.org_name or "").strip()}
        try:
            profiles = self.signal_store.list_profiles(org_name="")
        except Exception:
            profiles = []
        for profile in profiles:
            org_name = str(profile.get("org_name") or "").strip()
            if org_name:
                names.add(org_name)
        try:
            state = self.signal_store.get_state()
        except Exception:
            state = {}
        for key, context in dict(state.get("org_contexts") or {}).items():
            del key
            org_name = str((context or {}).get("org_name") or "").strip()
            if org_name:
                names.add(org_name)
        return sorted(name for name in names if name)

    def _match_known_org_scope(self, requested_org_name: str) -> str:
        requested = str(requested_org_name or "").strip()
        if not requested:
            return ""
        wanted = normalize_lookup(requested)
        exact_matches = [
            candidate
            for candidate in self._known_org_scopes()
            if normalize_lookup(candidate) == wanted
        ]
        if exact_matches:
            return exact_matches[0]
        compatible_matches = [
            candidate
            for candidate in self._known_org_scopes()
            if orgs_compatible(candidate, requested)
        ]
        if compatible_matches:
            compatible_matches.sort(key=lambda item: (len(normalize_lookup(item)), item))
            return compatible_matches[0]
        prefix_matches = [
            candidate
            for candidate in self._known_org_scopes()
            if wanted.startswith(normalize_lookup(candidate) + " ")
        ]
        if prefix_matches:
            prefix_matches.sort(key=lambda item: (-len(normalize_lookup(item)), item))
            return prefix_matches[0]
        return ""

    def _match_sender_domain_scope(self, sender_email: str) -> str:
        domain = _extract_email_domain(sender_email)
        labels = _domain_labels(domain)
        if not labels:
            return ""

        candidates: List[tuple[int, str]] = []
        known_scopes = self._known_org_scopes()
        for scope in known_scopes:
            scope_label = normalize_lookup(scope)
            scope_initialism = _org_initialism(scope).lower()
            for label in labels:
                score = 0
                if scope_label == label:
                    score = 100
                elif scope_initialism and scope_initialism == label:
                    score = 90
                elif scope_label.startswith(label) or label in scope_label:
                    score = 70
                if score:
                    candidates.append((score, scope))

        try:
            profiles = self.signal_store.list_profiles(org_name="")
        except Exception:
            profiles = []
        for profile in profiles:
            if str(profile.get("target_type") or "").strip().lower() != "organisation":
                continue
            org_name = str(profile.get("canonical_name") or profile.get("org_name") or "").strip()
            if not org_name:
                continue
            profile_labels = {_label for _label in _domain_labels(_extract_url_domain(str(profile.get("website_url") or "")))}
            for alias in profile.get("aliases") or []:
                alias_text = str(alias or "").strip()
                if alias_text:
                    profile_labels.add(normalize_lookup(alias_text))
                    alias_initialism = _org_initialism(alias_text).lower()
                    if alias_initialism:
                        profile_labels.add(alias_initialism)
            profile_labels.add(normalize_lookup(org_name))
            initialism = _org_initialism(org_name).lower()
            if initialism:
                profile_labels.add(initialism)

            for label in labels:
                if label in profile_labels:
                    matched = self._match_known_org_scope(org_name) or org_name
                    candidates.append((110, matched))

        if not candidates:
            return ""
        candidates.sort(key=lambda item: (-item[0], len(normalize_lookup(item[1])), item[1]))
        return candidates[0][1]

    def _resolve_message_routing(self, message: Dict[str, Any], persisted: Dict[str, Any]) -> Dict[str, Any]:
        raw_subject = str(message.get("subject") or "").strip()
        sender_email = str(message.get("from_email") or "").strip()
        requested_org_name = _extract_subject_entity_override(raw_subject)
        requested_depth = _extract_subject_depth_override(raw_subject) or "default"
        force_reingest = _extract_subject_force_override(raw_subject)
        subject_without_override = _strip_subject_force_override(
            _strip_subject_depth_override(_strip_subject_entity_override(raw_subject))
        )
        subject_without_override = _strip_subject_privacy_markers(subject_without_override)
        matched_org_name = self._match_known_org_scope(requested_org_name)
        if matched_org_name and requested_org_name:
            override_pattern = re.compile(
                rf"(?i)(?:^|[\s\[\]()|;>])(?:entity|org|organisation|organization)\s*:\s*{re.escape(matched_org_name)}\b"
            )
            refined_subject = override_pattern.sub(" ", raw_subject, count=1)
            refined_subject = _strip_subject_force_override(_strip_subject_depth_override(refined_subject))
            refined_subject = _strip_subject_privacy_markers(refined_subject)
            refined_subject = re.sub(r"\s*[|;>,]\s*", " ", refined_subject)
            refined_subject = re.sub(r"\s+", " ", refined_subject).strip(" -|:;,")
            if refined_subject:
                subject_without_override = refined_subject
                requested_org_name = matched_org_name
        clean_subject = _normalized_mail_subject(subject_without_override or raw_subject)
        sender_domain_org_name = ""
        effective_org_name = matched_org_name or self.config.org_name
        status = "default"
        if requested_org_name and matched_org_name:
            status = "matched_override"
        elif requested_org_name:
            status = "unmatched_override"
        else:
            sender_domain_org_name = self._match_sender_domain_scope(sender_email)
            if sender_domain_org_name:
                effective_org_name = sender_domain_org_name
                status = "matched_sender_domain"
        attachments = list(persisted.get("attachments") or [])
        has_document_attachment = any(
            str(item.get("kind") or "").strip().lower() == "document"
            for item in attachments
        )
        has_org_chart_image_attachment = any(
            str(item.get("kind") or "").strip().lower() == "image"
            and looks_like_org_chart_attachment(
                str(item.get("filename") or "").strip(),
                clean_subject,
            )
            for item in attachments
        )
        subject_org_hint = _subject_org_hint(clean_subject) if (has_document_attachment or has_org_chart_image_attachment) else ""
        if subject_org_hint and effective_org_name:
            subject_org_hint = _strip_scope_prefix_from_subject_hint(subject_org_hint, effective_org_name)
        return {
            "default_org_name": self.config.org_name,
            "requested_org_name": requested_org_name,
            "matched_org_name": matched_org_name,
            "sender_domain_org_name": sender_domain_org_name,
            "effective_org_name": effective_org_name,
            "status": status,
            "clean_subject": clean_subject,
            "subject_org_hint": subject_org_hint,
            "extraction_depth": requested_depth,
            "force_reingest": force_reingest,
            "has_document_attachment": has_document_attachment,
            "has_org_chart_image_attachment": has_org_chart_image_attachment,
        }

    def _build_extract_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        trace_id: str,
        routing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        routing = routing or {}
        tags = ["email_intake"]
        if any(item.get("kind") == "image" for item in persisted.get("attachments") or []):
            tags.append("image_attachment")
        if any(item.get("kind") == "document" for item in persisted.get("attachments") or []):
            tags.append("document_attachment")
        clean_subject = str(routing.get("clean_subject") or _normalized_mail_subject(message.get("subject", ""))).strip()
        subject_org_hint = str(routing.get("subject_org_hint") or "").strip()
        effective_org_name = str(routing.get("effective_org_name") or self.config.org_name).strip()
        extraction_depth = str(routing.get("extraction_depth") or "default").strip().lower() or "default"
        return {
            "org_name": effective_org_name,
            "source_system": self.config.source_system,
            "trace_id": trace_id,
            "signal_type": "email_intel",
            "original_subject": message.get("subject", ""),
            "submitted_by": self._effective_submitter_email(message),
            "message_id": message.get("message_id", ""),
            "received_at": message.get("received_at", ""),
            "subject": clean_subject,
            "raw_text": message.get("raw_text", ""),
            "html_text": message.get("html_text", ""),
            "primary_url": "",
            "text_note": "",
            "parsed_candidate_name": subject_org_hint,
            "parsed_candidate_employer": subject_org_hint,
            "target_type": "",
            "attachments": list(persisted.get("attachments") or []),
            "tags": tags,
            "extraction_depth": extraction_depth,
            "mailbox_routing": dict(routing),
        }

    def _subscriber_strategic_profile(self, scope_org_name: str = "") -> Dict[str, Any]:
        org_name = str(scope_org_name or self.config.org_name).strip()
        return dict(self.signal_store.get_org_context(org_name).get("org_strategic_profile") or {})

    @staticmethod
    def _choose_entity_primary_from_entities(
        output_data: Dict[str, Any],
        scope_org_name: str = "",
    ) -> Dict[str, str]:
        scope_key = normalize_lookup(scope_org_name)
        entities = list(output_data.get("entities") or [])
        org_candidates: List[tuple[float, Dict[str, str]]] = []
        person_candidates: List[tuple[float, Dict[str, str]]] = []

        for entity in entities:
            target_type = str(entity.get("target_type") or "").strip().lower()
            name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
            if not name:
                continue
            confidence = 0.0
            try:
                confidence = float(entity.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0
            candidate = {
                "target_type": target_type,
                "name": name,
                "employer": str(entity.get("current_employer") or entity.get("employer") or "").strip(),
            }
            if target_type == "organisation":
                score = confidence
                if scope_key and normalize_lookup(name) != scope_key:
                    score += 1.0
                org_candidates.append((score, candidate))
            elif target_type == "person":
                person_candidates.append((confidence, candidate))

        if org_candidates:
            return max(org_candidates, key=lambda item: item[0])[1]
        if person_candidates:
            return max(person_candidates, key=lambda item: item[0])[1]
        return {"target_type": "", "name": "", "employer": ""}

    def _assess_subscriber_fit(
        self,
        scope_org_name: str,
        message: Dict[str, Any],
        output_data: Dict[str, Any],
        strategic_doc: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = self._subscriber_strategic_profile(scope_org_name)
        if not profile:
            return {}

        priority_industries = [str(item).strip() for item in (profile.get("priority_industries") or []) if str(item).strip()]
        key_themes = [str(item).strip() for item in (profile.get("key_themes") or []) if str(item).strip()]
        strategic_objectives = [str(item).strip() for item in (profile.get("strategic_objectives") or []) if str(item).strip()]
        industries = [str(item).strip() for item in (profile.get("industries") or []) if str(item).strip()]
        description = str(profile.get("description") or "").strip()
        low_relevance_themes = [str(item).strip() for item in (profile.get("low_relevance_themes") or []) if str(item).strip()]

        if not any([priority_industries, key_themes, strategic_objectives, industries, description, low_relevance_themes]):
            return {}

        text_parts = [
            str(message.get("subject") or "").strip(),
            str(message.get("raw_text") or "").strip(),
            str(output_data.get("summary") or "").strip(),
            str((strategic_doc or {}).get("strategic_summary") or "").strip(),
            " ".join(str(item.get("headline") or "").strip() for item in ((strategic_doc or {}).get("strategic_signals") or []) if str(item.get("headline") or "").strip()),
            " ".join(str(item.get("snippet") or "").strip() for item in ((strategic_doc or {}).get("strategic_signals") or []) if str(item.get("snippet") or "").strip()),
        ]
        haystack = normalize_lookup(" ".join(part for part in text_parts if part))
        if not haystack:
            return {}
        haystack_tokens = _fit_phrase_token_set(haystack)
        haystack_strict_tokens = _fit_phrase_token_set(haystack, expand=False)

        def _matched(values: Sequence[str], *, strict: bool = False, expand: bool = True) -> List[str]:
            matches: List[str] = []
            for value in values:
                normalized = normalize_lookup(value)
                if not normalized:
                    continue
                if normalized in haystack:
                    matches.append(value)
                    continue
                phrase_tokens = _fit_phrase_token_set(value, expand=expand and not strict)
                if not phrase_tokens:
                    continue
                overlap = phrase_tokens & (haystack_strict_tokens if strict else haystack_tokens)
                if len(phrase_tokens) == 1:
                    if overlap:
                        matches.append(value)
                    continue
                if strict:
                    if len(overlap) >= 3 and len(overlap) / max(len(phrase_tokens), 1) >= 0.75:
                        matches.append(value)
                    continue
                if len(overlap) >= 2 or (overlap and len(overlap) / max(len(phrase_tokens), 1) >= 0.6):
                    matches.append(value)
            return matches

        matched_priority_industries = _matched(priority_industries, strict=True, expand=False)
        matched_industries = _matched(industries, strict=True, expand=False)
        matched_themes = _matched(key_themes)
        matched_objectives = _matched(strategic_objectives, strict=True)
        matched_low_relevance = _matched(low_relevance_themes, strict=True)

        score = (
            len(matched_priority_industries) * 3
            + len(matched_themes) * 3
            + len(matched_objectives) * 2
            + len(matched_industries) * 2
            - len(matched_low_relevance) * 2
        )
        if description:
            description_tokens = [
                normalize_lookup(token)
                for token in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", description)
                if len(token) > 4
            ]
            score += sum(1 for token in description_tokens[:8] if token and token in haystack)

        if score >= 6:
            fit_label = "high_fit"
        elif score >= 3:
            fit_label = "medium_fit"
        else:
            fit_label = "low_fit"

        return {
            "fit_label": fit_label,
            "fit_score": score,
            "matched_priority_industries": matched_priority_industries[:6],
            "matched_industries": matched_industries[:6],
            "matched_themes": matched_themes[:8],
            "matched_objectives": matched_objectives[:8],
            "matched_low_relevance_themes": matched_low_relevance[:6],
        }

    def _fit_commentary(
        self,
        scope_org_name: str,
        fit_assessment: Dict[str, Any],
        primary_name: str = "",
    ) -> str:
        fit = dict(fit_assessment or {})
        fit_label = str(fit.get("fit_label") or "").strip()
        if not fit_label:
            return ""

        org_label = str(scope_org_name or self.config.org_name).strip() or "the subscriber"
        primary_label = str(primary_name or "").strip()
        subject_label = primary_label or "this intelligence"
        if fit_label == "high_fit":
            opening = f"High fit for {org_label}: {subject_label} aligns strongly with current service and market priorities."
        elif fit_label == "medium_fit":
            opening = f"Moderate fit for {org_label}: {subject_label} overlaps with some current service and market priorities."
        else:
            opening = f"Lower fit for {org_label}: {subject_label} is less directly aligned with current service priorities."

        details: List[str] = []
        matched_priority = [str(item).strip() for item in (fit.get("matched_priority_industries") or []) if str(item).strip()]
        matched_themes = [str(item).strip() for item in (fit.get("matched_themes") or []) if str(item).strip()]
        matched_objectives = [str(item).strip() for item in (fit.get("matched_objectives") or []) if str(item).strip()]
        matched_low = [str(item).strip() for item in (fit.get("matched_low_relevance_themes") or []) if str(item).strip()]

        if matched_priority:
            details.append(f"Priority industry match: {', '.join(matched_priority[:3])}.")
        if matched_themes:
            details.append(f"Matched themes: {', '.join(matched_themes[:3])}.")
        if matched_objectives:
            details.append(f"Matched objectives: {', '.join(matched_objectives[:2])}.")
        if matched_low:
            details.append(f"Potentially lower-relevance themes also present: {', '.join(matched_low[:2])}.")

        return " ".join([opening] + details).strip()

    @staticmethod
    def _subscriber_signal_fit_score(item: Dict[str, Any], fit_assessment: Dict[str, Any]) -> int:
        fit = dict(fit_assessment or {})
        text_parts = [
            str(item.get("headline") or "").strip(),
            str(item.get("snippet") or "").strip(),
            str(item.get("evidence") or "").strip(),
        ]
        haystack = normalize_lookup(" ".join(part for part in text_parts if part))
        if not haystack:
            return 0
        haystack_tokens = _fit_phrase_token_set(haystack)
        score = 0
        category = normalize_lookup(str(item.get("category") or "").strip())
        for value in fit.get("matched_priority_industries") or []:
            phrase_tokens = _fit_phrase_token_set(value)
            overlap = phrase_tokens & haystack_tokens
            if overlap:
                score += 4
        for value in fit.get("matched_themes") or []:
            phrase_tokens = _fit_phrase_token_set(value)
            overlap = phrase_tokens & haystack_tokens
            if len(overlap) >= 2 or (overlap and len(phrase_tokens) <= 2):
                score += 3
        for value in fit.get("matched_objectives") or []:
            phrase_tokens = _fit_phrase_token_set(value, expand=False)
            overlap = phrase_tokens & haystack_tokens
            if len(overlap) >= 2 or (overlap and len(phrase_tokens) <= 2):
                score += 2
        for value in fit.get("matched_low_relevance_themes") or []:
            phrase_tokens = _fit_phrase_token_set(value, expand=False)
            overlap = phrase_tokens & haystack_tokens
            if overlap:
                score -= 4
        if category in {"community_commitment", "cultural_commitment", "social_impact"}:
            score -= 2
        if any(token in haystack_tokens for token in {"country", "community", "communities", "cultural", "aboriginal", "indigenous", "first", "nations", "elders"}):
            score -= 1
        return score

    @staticmethod
    def _looks_like_fragmented_strategy_signal(item: Dict[str, Any]) -> bool:
        headline = normalize_lookup(str(item.get("headline") or "").strip())
        if not headline:
            return True
        if len(headline.split()) >= 5 and any(marker in headline for marker in ("and our", "our customers", "our members", "every day")):
            return True
        return False

    @classmethod
    def _prioritize_strategic_signals_for_subscriber(
        cls,
        strategic_doc: Dict[str, Any],
        fit_assessment: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        signals = [
            dict(item)
            for item in strategic_doc.get("strategic_signals") or []
            if str(item.get("headline") or "").strip()
            and not cls._looks_like_fragmented_strategy_signal(item)
        ]
        if not signals:
            return [], []
        scored_signals: List[tuple[int, int, Dict[str, Any]]] = []
        for index, item in enumerate(signals):
            score = cls._subscriber_signal_fit_score(item, fit_assessment)
            scored_signals.append((score, index, item))
        background: List[str] = []
        low_signal_keys = {
            normalize_lookup(str(item.get("headline") or "").strip())
            for score, _, item in scored_signals
            if score < 0 and str(item.get("headline") or "").strip()
        }
        prioritized = [
            item
            for score, _, item in sorted(scored_signals, key=lambda entry: (-entry[0], entry[1]))
            if normalize_lookup(str(item.get("headline") or "").strip()) not in low_signal_keys
        ]
        if low_signal_keys:
            for theme in strategic_doc.get("themes") or []:
                label = _clean_display_label(str(theme).strip())
                if label and normalize_lookup(label) in low_signal_keys:
                    background.append(label)
        return prioritized, _dedupe_preserve_order(background)

    def _lookup_org_profile(self, org_name: str, scope_org_name: str = "") -> Dict[str, Any]:
        wanted = normalize_lookup(org_name)
        if not wanted:
            return {}
        effective_scope = str(scope_org_name or self.config.org_name).strip()
        for profile in self.signal_store.list_profiles(org_name=effective_scope):
            if str(profile.get("target_type") or "").strip().lower() != "organisation":
                continue
            if normalize_lookup(profile.get("canonical_name") or "") == wanted:
                return profile
            for alias in profile.get("aliases") or []:
                if normalize_lookup(alias) == wanted:
                    return profile
        return {}

    def _resolve_known_org_label(
        self,
        hint: str,
        output_data: Dict[str, Any],
        scope_org_name: str = "",
    ) -> str:
        cleaned_hint = _clean_display_label(hint)
        if not cleaned_hint:
            return ""

        for item in list(output_data.get("entities") or []) + list(output_data.get("organisations") or []):
            if str(item.get("target_type") or "").strip().lower() not in {"", "organisation", "organization"}:
                continue
            name = _clean_display_label(str(item.get("canonical_name") or item.get("name") or "").strip())
            match_type = _org_hint_match_type(cleaned_hint, name)
            if match_type == "exact":
                return name
            if match_type == "acronym":
                return name

        effective_scope = str(scope_org_name or self.config.org_name).strip()
        try:
            profiles = self.signal_store.list_profiles(org_name=effective_scope)
        except Exception:
            profiles = []
        for profile in profiles:
            if str(profile.get("target_type") or "").strip().lower() != "organisation":
                continue
            canonical = _clean_display_label(
                str(profile.get("canonical_name") or profile.get("org_name") or "").strip()
            )
            if not canonical:
                continue
            labels = [canonical] + [
                _clean_display_label(str(alias or "").strip())
                for alias in (profile.get("aliases") or [])
                if _clean_display_label(str(alias or "").strip())
            ]
            for label in labels:
                match_type = _org_hint_match_type(cleaned_hint, label)
                if match_type in {"exact", "acronym"}:
                    return canonical

        return cleaned_hint

    def _infer_industry_name(self, message: Dict[str, Any], entity: Dict[str, Any], scope_org_name: str = "") -> str:
        explicit = str(entity.get("industry") or "").strip()
        if explicit:
            return explicit

        org_name = str(entity.get("canonical_name") or entity.get("name") or entity.get("current_employer") or "").strip()
        org_profile = self._lookup_org_profile(org_name, scope_org_name=scope_org_name)
        affiliations = org_profile.get("industry_affiliations") or []
        if affiliations:
            first = affiliations[0]
            name = str(first.get("industry_name") or "").strip()
            if name:
                return name
        existing = str(org_profile.get("industry") or "").strip()
        if existing:
            return existing

        return classify_entity_industry(
            entity=entity,
            message=message,
            strategic_profile=self._subscriber_strategic_profile(scope_org_name),
            org_profile_lookup=lambda item: self._lookup_org_profile(item, scope_org_name=scope_org_name),
        )

    def _extract_note_urls(self, message: Dict[str, Any], markdown_text: str) -> List[Dict[str, str]]:
        combined_urls = _extract_urls_from_text(
            "\n".join(
                [
                    str(message.get("raw_text") or ""),
                    str(message.get("html_text") or ""),
                    str(markdown_text or ""),
                ]
            )
        )
        urls: List[Dict[str, str]] = []
        seen: set[str] = set()
        for url in combined_urls:
            if url in seen:
                continue
            seen.add(url)
            urls.append(
                {
                    "url": url,
                    "url_type": "reference",
                    "title": "",
                    "description": "",
                }
            )
        return urls

    @staticmethod
    def _choose_primary_entity(output_data: Dict[str, Any]) -> Dict[str, str]:
        for bucket_name, target_type in (("people", "person"), ("organisations", "organisation")):
            bucket = output_data.get(bucket_name) or []
            if not bucket:
                continue
            item = bucket[0]
            return {
                "target_type": target_type,
                "name": str(item.get("canonical_name") or item.get("name") or "").strip(),
                "employer": str(item.get("current_employer") or item.get("employer") or "").strip(),
            }
        return {"target_type": "", "name": "", "employer": ""}

    @staticmethod
    def _choose_subject_primary_entity(subject_org_hint: str, output_data: Dict[str, Any]) -> Dict[str, str]:
        wanted = normalize_lookup(subject_org_hint)
        if not wanted:
            return {"target_type": "", "name": "", "employer": ""}

        for item in output_data.get("entities") or []:
            if str(item.get("target_type") or "").strip().lower() != "organisation":
                continue
            name = str(item.get("canonical_name") or item.get("name") or "").strip()
            if name and normalize_lookup(name) == wanted:
                return {"target_type": "organisation", "name": name, "employer": ""}
            match_type = _org_hint_match_type(subject_org_hint, name)
            if match_type == "acronym":
                return {"target_type": "organisation", "name": name, "employer": ""}
            if match_type == "compatible":
                return {"target_type": "organisation", "name": subject_org_hint, "employer": ""}

        for item in output_data.get("organisations") or []:
            name = str(item.get("canonical_name") or item.get("name") or "").strip()
            if name and normalize_lookup(name) == wanted:
                return {"target_type": "organisation", "name": name, "employer": ""}
            match_type = _org_hint_match_type(subject_org_hint, name)
            if match_type == "acronym":
                return {"target_type": "organisation", "name": name, "employer": ""}
            if match_type == "compatible":
                return {"target_type": "organisation", "name": subject_org_hint, "employer": ""}

        for item in output_data.get("people") or []:
            employer = str(item.get("current_employer") or item.get("employer") or "").strip()
            if employer and normalize_lookup(employer) == wanted:
                return {"target_type": "organisation", "name": employer, "employer": ""}
            match_type = _org_hint_match_type(subject_org_hint, employer)
            if match_type == "acronym":
                return {"target_type": "organisation", "name": employer, "employer": ""}
            if match_type == "compatible":
                return {"target_type": "organisation", "name": subject_org_hint, "employer": ""}

        return {"target_type": "organisation", "name": subject_org_hint, "employer": ""}

    def _choose_document_primary_entity(
        self,
        output_data: Dict[str, Any],
        strategic_doc: Dict[str, Any],
    ) -> Dict[str, str]:
        strategic_org_name = str(strategic_doc.get("org_name") or "").strip()
        strategic_org_key = normalize_lookup(strategic_org_name)
        summary_text = " ".join(
            [
                str(output_data.get("summary") or ""),
                str(strategic_doc.get("strategic_summary") or ""),
                " ".join(str(item.get("excerpt") or "") for item in (output_data.get("attachments") or [])),
            ]
        )
        haystack = normalize_lookup(summary_text)

        candidates: List[tuple[int, Dict[str, str]]] = []
        for entity in output_data.get("entities") or []:
            if str(entity.get("target_type") or "").strip().lower() != "organisation":
                continue
            name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
            if not name or _looks_like_generic_document_name(name) or _looks_like_low_signal_org_label(name):
                continue
            name_key = normalize_lookup(name)
            score = 0
            if strategic_org_key and name_key == strategic_org_key:
                score += 10
            if haystack and name_key and name_key in haystack:
                score += 4
            evidence = normalize_lookup(str(entity.get("evidence") or ""))
            if any(token in evidence for token in ("strategic", "annual report", "authored", "foreword", "chief executive officer", "president", "organizational chart", "organisational chart")):
                score += 2
            if normalize_lookup(name) == normalize_lookup(self.config.org_name):
                score -= 3
            candidates.append(
                (
                    score,
                    {
                        "target_type": "organisation",
                        "name": name,
                        "employer": "",
                    },
                )
            )

        if candidates:
            best_score, best = max(candidates, key=lambda item: item[0])
            if best_score > 0:
                return best

        if strategic_org_name and not _looks_like_generic_document_name(strategic_org_name) and not _looks_like_low_signal_org_label(strategic_org_name):
            return {
                "target_type": "organisation",
                "name": strategic_org_name,
                "employer": "",
            }
        return {"target_type": "", "name": "", "employer": ""}

    @staticmethod
    def _infer_note_source_type(message: Dict[str, Any], output_data: Dict[str, Any], message_kind: str = "") -> str:
        subject = str(message.get("subject") or "").lower()
        attachment_names = " ".join(str(item.get("filename") or "").lower() for item in output_data.get("attachments") or [])
        strategic_doc = dict((output_data.get("processing_meta") or {}).get("strategic_doc") or {})
        doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
        text = " ".join(
            [
                subject,
                str(message.get("raw_text") or "").lower(),
                str(output_data.get("summary") or "").lower(),
                attachment_names,
                doc_type,
            ]
        )
        if doc_type in {"strategic_plan", "annual_report", "industry_report"}:
            return doc_type
        if "org chart" in text or "organisational chart" in text or "organization chart" in text:
            return "org_chart"
        if "annual report" in text:
            return "annual_report"
        if any(token in text for token in ("strategic plan", "strategic direction")) or "strategy" in subject:
            return "strategic_plan"
        if "industry report" in text or "sector report" in text:
            return "industry_report"
        if message_kind == "document_analysis":
            return "general"
        if any(token in text for token in ("meeting", "met with", "call with", "spoke with", "introduction", "intro")):
            return "meeting_note"
        return "general"

    @staticmethod
    def _infer_reference_type(message: Dict[str, Any], entity: Dict[str, Any]) -> str:
        context = " ".join(
            [
                str(entity.get("evidence") or ""),
                str(message.get("subject") or ""),
                str(message.get("raw_text") or ""),
            ]
        ).lower()
        if "introduc" in context:
            return "intro"
        if any(token in context for token in ("decision maker", "decision-maker", "head of", "chief", "executive")):
            return "decision_maker"
        if any(token in context for token in ("met", "meeting", "spoke with", "call with")):
            return "meeting"
        if any(token in context for token in ("lead", "leading", "owner")):
            return "lead"
        if str(entity.get("target_type") or "").strip().lower() == "industry":
            return "affiliation"
        return "mention"

    @staticmethod
    def _infer_signal_type(message: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        text = " ".join(
            [
                str(message.get("subject") or ""),
                str(message.get("raw_text") or ""),
                str(output_data.get("summary") or ""),
            ]
        ).lower()
        if "intro" in text or "introduc" in text:
            return "intro_offered"
        if any(token in text for token in ("strategic", "transformation", "roadmap", "priority")):
            return "strategic_intent"
        if any(token in text for token in ("restructure", "reorganisation", "reorg")):
            return "org_restructure"
        if any(token in text for token in ("appointed", "joins", "joined", "new role")):
            return "leadership_change"
        return "strategic_intent"

    @staticmethod
    def _confidence_label(value: Any) -> str:
        try:
            score = float(value)
        except Exception:
            score = 0.5
        if score >= 0.85:
            return "confirmed"
        if score >= 0.6:
            return "probable"
        return "speculative"

    @staticmethod
    def _enrich_note_markdown(
        markdown_text: str,
        strategic_doc: Dict[str, Any],
        note_summary: str = "",
        extraction_depth: str = "default",
        fit_assessment: Optional[Dict[str, Any]] = None,
        scope_org_name: str = "",
        local_only: bool = False,
    ) -> str:
        doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
        depth = str(extraction_depth or strategic_doc.get("extraction_depth") or "default").strip().lower()
        if depth not in {"brief", "default", "detailed"}:
            depth = "default"
        doc_types_with_curated_notes = {"strategic_plan", "annual_report", "industry_report"}
        base = _strip_managed_note_sections(markdown_text)
        strategic_signals = [
            item
            for item in strategic_doc.get("strategic_signals") or []
            if str(item.get("headline") or "").strip()
        ]
        background_themes: List[str] = []
        fit_label = str((fit_assessment or {}).get("fit_label") or "").strip().lower()
        if strategic_signals and fit_label in {"medium_fit", "high_fit"}:
            strategic_signals, background_themes = IntelMailboxPoller._prioritize_strategic_signals_for_subscriber(
                strategic_doc,
                fit_assessment or {},
            )
        stakeholders = [
            item
            for item in strategic_doc.get("key_stakeholders") or []
            if _should_keep_note_stakeholder(item, str(strategic_doc.get("org_name") or "").strip())
        ]
        performance_indicators = [
            item
            for item in strategic_doc.get("performance_indicators") or []
            if str(item.get("label") or "").strip()
        ]
        major_projects = [
            item
            for item in strategic_doc.get("major_projects") or []
            if str(item.get("name") or "").strip()
        ]
        kpi_focuses = [str(item).strip() for item in strategic_doc.get("kpi_focuses") or [] if str(item).strip()]
        kpi_focuses = [] if doc_type == "annual_report" else [item for item in kpi_focuses if _looks_like_high_signal_kpi_focus(item)]
        signal_limit = 2 if depth == "brief" else 5 if depth == "default" else 8
        stakeholder_limit = 3 if depth == "brief" else 6 if depth == "default" else 10
        performance_limit = 3 if depth == "brief" else 6 if depth == "default" else 10
        project_limit = 3 if depth == "brief" else 6 if depth == "default" else 10
        kpi_limit = 2 if depth == "brief" else 5 if depth == "default" else 6

        sections: List[str] = []
        if doc_type in doc_types_with_curated_notes and note_summary.strip():
            sections.append("## Summary\n\n" + note_summary.strip())
        if strategic_signals:
            heading = "## Priority Strategic Themes" if fit_label in {"medium_fit", "high_fit"} and scope_org_name else "## Strategic Insights"
            lines = [heading, ""]
            for item in strategic_signals[:signal_limit]:
                headline = _clean_display_label(str(item.get("headline") or "").strip())
                if not _should_keep_rendered_strategic_signal(headline, doc_type=doc_type):
                    continue
                raw_snippet = str(item.get("snippet") or item.get("evidence") or "").strip()
                if doc_type == "strategic_plan":
                    snippet = _compact_strategy_snippet(
                        raw_snippet,
                        local_only=local_only,
                        limit=220 if depth == "brief" else 320 if depth == "default" else 520,
                        allow_two_sentences=depth == "detailed",
                    )
                    if not snippet or _looks_like_low_quality_detail_sentence(snippet):
                        for point in (item.get("detail_points") or []):
                            replacement = _render_detail_point_text(point, local_only=local_only)
                            if replacement:
                                snippet = replacement
                                break
                else:
                    snippet = _clean_note_signal_snippet(raw_snippet, doc_type=doc_type)
                line = f"- {headline}"
                if snippet:
                    line += f": {snippet}"
                lines.append(line)
            sections.append("\n".join(lines))
        if doc_type == "strategic_plan" and depth == "detailed" and strategic_signals:
            lines = ["## Detailed Theme Notes", ""]
            detailed_items = 0
            for item in strategic_signals[: min(signal_limit, 5)]:
                headline = _clean_display_label(str(item.get("headline") or "").strip())
                if not _should_keep_rendered_strategic_signal(headline, doc_type=doc_type):
                    continue
                detail_points = [
                    point
                    for point in (item.get("detail_points") or [])
                    if str(point.get("heading") or "").strip() and str(point.get("snippet") or "").strip()
                ]
                if not detail_points:
                    raw_snippet = str(item.get("snippet") or item.get("evidence") or "").strip()
                    fallback_snippet = _compact_strategy_snippet(
                        raw_snippet,
                        local_only=local_only,
                        limit=520,
                        allow_two_sentences=True,
                    )
                    if not fallback_snippet:
                        continue
                    lines.extend([f"### {headline}", "", fallback_snippet, ""])
                    detailed_items += 1
                    continue
                lines.extend([f"### {headline}", ""])
                rendered_points: List[str] = []
                for point in detail_points[:3]:
                    point_snippet = _render_detail_point_text(point, local_only=local_only)
                    if not point_snippet:
                        continue
                    if point_snippet in rendered_points:
                        continue
                    rendered_points.append(point_snippet)
                for point_snippet in rendered_points[:3]:
                    lines.append(point_snippet)
                    lines.append("")
                if rendered_points:
                    detailed_items += 1
                else:
                    lines = lines[:-2]
            if detailed_items:
                sections.append("\n".join(lines).strip())
        if background_themes and depth != "brief":
            lines = ["## Background Themes", ""]
            for item in background_themes[: (3 if depth == "default" else 5)]:
                lines.append(f"- {_clean_display_label(item)}")
            sections.append("\n".join(lines))
        if performance_indicators:
            heading = "## Performance Snapshot" if doc_type == "annual_report" else "## Performance Indicators"
            lines = [heading, ""]
            for item in performance_indicators[:performance_limit]:
                label = _clean_display_label(str(item.get("label") or "").strip())
                value = _clean_display_label(str(item.get("value") or "").strip())
                evidence = _compact_note_evidence(str(item.get("evidence") or "").strip())
                line = f"- {label}"
                if value:
                    line += f": {value}"
                if evidence:
                    line += f" | {evidence}"
                lines.append(line)
            sections.append("\n".join(lines))
        if major_projects:
            lines = ["## Key Projects", ""]
            for item in major_projects[:project_limit]:
                name = _clean_display_label(str(item.get("name") or "").strip())
                value = _clean_display_label(str(item.get("value") or "").strip())
                evidence = _compact_note_evidence(str(item.get("evidence") or "").strip())
                line = f"- {name}"
                if value:
                    line += f": {value}"
                if evidence:
                    line += f" | {evidence}"
                lines.append(line)
            sections.append("\n".join(lines))
        if stakeholders:
            lines = ["## Key Stakeholders", ""]
            for item in stakeholders[:stakeholder_limit]:
                name = _clean_display_label(str(item.get("name") or "").strip())
                employer = _clean_display_label(str(item.get("current_employer") or "").strip())
                if _looks_like_low_signal_org_label(employer):
                    employer = _clean_display_label(str(strategic_doc.get("org_name") or "").strip())
                role = _clean_display_role(str(item.get("current_role") or "").strip(), employer)
                parts = [name]
                if role:
                    parts.append(role)
                if employer:
                    parts.append(employer)
                lines.append("- " + " | ".join(parts))
            sections.append("\n".join(lines))
        if kpi_focuses:
            lines = ["## KPI Focus Areas", ""]
            for item in kpi_focuses[:kpi_limit]:
                lines.append(f"- {_clean_display_label(item)}")
            sections.append("\n".join(lines))

        if not sections:
            return base
        if doc_type in doc_types_with_curated_notes:
            return "\n\n".join(sections).strip()
        if not base:
            return "\n\n".join(sections)
        return base + "\n\n" + "\n\n".join(sections)

    @staticmethod
    def _build_performance_signals(
        strategic_doc: Dict[str, Any],
        strategic_summary: str,
        note_summary: str,
        raw_text: str,
    ) -> List[Dict[str, str]]:
        indicators = [
            item
            for item in strategic_doc.get("performance_indicators") or []
            if str(item.get("label") or "").strip()
        ]
        signals: List[Dict[str, str]] = []
        for item in indicators[:4]:
            snippet = str(item.get("evidence") or item.get("value") or strategic_summary or note_summary or raw_text).strip()[:320]
            signals.append(
                {
                    "headline": str(item.get("label") or "Performance indicator").strip(),
                    "snippet": snippet,
                    "signal_type": "performance_snapshot",
                    "urgency": "medium",
                    "actionable": True,
                    "suggested_action": "Review this organisation-level performance signal for trend and positioning implications",
                }
            )
        return signals

    @staticmethod
    def _compose_note_summary(note_summary: str, strategic_summary: str, doc_type: str) -> str:
        base = str(note_summary or "").strip()
        strategic = str(strategic_summary or "").strip()
        if doc_type in {"strategic_plan", "annual_report", "industry_report"}:
            return base or strategic
        if not base:
            return strategic
        if strategic and normalize_lookup(strategic) not in normalize_lookup(base):
            return " ".join([base, strategic]).strip()
        return base

    @staticmethod
    def _build_note_original_text(
        message: Dict[str, Any],
        email_triage: Dict[str, Any],
        attachments_processed: Sequence[str],
    ) -> str:
        actionable_body = str(email_triage.get("actionable_body_text") or "").strip()
        if actionable_body:
            trimmed_actionable = _trim_mailbox_body_text(actionable_body)
            if trimmed_actionable:
                return trimmed_actionable
        raw_text = str(message.get("raw_text") or "").strip()
        if raw_text and not attachments_processed:
            trimmed_raw = _trim_mailbox_body_text(raw_text)
            if trimmed_raw:
                return trimmed_raw
        return _compact_mailbox_provenance(message)

    @staticmethod
    def _build_note_document_meta(
        *,
        note_source_type: str,
        primary_org_name: str,
        note_title: str,
        clean_subject: str,
        attachment_names: Sequence[str],
        attachment_fingerprints: Sequence[str],
        note_content: str,
        source_url: str = "",
        source_label: str = "Mailbox attachment",
    ) -> Dict[str, Any]:
        doc_type = str(note_source_type or "").strip().lower()
        if doc_type not in {"annual_report", "strategic_plan", "org_chart"}:
            return {}
        period_label = derive_period_label(doc_type, note_title, clean_subject, *attachment_names)
        return build_document_meta(
            doc_type=doc_type,
            target_org_name=primary_org_name,
            title=note_title,
            period_label=period_label,
            published_at="",
            content_fingerprint=build_content_fingerprint(
                attachment_fingerprints=attachment_fingerprints,
                text=note_content,
                source_url=source_url,
            ),
            source_url=source_url,
            source_label=source_label,
        )

    @staticmethod
    def _compose_org_chart_summary(
        primary_org_name: str,
        output_data: Dict[str, Any],
        markdown_text: str = "",
    ) -> str:
        org_name = _clean_display_label(primary_org_name)
        leaders = IntelMailboxPoller._collect_org_chart_leaders(org_name, output_data)
        functions = IntelMailboxPoller._collect_org_chart_functions(org_name, output_data, markdown_text)

        if not org_name:
            org_name = "the organisation"
        if not leaders:
            if functions:
                return f"Org chart for {org_name} outlining functional areas across {', '.join(functions[:4])}."
            return f"Org chart for {org_name}."

        role_terms = _dedupe_preserve_order(
            role
            for role in (str(item.get("role") or "").strip() for item in leaders)
            if role
        )
        leader_names = ", ".join(item["name"] for item in leaders[:4])
        summary = f"Org chart for {org_name} identifying {len(leaders)} senior leaders"
        if leader_names:
            summary += f", including {leader_names}"
        summary += "."
        if role_terms:
            summary += " Key roles include " + ", ".join(role_terms[:6]) + "."
        return summary

    @staticmethod
    def _compose_org_chart_note(
        primary_org_name: str,
        output_data: Dict[str, Any],
        markdown_text: str = "",
    ) -> str:
        org_name = _clean_display_label(primary_org_name)
        summary = IntelMailboxPoller._compose_org_chart_summary(org_name, output_data, markdown_text)
        people = IntelMailboxPoller._collect_org_chart_leaders(org_name, output_data)
        functions = IntelMailboxPoller._collect_org_chart_functions(org_name, output_data, markdown_text)
        leaders: List[str] = []
        for item in people:
            name = _clean_display_label(str(item.get("name") or "").strip())
            role = _clean_display_role(str(item.get("role") or "").strip(), org_name)
            line = f"- {name}"
            if role:
                line += f" | {role}"
            if org_name:
                line += f" | {org_name}"
            leaders.append(line)

        sections = ["## Summary", "", summary]
        if leaders:
            sections.extend(["", "## Leadership Team", ""])
            sections.extend(leaders[:10])
        elif functions:
            sections.extend(["", "## Functional Structure", ""])
            sections.extend(f"- {item}" for item in functions[:10])
        return "\n".join(sections).strip()

    @staticmethod
    def _collect_org_chart_leaders(primary_org_name: str, output_data: Dict[str, Any]) -> List[Dict[str, str]]:
        org_name = _clean_display_label(primary_org_name)
        org_key = normalize_lookup(org_name)
        raw_candidates = list(output_data.get("people") or [])
        if not raw_candidates:
            raw_candidates = [
                item
                for item in list(output_data.get("entities") or [])
                if str(item.get("target_type") or "").strip().lower() == "person"
            ]

        leaders: List[Dict[str, str]] = []
        seen_people: set[str] = set()
        for item in raw_candidates:
            name = _clean_display_label(str(item.get("canonical_name") or item.get("name") or "").strip())
            role = _clean_display_role(str(item.get("current_role") or item.get("role") or "").strip(), org_name)
            employer = _clean_display_label(str(item.get("current_employer") or item.get("employer") or "").strip())
            evidence = normalize_lookup(str(item.get("evidence") or "").strip())
            if not name:
                continue
            if not _looks_like_person_label(name):
                continue
            if normalize_lookup(name) in seen_people:
                continue
            if employer and org_key and not orgs_compatible(employer, org_name):
                continue
            if any(token in evidence for token in ("sender signature", "contact details", "credentials provided")):
                continue
            seen_people.add(normalize_lookup(name))
            leaders.append({"name": name, "role": role})
        return leaders

    @staticmethod
    def _collect_org_chart_functions(
        primary_org_name: str,
        output_data: Dict[str, Any],
        markdown_text: str = "",
    ) -> List[str]:
        org_name = _clean_display_label(primary_org_name)
        excluded_org_labels = {
            normalize_lookup(label)
            for label in [org_name]
            if str(label or "").strip()
        }
        for item in list(output_data.get("organisations") or []) + list(output_data.get("entities") or []):
            if str(item.get("target_type") or "").strip().lower() not in {"", "organisation", "organization"}:
                continue
            label = _clean_display_label(str(item.get("canonical_name") or item.get("name") or "").strip())
            if label:
                excluded_org_labels.add(normalize_lookup(label))
        lines: List[str] = []
        for value in (
            markdown_text,
            str(output_data.get("summary") or ""),
            *[
                str(item.get("excerpt") or "")
                for item in (output_data.get("attachments") or [])
                if str(item.get("excerpt") or "").strip()
            ],
        ):
            for raw_line in str(value or "").splitlines():
                cleaned = _clean_display_label(raw_line)
                if cleaned:
                    lines.append(cleaned)

        functions: List[str] = []
        seen: set[str] = set()
        org_key = normalize_lookup(org_name)
        for line in lines:
            cleaned_line = _clean_org_chart_function_label(line, org_name)
            lowered = normalize_lookup(cleaned_line)
            if not lowered:
                continue
            if org_key and lowered == org_key:
                continue
            if lowered in excluded_org_labels:
                continue
            if any(marker in lowered for marker in _ORG_CHART_FUNCTION_STOPWORDS):
                continue
            if "http" in lowered or "@" in lowered:
                continue
            if len(cleaned_line) > 90:
                continue
            if any(char.isdigit() for char in cleaned_line):
                continue
            words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", cleaned_line) if word]
            if len(words) < 2 or len(words) > 10:
                continue
            looks_like_function = any(marker in lowered for marker in _ORG_CHART_FUNCTION_MARKERS) or any(
                lowered.startswith(prefix) for prefix in _ORG_CHART_FUNCTION_ROLE_PREFIXES
            )
            if not looks_like_function:
                continue
            normalized = lowered.strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            functions.append(cleaned_line)
        return functions

    @staticmethod
    def _derive_note_title(
        clean_subject: str,
        strategic_doc: Dict[str, Any],
        output_data: Dict[str, Any],
        primary: Dict[str, str],
    ) -> str:
        subject_match = re.match(
            r"^\s*(org(?:anisation|anization)?\s+chart|strategy|strategic\s+plan|annual\s+report)\s+for\s+(.+?)\s*$",
            clean_subject,
            flags=re.IGNORECASE,
        )
        if subject_match:
            doc_phrase = normalize_lookup(subject_match.group(1))
            subject_org = _clean_display_label(subject_match.group(2))
            primary_org = _clean_display_label(primary.get("name") or "")
            if primary_org and _org_hint_match_type(subject_org, primary_org) == "acronym":
                subject_org = primary_org
            if "chart" in doc_phrase:
                return _clean_display_label(f"{subject_org} org chart")
            if "annual report" in doc_phrase:
                return _clean_display_label(f"{subject_org} Annual Report")
            return _clean_display_label(f"{subject_org} Strategic Plan")
        doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
        org_name = str(strategic_doc.get("org_name") or primary.get("name") or "").strip()
        attachments = list(output_data.get("attachments") or [])
        attachment_names = [str(item.get("filename") or "").strip() for item in attachments if str(item.get("filename") or "").strip()]
        attachment_excerpts = [str(item.get("excerpt") or "").strip() for item in attachments if str(item.get("excerpt") or "").strip()]
        title_context = " ".join([clean_subject, *attachment_names, *attachment_excerpts])

        if doc_type in {"strategic_plan", "annual_report", "industry_report"} and org_name and (
            _looks_like_org_acronym(clean_subject) or normalize_lookup(clean_subject) == normalize_lookup(org_name)
        ):
            label = _document_label(doc_type, title_context)
            year_label = _extract_document_year_label(clean_subject, *attachment_names, *attachment_excerpts)
            parts = [org_name, label]
            if year_label:
                parts.append(year_label)
            return _clean_display_label(" ".join(parts))

        if _looks_like_useful_subject_title(clean_subject):
            return clean_subject

        if doc_type in {"strategic_plan", "annual_report", "industry_report"} and org_name:
            label = _document_label(doc_type, title_context)
            year_label = _extract_document_year_label(clean_subject, *attachment_names, *attachment_excerpts)
            parts = [org_name, label]
            if year_label:
                parts.append(year_label)
            return _clean_display_label(" ".join(parts))

        return str(clean_subject or org_name or primary.get("name") or "").strip()

    def _build_ingest_note_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        output_data: Dict[str, Any],
        signal: Dict[str, Any],
        markdown_text: str,
        message_kind: str,
        scope_org_name: str = "",
        routing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        routing = dict(routing or {})
        scope_org_name = str(scope_org_name or routing.get("effective_org_name") or self.config.org_name).strip()
        extraction_depth = str(routing.get("extraction_depth") or "default").strip().lower() or "default"
        processing_meta = dict(output_data.get("processing_meta") or {})
        email_triage = dict(processing_meta.get("email_triage") or {})
        clean_subject = str(
            routing.get("clean_subject")
            or email_triage.get("clean_subject")
            or _normalized_mail_subject(message.get("subject", ""))
        ).strip()
        subject_org_hint = str(routing.get("subject_org_hint") or "").strip()
        resolved_subject_org_hint = self._resolve_known_org_label(subject_org_hint, output_data, scope_org_name=scope_org_name)
        strategic_doc = dict(processing_meta.get("strategic_doc") or {})
        doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
        strategic_org_name = str(strategic_doc.get("org_name") or "").strip()
        if (
            resolved_subject_org_hint
            and (
                not strategic_org_name
                or _looks_like_low_signal_org_label(strategic_org_name)
                or (
                    subject_org_hint
                    and _looks_like_org_acronym(subject_org_hint)
                    and normalize_lookup(subject_org_hint) in normalize_lookup(strategic_org_name)
                )
                or _org_hint_match_type(resolved_subject_org_hint, strategic_org_name) in {"exact", "compatible", "acronym"}
            )
        ):
            strategic_doc["org_name"] = resolved_subject_org_hint
            for bucket_name in ("key_stakeholders", "leadership_people"):
                updated_items: List[Dict[str, Any]] = []
                for item in strategic_doc.get(bucket_name) or []:
                    updated = dict(item)
                    employer = str(updated.get("current_employer") or "").strip()
                    if employer and (
                        _looks_like_low_signal_org_label(employer)
                        or _org_hint_match_type(resolved_subject_org_hint, employer) in {"exact", "compatible", "acronym"}
                    ):
                        updated["current_employer"] = resolved_subject_org_hint
                    updated_items.append(updated)
                if updated_items:
                    strategic_doc[bucket_name] = updated_items
            strategic_org_name = resolved_subject_org_hint
        primary = self._choose_primary_entity(output_data)
        can_use_document_subject_hint = message_kind in {"document_analysis", "org_chart"} or bool(routing.get("has_org_chart_image_attachment"))
        subject_primary = self._choose_subject_primary_entity(resolved_subject_org_hint, output_data) if can_use_document_subject_hint else {"target_type": "", "name": "", "employer": ""}
        document_primary = self._choose_document_primary_entity(output_data, strategic_doc) if can_use_document_subject_hint else {"target_type": "", "name": "", "employer": ""}
        if subject_primary.get("name") and document_primary.get("name"):
            subject_key = normalize_lookup(subject_primary.get("name") or "")
            document_key = normalize_lookup(document_primary.get("name") or "")
            match_type = _org_hint_match_type(subject_primary.get("name") or "", document_primary.get("name") or "")
            if (
                subject_key and document_key
                and subject_key != document_key
                and (
                    subject_key in document_key
                    or match_type in {"compatible", "acronym"}
                    or orgs_compatible(subject_primary.get("name") or "", document_primary.get("name") or "")
                    or (subject_org_hint and not _looks_like_org_acronym(subject_org_hint))
                )
            ):
                document_primary = {"target_type": "", "name": "", "employer": ""}
        if message_kind == "org_chart" or bool(routing.get("has_org_chart_image_attachment")):
            if subject_primary.get("name"):
                primary = subject_primary
            elif document_primary.get("name"):
                primary = document_primary
        elif document_primary.get("name"):
            primary = document_primary
        elif subject_primary.get("name") and (
            not primary.get("name")
            or primary.get("target_type") != "organisation"
            or orgs_compatible(primary.get("name") or "", subject_primary.get("name") or "")
        ):
            primary = subject_primary
        elif strategic_org_name and not _looks_like_generic_document_name(strategic_org_name) and (
            not primary.get("name")
            or (
                can_use_document_subject_hint and primary.get("target_type") != "organisation"
            )
            or (
                primary.get("target_type") == "person"
                and normalize_lookup(primary.get("employer") or "") == normalize_lookup(scope_org_name)
            )
        ):
            primary = {
                "target_type": "organisation",
                "name": strategic_org_name,
                "employer": "",
            }
        entity_primary = self._choose_entity_primary_from_entities(output_data, scope_org_name=scope_org_name)
        if entity_primary.get("name") and (
            not primary.get("name")
            or (
                primary.get("target_type") == "organisation"
                and normalize_lookup(primary.get("name") or "") == normalize_lookup(scope_org_name)
                and entity_primary.get("target_type") == "organisation"
                and normalize_lookup(entity_primary.get("name") or "") != normalize_lookup(scope_org_name)
            )
        ):
            primary = entity_primary
        if (
            doc_type in {"strategic_plan", "annual_report", "industry_report"}
            and strategic_org_name
            and primary.get("target_type") == "organisation"
            and orgs_compatible(primary.get("name") or "", strategic_org_name)
        ):
            primary = {
                "target_type": "organisation",
                "name": strategic_org_name,
                "employer": "",
            }
        entities = list(output_data.get("entities") or [])
        primary_key = normalize_lookup(primary.get("name", ""))
        referenced_entities: List[Dict[str, Any]] = []
        seen_refs: set[tuple[str, str]] = set()

        for entity in entities:
            name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
            target_type = str(entity.get("target_type") or "").strip().lower() or "person"
            if not name:
                continue
            dedupe = (normalize_lookup(name), target_type)
            if dedupe in seen_refs or (target_type == primary.get("target_type") and normalize_lookup(name) == primary_key):
                continue
            seen_refs.add(dedupe)
            current_employer = _clean_display_label(str(entity.get("current_employer") or "").strip())
            current_role = _clean_display_role(str(entity.get("current_role") or "").strip(), current_employer)
            referenced_entities.append(
                {
                    "name": _clean_display_label(name),
                    "target_type": target_type,
                    "current_employer": current_employer,
                    "current_role": current_role,
                    "reference_type": self._infer_reference_type(message, entity),
                    "confidence": self._confidence_label(entity.get("confidence")),
                    "context": str(entity.get("evidence") or "").strip()[:240],
                }
            )
            industry_name = self._infer_industry_name(message, entity, scope_org_name=scope_org_name)
            if target_type == "organisation" and industry_name:
                industry_key = (normalize_lookup(industry_name), "industry")
                if industry_key not in seen_refs:
                    seen_refs.add(industry_key)
                    referenced_entities.append(
                        {
                            "name": industry_name,
                            "target_type": "industry",
                            "reference_type": "affiliation",
                            "confidence": self._confidence_label(entity.get("confidence")),
                        "context": f"{name} is associated with the {industry_name} sector",
                    }
                )

        primary_employer = str(primary.get("employer") or "").strip()
        if strategic_org_name and not _looks_like_generic_document_name(strategic_org_name):
            strategic_org_key = (normalize_lookup(strategic_org_name), "organisation")
            if strategic_org_key not in seen_refs and normalize_lookup(strategic_org_name) != primary_key:
                seen_refs.add(strategic_org_key)
                referenced_entities.append(
                    {
                        "name": strategic_org_name,
                        "target_type": "organisation",
                        "reference_type": "mention",
                        "confidence": "confirmed",
                        "context": "Document title/content indicates this organisation is the subject of the strategic document",
                    }
                )
        if primary.get("target_type") == "person" and primary_employer:
            employer_key = (normalize_lookup(primary_employer), "organisation")
            if employer_key not in seen_refs:
                seen_refs.add(employer_key)
                referenced_entities.append(
                    {
                        "name": primary_employer,
                        "target_type": "organisation",
                        "reference_type": "meeting",
                        "confidence": "confirmed",
                        "context": f"Primary contact is affiliated with {primary_employer}",
                    }
                )

        matched_names = []
        suggested_names = []
        for match in output_data.get("matches") or []:
            candidate_name = str(match.get("candidate_name") or "").strip()
            if candidate_name and match.get("matched"):
                matched_names.append(candidate_name)
            elif candidate_name:
                suggested_names.append(candidate_name)

        note_date = ""
        received_at = str(message.get("received_at") or "").strip()
        if received_at:
            note_date = received_at[:10]

        note_source_type = self._infer_note_source_type(message, output_data, message_kind)
        note_summary = str(output_data.get("summary") or "").strip()
        strategic_summary = str(strategic_doc.get("strategic_summary") or "").strip()
        strategic_signals = list(strategic_doc.get("strategic_signals") or [])
        note_title = self._derive_note_title(clean_subject, strategic_doc, output_data, primary)
        note_summary = self._compose_note_summary(note_summary, strategic_summary, doc_type)
        note_summary = _normalize_document_summary_org_label(note_summary, resolved_subject_org_hint, doc_type)
        llm_policy = dict(output_data.get("llm_policy") or {})
        fit_assessment = self._assess_subscriber_fit(
            scope_org_name,
            message,
            output_data,
            strategic_doc=strategic_doc,
        )
        attachments_processed = [
            str(item.get("filename") or "").strip()
            for item in output_data.get("attachments") or []
            if str(item.get("status") or "").strip().lower() == "processed"
            and str(item.get("filename") or "").strip()
        ]
        attachment_fingerprints = self._build_attachment_fingerprints(persisted, output_data)
        if note_source_type == "org_chart":
            org_chart_name = str(primary.get("name") or subject_org_hint or strategic_org_name or "").strip()
            note_summary = self._compose_org_chart_summary(org_chart_name, output_data, markdown_text)
            note_content = self._compose_org_chart_note(org_chart_name, output_data, markdown_text)
        else:
            note_content = self._enrich_note_markdown(
                markdown_text,
                strategic_doc,
                note_summary,
                extraction_depth=extraction_depth,
                fit_assessment=fit_assessment,
                scope_org_name=scope_org_name,
                local_only=bool(llm_policy.get("local_only")),
            )
        fit_commentary = self._fit_commentary(scope_org_name, fit_assessment, primary.get("name") or "")
        if fit_commentary:
            fit_section = f"## Subscriber Fit\n\n{fit_commentary}"
            note_content = f"{str(note_content or '').rstrip()}\n\n{fit_section}".strip()
        note_content = self._finalize_note_markdown(note_content, llm_policy)
        document_meta = self._build_note_document_meta(
            note_source_type=note_source_type,
            primary_org_name=str(primary.get("name") or resolved_subject_org_hint or strategic_org_name or "").strip(),
            note_title=note_title,
            clean_subject=clean_subject,
            attachment_names=attachments_processed,
            attachment_fingerprints=attachment_fingerprints,
            note_content=note_content,
            source_url="",
            source_label="Mailbox attachment",
        )
        signals = [
            {
                "headline": str(note_title or strategic_org_name or "Mailbox note").strip(),
                "snippet": (note_summary or str(message.get("raw_text") or "").strip())[:320],
                "signal_type": self._infer_signal_type(message, output_data),
                "urgency": "medium",
                "actionable": True if output_data.get("target_update_suggestions") or strategic_signals or strategic_doc.get("themes") or strategic_doc.get("initiatives") else False,
                "suggested_action": "Review referenced entities and follow up on any introductions or strategic leads",
            }
        ]
        for item in strategic_signals[:5]:
            signals.append(
                {
                    "headline": str(item.get("headline") or "Strategic signal").strip(),
                    "snippet": str(item.get("snippet") or strategic_summary or note_summary or str(message.get("raw_text") or "").strip())[:320],
                    "signal_type": "strategic_intent",
                    "urgency": "medium",
                    "actionable": True,
                    "suggested_action": "Review whether this strategic signal creates an engagement, policy, or positioning opportunity",
                }
            )
        if not strategic_signals:
            for theme in (strategic_doc.get("themes") or [])[:3]:
                signals.append(
                    {
                        "headline": f"Strategic theme: {theme}",
                        "snippet": (strategic_summary or note_summary or str(message.get("raw_text") or "").strip())[:320],
                        "signal_type": "strategic_intent",
                        "urgency": "medium",
                        "actionable": True,
                        "suggested_action": "Review whether this strategic theme creates an engagement or positioning opportunity",
                    }
                )
        signals.extend(
            self._build_performance_signals(
                strategic_doc,
                strategic_summary,
                note_summary,
                str(message.get("raw_text") or "").strip(),
            )
        )
        for item in (strategic_doc.get("major_projects") or [])[:4]:
            project_name = str(item.get("name") or "").strip()
            project_value = str(item.get("value") or "").strip()
            project_evidence = str(item.get("evidence") or strategic_summary or note_summary or "").strip()
            if not project_name:
                continue
            snippet = project_name
            if project_value:
                snippet += f": {project_value}"
            if project_evidence:
                snippet += f" | {project_evidence}"
            signals.append(
                {
                    "headline": f"Major project: {project_name}",
                    "snippet": snippet[:320],
                    "signal_type": "strategic_investment",
                    "urgency": "medium",
                    "actionable": True,
                    "suggested_action": "Review whether this capital or program investment creates a delivery, partnership, or transformation opportunity",
                }
            )
        relationship_paths = self.signal_store.find_relationship_paths(
            org_name=scope_org_name,
            target_names=[
                primary.get("name", ""),
                *[str(item.get("name") or "").strip() for item in referenced_entities if str(item.get("target_type") or "").strip().lower() == "person"],
            ],
            max_hops=4,
            limit=5,
        )

        return {
            "action": "ingest_intel_note",
            "secret": self.config.callback_secret,
            "org_name": scope_org_name,
            "mailbox_routing": {
                "default_org_name": str(routing.get("default_org_name") or self.config.org_name).strip(),
                "requested_org_name": str(routing.get("requested_org_name") or "").strip(),
                "matched_org_name": str(routing.get("matched_org_name") or "").strip(),
                "sender_domain_org_name": str(routing.get("sender_domain_org_name") or "").strip(),
                "effective_org_name": scope_org_name,
                "status": str(routing.get("status") or "default").strip(),
                "subject_org_hint": subject_org_hint,
                "extraction_depth": extraction_depth,
                "force_reingest": bool(routing.get("force_reingest")),
            },
            "note": {
                "source_type": note_source_type,
                "title": note_title,
                "content": note_content,
                "original_text": self._build_note_original_text(message, email_triage, attachments_processed),
                "submitted_by": self._effective_submitter_email(message),
                "note_date": note_date,
                "attachments_processed": attachments_processed,
                "attachment_fingerprints": attachment_fingerprints,
                "extraction_depth": extraction_depth,
            },
            "document_meta": document_meta,
            "primary_entity": {
                "name": primary.get("name", ""),
                "target_type": primary.get("target_type", ""),
                "current_employer": primary_employer,
                "current_role": "",
                "linkedin_url": "",
                "tags": ["existing-relationship"] if matched_names else [],
            },
            "referenced_entities": referenced_entities,
            "urls": self._extract_note_urls(message, note_content),
            "signals": signals,
            "fit_assessment": fit_assessment,
            "graph_enrichment": {
                "existing_connections": sorted(set(matched_names)),
                "new_profiles_suggested": sorted(set(suggested_names)),
                "relationship_paths": relationship_paths,
            },
        }

    def _build_attachment_fingerprints(
        self,
        persisted: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> List[str]:
        processed_names = {
            str(item.get("filename") or "").strip()
            for item in output_data.get("attachments") or []
            if str(item.get("status") or "").strip().lower() == "processed"
            and str(item.get("filename") or "").strip()
        }
        fingerprints: List[str] = []
        for attachment in persisted.get("attachments") or []:
            filename = str(attachment.get("filename") or "").strip()
            if processed_names and filename not in processed_names:
                continue
            stored_path = str(attachment.get("stored_path") or "").strip()
            digest = ""
            if stored_path and Path(stored_path).exists():
                try:
                    digest = hashlib.sha1(Path(stored_path).read_bytes()).hexdigest()
                except Exception:
                    digest = ""
            if not digest:
                fallback = "|".join(
                    [
                        filename,
                        str(attachment.get("mime_type") or "").strip(),
                        str(attachment.get("size_bytes") or "").strip(),
                    ]
                ).strip()
                if fallback:
                    digest = hashlib.sha1(fallback.encode("utf-8", "ignore")).hexdigest()
            if digest:
                fingerprints.append(f"{filename}:{digest}")
        return sorted(set(fingerprints))

    def _build_duplicate_delivery(
        self,
        duplicate: Dict[str, Any],
        note_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        intel_id = str(duplicate.get("existing_intel_id") or "").strip()
        if not intel_id:
            intel_id = str(((duplicate.get("delivery") or {}).get("response") or {}).get("intel_id") or "").strip()
        return {
            "status": "duplicate_local",
            "duplicate_of_trace_id": str(duplicate.get("trace_id") or "").strip(),
            "similarity": float(duplicate.get("similarity") or 0.0),
            "response": {
                "status": "duplicate",
                "intel_id": intel_id,
                "primary_entity": ((note_payload.get("primary_entity") or {}).get("name") or "").strip(),
            },
        }

    def _build_document_duplicate_delivery(
        self,
        document_meta: Dict[str, Any],
        note_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        existing = dict(document_meta.get("existing_record") or {})
        intel_id = str(existing.get("latest_intel_id") or "").strip()
        return {
            "status": "duplicate_document",
            "document_status": str(document_meta.get("status") or "").strip(),
            "response": {
                "status": "duplicate_document",
                "intel_id": intel_id,
                "primary_entity": ((note_payload.get("primary_entity") or {}).get("name") or "").strip(),
            },
        }

    def _ingest_signal(
        self,
        message: Dict[str, Any],
        output_data: Dict[str, Any],
        trace_id: str,
        scope_org_name: str = "",
        clean_subject: str = "",
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        primary = self._choose_primary_entity(output_data)
        if not primary.get("name"):
            primary = self._choose_entity_primary_from_entities(output_data, scope_org_name=scope_org_name)
        payload = {
            "org_name": scope_org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "signal_type": "email_intel",
            "submitted_by": self._effective_submitter_email(message),
            "message_id": message.get("message_id", ""),
            "received_at": message.get("received_at", ""),
            "subject": clean_subject or message.get("subject", ""),
            "raw_text": message.get("raw_text", ""),
            "primary_url": "",
            "text_note": "",
            "parsed_candidate_name": primary["name"],
            "parsed_candidate_employer": primary["employer"],
            "target_type": primary["target_type"],
            "notification_kind": "mailbox_auto_extract",
            "tags": ["email_intake"],
        }
        return self.signal_store.ingest_signal(payload)

    def _build_result_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        trace_id: str,
        output_data: Dict[str, Any],
        signal: Dict[str, Any],
        scope_org_name: str = "",
        routing: Optional[Dict[str, Any]] = None,
        fit_assessment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        routing = dict(routing or {})
        output_entities = list(output_data.get("entities") or [])
        if not output_entities:
            for candidate in output_data.get("people") or []:
                output_entities.append(
                    {
                        "name": str(candidate.get("canonical_name") or candidate.get("name") or "").strip(),
                        "canonical_name": str(candidate.get("canonical_name") or candidate.get("name") or "").strip(),
                        "target_type": "person",
                        "current_employer": str(candidate.get("current_employer") or candidate.get("employer") or "").strip(),
                        "current_role": str(candidate.get("current_role") or candidate.get("title") or "").strip(),
                        "email": str(candidate.get("email") or "").strip(),
                        "linkedin_url": str(candidate.get("linkedin_url") or "").strip(),
                        "confidence": candidate.get("confidence"),
                        "evidence": candidate.get("evidence", ""),
                    }
                )
            for candidate in output_data.get("organisations") or []:
                output_entities.append(
                    {
                        "name": str(candidate.get("canonical_name") or candidate.get("name") or "").strip(),
                        "canonical_name": str(candidate.get("canonical_name") or candidate.get("name") or "").strip(),
                        "target_type": "organisation",
                        "website_url": str(candidate.get("website_url") or "").strip(),
                        "industry": str(candidate.get("industry") or "").strip(),
                        "parent_entity": str(candidate.get("parent_entity") or "").strip(),
                        "confidence": candidate.get("confidence"),
                        "evidence": candidate.get("evidence", ""),
                    }
                )
        output_suggestions = output_data.get("target_update_suggestions") or []
        output_warnings = output_data.get("warnings") or []
        fit_payload = dict(fit_assessment or {})
        processing_meta = dict(output_data.get("processing_meta") or {})
        return {
            "result_type": "intel_extract_result",
            "org_name": scope_org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "message_id": message.get("message_id", ""),
            "intel_id": f"mail_{persisted.get('message_key', '')}",
            "raw_summary": str(output_data.get("summary") or ""),
            "entities": output_entities,
            "target_update_suggestions": output_suggestions,
            "suggested_targets": [],
            "warnings": output_warnings,
            "fit_assessment": fit_payload,
            "processing_meta": processing_meta,
            "mailbox_message": {
                "subject": message.get("subject", ""),
                "from_email": message.get("from_email", ""),
                "from_name": message.get("from_name", ""),
                "received_at": message.get("received_at", ""),
                "raw_path": persisted.get("raw_path", ""),
                "attachments": persisted.get("attachments") or [],
            },
            "signal": {
                "signal_id": signal.get("signal_id", ""),
                "matched_profile_keys": signal.get("matched_profile_keys") or [],
                "needs_review": bool(signal.get("needs_review")),
            },
            "llm_policy": dict(output_data.get("llm_policy") or {}),
            "mailbox_routing": {
                "default_org_name": str(routing.get("default_org_name") or self.config.org_name).strip(),
                "requested_org_name": str(routing.get("requested_org_name") or "").strip(),
                "matched_org_name": str(routing.get("matched_org_name") or "").strip(),
                "sender_domain_org_name": str(routing.get("sender_domain_org_name") or "").strip(),
                "effective_org_name": scope_org_name,
                "status": str(routing.get("status") or "default").strip(),
                "subject_org_hint": str(routing.get("subject_org_hint") or "").strip(),
                "extraction_depth": str(routing.get("extraction_depth") or "default").strip().lower() or "default",
                "force_reingest": bool(routing.get("force_reingest")),
            },
            "output_data": output_data,
        }

    @staticmethod
    def _build_website_payload(delivery_payload: Dict[str, Any], output_data: Dict[str, Any]) -> Dict[str, Any]:
        website_payload = dict(delivery_payload or {})
        website_payload["secret"] = "[redacted]"
        website_payload["processing_meta"] = dict(output_data.get("processing_meta") or {})
        return website_payload

    def _send_reply(
        self,
        message: Dict[str, Any],
        subject: str,
        body: str,
    ) -> Dict[str, Any]:
        recipient = self._reply_recipient(message)
        if not recipient:
            return {"status": "skipped", "reason": "self_recipient"}
        try:
            return self.reply_client.send(
                to_email=recipient,
                subject=subject,
                body=body,
                in_reply_to=message.get("message_id", ""),
            )
        except Exception as exc:
            logger.warning("Intel mailbox reply send failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

    @staticmethod
    def _reply_excerpt(value: str, limit: int = 3000) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > limit:
            text = text[: limit - 3].rstrip() + "..."
        return text

    def _build_intel_reply(
        self,
        message: Dict[str, Any],
        delivery_payload: Dict[str, Any],
        routing: Dict[str, Any],
    ) -> tuple[str, str]:
        note = dict(delivery_payload.get("note") or {})
        primary = dict(delivery_payload.get("primary_entity") or {})
        fit = dict(delivery_payload.get("fit_assessment") or {})
        title = str(
            note.get("title")
            or delivery_payload.get("raw_summary")
            or ((delivery_payload.get("output_data") or {}).get("summary") or "")
            or "Extracted intelligence"
        ).strip()
        content = self._reply_excerpt(str(note.get("content") or "").strip())
        scope_org = str(delivery_payload.get("org_name") or routing.get("effective_org_name") or self.config.org_name).strip()
        primary_name = str(primary.get("name") or "").strip()
        if not primary_name:
            for item in (delivery_payload.get("entities") or []):
                candidate_name = str(item.get("canonical_name") or item.get("name") or "").strip()
                if candidate_name:
                    primary_name = candidate_name
                    break
        extraction_depth = str((routing or {}).get("extraction_depth") or note.get("extraction_depth") or "default").strip().lower() or "default"
        fit_label = str(fit.get("fit_label") or "").strip()
        reply_limit = 1200 if extraction_depth == "brief" else 3000 if extraction_depth == "default" else 12000
        content = self._reply_excerpt(str(note.get("content") or "").strip(), limit=reply_limit)
        if not content:
            fallback_summary = str(
                delivery_payload.get("raw_summary")
                or ((delivery_payload.get("output_data") or {}).get("summary") or "")
            ).strip()
            if fallback_summary:
                content = self._reply_excerpt(fallback_summary, limit=reply_limit)
            else:
                entity_names = [
                    str(item.get("canonical_name") or item.get("name") or "").strip()
                    for item in (delivery_payload.get("entities") or [])
                    if str(item.get("canonical_name") or item.get("name") or "").strip()
                ]
                if entity_names:
                    content = f"Entities extracted: {', '.join(entity_names[:8])}"

        subject = f"Re: {str(message.get('subject') or '').strip() or title}"
        lines = [
            f"Cortex processed your submission for {scope_org}.",
            "",
            f"Depth: {extraction_depth}",
        ]
        if primary_name:
            lines.append(f"Primary entity: {primary_name}")
        if fit_label:
            lines.append(f"Subscriber fit: {fit_label}")
        lines.extend(
            [
                "",
                f"Title: {title}",
                "",
            ]
        )
        if content:
            lines.append(content)
        else:
            lines.append("No note content was generated.")
        return subject, "\n".join(lines).strip()

    def _build_csv_result_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        trace_id: str,
        csv_result: Dict[str, Any],
        reply_delivery: Dict[str, Any],
        scope_org_name: str = "",
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        return {
            "result_type": "csv_profile_import_result",
            "org_name": scope_org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "message_id": message.get("message_id", ""),
            "mailbox_message": {
                "subject": message.get("subject", ""),
                "from_email": message.get("from_email", ""),
                "from_name": message.get("from_name", ""),
                "received_at": message.get("received_at", ""),
                "raw_path": persisted.get("raw_path", ""),
                "attachments": persisted.get("attachments") or [],
            },
            "csv_import": {
                "filename": csv_result.get("filename", ""),
                "row_count": int(csv_result.get("row_count") or 0),
                "dry_run": bool(csv_result.get("dry_run")),
                "created": int(csv_result.get("created") or 0),
                "updated": int(csv_result.get("updated") or 0),
                "skipped": int(csv_result.get("skipped") or 0),
                "errors": list(csv_result.get("errors") or []),
                "api_result": dict(csv_result.get("api_result") or {}),
            },
            "reply": {
                "subject": csv_result.get("reply_subject", ""),
                "body": csv_result.get("reply_body", ""),
                "delivery": reply_delivery,
            },
        }

    def _build_csv_failure_payload(
        self,
        message: Dict[str, Any],
        persisted: Dict[str, Any],
        trace_id: str,
        error_message: str,
        reply_subject: str,
        reply_body: str,
        reply_delivery: Dict[str, Any],
        scope_org_name: str = "",
    ) -> Dict[str, Any]:
        scope_org_name = str(scope_org_name or self.config.org_name).strip()
        return {
            "result_type": "csv_profile_import_result",
            "org_name": scope_org_name,
            "trace_id": trace_id,
            "source_system": self.config.source_system,
            "message_id": message.get("message_id", ""),
            "mailbox_message": {
                "subject": message.get("subject", ""),
                "from_email": message.get("from_email", ""),
                "from_name": message.get("from_name", ""),
                "received_at": message.get("received_at", ""),
                "raw_path": persisted.get("raw_path", ""),
                "attachments": persisted.get("attachments") or [],
            },
            "csv_import": {
                "status": "failed",
                "error": error_message,
            },
            "reply": {
                "subject": reply_subject,
                "body": reply_body,
                "delivery": reply_delivery,
            },
        }

    def _process_message(self, raw_bytes: bytes) -> Optional[Dict[str, Any]]:
        message = parse_email_bytes(raw_bytes)
        suppression_reason = self._mailbox_suppression_reason(message)
        if suppression_reason:
            logger.info(
                "Skipping intel mailbox message suppressed as %s: from=%s subject=%s",
                suppression_reason,
                message.get("from_email", ""),
                message.get("subject", ""),
            )
            return None
        if not self._allowed_sender(message.get("from_email", "")):
            logger.info("Skipping intel mailbox message from unapproved sender: %s", message.get("from_email", ""))
            return None

        persisted = self.store.persist_message(message, raw_bytes, message.get("attachments") or [])
        message_id = str(message.get("message_id") or "").strip()
        if message_id and self.store.has_processed_message(message_id):
            return None

        trace_seed = f"{message_id}|{message.get('subject','')}|{message.get('received_at','')}"
        trace_id = f"trace-{hashlib.sha1(trace_seed.encode('utf-8', 'ignore')).hexdigest()[:32]}"

        message_kind = classify_mailbox_message(message, persisted)
        routing = self._resolve_message_routing(message, persisted)
        effective_org_name = str(routing.get("effective_org_name") or self.config.org_name).strip()
        clean_subject = str(routing.get("clean_subject") or _normalized_mail_subject(message.get("subject", ""))).strip()

        if message_kind == "csv_profile_import":
            try:
                csv_result = self.csv_importer.process_message(message, persisted, effective_org_name)
                reply_delivery = self._send_reply(message, csv_result["reply_subject"], csv_result["reply_body"])
                result_payload = self._build_csv_result_payload(
                    message,
                    persisted,
                    trace_id,
                    csv_result,
                    reply_delivery,
                    scope_org_name=effective_org_name,
                )
                result_payload["mailbox_routing"] = {
                    "default_org_name": self.config.org_name,
                    "requested_org_name": str(routing.get("requested_org_name") or "").strip(),
                    "matched_org_name": str(routing.get("matched_org_name") or "").strip(),
                    "effective_org_name": effective_org_name,
                    "status": str(routing.get("status") or "default").strip(),
                    "subject_org_hint": str(routing.get("subject_org_hint") or "").strip(),
                }
                delivery = {
                    "status": "processed",
                    "created": int(csv_result.get("created") or 0),
                    "updated": int(csv_result.get("updated") or 0),
                    "skipped": int(csv_result.get("skipped") or 0),
                    "dry_run": bool(csv_result.get("dry_run")),
                    "reply": reply_delivery,
                }
                self.store.record_processed(persisted["message_key"], trace_id, result_payload, delivery)
                return {
                    "message_id": message.get("message_id", ""),
                    "trace_id": trace_id,
                    "signal_id": "",
                    "delivery": delivery,
                    "entity_count": 0,
                    "entity_names": [],
                    "update_suggestion_count": 0,
                    "warning_count": len(csv_result.get("errors") or []),
                    "warnings": list(csv_result.get("errors") or []),
                }
            except CsvProfileImportError as exc:
                reply_subject = self.csv_importer._reply_subject(message.get("subject", ""), dry_run="dry run" in str(message.get("subject", "")).lower(), ok=False)
                reply_body = str(exc)
                reply_delivery = self._send_reply(message, reply_subject, reply_body)
                result_payload = self._build_csv_failure_payload(
                    message,
                    persisted,
                    trace_id,
                    str(exc),
                    reply_subject,
                    reply_body,
                    reply_delivery,
                    scope_org_name=effective_org_name,
                )
                result_payload["mailbox_routing"] = {
                    "default_org_name": self.config.org_name,
                    "requested_org_name": str(routing.get("requested_org_name") or "").strip(),
                    "matched_org_name": str(routing.get("matched_org_name") or "").strip(),
                    "effective_org_name": effective_org_name,
                    "status": str(routing.get("status") or "default").strip(),
                    "subject_org_hint": str(routing.get("subject_org_hint") or "").strip(),
                }
                delivery = {"status": "failed", "error": str(exc), "reply": reply_delivery}
                self.store.record_processed(persisted["message_key"], trace_id, result_payload, delivery)
                return {
                    "message_id": message.get("message_id", ""),
                    "trace_id": trace_id,
                    "signal_id": "",
                    "delivery": delivery,
                    "entity_count": 0,
                    "entity_names": [],
                    "update_suggestion_count": 0,
                    "warning_count": 1,
                    "warnings": [str(exc)],
                }

        payload = self._build_extract_payload(message, persisted, trace_id, routing=routing)
        output_data, _output_file, processing_meta = self.note_processor.process(payload, message_kind)
        output_data["processing_meta"] = processing_meta
        markdown_text = ""
        if _output_file and Path(_output_file).exists():
            markdown_text = Path(_output_file).read_text(encoding="utf-8", errors="ignore")
        reply_delivery: Dict[str, Any] = {"status": "skipped"}
        reply_subject = ""
        reply_body = ""
        if message_kind == "intel_extract":
            signal = self._ingest_signal(message, output_data, trace_id, scope_org_name=effective_org_name, clean_subject=clean_subject)
            result_payload = self._build_result_payload(
                message,
                persisted,
                trace_id,
                output_data,
                signal,
                scope_org_name=effective_org_name,
                routing=routing,
            )
            delivery = self.result_client.deliver(persisted["message_key"], result_payload)
            reply_subject, reply_body = self._build_intel_reply(message, result_payload, routing)
            reply_delivery = self._send_reply(message, reply_subject, reply_body)
        else:
            delivery_payload = self._build_ingest_note_payload(
                message,
                persisted,
                output_data,
                {},
                markdown_text,
                message_kind,
                scope_org_name=effective_org_name,
                routing=routing,
            )
            document_meta = dict(delivery_payload.get("document_meta") or {})
            if document_meta:
                document_meta = self.signal_store.classify_document_meta(effective_org_name, document_meta)
                delivery_payload["document_meta"] = {
                    key: value for key, value in document_meta.items() if key != "existing_record"
                }
            strict_duplicate = bool(
                document_meta
                and str(document_meta.get("doc_type") or "").strip().lower() in STRICT_DOC_TYPES
                and str(document_meta.get("status") or "").strip().lower() == "known_same"
            )
            if strict_duplicate and bool(routing.get("force_reingest")):
                strict_duplicate = False
                document_meta["status"] = "changed_document"
                document_meta["ingest_recommendation"] = "ingest"
                delivery_payload["document_meta"] = {
                    key: value for key, value in document_meta.items() if key != "existing_record"
                }
            if strict_duplicate:
                signal = {}
                result_payload = self._build_result_payload(
                    message,
                    persisted,
                    trace_id,
                    output_data,
                    signal,
                    scope_org_name=effective_org_name,
                    routing=routing,
                    fit_assessment=delivery_payload.get("fit_assessment") or {},
                )
                result_payload["website_payload"] = self._build_website_payload(delivery_payload, output_data)
                result_payload["document_meta"] = delivery_payload.get("document_meta") or {}
                delivery = self._build_document_duplicate_delivery(document_meta, delivery_payload)
                self.signal_store.register_document_meta(
                    effective_org_name,
                    delivery_payload.get("document_meta") or {},
                    latest_trace_id=trace_id,
                    latest_intel_id=str(((document_meta.get("existing_record") or {}).get("latest_intel_id") or "")).strip(),
                )
            else:
                signal = self._ingest_signal(message, output_data, trace_id, scope_org_name=effective_org_name, clean_subject=clean_subject)
                result_payload = self._build_result_payload(
                    message,
                    persisted,
                    trace_id,
                    output_data,
                    signal,
                    scope_org_name=effective_org_name,
                    routing=routing,
                    fit_assessment=delivery_payload.get("fit_assessment") or {},
                )
                result_payload["website_payload"] = self._build_website_payload(delivery_payload, output_data)
                result_payload["document_meta"] = delivery_payload.get("document_meta") or {}
                delivery = self.result_client.deliver(
                    persisted["message_key"],
                    result_payload,
                    delivery_payload=delivery_payload,
                    callback_url_override=self.config.note_callback_url or self.config.callback_url,
                )
                response = dict(delivery.get("response") or {})
                if response.get("intel_id"):
                    reconciliation = self.signal_store.reconcile_intel_note_delivery(
                        org_name=effective_org_name,
                        trace_id=trace_id,
                        payload=delivery_payload,
                        response=response,
                    )
                    result_payload["graph_reconciliation"] = reconciliation
                if delivery_payload.get("document_meta"):
                    self.signal_store.register_document_meta(
                        effective_org_name,
                        delivery_payload.get("document_meta") or {},
                        latest_trace_id=trace_id,
                        latest_intel_id=str(response.get("intel_id") or "").strip(),
                    )
            reply_subject, reply_body = self._build_intel_reply(message, delivery_payload, routing)
            reply_delivery = self._send_reply(message, reply_subject, reply_body)
        result_payload["reply"] = {
            "subject": reply_subject,
            "body": reply_body,
            "delivery": reply_delivery,
        }
        if isinstance(delivery, dict):
            delivery["reply"] = reply_delivery
        self.store.record_processed(persisted["message_key"], trace_id, result_payload, delivery)
        entity_names = [
            str(item.get("canonical_name") or item.get("name") or "").strip()
            for item in (result_payload.get("entities") or [])
            if str(item.get("canonical_name") or item.get("name") or "").strip()
        ]
        return {
            "message_id": message.get("message_id", ""),
            "trace_id": trace_id,
            "signal_id": signal.get("signal_id", ""),
            "delivery": delivery,
            "entity_count": int(output_data.get("entity_count") or 0),
            "entity_names": entity_names,
            "update_suggestion_count": len(output_data.get("target_update_suggestions") or []),
            "warning_count": len(output_data.get("warnings") or []),
            "warnings": list(output_data.get("warnings") or []),
        }

    def poll_once(self) -> Dict[str, Any]:
        self.config.validate()
        processed = 0
        skipped = 0
        failures = 0
        results: List[Dict[str, Any]] = []

        client = self.imap_factory(self.config.host, self.config.port)
        try:
            client.login(self.config.username, self.config.password)
            client.select(self.config.folder)
            status, data = client.search(None, self.config.search_criteria)
            if status != "OK":
                raise RuntimeError(f"IMAP search failed: {status}")
            message_ids = [item for item in (data[0] or b"").split() if item][: self.config.poll_limit]

            for imap_id in message_ids:
                status, parts = client.fetch(imap_id, "(BODY.PEEK[])")
                if status != "OK":
                    failures += 1
                    continue
                raw_bytes = b""
                for part in parts or []:
                    if isinstance(part, tuple) and len(part) > 1 and isinstance(part[1], (bytes, bytearray)):
                        raw_bytes = bytes(part[1])
                        break
                if not raw_bytes:
                    failures += 1
                    continue
                parsed = parse_email_bytes(raw_bytes)
                suppression_reason = self._mailbox_suppression_reason(parsed)
                if suppression_reason:
                    skipped += 1
                    logger.info(
                        "Skipping intel mailbox message suppressed as %s: from=%s subject=%s",
                        suppression_reason,
                        parsed.get("from_email", ""),
                        parsed.get("subject", ""),
                    )
                    if self.config.mark_seen_on_success:
                        self._mark_imap_seen(client, imap_id)
                    continue
                if parsed.get("message_id") and self.store.has_processed_message(parsed["message_id"]):
                    skipped += 1
                    if self.config.mark_seen_on_success:
                        self._mark_imap_seen(client, imap_id)
                    continue
                try:
                    result = self._process_message(raw_bytes)
                    if result:
                        processed += 1
                        results.append(result)
                        if self.config.mark_seen_on_success:
                            self._mark_imap_seen(client, imap_id)
                    else:
                        skipped += 1
                except Exception as exc:
                    failures += 1
                    logger.exception("Intel mailbox message processing failed")
                    try:
                        message_id = parsed.get("message_id") or ""
                        message_key = _derive_message_key(message_id, raw_bytes)
                        self.store.record_failure(message_key, str(exc))
                    except Exception:
                        pass
        finally:
            try:
                client.logout()
            except Exception:
                pass

        return {
            "processed": processed,
            "skipped": skipped,
            "failures": failures,
            "results": results,
        }
