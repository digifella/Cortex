from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from cortex_engine.config_manager import ConfigManager
from cortex_engine.graph_manager import EnhancedGraphManager
from cortex_engine.stakeholder_signal_matcher import match_signal_to_profiles, normalize_lookup
from cortex_engine.target_update_detector import detect_profile_change_artifacts
from cortex_engine.utils import convert_windows_to_wsl_path

_ORG_STOPWORDS = {
    "pty",
    "ltd",
    "limited",
    "llc",
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "group",
    "consulting",
    "consultancy",
    "services",
    "service",
    "holdings",
    "the",
    "and",
}
_DEFAULT_OLLAMA_WATCH_MODEL = "qwen3.5:9b"
_OLLAMA_WATCH_MODEL_FALLBACKS = (
    "qwen3.5:9b",
    "qwen3.5:9b-q8_0",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:14b",
    "mistral-small3.2:latest",
    "mistral:latest",
)
_DEFAULT_OLLAMA_WATCH_TIMEOUT_SECONDS = 300
_LOW_VALUE_PERSON_SIGNAL_DOMAINS = {
    "imdb.com",
    "www.imdb.com",
    "ballotpedia.org",
    "www.ballotpedia.org",
    "contactout.com",
    "www.contactout.com",
    "rocketreach.co",
    "www.rocketreach.co",
    "zoominfo.com",
    "www.zoominfo.com",
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "medium.com",
    "www.medium.com",
}
_GRAPH_EDGE_BASE_WEIGHTS = {
    "linkedin_connection": 0.95,
    "works_at": 0.9,
    "belongs_to_industry": 0.86,
    "strategic_focus": 0.84,
    "tracked_by": 0.88,
    "affiliated_with": 0.8,
    "alumni_of": 0.75,
    "org_alumni_context": 0.72,
    "strategic_theme": 0.68,
    "co_mentioned": 0.6,
    "mentions_profile": 0.55,
    "mentions_organization": 0.5,
    "published_by": 0.35,
}

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_EMPTY_DIGEST_SECTION_HEADINGS = {
    "alumni context",
    "weak linkage context",
}

_EMPTY_DIGEST_SECTION_PATTERNS = (
    "no relevant alumni context",
    "no alumni context",
    "no alumni-related",
    "no alumni or weak linkage",
    "no weak linkage context",
    "no weak linkage",
    "no such data exists",
    "not provided for the organization",
    "not provided for the organisation",
    "none provided",
)


def _strip_empty_digest_sections(markdown: str) -> str:
    text = str(markdown or "").strip()
    if not text:
        return text

    lines = text.splitlines()
    kept: List[str] = []
    current_heading: Optional[str] = None
    current_section: List[str] = []

    def _flush_section() -> None:
        nonlocal current_heading, current_section
        if current_heading is None:
            if current_section:
                kept.extend(current_section)
            current_section = []
            return

        heading_key = normalize_lookup(current_heading).replace("-", " ").strip()
        section_text = "\n".join(current_section).strip().lower()
        should_drop = (
            heading_key in _EMPTY_DIGEST_SECTION_HEADINGS
            and any(pattern in section_text for pattern in _EMPTY_DIGEST_SECTION_PATTERNS)
            and "http://" not in section_text
            and "https://" not in section_text
            and "sig_" not in section_text
        )
        if not should_drop:
            kept.extend(current_section)
        current_heading = None
        current_section = []

    heading_re = re.compile(r"^(#{2,6})\s+(.*\S)\s*$")
    for line in lines:
        match = heading_re.match(line)
        if match:
            _flush_section()
            current_heading = match.group(2).strip()
            current_section = [line]
        else:
            current_section.append(line)
    _flush_section()

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _normalize_address(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key in ("street", "city", "state", "postcode", "country"):
        item = str(value.get(key) or "").strip()
        if item:
            normalized[key] = item
    return normalized


def _normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"active", "inactive", "watch", "former"}:
        return text
    return "active"


def _normalize_watch_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"off", "watch"}:
        return text
    return "off"


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


def _coerce_primary_flag(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "true", "yes", "on"} else 0


def _normalize_affiliations(
    raw_affiliations: Any,
    current_employer: str = "",
    current_role: str = "",
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in raw_affiliations or []:
        if not isinstance(item, dict):
            continue
        org_name_text = str(item.get("org_name_text") or item.get("org_name") or "").strip()
        if not org_name_text:
            continue
        role = str(item.get("role") or "").strip()
        affiliation_type = _normalize_affiliation_type(item.get("affiliation_type") or item.get("type"))
        dedupe_key = (
            normalize_lookup(org_name_text),
            normalize_lookup(role),
            affiliation_type,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(
            {
                "org_name_text": org_name_text,
                "role": role,
                "affiliation_type": affiliation_type,
                "confidence": _normalize_affiliation_confidence(item.get("confidence")),
                "is_primary": _coerce_primary_flag(item.get("is_primary")),
                "start_date": str(item.get("start_date") or "").strip(),
                "end_date": str(item.get("end_date") or "").strip(),
                "source": str(item.get("source") or "").strip(),
            }
        )

    fallback_employer = str(current_employer or "").strip()
    if not normalized and fallback_employer:
        normalized.append(
            {
                "org_name_text": fallback_employer,
                "role": str(current_role or "").strip(),
                "affiliation_type": "current",
                "confidence": "confirmed",
                "is_primary": 1,
                "start_date": "",
                "end_date": "",
                "source": "",
            }
        )

    if normalized and not any(item.get("is_primary") for item in normalized):
        normalized[0]["is_primary"] = 1

    primary_seen = False
    for item in normalized:
        if item.get("is_primary") and not primary_seen:
            primary_seen = True
            item["is_primary"] = 1
        else:
            item["is_primary"] = 0

    return normalized


def _primary_affiliation(affiliations: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in affiliations:
        if item.get("is_primary"):
            return item
    return affiliations[0] if affiliations else {}


def _employers_from_affiliations(affiliations: List[Dict[str, Any]]) -> List[str]:
    employers: List[str] = []
    seen: set[str] = set()
    for item in affiliations:
        name = str(item.get("org_name_text") or "").strip()
        normalized = normalize_lookup(name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        employers.append(name)
    return employers


def _normalize_org_name_list(values: Any) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        key = normalize_lookup(text)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _normalize_profile_alumni(values: Any) -> List[str]:
    return _normalize_org_name_list(values)


def _normalize_linkedin_connections(values: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in values or []:
        if not isinstance(item, dict):
            continue
        member = str(item.get("member") or item.get("email") or "").strip()
        degree = str(item.get("degree") or "").strip()
        if not member:
            continue
        key = (member.lower(), degree.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"member": member, "degree": degree})
    return normalized


def _normalize_industry_affiliations(values: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in values or []:
        if not isinstance(item, dict):
            continue
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
        dedupe_key = (industry_profile_key.lower(), normalize_lookup(industry_name), role.lower())
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


def _normalize_strategic_profile(values: Any) -> Dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    return {
        "description": str(values.get("description") or "").strip(),
        "industries": _normalize_org_name_list(values.get("industries") or []),
        "priority_industries": _normalize_org_name_list(values.get("priority_industries") or []),
        "key_themes": _normalize_org_name_list(values.get("key_themes") or []),
        "strategic_objectives": _normalize_org_name_list(values.get("strategic_objectives") or []),
        "updated_at": str(values.get("updated_at") or "").strip(),
    }


def _signal_visible_to_org(signal: Dict[str, Any], org_name: str) -> bool:
    if not org_name:
        return True
    if orgs_compatible(signal.get("org_name", ""), org_name):
        return True
    visible_to = _normalize_org_name_list(signal.get("visible_to_orgs") or [])
    shared_with = _normalize_org_name_list(signal.get("shared_with_orgs") or [])
    candidates = [*visible_to, *shared_with]
    return any(orgs_compatible(candidate, org_name) for candidate in candidates)


def _parse_timestamp(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for candidate in (text, text.replace(" ", "T")):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _stakeholder_graph_node_id(kind: str, key: str) -> str:
    return f"stakeholder_{kind}:{normalize_lookup(key)}"


def _profile_graph_node_id(profile: Dict[str, Any]) -> str:
    target_type = str(profile.get("target_type") or "").strip().lower()
    profile_kind = _target_type_graph_kind(target_type)
    return _stakeholder_graph_node_id(
        profile_kind,
        str(profile.get("profile_key") or profile.get("canonical_name") or ""),
    )


def _target_type_graph_kind(target_type: str) -> str:
    normalized = str(target_type or "").strip().lower()
    if normalized == "person":
        return "person"
    if normalized == "industry":
        return "industry"
    return "entity"


def _text_contains_org_hint(text: str, org_name: str) -> bool:
    haystack = normalize_lookup(text)
    needle = normalize_lookup(org_name)
    if not haystack or not needle:
        return False
    if len(needle) <= 3:
        return needle in haystack.split()
    return needle in haystack


def _weak_org_link_match(left: str, right: str) -> bool:
    left_norm = normalize_lookup(left)
    right_norm = normalize_lookup(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if len(left_norm) > 3 and left_norm in right_norm:
        return True
    if len(right_norm) > 3 and right_norm in left_norm:
        return True
    left_tokens = _org_tokens(left)
    right_tokens = _org_tokens(right)
    if left_tokens and right_tokens:
        smaller, larger = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
        if smaller.issubset(larger):
            return True
    return False


def _find_alumni_hits(signal: Dict[str, Any], alumni_names: List[str]) -> List[str]:
    if not alumni_names:
        return []
    candidate_employer = str(signal.get("parsed_candidate_employer") or "").strip()
    text_fields = [
        str(signal.get("subject") or ""),
        str(signal.get("text_note") or ""),
        str(signal.get("raw_text") or ""),
        str(signal.get("primary_url") or ""),
    ]
    hits: List[str] = []
    for alumni_name in alumni_names:
        if candidate_employer and _weak_org_link_match(candidate_employer, alumni_name):
            hits.append(alumni_name)
            continue
        if any(_text_contains_org_hint(text, alumni_name) for text in text_fields if text):
            hits.append(alumni_name)
    return _normalize_org_name_list(hits)


def _find_profile_alumni_links(profile: Dict[str, Any], alumni_names: List[str]) -> List[str]:
    if not alumni_names or not profile:
        return []
    candidates = [
        str(profile.get("current_employer") or "").strip(),
        *(str(item).strip() for item in profile.get("known_employers") or [] if str(item).strip()),
    ]
    for affiliation in profile.get("affiliations") or []:
        if isinstance(affiliation, dict):
            candidates.append(str(affiliation.get("org_name_text") or "").strip())
    hits: List[str] = []
    for alumni_name in alumni_names:
        if any(_weak_org_link_match(candidate, alumni_name) for candidate in candidates if candidate):
            hits.append(alumni_name)
    return _normalize_org_name_list(hits)


def _upsert_primary_affiliation(
    affiliations: List[Dict[str, Any]],
    org_name_text: str,
    role: str = "",
    confidence: str = "confirmed",
    source: str = "",
) -> List[Dict[str, Any]]:
    wanted_org = str(org_name_text or "").strip()
    if not wanted_org:
        return affiliations

    normalized_existing = _normalize_affiliations(affiliations)
    wanted_lookup = normalize_lookup(wanted_org)
    updated: List[Dict[str, Any]] = []
    found = False
    for item in normalized_existing:
        next_item = dict(item)
        if normalize_lookup(item.get("org_name_text") or "") == wanted_lookup:
            next_item["org_name_text"] = wanted_org
            if role:
                next_item["role"] = str(role).strip()
            next_item["confidence"] = _normalize_affiliation_confidence(confidence)
            next_item["source"] = str(source or item.get("source") or "").strip()
            next_item["is_primary"] = 1
            found = True
        else:
            next_item["is_primary"] = 0
        updated.append(next_item)

    if not found:
        updated.insert(
            0,
            {
                "org_name_text": wanted_org,
                "role": str(role or "").strip(),
                "affiliation_type": "current",
                "confidence": _normalize_affiliation_confidence(confidence),
                "is_primary": 1,
                "start_date": "",
                "end_date": "",
                "source": str(source or "").strip(),
            },
        )
    return _normalize_affiliations(updated)


def _ollama_watch_timeout_seconds() -> int:
    raw_value = str(os.environ.get("CORTEX_WATCH_OLLAMA_TIMEOUT") or "").strip()
    if raw_value:
        try:
            value = int(raw_value)
            if value > 0:
                return value
        except Exception:
            logger.warning("Invalid CORTEX_WATCH_OLLAMA_TIMEOUT=%r; using default %s", raw_value, _DEFAULT_OLLAMA_WATCH_TIMEOUT_SECONDS)
    return _DEFAULT_OLLAMA_WATCH_TIMEOUT_SECONDS


def _resolve_storage_root(base_path: Optional[Path] = None) -> Path:
    if base_path is not None:
        root = Path(base_path)
        root.mkdir(parents=True, exist_ok=True)
        return root

    config = ConfigManager().get_config()
    raw_db_path = str(config.get("ai_database_path") or "").strip()
    if not raw_db_path:
        raise RuntimeError("ai_database_path is not configured; Cortex stakeholder signal store cannot initialize")

    safe_db_path = raw_db_path if os.path.exists("/.dockerenv") else convert_windows_to_wsl_path(raw_db_path)
    root = Path(safe_db_path) / "stakeholder_intel"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _org_tokens(value: str) -> set[str]:
    normalized = normalize_lookup(value)
    return {token for token in normalized.split() if token and token not in _ORG_STOPWORDS}


def _normalize_domain(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        from urllib.parse import urlparse

        parsed = urlparse(raw)
    except Exception:
        return ""
    host = (parsed.netloc or parsed.path or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _signal_domain(signal: Dict[str, Any]) -> str:
    return _normalize_domain(str(signal.get("primary_url") or "").strip())


def _signal_text_haystack(signal: Dict[str, Any]) -> str:
    return normalize_lookup(
        " ".join(
            [
                str(signal.get("subject") or ""),
                str(signal.get("raw_text") or ""),
                str(signal.get("text_note") or ""),
            ]
        )
    )


def _has_employer_anchor(signal: Dict[str, Any], top_match: Dict[str, Any]) -> bool:
    reasons = [str(item).strip().lower() for item in top_match.get("reasons") or []]
    if any("employer" in reason for reason in reasons):
        return True

    employers = [
        str(top_match.get("current_employer") or "").strip(),
        *(str(item).strip() for item in top_match.get("known_employers") or [] if str(item).strip()),
    ]
    for affiliation in top_match.get("affiliations") or []:
        if isinstance(affiliation, dict):
            org_name = str(affiliation.get("org_name_text") or "").strip()
            if org_name:
                employers.append(org_name)

    haystack = _signal_text_haystack(signal)
    for employer in employers:
        if employer and _text_contains_org_hint(haystack, employer):
            return True
    return False


def _digest_signal_identity_strength(signal: Dict[str, Any]) -> str:
    top_match = (signal.get("matches") or [{}])[0]
    if not top_match:
        return "unmatched"

    reasons = [str(item).strip().lower() for item in top_match.get("reasons") or []]
    score = float(top_match.get("score") or 0.0)
    domain = _signal_domain(signal)

    if "exact linkedin url match" in reasons:
        return "verified"
    if any(
        token in reason
        for reason in reasons
        for token in ("candidate employer match", "employer found in text", "current role found in text", "affiliation role found in text")
    ):
        return "anchored"
    if _has_employer_anchor(signal, top_match):
        return "anchored"
    if score >= 0.9:
        return "strong"
    if str(signal.get("parsed_candidate_employer") or "").strip() and domain not in _LOW_VALUE_PERSON_SIGNAL_DOMAINS:
        return "anchored"
    if domain and domain not in _LOW_VALUE_PERSON_SIGNAL_DOMAINS and score >= 0.82:
        return "probable"
    return "weak"


def _should_include_signal_in_digest(signal: Dict[str, Any], report_depth: str) -> bool:
    top_match = (signal.get("matches") or [{}])[0]
    if not top_match:
        return bool(signal.get("alumni_hits"))

    if str(top_match.get("target_type") or "").strip().lower() != "person":
        return True

    identity_strength = _digest_signal_identity_strength(signal)
    signal["digest_identity_strength"] = identity_strength
    domain = _signal_domain(signal)

    # Directory/profile-farm domains are too noisy for human-facing digests,
    # even when they incidentally contain an employer hint.
    if domain in _LOW_VALUE_PERSON_SIGNAL_DOMAINS:
        return False
    if identity_strength in {"verified", "anchored", "strong"}:
        return True
    if identity_strength == "probable":
        return report_depth == "summary"
    return False


def orgs_compatible(left: str, right: str) -> bool:
    left_norm = normalize_lookup(left)
    right_norm = normalize_lookup(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_tokens = _org_tokens(left)
    right_tokens = _org_tokens(right)
    if not left_tokens or not right_tokens:
        return left_norm == right_norm

    if left_tokens == right_tokens:
        return True
    smaller, larger = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    return smaller.issubset(larger)


class StakeholderSignalStore:
    """JSON-backed store for stakeholder profiles, incoming signals, and generated artefacts."""

    def __init__(self, base_path: Optional[Path] = None):
        self.root = _resolve_storage_root(base_path)
        self.state_path = self.root / "state.json"
        self.raw_dir = self.root / "raw_signals"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._graph_manager: Optional[EnhancedGraphManager] = None
        self._graph_name_index: Optional[Dict[str, set[str]]] = None
        if not self.state_path.exists():
            self._write_state(self._initial_state())

    @staticmethod
    def _initial_state() -> Dict[str, Any]:
        return {
            "updated_at": _utc_now_iso(),
            "profiles": [],
            "signals": [],
            "observed_facts": [],
            "update_suggestions": [],
            "intel_notes": [],
            "org_contexts": {},
        }

    def _read_state(self) -> Dict[str, Any]:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            payload = json.loads(raw) if raw.strip() else {}
            if not isinstance(payload, dict):
                return self._initial_state()
            payload.setdefault("profiles", [])
            payload.setdefault("signals", [])
            payload.setdefault("observed_facts", [])
            payload.setdefault("update_suggestions", [])
            payload.setdefault("intel_notes", [])
            payload.setdefault("org_contexts", {})
            payload.setdefault("updated_at", _utc_now_iso())
            return payload
        except Exception:
            return self._initial_state()

    def _write_state(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = _utc_now_iso()
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="stakeholder_intel_", suffix=".json", dir=str(self.root))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=True, indent=2, sort_keys=True)
            os.replace(tmp_path, self.state_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        return self._read_state()

    def list_profiles(self, org_name: str = "") -> List[Dict[str, Any]]:
        state = self._read_state()
        profiles = list(state.get("profiles") or [])
        if org_name:
            profiles = [p for p in profiles if orgs_compatible(p.get("org_name", ""), org_name)]
        return sorted(profiles, key=lambda item: (item.get("org_name", ""), item.get("canonical_name", "")))

    def list_signals(self, org_name: str = "", matched_only: bool = False, limit: int = 200) -> List[Dict[str, Any]]:
        state = self._read_state()
        signals = list(state.get("signals") or [])
        if org_name:
            signals = [s for s in signals if _signal_visible_to_org(s, org_name)]
        if matched_only:
            signals = [s for s in signals if s.get("matches")]
        signals = sorted(signals, key=lambda item: item.get("received_at", ""), reverse=True)
        return signals[:limit]

    def list_intel_notes(self, org_name: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        state = self._read_state()
        notes = list(state.get("intel_notes") or [])
        if org_name:
            notes = [item for item in notes if orgs_compatible(item.get("org_name", ""), org_name)]
        return sorted(notes, key=lambda item: item.get("note_date", ""), reverse=True)[:limit]

    def list_update_suggestions(self, org_name: str = "", status: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        state = self._read_state()
        suggestions = list(state.get("update_suggestions") or [])
        if org_name:
            suggestions = [item for item in suggestions if orgs_compatible(item.get("org_name", ""), org_name)]
        if status:
            status_norm = str(status).strip().lower()
            suggestions = [item for item in suggestions if str(item.get("status") or "").strip().lower() == status_norm]
        suggestions = sorted(suggestions, key=lambda item: item.get("created_at", ""), reverse=True)
        return suggestions[:limit]

    def list_observed_facts(self, org_name: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        state = self._read_state()
        facts = list(state.get("observed_facts") or [])
        if org_name:
            facts = [item for item in facts if orgs_compatible(item.get("org_name", ""), org_name)]
        return sorted(facts, key=lambda item: item.get("created_at", ""), reverse=True)[:limit]

    def get_profile(self, profile_key: str) -> Optional[Dict[str, Any]]:
        wanted = str(profile_key).strip()
        if not wanted:
            return None
        state = self._read_state()
        for profile in state.get("profiles") or []:
            if profile.get("profile_key") == wanted:
                return dict(profile)
        return None

    def get_org_context(self, org_name: str) -> Dict[str, Any]:
        state = self._read_state()
        key = normalize_lookup(org_name)
        context = dict((state.get("org_contexts") or {}).get(key) or {})
        context.setdefault("org_name", org_name)
        context.setdefault("org_alumni", [])
        context.setdefault("org_strategic_profile", {})
        return context

    def _graph_view_node_type(
        self,
        node_id: str,
        attrs: Dict[str, Any],
        tracked_profile_keys: set[str],
    ) -> str:
        entity_type = str(attrs.get("entity_type") or "").strip()
        target_type = str(attrs.get("target_type") or "").strip().lower()
        if entity_type == "industry" or target_type == "industry":
            return "industry"
        if entity_type == "entity" and str(attrs.get("profile_key") or "").strip() in tracked_profile_keys:
            return "tracked_org"
        if entity_type == "entity":
            return "organization"
        return entity_type or "unknown"

    def _graph_view_allowed_node(
        self,
        attrs: Dict[str, Any],
        *,
        include_signals: bool,
        include_sources: bool,
        include_lab_members: bool,
        include_alumni: bool,
    ) -> bool:
        entity_type = str(attrs.get("entity_type") or "").strip()
        if entity_type == "stakeholder_signal" and not include_signals:
            return False
        if entity_type == "source" and not include_sources:
            return False
        if entity_type == "lab_member" and not include_lab_members:
            return False
        if entity_type == "alumni_group" and not include_alumni:
            return False
        return entity_type in {
            "person",
            "entity",
            "industry",
            "organization",
            "alumni_group",
            "lab_member",
            "subscriber_org",
            "stakeholder_signal",
            "source",
        }

    def _graph_view_edge_weight(self, relationship: str, edge_attrs: Dict[str, Any]) -> float:
        base = _GRAPH_EDGE_BASE_WEIGHTS.get(relationship, 0.4)
        confidence = edge_attrs.get("confidence")
        if confidence is None:
            confidence_score = base * 10.0
        else:
            try:
                confidence_value = float(confidence)
            except Exception:
                confidence_value = 0.0
            confidence_score = confidence_value * 10.0 if confidence_value <= 1.0 else confidence_value
        return round(min(1.0, (base * 0.65) + ((max(0.0, min(10.0, confidence_score)) / 10.0) * 0.35)), 3)

    def _graph_view_actionability_score(self, relationship: str) -> float:
        if relationship == "linkedin_connection":
            return 9.2
        if relationship == "belongs_to_industry":
            return 8.0
        if relationship in {"works_at", "tracked_by"}:
            return 7.8
        if relationship in {"alumni_of", "org_alumni_context"}:
            return 7.1
        if relationship == "co_mentioned":
            return 5.8
        return 4.5

    def _graph_view_signal_counters(
        self,
        signals: List[Dict[str, Any]],
        since_ts: str,
    ) -> tuple[Counter, Counter]:
        since_dt = _parse_timestamp(since_ts)
        total_counts: Counter = Counter()
        recent_counts: Counter = Counter()
        for signal in signals:
            signal_dt = _parse_timestamp(signal.get("received_at", ""))
            for profile_key in [str(item).strip() for item in signal.get("matched_profile_keys") or [] if str(item).strip()]:
                total_counts[profile_key] += 1
                if since_dt is None or (signal_dt and signal_dt >= since_dt):
                    recent_counts[profile_key] += 1
        return total_counts, recent_counts

    def _graph_view_focus_profiles(
        self,
        profiles: List[Dict[str, Any]],
        *,
        view_mode: str,
        profile_keys: List[str],
        focus_profile_key: str,
    ) -> List[Dict[str, Any]]:
        profiles_by_key = {str(profile.get("profile_key") or "").strip(): profile for profile in profiles}
        if view_mode == "ego" and focus_profile_key:
            wanted = profiles_by_key.get(focus_profile_key)
            return [wanted] if wanted else []
        if view_mode == "industry_network":
            if focus_profile_key:
                wanted = profiles_by_key.get(focus_profile_key)
                return [wanted] if wanted else []
            industry_profiles = [profile for profile in profiles if str(profile.get("target_type") or "").strip().lower() == "industry"]
            return industry_profiles or profiles
        if profile_keys:
            return [profiles_by_key[key] for key in profile_keys if key in profiles_by_key]
        watched = [profile for profile in profiles if str(profile.get("watch_status") or "").strip().lower() == "watch"]
        return watched or profiles

    def build_graph_view(
        self,
        org_name: str,
        view_mode: str = "watch_network",
        profile_keys: Optional[List[str]] = None,
        child_profile_keys: Optional[List[str]] = None,
        focus_profile_key: str = "",
        focus_org_name: str = "",
        since_ts: str = "",
        max_hops: int = 2,
        max_nodes: int = 100,
        max_edges: int = 200,
        include_signals: bool = True,
        include_sources: bool = False,
        include_lab_members: bool = True,
        include_alumni: bool = True,
        include_unwatched_bridges: bool = False,
        edge_types: Optional[List[str]] = None,
        min_edge_weight: float = 0.2,
        min_confidence: float = 5.0,
        layout_hint: str = "force",
        top_k_paths: int = 5,
    ) -> Dict[str, Any]:
        graph_manager = self._load_graph_manager()
        profiles = self.list_profiles(org_name=org_name)
        child_profile_keys = [str(item).strip() for item in child_profile_keys or [] if str(item).strip()]
        if not include_unwatched_bridges:
            profiles = [
                profile
                for profile in profiles
                if str(profile.get("watch_status") or "").strip().lower() == "watch"
                or str(profile.get("profile_key") or "").strip() in {str(item).strip() for item in profile_keys or [] if str(item).strip()}
                or str(profile.get("profile_key") or "").strip() in set(child_profile_keys)
                or str(profile.get("profile_key") or "").strip() == str(focus_profile_key or "").strip()
            ] or profiles
        tracked_profile_keys = {
            str(profile.get("profile_key") or "").strip()
            for profile in profiles
            if str(profile.get("profile_key") or "").strip()
        }
        focus_profiles = self._graph_view_focus_profiles(
            profiles,
            view_mode=view_mode,
            profile_keys=[str(item).strip() for item in [*(profile_keys or []), *child_profile_keys] if str(item).strip()],
            focus_profile_key=str(focus_profile_key or "").strip(),
        )
        org_context = self.get_org_context(org_name)
        signals = self.list_signals(org_name=org_name, matched_only=False, limit=2000)
        total_signal_counts, recent_signal_counts = self._graph_view_signal_counters(signals, since_ts)

        if graph_manager is None:
            graph_manager = self._load_graph_manager(create=True)
        graph = graph_manager.graph if graph_manager is not None else nx.Graph()
        allowed_edge_types = set(edge_types or _GRAPH_EDGE_BASE_WEIGHTS.keys())

        subscriber_node = _stakeholder_graph_node_id("subscriber_org", org_name)
        start_nodes: List[str] = []
        if subscriber_node in graph:
            start_nodes.append(subscriber_node)
        for profile in focus_profiles:
            profile_node = _profile_graph_node_id(profile)
            if profile_node in graph and profile_node not in start_nodes:
                start_nodes.append(profile_node)
        if focus_org_name:
            org_node = _stakeholder_graph_node_id("organization", focus_org_name)
            if org_node in graph and org_node not in start_nodes:
                start_nodes.append(org_node)
            for profile in profiles:
                if profile.get("target_type") == "organisation" and _weak_org_link_match(profile.get("canonical_name", ""), focus_org_name):
                    profile_node = _profile_graph_node_id(profile)
                    if profile_node in graph and profile_node not in start_nodes:
                        start_nodes.append(profile_node)
        if not start_nodes:
            for profile in focus_profiles[:5]:
                profile_node = _profile_graph_node_id(profile)
                if profile_node not in start_nodes:
                    start_nodes.append(profile_node)

        visited: set[str] = set()
        node_hops: Dict[str, int] = {}
        queue: List[tuple[str, int]] = [(node_id, 0) for node_id in start_nodes]
        collected_edges: set[tuple[str, str]] = set()

        while queue and len(visited) < max_nodes:
            node_id, hops = queue.pop(0)
            if node_id in visited or node_id not in graph:
                continue
            attrs = dict(graph.nodes.get(node_id, {}))
            if not self._graph_view_allowed_node(
                attrs,
                include_signals=include_signals,
                include_sources=include_sources,
                include_lab_members=include_lab_members,
                include_alumni=include_alumni,
            ):
                continue
            visited.add(node_id)
            node_hops[node_id] = hops
            if hops >= max_hops:
                continue
            for neighbor in graph.neighbors(node_id):
                if len(collected_edges) >= max_edges:
                    break
                neighbor_attrs = dict(graph.nodes.get(neighbor, {}))
                if not self._graph_view_allowed_node(
                    neighbor_attrs,
                    include_signals=include_signals,
                    include_sources=include_sources,
                    include_lab_members=include_lab_members,
                    include_alumni=include_alumni,
                ):
                    continue
                edge_attrs = dict(graph.get_edge_data(node_id, neighbor) or {})
                relationship = str(edge_attrs.get("relationship") or "").strip()
                if relationship not in allowed_edge_types:
                    continue
                edge_key = tuple(sorted((node_id, neighbor)))
                if edge_key not in collected_edges:
                    collected_edges.add(edge_key)
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))

        focus_node_ids = [node_id for node_id in start_nodes if node_id in visited]
        focus_node_set = set(focus_node_ids)
        profiles_by_key = {
            str(profile.get("profile_key") or "").strip(): profile
            for profile in profiles
            if str(profile.get("profile_key") or "").strip()
        }

        nodes: List[Dict[str, Any]] = []
        node_lookup: Dict[str, Dict[str, Any]] = {}
        for node_id in sorted(visited):
            attrs = dict(graph.nodes.get(node_id, {}))
            profile_key = str(attrs.get("profile_key") or "").strip()
            profile = profiles_by_key.get(profile_key)
            node_type = self._graph_view_node_type(node_id, attrs, tracked_profile_keys)
            watched = bool(profile and str(profile.get("watch_status") or "").strip().lower() == "watch")
            recent_signal_count = recent_signal_counts.get(profile_key, 0) if profile_key else 0
            total_signal_count = total_signal_counts.get(profile_key, 0) if profile_key else 0
            importance = 3.5
            if watched:
                importance += 2.0
            if node_id in focus_node_set:
                importance += 1.5
            importance += min(2.0, recent_signal_count * 0.4)
            importance += min(1.5, total_signal_count * 0.15)
            if node_type in {"lab_member", "alumni_group"}:
                importance += 0.5
            confidence_score = 6.0
            if profile_key and total_signal_count:
                confidence_score = min(10.0, 5.5 + (recent_signal_count * 0.5) + min(2.0, total_signal_count * 0.2))
            meta: Dict[str, Any] = {
                "entity_type": str(attrs.get("entity_type") or "").strip(),
                "graph_hops": node_hops.get(node_id, 0),
            }
            if profile:
                meta.update(
                    {
                        "target_type": profile.get("target_type", ""),
                        "current_employer": profile.get("current_employer", ""),
                        "current_role": profile.get("current_role", ""),
                        "alumni": profile.get("alumni") or [],
                        "known_employers": profile.get("known_employers") or [],
                    }
                )
            subtitle = ""
            if profile:
                subtitle = str(profile.get("current_employer") or profile.get("current_role") or "").strip()
            elif node_type == "alumni_group":
                subtitle = "Alumni group"
            elif node_type == "lab_member":
                subtitle = "Lab member"
            node_payload = {
                "id": node_id,
                "type": node_type,
                "label": str(attrs.get("name") or profile.get("canonical_name") if profile else attrs.get("name") or node_id).strip() if attrs or profile else node_id,
                "subtitle": subtitle,
                "org_name": org_name,
                "profile_key": profile_key,
                "watch_status": str(profile.get("watch_status") or "").strip() if profile else "",
                "is_focus": node_id in focus_node_set,
                "is_watched": watched,
                "importance_score": round(min(10.0, importance), 1),
                "confidence_score": round(confidence_score, 1),
                "signal_count": total_signal_count,
                "recent_signal_count": recent_signal_count,
                "tags": [tag for tag in ["focus" if node_id in focus_node_set else "", "watched" if watched else ""] if tag],
                "meta": meta,
            }
            nodes.append(node_payload)
            node_lookup[node_id] = node_payload

        edges: List[Dict[str, Any]] = []
        for idx, (left, right) in enumerate(sorted(collected_edges)):
            if left not in node_lookup or right not in node_lookup:
                continue
            edge_attrs = dict(graph.get_edge_data(left, right) or {})
            relationship = str(edge_attrs.get("relationship") or "").strip()
            weight = self._graph_view_edge_weight(relationship, edge_attrs)
            confidence_value = edge_attrs.get("confidence")
            try:
                confidence_score = float(confidence_value) * 10.0 if float(confidence_value) <= 1.0 else float(confidence_value)
            except Exception:
                confidence_score = round(weight * 10.0, 1)
            if weight < float(min_edge_weight) or confidence_score < float(min_confidence):
                continue
            edges.append(
                {
                    "id": f"edge_{idx + 1}",
                    "source": left,
                    "target": right,
                    "type": relationship,
                    "label": relationship.replace("_", " "),
                    "weight": weight,
                    "confidence_score": round(min(10.0, confidence_score), 1),
                    "recency_score": 8.0 if relationship in {"co_mentioned", "mentions_profile", "mentions_organization"} else 6.5,
                    "actionability_score": self._graph_view_actionability_score(relationship),
                    "evidence_count": 1,
                    "is_direct": left in focus_node_set or right in focus_node_set,
                    "is_inferred": False,
                    "path_role": relationship,
                    "meta": {
                        key: value
                        for key, value in edge_attrs.items()
                        if key not in {"relationship"}
                    },
                }
            )

        edge_id_by_nodes = {
            tuple(sorted((edge["source"], edge["target"]))): edge["id"]
            for edge in edges
        }
        subgraph = graph.subgraph([node["id"] for node in nodes]).copy()

        tracked_profile_nodes = {
            _profile_graph_node_id(profile): profile
            for profile in profiles
            if _profile_graph_node_id(profile) in subgraph
        }
        lab_member_nodes = [node_id for node_id, attrs in subgraph.nodes(data=True) if str(attrs.get("entity_type") or "").strip() == "lab_member"]
        alumni_nodes = [node_id for node_id, attrs in subgraph.nodes(data=True) if str(attrs.get("entity_type") or "").strip() == "alumni_group"]

        paths: List[Dict[str, Any]] = []
        for member_node in lab_member_nodes:
            for profile_node in tracked_profile_nodes:
                try:
                    path_nodes = nx.shortest_path(subgraph, member_node, profile_node)
                except Exception:
                    continue
                if len(path_nodes) - 1 > max_hops:
                    continue
                edge_ids = [
                    edge_id_by_nodes.get(tuple(sorted((path_nodes[i], path_nodes[i + 1]))), "")
                    for i in range(len(path_nodes) - 1)
                ]
                path_strength = 0.0
                if edge_ids:
                    matching_edges = [edge for edge in edges if edge["id"] in edge_ids]
                    if matching_edges:
                        path_strength = round(sum(edge["weight"] for edge in matching_edges) / len(matching_edges), 3)
                paths.append(
                    {
                        "id": f"path_warm_{len(paths) + 1}",
                        "type": "warm_intro",
                        "source_node_id": member_node,
                        "target_node_id": profile_node,
                        "hop_count": len(path_nodes) - 1,
                        "strength": path_strength,
                        "explanation": f"{self._graph_node_label(member_node)} -> {self._graph_node_label(profile_node)} warm path",
                        "node_ids": path_nodes,
                        "edge_ids": [item for item in edge_ids if item],
                    }
                )
                if len(paths) >= top_k_paths:
                    break
            if len(paths) >= top_k_paths:
                break

        for idx, left_node in enumerate(sorted(tracked_profile_nodes)):
            for right_node in sorted(list(tracked_profile_nodes))[idx + 1:]:
                shared_alumni = self._graph_common_neighbor_details(left_node, right_node, {"alumni_group"})
                if shared_alumni:
                    paths.append(
                        {
                            "id": f"path_alumni_{len(paths) + 1}",
                            "type": "shared_alumni",
                            "source_node_id": left_node,
                            "target_node_id": right_node,
                            "hop_count": 2,
                            "strength": 0.72,
                            "explanation": f"Shared alumni bridge via {shared_alumni[0]['name']}",
                            "node_ids": [left_node, shared_alumni[0]["node_id"], right_node],
                            "edge_ids": [
                                edge_id_by_nodes.get(tuple(sorted((left_node, shared_alumni[0]["node_id"]))), ""),
                                edge_id_by_nodes.get(tuple(sorted((shared_alumni[0]["node_id"], right_node))), ""),
                            ],
                        }
                    )
                    if len(paths) >= top_k_paths:
                        break
            if len(paths) >= top_k_paths:
                break

        insights: List[Dict[str, Any]] = []
        if lab_member_nodes and paths:
            first_path = next((path for path in paths if path["type"] == "warm_intro"), None)
            if first_path:
                insights.append(
                    {
                        "id": "insight_warm_intro",
                        "type": "warm_intro",
                        "severity": "medium",
                        "score": 8.4,
                        "title": "Warm introduction path available",
                        "summary": first_path["explanation"],
                        "node_ids": first_path["node_ids"],
                        "edge_ids": [item for item in first_path["edge_ids"] if item],
                        "source_signal_ids": [],
                    }
                )
        if alumni_nodes:
            insights.append(
                {
                    "id": "insight_alumni_cluster",
                    "type": "alumni_cluster",
                    "severity": "medium",
                    "score": 7.6,
                    "title": "Shared alumni bridges detected",
                    "summary": f"{len(alumni_nodes)} alumni-group nodes connect this scoped network.",
                    "node_ids": alumni_nodes[:5],
                    "edge_ids": [],
                    "source_signal_ids": [],
                }
            )
        co_mentioned_edges = [edge for edge in edges if edge["type"] == "co_mentioned"]
        if co_mentioned_edges:
            insights.append(
                {
                    "id": "insight_cross_target",
                    "type": "cross_target_overlap",
                    "severity": "low",
                    "score": 6.9,
                    "title": "Cross-target co-mentions present",
                    "summary": f"{len(co_mentioned_edges)} direct co-mention link(s) found in the scoped network.",
                    "node_ids": [],
                    "edge_ids": [edge["id"] for edge in co_mentioned_edges[:5]],
                    "source_signal_ids": [],
                }
            )

        generated_at = _utc_now_iso()
        graph_id = f"graph_{hashlib.sha1(f'{org_name}|{view_mode}|{generated_at}'.encode('utf-8')).hexdigest()[:16]}"
        output_dir = self.root / "graph_views"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{graph_id}.json"

        payload = {
            "graph_id": graph_id,
            "org_name": org_name,
            "view_mode": view_mode,
            "generated_at": generated_at,
            "filters": {
                "since_ts": since_ts,
                "max_hops": max_hops,
                "max_nodes": max_nodes,
                "max_edges": max_edges,
                "include_signals": include_signals,
                "include_sources": include_sources,
                "include_lab_members": include_lab_members,
                "include_alumni": include_alumni,
                "include_unwatched_bridges": include_unwatched_bridges,
                "min_edge_weight": min_edge_weight,
                "min_confidence": min_confidence,
                "layout_hint": layout_hint,
                "edge_types": sorted(allowed_edge_types),
            },
            "summary": {
                "watched_people": len([node for node in nodes if node["type"] == "person" and node["is_watched"]]),
                "watched_orgs": len([node for node in nodes if node["type"] == "tracked_org"]),
                "watched_industries": len([node for node in nodes if node["type"] == "industry"]),
                "alumni_groups": len(alumni_nodes),
                "warm_intro_paths": len([path for path in paths if path["type"] == "warm_intro"]),
                "cross_target_links": len([edge for edge in edges if edge["type"] == "co_mentioned"]),
            },
            "nodes": nodes,
            "edges": edges,
            "paths": paths[:top_k_paths],
            "insights": insights[:8],
            "legend": {
                "node_types": {
                    "person": "Watched stakeholder",
                    "tracked_org": "Tracked organisation profile",
                    "industry": "Industry profile",
                    "organization": "Organisation bridge",
                    "alumni_group": "Alumni group",
                    "lab_member": "Lab member connector",
                    "subscriber_org": "Subscriber organisation",
                    "stakeholder_signal": "Signal overlay",
                    "source": "Source node",
                },
                "edge_types": {
                    edge_type: edge_type.replace("_", " ")
                    for edge_type in sorted(allowed_edge_types)
                },
            },
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "graph_id": graph_id,
            "org_name": org_name,
            "view_mode": view_mode,
            "generated_at": generated_at,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "focus_node_ids": focus_node_ids,
            "output_path": str(output_path),
            "has_paths": bool(paths),
            "has_signal_overlay": bool(include_signals),
            "summary": payload["summary"],
        }

    def _graph_file_path(self) -> Path:
        return self.root.parent / "knowledge_cortex.gpickle"

    def _load_graph_manager(self, create: bool = False) -> Optional[EnhancedGraphManager]:
        if self._graph_manager is not None:
            return self._graph_manager
        graph_path = self._graph_file_path()
        if not graph_path.exists() and not create:
            self._graph_manager = None
            return None
        self._graph_manager = EnhancedGraphManager(str(graph_path))
        return self._graph_manager

    def _save_graph_manager(self) -> None:
        if self._graph_manager is None:
            return
        self._graph_manager.save_graph()
        self._graph_name_index = None

    def _graph_exact_nodes(self, term: str) -> List[str]:
        lookup = normalize_lookup(term)
        if not lookup:
            return []
        manager = self._load_graph_manager()
        if manager is None:
            return []
        if self._graph_name_index is None:
            index: Dict[str, set[str]] = defaultdict(set)
            for node_id, attrs in manager.graph.nodes(data=True):
                index[normalize_lookup(node_id)].add(node_id)
                for candidate in (
                    attrs.get("name"),
                    attrs.get("file_name"),
                    attrs.get("title"),
                ):
                    normalized = normalize_lookup(candidate or "")
                    if normalized:
                        index[normalized].add(node_id)
            self._graph_name_index = index
        return sorted(self._graph_name_index.get(lookup, set()))

    def _graph_node_label(self, node_id: str) -> str:
        manager = self._load_graph_manager()
        if manager is None:
            return node_id
        attrs = manager.graph.nodes.get(node_id, {})
        return str(attrs.get("name") or attrs.get("profile_key") or node_id).strip() or node_id

    def _graph_neighbor_details(self, node_id: str, allowed_types: Optional[set[str]] = None) -> List[Dict[str, Any]]:
        manager = self._load_graph_manager()
        if manager is None or node_id not in manager.graph:
            return []
        details: List[Dict[str, Any]] = []
        for neighbor in manager.graph.neighbors(node_id):
            attrs = dict(manager.graph.nodes.get(neighbor, {}))
            entity_type = str(attrs.get("entity_type") or "").strip()
            if allowed_types and entity_type not in allowed_types:
                continue
            details.append(
                {
                    "node_id": neighbor,
                    "name": str(attrs.get("name") or attrs.get("profile_key") or neighbor).strip() or neighbor,
                    "entity_type": entity_type,
                    "edge": dict(manager.graph.get_edge_data(node_id, neighbor) or {}),
                }
            )
        return details

    def _graph_common_neighbor_details(
        self,
        left_node: str,
        right_node: str,
        allowed_types: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        manager = self._load_graph_manager()
        if manager is None or left_node not in manager.graph or right_node not in manager.graph:
            return []
        shared = set(manager.graph.neighbors(left_node)).intersection(set(manager.graph.neighbors(right_node)))
        details: List[Dict[str, Any]] = []
        for neighbor in shared:
            attrs = dict(manager.graph.nodes.get(neighbor, {}))
            entity_type = str(attrs.get("entity_type") or "").strip()
            if allowed_types and entity_type not in allowed_types:
                continue
            details.append(
                {
                    "node_id": neighbor,
                    "name": str(attrs.get("name") or attrs.get("profile_key") or neighbor).strip() or neighbor,
                    "entity_type": entity_type,
                    "left_edge": dict(manager.graph.get_edge_data(left_node, neighbor) or {}),
                    "right_edge": dict(manager.graph.get_edge_data(right_node, neighbor) or {}),
                }
            )
        return sorted(details, key=lambda item: (item.get("entity_type", ""), normalize_lookup(item.get("name", ""))))

    def _graph_path_summary(
        self,
        left_node: str,
        right_node: str,
        max_hops: int = 4,
        ignore_entity_types: Optional[set[str]] = None,
    ) -> str:
        manager = self._load_graph_manager()
        if manager is None or left_node not in manager.graph or right_node not in manager.graph:
            return ""
        try:
            path = nx.shortest_path(manager.graph, left_node, right_node)
        except Exception:
            return ""
        if len(path) - 1 > max_hops:
            return ""

        connector_nodes = path[1:-1]
        if ignore_entity_types:
            connector_nodes = [
                node
                for node in connector_nodes
                if str(manager.graph.nodes.get(node, {}).get("entity_type") or "").strip() not in ignore_entity_types
            ]
        if not connector_nodes:
            return ""

        connector_labels = [self._graph_node_label(node) for node in connector_nodes if self._graph_node_label(node)]
        if not connector_labels:
            return ""
        return f"graph path via {' -> '.join(connector_labels[:3])}"

    def _graph_shortest_path_summary(self, left_terms: List[str], right_terms: List[str], max_hops: int = 4) -> str:
        manager = self._load_graph_manager()
        if manager is None:
            return ""
        left_nodes = [node for term in left_terms for node in self._graph_exact_nodes(term)]
        right_nodes = [node for term in right_terms for node in self._graph_exact_nodes(term)]
        if not left_nodes or not right_nodes:
            return ""
        best_length: Optional[int] = None
        best_pair: tuple[str, str] | None = None
        best_path_summary = ""
        for left_node in left_nodes[:5]:
            for right_node in right_nodes[:5]:
                try:
                    length = nx.shortest_path_length(manager.graph, left_node, right_node)
                except Exception:
                    continue
                if length > max_hops:
                    continue
                path_summary = self._graph_path_summary(
                    left_node,
                    right_node,
                    max_hops=max_hops,
                    ignore_entity_types={"subscriber_org"},
                )
                if best_length is None or length < best_length:
                    best_length = length
                    best_pair = (left_node, right_node)
                    best_path_summary = path_summary
        if best_length is None or best_pair is None:
            return ""
        if best_path_summary:
            return best_path_summary
        return f"knowledge-graph path length {best_length} between {best_pair[0]} and {best_pair[1]}"

    def find_relationship_paths(
        self,
        org_name: str,
        target_names: List[str],
        max_hops: int = 4,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        manager = self._load_graph_manager()
        if manager is None:
            return []

        source_nodes: List[str] = []
        subscriber_node = _stakeholder_graph_node_id("subscriber_org", org_name)
        if subscriber_node in manager.graph:
            source_nodes.append(subscriber_node)
        for node_id, attrs in manager.graph.nodes(data=True):
            if str(attrs.get("entity_type") or "").strip() == "lab_member":
                source_nodes.append(node_id)

        target_nodes: List[str] = []
        for target_name in target_names or []:
            target_nodes.extend(self._graph_exact_nodes(target_name))

        seen: set[tuple[str, str]] = set()
        output: List[Dict[str, Any]] = []
        for source_node in source_nodes[:20]:
            for target_node in target_nodes[:20]:
                pair = (source_node, target_node)
                if pair in seen or source_node == target_node:
                    continue
                seen.add(pair)
                try:
                    path_nodes = nx.shortest_path(manager.graph, source_node, target_node)
                except Exception:
                    continue
                hop_count = len(path_nodes) - 1
                if hop_count <= 0 or hop_count > max_hops:
                    continue
                via_nodes = [self._graph_node_label(node) for node in path_nodes[1:-1] if self._graph_node_label(node)]
                output.append(
                    {
                        "from": self._graph_node_label(source_node),
                        "to": self._graph_node_label(target_node),
                        "via": via_nodes[0] if via_nodes else "",
                        "strength": "warm_intro" if hop_count <= 3 else "indirect",
                        "hop_count": hop_count,
                    }
                )

        output.sort(key=lambda item: (item.get("hop_count", 99), item.get("from", ""), item.get("to", "")))
        return output[: max(0, int(limit))]

    def reconcile_intel_note_delivery(
        self,
        org_name: str,
        trace_id: str,
        payload: Dict[str, Any],
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        note = dict(payload.get("note") or {})
        primary = dict(payload.get("primary_entity") or {})
        intel_id = str(response.get("intel_id") or "").strip() or f"note_{hashlib.sha1(str(trace_id).encode('utf-8')).hexdigest()[:12]}"
        note_title = str(note.get("title") or "").strip()
        note_date = str(note.get("note_date") or "").strip()
        primary_name = str(primary.get("name") or "").strip()
        content_hash = hashlib.sha1(
            "|".join([org_name, note_date, primary_name, str(note.get("content") or note.get("original_text") or "")]).encode("utf-8", "ignore")
        ).hexdigest()[:20]

        state = self._read_state()
        intel_notes = [item for item in state.get("intel_notes") or [] if str(item.get("intel_id") or "").strip() != intel_id]
        intel_notes.append(
            {
                "intel_id": intel_id,
                "org_name": org_name,
                "trace_id": trace_id,
                "title": note_title,
                "note_date": note_date,
                "primary_entity_name": primary_name,
                "primary_target_type": str(primary.get("target_type") or "").strip(),
                "content_hash": content_hash,
                "submitted_by": str(note.get("submitted_by") or "").strip(),
                "website_response": dict(response or {}),
                "created_at": _utc_now_iso(),
            }
        )
        state["intel_notes"] = sorted(intel_notes, key=lambda item: item.get("created_at", ""), reverse=True)
        self._write_state(state)

        manager = self._load_graph_manager(create=True)
        if manager is None:
            return {"intel_id": intel_id, "linked_entities": 0}

        note_node = _stakeholder_graph_node_id("intel_note", intel_id)
        manager.add_entity(
            note_node,
            "intel_note",
            name=note_title or primary_name or intel_id,
            intel_id=intel_id,
            trace_id=trace_id,
            org_name=org_name,
            note_date=note_date,
            submitted_by=str(note.get("submitted_by") or "").strip(),
            source="ingest_intel_note",
        )

        linked_entities = 0
        entity_specs: List[Dict[str, Any]] = []
        if primary_name:
            entity_specs.append(
                {
                    "name": primary_name,
                    "target_type": str(primary.get("target_type") or "").strip(),
                    "relationship": "note_primary",
                }
            )
        for item in payload.get("referenced_entities") or []:
            if not isinstance(item, dict):
                continue
            entity_specs.append(
                {
                    "name": str(item.get("name") or "").strip(),
                    "target_type": str(item.get("target_type") or "").strip(),
                    "relationship": "intel_reference",
                    "reference_type": str(item.get("reference_type") or "").strip(),
                    "confidence": str(item.get("confidence") or "").strip(),
                }
            )

        profile_by_name = {
            normalize_lookup(profile.get("canonical_name") or ""): profile
            for profile in self.list_profiles(org_name=org_name)
        }
        for entity in entity_specs:
            entity_name = str(entity.get("name") or "").strip()
            if not entity_name:
                continue
            lookup = normalize_lookup(entity_name)
            profile = profile_by_name.get(lookup)
            if profile:
                entity_node = _profile_graph_node_id(profile)
            else:
                target_type = str(entity.get("target_type") or "organization").strip().lower() or "organization"
                graph_kind = _target_type_graph_kind(target_type)
                entity_node = _stakeholder_graph_node_id(graph_kind, entity_name)
                manager.add_entity(
                    entity_node,
                    graph_kind,
                    name=entity_name,
                    target_type=target_type,
                    source="ingest_intel_note",
                )
            manager.add_relationship(
                note_node,
                entity_node,
                relationship=str(entity.get("relationship") or "intel_reference"),
                reference_type=str(entity.get("reference_type") or "").strip(),
                confidence=str(entity.get("confidence") or "").strip(),
            )
            linked_entities += 1

        self._save_graph_manager()
        return {"intel_id": intel_id, "linked_entities": linked_entities}

    def _upsert_profile_graph(self, profile: Dict[str, Any], subscriber_org: str) -> None:
        manager = self._load_graph_manager(create=True)
        if manager is None:
            return

        target_type = str(profile.get("target_type") or "").strip().lower()
        profile_kind = _target_type_graph_kind(target_type)
        profile_node = _stakeholder_graph_node_id(profile_kind, str(profile.get("profile_key") or profile.get("canonical_name") or ""))
        manager.add_entity(
            profile_node,
            profile_kind,
            name=str(profile.get("canonical_name") or "").strip(),
            profile_key=str(profile.get("profile_key") or "").strip(),
            org_name=str(profile.get("org_name") or "").strip(),
            target_type=str(profile.get("target_type") or "").strip(),
            watch_status=str(profile.get("watch_status") or "").strip(),
            source="stakeholder_profile_sync",
        )

        subscriber_node = _stakeholder_graph_node_id("subscriber_org", subscriber_org)
        manager.add_entity(
            subscriber_node,
            "subscriber_org",
            name=str(subscriber_org).strip(),
            org_name=str(subscriber_org).strip(),
            source="stakeholder_profile_sync",
        )
        manager.add_relationship(profile_node, subscriber_node, relationship="tracked_by")

        for affiliation in profile.get("affiliations") or []:
            if not isinstance(affiliation, dict):
                continue
            org_name_text = str(affiliation.get("org_name_text") or "").strip()
            if not org_name_text:
                continue
            org_node = _stakeholder_graph_node_id("organization", org_name_text)
            manager.add_entity(
                org_node,
                "organization",
                name=org_name_text,
                source="stakeholder_profile_sync",
            )
            relationship = "works_at" if str(affiliation.get("affiliation_type") or "") == "current" else "affiliated_with"
            manager.add_relationship(
                profile_node,
                org_node,
                relationship=relationship,
                role=str(affiliation.get("role") or "").strip(),
                affiliation_type=str(affiliation.get("affiliation_type") or "").strip(),
                confidence=str(affiliation.get("confidence") or "").strip(),
                is_primary=int(affiliation.get("is_primary") or 0),
            )

        for industry_affiliation in profile.get("industry_affiliations") or []:
            if not isinstance(industry_affiliation, dict):
                continue
            industry_key = str(industry_affiliation.get("industry_profile_key") or "").strip()
            industry_name = str(industry_affiliation.get("industry_name") or "").strip()
            if not industry_key and not industry_name:
                continue
            industry_node = _stakeholder_graph_node_id("industry", industry_key or industry_name)
            manager.add_entity(
                industry_node,
                "industry",
                name=industry_name or industry_key,
                profile_key=industry_key,
                target_type="industry",
                source="stakeholder_profile_sync",
            )
            manager.add_relationship(
                profile_node,
                industry_node,
                relationship="belongs_to_industry",
                role=str(industry_affiliation.get("role") or "").strip(),
                affiliation_type=str(industry_affiliation.get("affiliation_type") or "").strip(),
                source=str(industry_affiliation.get("source") or "").strip(),
            )

        if target_type == "industry":
            for theme in profile.get("key_themes") or []:
                theme_text = str(theme or "").strip()
                if not theme_text:
                    continue
                theme_node = _stakeholder_graph_node_id("industry_theme", theme_text)
                manager.add_entity(theme_node, "source", name=theme_text, source="stakeholder_profile_sync")
                manager.add_relationship(profile_node, theme_node, relationship="published_by")

        for alumni_name in profile.get("alumni") or []:
            alumni_text = str(alumni_name or "").strip()
            if not alumni_text:
                continue
            alumni_node = _stakeholder_graph_node_id("alumni", alumni_text)
            manager.add_entity(alumni_node, "alumni_group", name=alumni_text, source="stakeholder_profile_sync")
            manager.add_relationship(profile_node, alumni_node, relationship="alumni_of")

        for connection in profile.get("linkedin_connections") or []:
            if not isinstance(connection, dict):
                continue
            member = str(connection.get("member") or "").strip()
            if not member:
                continue
            member_node = _stakeholder_graph_node_id("member", member)
            manager.add_entity(member_node, "lab_member", name=member, email=member, source="stakeholder_profile_sync")
            manager.add_relationship(
                member_node,
                profile_node,
                relationship="linkedin_connection",
                degree=str(connection.get("degree") or "").strip(),
            )

    def _upsert_org_context_graph(self, org_name: str, org_alumni: List[str], org_strategic_profile: Optional[Dict[str, Any]] = None) -> None:
        manager = self._load_graph_manager(create=True)
        if manager is None:
            return
        subscriber_node = _stakeholder_graph_node_id("subscriber_org", org_name)
        manager.add_entity(
            subscriber_node,
            "subscriber_org",
            name=str(org_name).strip(),
            org_name=str(org_name).strip(),
            source="stakeholder_profile_sync",
        )
        for alumni_name in org_alumni or []:
            alumni_text = str(alumni_name or "").strip()
            if not alumni_text:
                continue
            alumni_node = _stakeholder_graph_node_id("alumni", alumni_text)
            manager.add_entity(alumni_node, "alumni_group", name=alumni_text, source="stakeholder_profile_sync")
            manager.add_relationship(subscriber_node, alumni_node, relationship="org_alumni_context")

        strategic_profile = _normalize_strategic_profile(org_strategic_profile)
        for industry_name in strategic_profile.get("industries") or []:
            industry_text = str(industry_name or "").strip()
            if not industry_text:
                continue
            industry_node = _stakeholder_graph_node_id("industry", industry_text)
            manager.add_entity(
                industry_node,
                "industry",
                name=industry_text,
                target_type="industry",
                source="stakeholder_profile_sync",
            )
            manager.add_relationship(subscriber_node, industry_node, relationship="strategic_focus")

        for theme_name in strategic_profile.get("key_themes") or []:
            theme_text = str(theme_name or "").strip()
            if not theme_text:
                continue
            theme_node = _stakeholder_graph_node_id("industry_theme", theme_text)
            manager.add_entity(theme_node, "source", name=theme_text, source="stakeholder_profile_sync")
            manager.add_relationship(subscriber_node, theme_node, relationship="strategic_theme")

    def _upsert_signal_graph(self, signal: Dict[str, Any]) -> None:
        manager = self._load_graph_manager(create=True)
        if manager is None:
            return

        signal_node = _stakeholder_graph_node_id("signal", str(signal.get("signal_id") or ""))
        manager.add_entity(
            signal_node,
            "stakeholder_signal",
            name=str(signal.get("subject") or signal.get("signal_id") or "").strip(),
            signal_id=str(signal.get("signal_id") or "").strip(),
            source_system=str(signal.get("source_system") or "").strip(),
            received_at=str(signal.get("received_at") or "").strip(),
            primary_url=str(signal.get("primary_url") or "").strip(),
        )

        for match in signal.get("matches") or []:
            profile_key = str(match.get("profile_key") or "").strip()
            canonical_name = str(match.get("canonical_name") or "").strip()
            if not profile_key and not canonical_name:
                continue
            match_kind = _target_type_graph_kind(str(match.get("target_type") or "").strip().lower())
            profile_node = _stakeholder_graph_node_id(match_kind, profile_key or canonical_name)
            manager.add_entity(
                profile_node,
                match_kind,
                name=canonical_name,
                profile_key=profile_key,
                source="signal_ingest",
            )
            manager.add_relationship(
                signal_node,
                profile_node,
                relationship="mentions_profile",
                confidence=float(match.get("score") or 0.0),
            )

        employer = str(signal.get("parsed_candidate_employer") or "").strip()
        if employer:
            employer_node = _stakeholder_graph_node_id("organization", employer)
            manager.add_entity(employer_node, "organization", name=employer, source="signal_ingest")
            manager.add_relationship(signal_node, employer_node, relationship="mentions_organization")

        for industry_name in signal.get("key_themes") or []:
            industry_text = str(industry_name or "").strip()
            if not industry_text:
                continue
            industry_node = _stakeholder_graph_node_id("industry_theme", industry_text)
            manager.add_entity(industry_node, "source", name=industry_text, source="signal_ingest")
            manager.add_relationship(signal_node, industry_node, relationship="published_by")

        source_name = str(signal.get("source_name") or "").strip()
        if source_name:
            source_node = _stakeholder_graph_node_id("source", source_name)
            manager.add_entity(source_node, "source", name=source_name, source="signal_ingest")
            manager.add_relationship(signal_node, source_node, relationship="published_by")

        if len(signal.get("matched_profile_keys") or []) > 1:
            kind_by_key = {
                str(match.get("profile_key") or "").strip(): _target_type_graph_kind(str(match.get("target_type") or "").strip().lower())
                for match in signal.get("matches") or []
                if str(match.get("profile_key") or "").strip()
            }
            matched = [str(item).strip() for item in signal.get("matched_profile_keys") or [] if str(item).strip()]
            for idx, left in enumerate(matched):
                for right in matched[idx + 1:]:
                    left_node = _stakeholder_graph_node_id(kind_by_key.get(left, "entity"), left)
                    right_node = _stakeholder_graph_node_id(kind_by_key.get(right, "entity"), right)
                    manager.add_relationship(left_node, right_node, relationship="co_mentioned")

    def rebuild_stakeholder_graph(self) -> Dict[str, Any]:
        state = self._read_state()
        manager = self._load_graph_manager(create=True)
        if manager is None:
            return {"profiles_processed": 0, "signals_processed": 0, "org_contexts_processed": 0}

        profiles_processed = 0
        signals_processed = 0
        org_contexts_processed = 0

        for org_context in (state.get("org_contexts") or {}).values():
            if not isinstance(org_context, dict):
                continue
            org_name = str(org_context.get("org_name") or "").strip()
            if not org_name:
                continue
            self._upsert_org_context_graph(
                org_name,
                _normalize_org_name_list(org_context.get("org_alumni") or []),
                _normalize_strategic_profile(org_context.get("org_strategic_profile") or {}),
            )
            org_contexts_processed += 1

        for profile in state.get("profiles") or []:
            if not isinstance(profile, dict):
                continue
            org_name = str(profile.get("org_name") or "").strip()
            if not org_name:
                continue
            self._upsert_profile_graph(profile, subscriber_org=org_name)
            profiles_processed += 1

        for signal in state.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            self._upsert_signal_graph(signal)
            signals_processed += 1

        self._save_graph_manager()
        return {
            "profiles_processed": profiles_processed,
            "signals_processed": signals_processed,
            "org_contexts_processed": org_contexts_processed,
            "graph_nodes": manager.graph.number_of_nodes(),
            "graph_edges": manager.graph.number_of_edges(),
        }

    def save_profile(self, profile_key: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wanted = str(profile_key).strip()
        if not wanted:
            return None
        state = self._read_state()
        profiles = list(state.get("profiles") or [])
        updated_profile: Optional[Dict[str, Any]] = None
        for profile in profiles:
            if profile.get("profile_key") != wanted:
                continue
            for field_name, field_value in updates.items():
                if field_name == "external_profile_id":
                    next_value = str(field_value or "").strip()
                    if not next_value:
                        continue
                    profile[field_name] = next_value
                    continue
                if field_name == "address":
                    profile[field_name] = _normalize_address(field_value)
                    continue
                if field_name == "status":
                    profile[field_name] = _normalize_status(field_value)
                    continue
                if field_name == "watch_status":
                    profile[field_name] = _normalize_watch_status(field_value)
                    continue
                profile[field_name] = field_value
            profile["updated_at"] = _utc_now_iso()
            updated_profile = dict(profile)
            break
        if updated_profile is None:
            return None
        state["profiles"] = profiles
        self._write_state(state)
        return updated_profile

    def delete_profile(self, profile_key: str) -> bool:
        wanted = str(profile_key).strip()
        if not wanted:
            return False
        state = self._read_state()
        profiles = [profile for profile in state.get("profiles") or [] if profile.get("profile_key") != wanted]
        if len(profiles) == len(state.get("profiles") or []):
            return False

        suggestions = [
            item for item in state.get("update_suggestions") or []
            if item.get("profile_key") != wanted
        ]
        facts = [
            item for item in state.get("observed_facts") or []
            if item.get("profile_key") != wanted
        ]
        state["profiles"] = profiles
        state["update_suggestions"] = suggestions
        state["observed_facts"] = facts
        self._write_state(state)
        return True

    def delete_signal(self, signal_id: str) -> bool:
        wanted = str(signal_id).strip()
        if not wanted:
            return False
        state = self._read_state()
        signals = list(state.get("signals") or [])
        target_signal = next((signal for signal in signals if signal.get("signal_id") == wanted), None)
        if target_signal is None:
            return False

        state["signals"] = [signal for signal in signals if signal.get("signal_id") != wanted]
        fact_ids = set(target_signal.get("observed_fact_ids") or [])
        suggestion_ids = set(target_signal.get("update_suggestion_ids") or [])
        state["observed_facts"] = [
            item for item in state.get("observed_facts") or []
            if item.get("fact_id") not in fact_ids and item.get("signal_id") != wanted
        ]
        state["update_suggestions"] = [
            item for item in state.get("update_suggestions") or []
            if item.get("suggestion_id") not in suggestion_ids and item.get("signal_id") != wanted
        ]
        self._write_state(state)
        return True

    def review_update_suggestion(
        self,
        suggestion_id: str,
        status: str,
        reviewed_by: str = "streamlit",
        apply_to_profile: bool = False,
    ) -> Optional[Dict[str, Any]]:
        wanted = str(suggestion_id).strip()
        if not wanted:
            return None
        next_status = str(status).strip().lower()
        if next_status not in {"accepted", "rejected", "pending"}:
            raise ValueError(f"Unsupported suggestion status: {status!r}")

        state = self._read_state()
        suggestions = list(state.get("update_suggestions") or [])
        updated_item: Optional[Dict[str, Any]] = None
        for item in suggestions:
            if item.get("suggestion_id") != wanted:
                continue
            item["status"] = next_status
            item["reviewed_at"] = _utc_now_iso()
            item["reviewed_by"] = reviewed_by
            updated_item = item
            break
        if updated_item is None:
            return None
        if next_status == "accepted" and apply_to_profile and updated_item.get("profile_key"):
            for profile in state.get("profiles") or []:
                if profile.get("profile_key") != updated_item.get("profile_key"):
                    continue
                profile[updated_item["field_name"]] = updated_item.get("proposed_value", "")
                if updated_item["field_name"] == "current_employer":
                    employers = [str(item).strip() for item in profile.get("known_employers") or [] if str(item).strip()]
                    new_employer = str(updated_item.get("proposed_value") or "").strip()
                    if new_employer and new_employer not in employers:
                        employers.append(new_employer)
                    profile["known_employers"] = employers
                    profile["affiliations"] = _upsert_primary_affiliation(
                        profile.get("affiliations") or [],
                        org_name_text=new_employer,
                        role=str(profile.get("current_role") or "").strip(),
                        source="accepted_update",
                    )
                elif updated_item["field_name"] == "current_role":
                    primary = _primary_affiliation(_normalize_affiliations(profile.get("affiliations") or []))
                    if primary and str(profile.get("current_employer") or "").strip():
                        profile["affiliations"] = _upsert_primary_affiliation(
                            profile.get("affiliations") or [],
                            org_name_text=str(profile.get("current_employer") or "").strip(),
                            role=str(updated_item.get("proposed_value") or "").strip(),
                            source="accepted_update",
                        )
                profile["updated_at"] = _utc_now_iso()
                break
        state["update_suggestions"] = suggestions
        self._write_state(state)
        return updated_item

    def upsert_profiles(
        self,
        org_name: str,
        profiles: List[Dict[str, Any]],
        org_alumni: Optional[List[str]] = None,
        org_strategic_profile: Optional[Dict[str, Any]] = None,
        source: str = "",
        trace_id: str = "",
        replace_org_scope: bool = False,
    ) -> Dict[str, Any]:
        state = self._read_state()
        existing_profiles = list(state.get("profiles") or [])
        by_profile_key = {
            profile["profile_key"]: profile for profile in existing_profiles if profile.get("profile_key")
        }
        by_external_profile_id = {
            str(profile.get("external_profile_id") or "").strip(): profile
            for profile in existing_profiles
            if str(profile.get("external_profile_id") or "").strip()
        }
        added = 0
        updated = 0
        synced_profile_keys: set[str] = set()

        for raw_profile in profiles:
            profile = self._normalize_profile(raw_profile, org_name=org_name, source=source, trace_id=trace_id)
            external_profile_id = str(profile.get("external_profile_id") or "").strip()
            current = None
            if external_profile_id:
                current = by_external_profile_id.get(external_profile_id)
            if current is None:
                current = by_profile_key.get(profile["profile_key"])
            if current is None:
                current = self._match_legacy_profile(existing_profiles, profile)
            if current:
                profile["profile_key"] = current.get("profile_key", profile["profile_key"])
                profile["created_at"] = current.get("created_at", profile["created_at"])
                if not external_profile_id:
                    profile["external_profile_id"] = str(current.get("external_profile_id") or "").strip()
                updated += 1
            else:
                added += 1
            by_profile_key[profile["profile_key"]] = profile
            synced_profile_keys.add(str(profile["profile_key"]))
            if str(profile.get("external_profile_id") or "").strip():
                by_external_profile_id[str(profile["external_profile_id"]).strip()] = profile
            self._upsert_profile_graph(profile, subscriber_org=org_name)

        if replace_org_scope:
            retained_keys = set(by_profile_key.keys())
            for existing in existing_profiles:
                same_org = orgs_compatible(existing.get("org_name", ""), org_name)
                same_source = str(existing.get("source") or "").strip() == str(source or "").strip()
                if same_org and same_source and existing.get("profile_key") not in synced_profile_keys:
                    retained_keys.discard(existing.get("profile_key"))
            state["profiles"] = sorted(
                [profile for key, profile in by_profile_key.items() if key in retained_keys],
                key=lambda item: (item.get("org_name", ""), item.get("canonical_name", "")),
            )
        else:
            state["profiles"] = sorted(
                by_profile_key.values(),
                key=lambda item: (item.get("org_name", ""), item.get("canonical_name", "")),
            )
        context_key = normalize_lookup(org_name)
        org_contexts = dict(state.get("org_contexts") or {})
        existing_context = dict(org_contexts.get(context_key) or {})
        strategic_profile = _normalize_strategic_profile(
            org_strategic_profile if org_strategic_profile is not None else existing_context.get("org_strategic_profile") or {}
        )
        org_contexts[context_key] = {
            "org_name": org_name,
            "org_alumni": _normalize_org_name_list(org_alumni if org_alumni is not None else existing_context.get("org_alumni") or []),
            "org_strategic_profile": strategic_profile,
            "source": str(source or existing_context.get("source") or "").strip(),
            "trace_id": str(trace_id or existing_context.get("trace_id") or "").strip(),
            "updated_at": _utc_now_iso(),
        }
        state["org_contexts"] = org_contexts
        self._upsert_org_context_graph(
            org_name,
            state["org_contexts"][context_key].get("org_alumni") or [],
            state["org_contexts"][context_key].get("org_strategic_profile") or {},
        )
        self._write_state(state)
        self._save_graph_manager()
        return {
            "org_name": org_name,
            "profile_count": len(state["profiles"]),
            "added": added,
            "updated": updated,
            "org_alumni_count": len(state["org_contexts"][context_key].get("org_alumni") or []),
            "org_strategic_industry_count": len(state["org_contexts"][context_key].get("org_strategic_profile", {}).get("industries") or []),
        }

    @staticmethod
    def _match_legacy_profile(existing_profiles: List[Dict[str, Any]], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        target_type = str(profile.get("target_type") or "").strip().lower()
        canonical_name = normalize_lookup(profile.get("canonical_name") or "")
        linkedin_url = normalize_lookup(profile.get("linkedin_url") or "")
        website_url = normalize_lookup(profile.get("website_url") or "")
        current_employer = normalize_lookup(profile.get("current_employer") or "")
        candidates: List[tuple[int, Dict[str, Any]]] = []

        for existing in existing_profiles:
            if str(existing.get("target_type") or "").strip().lower() != target_type:
                continue
            score = 0
            if linkedin_url and normalize_lookup(existing.get("linkedin_url") or "") == linkedin_url:
                score += 4
            if website_url and normalize_lookup(existing.get("website_url") or "") == website_url:
                score += 4
            if canonical_name and normalize_lookup(existing.get("canonical_name") or "") == canonical_name:
                score += 3
            if current_employer and normalize_lookup(existing.get("current_employer") or "") == current_employer:
                score += 1
            if score > 0:
                candidates.append((score, existing))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1].get("updated_at", "")), reverse=True)
        return candidates[0][1]

    def _watch_signal_payload(self, payload: Dict[str, Any], item: Dict[str, Any], index: int) -> Dict[str, Any]:
        target_name = str(item.get("target_name") or item.get("name") or item.get("target") or "").strip()
        target_type = str(item.get("target_type") or item.get("type") or "person").strip().lower() or "person"
        employer = str(item.get("current_employer") or item.get("employer") or "").strip()
        headline = str(item.get("headline") or item.get("subject") or target_name or f"watch-signal-{index+1}").strip()
        snippet = str(item.get("snippet") or item.get("summary") or item.get("text_note") or item.get("raw_text") or "").strip()
        url = str(item.get("url") or item.get("primary_url") or item.get("source_url") or "").strip()
        received_at = str(item.get("date") or item.get("received_at") or _utc_now_iso()).strip()
        source_name = str(item.get("source_name") or item.get("publication") or "").strip()
        message_id = str(item.get("message_id") or f"{payload.get('source_job') or payload.get('trace_id') or 'watch'}:{index}:{headline[:64]}").strip()
        return {
            "org_name": payload["org_name"],
            "trace_id": str(payload.get("trace_id") or "").strip(),
            "source_system": str(payload.get("source_system") or payload.get("source") or "market_radar_watch").strip() or "market_radar_watch",
            "source_job": str(payload.get("source_job") or "").strip(),
            "source_org_name": str(item.get("source_org_name") or payload.get("source_org_name") or payload["org_name"]).strip(),
            "visible_to_orgs": item.get("visible_to_orgs") or payload.get("visible_to_orgs") or [],
            "shared_with_orgs": item.get("shared_with_orgs") or payload.get("shared_with_orgs") or [],
            "scope_profile_key": str(item.get("scope_profile_key") or payload.get("scope_profile_key") or "").strip(),
            "child_profile_keys": list(item.get("child_profile_keys") or payload.get("child_profile_keys") or []),
            "child_org_names": list(item.get("child_org_names") or payload.get("child_org_names") or []),
            "key_themes": list(item.get("key_themes") or payload.get("key_themes") or []),
            "regulatory_context": str(item.get("regulatory_context") or payload.get("regulatory_context") or "").strip(),
            "market_size": str(item.get("market_size") or payload.get("market_size") or "").strip(),
            "signal_type": "watch_report_signal",
            "target_type": target_type,
            "target_name": target_name,
            "parsed_candidate_name": target_name if target_type == "person" else target_name,
            "parsed_candidate_employer": employer,
            "subject": headline,
            "raw_text": snippet,
            "text_note": snippet,
            "primary_url": url,
            "received_at": received_at,
            "message_id": message_id,
            "submitted_by": str(payload.get("submitted_by") or "").strip(),
            "source_name": source_name,
            "source_type": str(item.get("source_type") or "").strip().lower(),
            "confidence_hint": str(item.get("confidence_hint") or "").strip().lower(),
            "tags": list(item.get("tags") or []),
            "watch_report_meta": dict(payload.get("watch_report_meta") or {}) if isinstance(payload.get("watch_report_meta"), dict) else {},
        }

    def ingest_watch_signals(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        signal_ids: List[str] = []
        matched_signal_count = 0
        touched_profiles: set[str] = set()
        for index, item in enumerate(payload.get("signals") or []):
            signal = self.ingest_signal(self._watch_signal_payload(payload, item, index))
            signal_ids.append(signal["signal_id"])
            if signal.get("matched_profile_keys"):
                matched_signal_count += 1
                touched_profiles.update(str(key).strip() for key in signal.get("matched_profile_keys") or [] if str(key).strip())
        return {
            "signal_count": len(payload.get("signals") or []),
            "ingested_count": len(signal_ids),
            "matched_signal_count": matched_signal_count,
            "profiles_touched": sorted(touched_profiles),
            "signal_ids": signal_ids,
        }

    def ingest_signal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._read_state()
        signal = self._normalize_signal(payload)
        signals = list(state.get("signals") or [])

        for existing in signals:
            if existing.get("signal_hash") == signal["signal_hash"]:
                return existing

        profiles = [
            profile for profile in state.get("profiles") or []
            if orgs_compatible(profile.get("org_name", ""), signal.get("org_name", ""))
        ]
        signal["matches"] = match_signal_to_profiles(signal, profiles)
        signal["matched_profile_keys"] = [match["profile_key"] for match in signal["matches"]]
        signal["needs_review"] = bool(signal["matches"]) and signal["matches"][0]["score"] < 0.75
        signal["raw_file"] = str(self._write_raw_signal_markdown(signal))

        artifacts = detect_profile_change_artifacts(signal, signal["matches"])
        signal["observed_fact_ids"] = [item["fact_id"] for item in artifacts["observed_facts"]]
        signal["update_suggestion_ids"] = [item["suggestion_id"] for item in artifacts["update_suggestions"]]

        signals.append(signal)
        state["signals"] = sorted(signals, key=lambda item: item.get("received_at", ""), reverse=True)
        state["observed_facts"] = self._merge_unique_items(
            state.get("observed_facts") or [],
            artifacts["observed_facts"],
            key_field="fact_id",
            extra_fields={"org_name": signal["org_name"], "created_at": _utc_now_iso()},
        )
        state["update_suggestions"] = self._merge_unique_items(
            state.get("update_suggestions") or [],
            artifacts["update_suggestions"],
            key_field="suggestion_id",
            extra_fields={"org_name": signal["org_name"], "created_at": _utc_now_iso()},
        )
        self._write_state(state)
        self._upsert_signal_graph(signal)
        self._save_graph_manager()
        return signal

    def generate_digest(
        self,
        org_name: str,
        since_ts: str = "",
        scope_type: str = "org",
        scope_profile_key: str = "",
        profile_keys: Optional[List[str]] = None,
        child_profile_keys: Optional[List[str]] = None,
        child_org_names: Optional[List[str]] = None,
        key_themes: Optional[List[str]] = None,
        regulatory_context: str = "",
        market_size: str = "",
        shared_with_orgs: Optional[List[str]] = None,
        member_alumni: Optional[List[str]] = None,
        org_alumni: Optional[List[str]] = None,
        report_depth: str = "detailed",
        digest_tier: str = "standard",
        priority_profile_keys: Optional[List[str]] = None,
        deep_analysis: bool = False,
        max_items: int = 25,
        include_needs_review: bool = True,
        matched_only: bool = True,
        llm_synthesis: bool = False,
        llm_provider: str = "ollama",
        llm_model: str = "",
    ) -> Dict[str, Any]:
        state = self._read_state()
        scope_type = str(scope_type or "org").strip().lower() or "org"
        if scope_type not in {"org", "industry"}:
            scope_type = "org"
        scope_profile_key = str(scope_profile_key or "").strip()
        child_profile_keys = [str(item).strip() for item in child_profile_keys or [] if str(item).strip()]
        child_org_names = _normalize_org_name_list(child_org_names or [])
        key_themes = _normalize_org_name_list(key_themes or [])
        shared_with_orgs = _normalize_org_name_list(shared_with_orgs or [])
        industry_profile = self.get_profile(scope_profile_key) if scope_type == "industry" and scope_profile_key else None
        if industry_profile:
            key_themes = _normalize_org_name_list([*key_themes, *(industry_profile.get("key_themes") or [])])
            regulatory_context = str(regulatory_context or industry_profile.get("regulatory_context") or "").strip()
            market_size = str(market_size or industry_profile.get("market_size") or "").strip()

        stored_org_alumni = self.get_org_context(org_name).get("org_alumni") or []
        member_alumni = _normalize_org_name_list(member_alumni)
        org_alumni = _normalize_org_name_list(org_alumni if org_alumni is not None else stored_org_alumni)
        alumni_context = _normalize_org_name_list([*member_alumni, *org_alumni])

        signals = self.list_signals(org_name=org_name, matched_only=False, limit=2000)
        since_dt = _parse_timestamp(since_ts)
        if since_ts:
            signals = [
                signal for signal in signals
                if (
                    (_parse_timestamp(signal.get("received_at", "")) or datetime.min.replace(tzinfo=timezone.utc))
                    >= (since_dt or datetime.min.replace(tzinfo=timezone.utc))
                )
            ]
        enriched_signals: List[Dict[str, Any]] = []
        for signal in signals:
            enriched = dict(signal)
            enriched["alumni_hits"] = _find_alumni_hits(enriched, alumni_context)
            top_match = (enriched.get("matches") or [{}])[0]
            enriched["profile_alumni_links"] = _find_profile_alumni_links(top_match, alumni_context) if top_match else []
            enriched_signals.append(enriched)
        signals = enriched_signals
        scope_profile_keys = [str(item).strip() for item in profile_keys or [] if str(item).strip()]
        if scope_type == "industry":
            scope_profile_keys = _normalize_org_name_list([*scope_profile_keys, *child_profile_keys])  # type: ignore[arg-type]
            scope_profile_keys = [item for item in scope_profile_keys if item]
            if scope_profile_key and scope_profile_key not in scope_profile_keys:
                scope_profile_keys.append(scope_profile_key)
        if scope_profile_keys:
            wanted = set(scope_profile_keys)
            signals = [
                signal
                for signal in signals
                if wanted.intersection(set(signal.get("matched_profile_keys") or []))
                or (
                    scope_type == "industry"
                    and (
                        any(_text_contains_org_hint(" ".join([signal.get("subject", ""), signal.get("raw_text", ""), signal.get("text_note", "")]), item) for item in child_org_names)
                        or any(_text_contains_org_hint(" ".join([signal.get("subject", ""), signal.get("raw_text", ""), signal.get("text_note", "")]), item) for item in key_themes)
                        or str(signal.get("scope_profile_key") or "").strip() == scope_profile_key
                    )
                )
            ]
        if matched_only:
            signals = [
                signal
                for signal in signals
                if signal.get("matches")
                or signal.get("alumni_hits")
                or (
                    scope_type == "industry"
                    and (
                        str(signal.get("scope_profile_key") or "").strip() == scope_profile_key
                        or any(_text_contains_org_hint(" ".join([signal.get("subject", ""), signal.get("raw_text", ""), signal.get("text_note", "")]), item) for item in child_org_names)
                        or any(_text_contains_org_hint(" ".join([signal.get("subject", ""), signal.get("raw_text", ""), signal.get("text_note", "")]), item) for item in key_themes)
                    )
                )
            ]
        if not include_needs_review:
            signals = [signal for signal in signals if not signal.get("needs_review")]
        signals = [
            signal
            for signal in signals
            if _should_include_signal_in_digest(signal, report_depth)
            or (
                scope_type == "industry"
                and (
                    str(signal.get("scope_profile_key") or "").strip() == scope_profile_key
                    or any(_text_contains_org_hint(" ".join([signal.get("subject", ""), signal.get("raw_text", ""), signal.get("text_note", "")]), item) for item in child_org_names)
                    or any(_text_contains_org_hint(" ".join([signal.get("subject", ""), signal.get("raw_text", ""), signal.get("text_note", "")]), item) for item in key_themes)
                )
            )
        ]
        selected_profile_keys = scope_profile_keys
        signals, intelligence_context = self._enrich_digest_signals(
            signals,
            state=state,
            org_name=org_name,
            profile_keys=selected_profile_keys,
            alumni_context=alumni_context,
            since_ts=since_ts,
        )
        signals = sorted(
            signals,
            key=lambda item: (
                float(item.get("intelligence_score") or 0.0),
                float(item.get("confidence_score") or 0.0),
                2 if item.get("matches") else 0,
                1 if item.get("alumni_hits") else 0,
                item.get("received_at", ""),
            ),
            reverse=True,
        )

        signals = signals[:max_items]
        intelligence_context["scoring_lines"] = self._build_signal_priority_summary(signals)
        generated_at = _utc_now_iso()
        digest_id = f"digest_{hashlib.sha1(f'{org_name}|{since_ts}|{len(signals)}|{generated_at}'.encode('utf-8')).hexdigest()[:16]}"
        digest_path = self.root / "digests"
        digest_path.mkdir(parents=True, exist_ok=True)
        output_path = digest_path / f"{digest_id}.md"
        profiles_covered = len(
            {
                str(profile_key).strip()
                for signal in signals
                for profile_key in signal.get("matched_profile_keys") or []
                if str(profile_key).strip()
            }
        )
        period_start = since_ts or (signals[-1].get("received_at", "") if signals else "")
        period_end = generated_at
        llm_synthesised = False
        actual_llm_provider = str(llm_provider or "ollama").strip().lower() or "ollama"
        actual_llm_model = str(llm_model or "").strip()
        report_depth = str(report_depth or "detailed").strip().lower() or "detailed"
        digest_tier = str(digest_tier or "standard").strip().lower() or "standard"
        priority_profile_keys = [str(item).strip() for item in priority_profile_keys or [] if str(item).strip()]
        if report_depth not in {"summary", "detailed", "strategic"}:
            report_depth = "detailed"
        if digest_tier not in {"priority", "standard"}:
            digest_tier = "standard"

        body_lines = self._build_mechanical_digest_lines(
            signals,
            report_depth=report_depth,
            since_ts=since_ts,
            generated_at=generated_at,
            alumni_context=alumni_context,
            intelligence_context=intelligence_context,
            scope_type=scope_type,
            scope_profile=industry_profile or {},
            child_org_names=child_org_names,
            key_themes=key_themes,
            regulatory_context=regulatory_context,
            market_size=market_size,
        )
        if llm_synthesis and signals:
            raw_data = self._prepare_digest_data(
                signals,
                state=state,
                org_name=org_name,
                scope_type=scope_type,
                scope_profile=industry_profile or {},
                child_org_names=child_org_names,
                key_themes=key_themes,
                regulatory_context=regulatory_context,
                market_size=market_size,
                member_alumni=member_alumni,
                org_alumni=org_alumni,
                intelligence_context=intelligence_context,
                report_depth=report_depth,
                digest_tier=digest_tier,
                priority_profile_keys=priority_profile_keys,
                deep_analysis=deep_analysis,
            )
            try:
                synthesised, actual_llm_provider, actual_llm_model = self._llm_synthesise(
                    raw_data=raw_data,
                    org_name=org_name,
                    scope_type=scope_type,
                    scope_profile_name=str((industry_profile or {}).get("canonical_name") or "").strip(),
                    provider=llm_provider,
                    model=llm_model,
                    report_depth=report_depth,
                )
            except TypeError:
                synthesised, actual_llm_provider, actual_llm_model = self._llm_synthesise(
                    raw_data,
                    org_name,
                    llm_provider,
                    llm_model,
                    report_depth,
                )
            if synthesised:
                body_lines = _strip_empty_digest_sections(synthesised).splitlines()
                llm_synthesised = True

        escalate = False
        escalate_profiles: List[str] = []
        escalate_reason = ""
        if report_depth == "summary":
            escalate, escalate_profiles, escalate_reason = self._detect_digest_escalation(signals)

        lines = [
            "---",
            f"digest_id: {digest_id}",
            f"org_name: {org_name}",
            f"generated_at: {generated_at}",
            f"since_ts: {since_ts}",
            f"scope_type: {scope_type}",
            f"scope_profile_key: {scope_profile_key}",
            f"signal_count: {len(signals)}",
            f"report_depth: {report_depth}",
            f"digest_tier: {digest_tier}",
            f"deep_analysis: {str(bool(deep_analysis)).lower()}",
            f"member_alumni_count: {len(member_alumni)}",
            f"org_alumni_count: {len(org_alumni)}",
            f"llm_synthesised: {str(llm_synthesised).lower()}",
            f"llm_provider: {actual_llm_provider}",
            f"llm_model: {actual_llm_model}",
            "---",
            "",
            "# Strategic Intelligence Brief" if report_depth == "strategic" else ("# Industry Intelligence Digest" if scope_type == "industry" else "# Stakeholder Intelligence Digest"),
            "",
            f"Organisation: {org_name}",
            f"Generated: {generated_at}",
            "",
        ]
        if scope_type == "industry":
            lines.extend([
                f"Scope: {str((industry_profile or {}).get('canonical_name') or scope_profile_key or 'Industry').strip()}",
                "",
            ])
        if alumni_context:
            lines.extend([f"Alumni context: {', '.join(alumni_context)}", ""])
        lines.extend(body_lines)

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "digest_id": digest_id,
            "org_name": org_name,
            "signal_count": len(signals),
            "signals": signals,
            "output_path": str(output_path),
            "scope_type": scope_type,
            "scope_profile_key": scope_profile_key,
            "child_profile_keys": child_profile_keys,
            "child_org_names": child_org_names,
            "key_themes": key_themes,
            "regulatory_context": regulatory_context,
            "market_size": market_size,
            "shared_with_orgs": shared_with_orgs,
            "llm_synthesised": llm_synthesised,
            "profiles_covered": profiles_covered,
            "member_alumni": member_alumni,
            "org_alumni": org_alumni,
            "period_start": period_start,
            "period_end": period_end,
            "report_depth": report_depth,
            "digest_tier": digest_tier,
            "deep_analysis": bool(deep_analysis),
            "escalate": escalate,
            "escalate_profiles": escalate_profiles,
            "escalate_reason": escalate_reason,
            "llm_provider": actual_llm_provider,
            "llm_model": actual_llm_model,
            "relationship_intelligence_count": sum(
                len(items)
                for items in intelligence_context.get("relationship_context", {}).values()
            ),
            "temporal_pattern_count": len(intelligence_context.get("temporal_lines") or []),
            "top_signal_scores": [
                {
                    "signal_id": signal.get("signal_id", ""),
                    "intelligence_score": signal.get("intelligence_score", 0),
                    "confidence_score": signal.get("confidence_score", 0),
                    "confidence_band": signal.get("confidence_band", ""),
                }
                for signal in signals[:5]
            ],
        }

    def _build_mechanical_digest_lines(
        self,
        signals: List[Dict[str, Any]],
        report_depth: str,
        since_ts: str,
        generated_at: str,
        alumni_context: List[str],
        intelligence_context: Dict[str, Any],
        scope_type: str = "org",
        scope_profile: Optional[Dict[str, Any]] = None,
        child_org_names: Optional[List[str]] = None,
        key_themes: Optional[List[str]] = None,
        regulatory_context: str = "",
        market_size: str = "",
    ) -> List[str]:
        if not signals:
            return ["No matching signals for this window."]

        if scope_type == "industry":
            return self._build_industry_digest_lines(
                signals,
                report_depth=report_depth,
                since_ts=since_ts,
                generated_at=generated_at,
                intelligence_context=intelligence_context,
                scope_profile=scope_profile or {},
                child_org_names=child_org_names or [],
                key_themes=key_themes or [],
                regulatory_context=regulatory_context,
                market_size=market_size,
            )

        if report_depth == "summary":
            return self._build_summary_digest_lines(
                signals,
                since_ts=since_ts,
                generated_at=generated_at,
                alumni_context=alumni_context,
                intelligence_context=intelligence_context,
            )
        if report_depth == "strategic":
            return self._build_strategic_digest_lines(signals, alumni_context=alumni_context, intelligence_context=intelligence_context)
        return self._build_detailed_digest_lines(signals, alumni_context=alumni_context, intelligence_context=intelligence_context)

    def _build_industry_digest_lines(
        self,
        signals: List[Dict[str, Any]],
        report_depth: str,
        since_ts: str,
        generated_at: str,
        intelligence_context: Dict[str, Any],
        scope_profile: Dict[str, Any],
        child_org_names: List[str],
        key_themes: List[str],
        regulatory_context: str,
        market_size: str,
    ) -> List[str]:
        lines: List[str] = [
            "## Sector Overview",
            f"**Period:** {since_ts or 'All available'} -> {generated_at}",
            f"**Signals:** {len(signals)}",
        ]
        if child_org_names:
            lines.append(f"**Child organisations:** {', '.join(child_org_names[:8])}")
        if key_themes:
            lines.append(f"**Key themes:** {', '.join(key_themes[:8])}")
        if regulatory_context:
            lines.append(f"**Regulatory context:** {regulatory_context}")
        if market_size:
            lines.append(f"**Market size:** {market_size}")

        org_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        stakeholder_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for signal in signals:
            top_match = (signal.get("matches") or [{}])[0]
            stakeholder_name = str(top_match.get("canonical_name") or signal.get("parsed_candidate_name") or "").strip()
            employer = str(top_match.get("current_employer") or signal.get("parsed_candidate_employer") or "").strip()
            if employer:
                org_buckets[employer].append(signal)
            if stakeholder_name:
                stakeholder_buckets[stakeholder_name].append(signal)

        lines.extend(["", "## Key Organisation Updates"])
        for org_name, items in sorted(org_buckets.items(), key=lambda item: (-len(item[1]), item[0].lower()))[:6]:
            lines.append(f"### {org_name}")
            for signal in items[: min(3, 2 if report_depth == 'summary' else 3)]:
                lines.append(
                    f"- **{signal.get('subject', signal.get('signal_id', 'signal'))}** "
                    f"[{self._classify_signal_category(signal)} | conf {signal.get('confidence_score', 0):.1f}/10]"
                )

        lines.extend(["", "## Stakeholder Movements"])
        for name, items in sorted(stakeholder_buckets.items(), key=lambda item: (-len(item[1]), item[0].lower()))[:6]:
            trend = self._summarise_signal_trend(items)
            lines.append(f"- **{name}**: {trend}")

        regulatory_signals = [signal for signal in signals if self._classify_signal_category(signal) == "Regulatory"]
        if regulatory_signals or regulatory_context:
            lines.extend(["", "## Regulatory & Policy Changes"])
            if regulatory_context:
                lines.append(f"- Sector backdrop: {regulatory_context}")
            for signal in regulatory_signals[:5]:
                lines.append(f"- **{signal.get('subject', signal.get('signal_id', 'signal'))}**")

        if key_themes:
            lines.extend(["", "## Emerging Themes & Opportunities"])
            for theme in key_themes[:6]:
                theme_hits = [
                    signal for signal in signals
                    if _text_contains_org_hint(" ".join([signal.get("subject", ""), signal.get("raw_text", ""), signal.get("text_note", "")]), theme)
                ]
                if theme_hits:
                    lines.append(f"- **{theme}** appears in {len(theme_hits)} current-window signal(s)")

        relationship_context = intelligence_context.get("relationship_context", {})
        if any(relationship_context.get(key) for key in ("network_lines", "cross_target_lines", "talent_flow_lines")):
            lines.extend(["", "## Relationship Intelligence"])
            for key in ("network_lines", "cross_target_lines", "talent_flow_lines"):
                for item in relationship_context.get(key) or []:
                    lines.append(f"- {item}")

        if intelligence_context.get("temporal_lines"):
            lines.extend(["", "## Temporal Patterns"])
            lines.extend(f"- {item}" for item in intelligence_context.get("temporal_lines") or [])

        if report_depth == "strategic":
            lines.extend([
                "",
                "## Strategic Implications",
                f"- {self._summarise_signal_trend(signals)}",
                f"- Industry focus: {str(scope_profile.get('description') or 'Sector-level activity is consolidating around the highest-signal organisations and people.').strip()}",
            ])
        return lines

    def _build_summary_digest_lines(
        self,
        signals: List[Dict[str, Any]],
        since_ts: str,
        generated_at: str,
        alumni_context: List[str],
        intelligence_context: Dict[str, Any],
    ) -> List[str]:
        lines = [
            f"**Period:** {since_ts or 'All available'} -> {generated_at}",
            f"**Profiles:** {len({key for signal in signals for key in signal.get('matched_profile_keys') or [] if key})} | **Signals:** {len(signals)}",
            "",
            "## Key Signals",
        ]
        if alumni_context:
            lines.append(f"_Alumni watchlist:_ {', '.join(alumni_context)}")
        for signal in signals:
            top_match = (signal.get("matches") or [{}])[0]
            stakeholder_name = top_match.get("canonical_name") or signal.get("parsed_candidate_name") or signal.get("signal_id")
            summary_text = str(signal.get("subject") or signal.get("text_note") or signal.get("raw_text") or "").strip()
            suffix = ""
            if signal.get("alumni_hits"):
                suffix = f" [alumni: {', '.join(signal.get('alumni_hits') or [])}]"
            lines.append(
                f"- **{stakeholder_name}**: {' '.join(summary_text.split())[:180]}{suffix} "
                f"[conf {signal.get('confidence_band', 'Low')} {signal.get('confidence_score', 0):.1f}/10 | action {signal.get('intelligence_score', 0):.1f}/10]"
            )
        if intelligence_context.get("temporal_lines"):
            lines.extend(["", "## Temporal Patterns"])
            lines.extend(f"- {item}" for item in intelligence_context.get("temporal_lines") or [])
        if intelligence_context.get("scoring_lines"):
            lines.extend(["", "## Priority Signals"])
            lines.extend(intelligence_context.get("scoring_lines") or [])
        return lines

    def _group_signals_by_stakeholder(self, signals: List[Dict[str, Any]]) -> List[tuple[str, List[Dict[str, Any]]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for signal in signals:
            top_match = (signal.get("matches") or [{}])[0]
            name = str(top_match.get("canonical_name") or signal.get("parsed_candidate_name") or "Unmatched").strip() or "Unmatched"
            grouped.setdefault(name, []).append(signal)
        return sorted(grouped.items(), key=lambda item: item[0].lower())

    def _build_detailed_digest_lines(
        self,
        signals: List[Dict[str, Any]],
        alumni_context: List[str],
        intelligence_context: Dict[str, Any],
    ) -> List[str]:
        lines: List[str] = []
        for stakeholder_name, items in self._group_signals_by_stakeholder(signals):
            top_match = (items[0].get("matches") or [{}])[0]
            role = str(top_match.get("current_role") or "").strip()
            employer = str(top_match.get("current_employer") or "").strip()
            lines.append(f"## {stakeholder_name}")
            if role or employer:
                if role and employer:
                    lines.append(f"**Role:** {role} at {employer}")
                else:
                    lines.append(f"**Role:** {role or employer}")
            profile_alumni_links = _normalize_org_name_list(
                alumni_name
                for item in items
                for alumni_name in item.get("profile_alumni_links") or []
            )
            if profile_alumni_links:
                lines.append(f"**Alumni linkage:** {', '.join(profile_alumni_links)}")
            lines.extend(["", "### Recent Signals"])
            for signal in items:
                headline = str(signal.get("subject") or signal.get("signal_id") or "").strip()
                lines.append(
                    f"- **{headline}** ({signal.get('received_at', '')}) - {self._classify_signal_category(signal)} "
                    f"[conf {signal.get('confidence_band', 'Low')} {signal.get('confidence_score', 0):.1f}/10 | "
                    f"relevance {signal.get('relevance_score', 0):.1f} | urgency {signal.get('urgency_score', 0):.1f} | "
                    f"action {signal.get('actionability_score', 0):.1f}]"
                )
                analysis = str(signal.get("text_note") or signal.get("raw_text") or "").strip()
                if analysis:
                    lines.append(f"  {' '.join(analysis.split())[:260]}")
                if signal.get("alumni_hits"):
                    lines.append(f"  Alumni context: {', '.join(signal.get('alumni_hits') or [])}")
                if signal.get("confidence_reason"):
                    lines.append(f"  Confidence note: {signal.get('confidence_reason')}")
            lines.extend(["", "### Trend Notes", self._summarise_signal_trend(items), "", "---"])
        if lines and lines[-1] == "---":
            lines = lines[:-1]

        relationship_context = intelligence_context.get("relationship_context", {})
        if any(relationship_context.get(key) for key in ("network_lines", "warm_intro_lines", "cross_target_lines", "talent_flow_lines")):
            lines.extend(["", "## Relationship Intelligence"])
            if relationship_context.get("network_lines"):
                lines.extend(["", "### Network Connections"])
                lines.extend(f"- {item}" for item in relationship_context.get("network_lines") or [])
            if relationship_context.get("warm_intro_lines"):
                lines.extend(["", "### Warm Introduction Paths"])
                lines.extend(f"- {item}" for item in relationship_context.get("warm_intro_lines") or [])
            if relationship_context.get("cross_target_lines"):
                lines.extend(["", "### Cross-Target Connections"])
                lines.extend(f"- {item}" for item in relationship_context.get("cross_target_lines") or [])
            if relationship_context.get("talent_flow_lines"):
                lines.extend(["", "### Talent Flow"])
                lines.extend(f"- {item}" for item in relationship_context.get("talent_flow_lines") or [])

        if intelligence_context.get("temporal_lines"):
            lines.extend(["", "## Temporal Patterns"])
            lines.extend(f"- {item}" for item in intelligence_context.get("temporal_lines") or [])
        if intelligence_context.get("scoring_lines"):
            lines.extend(["", "## Highest-Scoring Signals"])
            lines.extend(intelligence_context.get("scoring_lines") or [])
        return lines

    def _build_strategic_digest_lines(
        self,
        signals: List[Dict[str, Any]],
        alumni_context: List[str],
        intelligence_context: Dict[str, Any],
    ) -> List[str]:
        lines: List[str] = ["## Executive Summary", self._summarise_signal_trend(signals), ""]
        if alumni_context:
            lines.extend([f"Alumni-informed context: {', '.join(alumni_context)}", ""])
        lines.extend(self._build_detailed_digest_lines(signals, alumni_context=alumni_context, intelligence_context=intelligence_context))
        lines.extend(
            [
                "",
                "## Suggested Follow-Up Actions",
                "1. Validate the highest-confidence stakeholder changes against source material.",
                "2. Review pending profile updates for profiles flagged in this digest.",
                "3. Use warm-introduction paths and shared alumni links before cold outreach where available.",
                "4. Escalate competitor, regulatory, or leadership signals for analyst follow-up.",
                "",
                "## Risk & Opportunity Flags",
                "| Flag | Type | Stakeholder | Detail |",
                "|------|------|-------------|--------|",
            ]
        )
        for signal in signals[:5]:
            top_match = (signal.get("matches") or [{}])[0]
            flag_type = "Risk" if self._classify_signal_category(signal) in {"Regulatory", "Transaction"} else "Opportunity"
            stakeholder_name = top_match.get("canonical_name") or signal.get("parsed_candidate_name") or "Unknown"
            detail = " ".join(str(signal.get("subject") or "").split())[:90]
            lines.append(f"| ! | {flag_type} | {stakeholder_name} | {detail} |")
        lines.extend(["", "## Market Positioning Notes", self._summarise_signal_trend(signals)])
        return lines

    def _classify_signal_category(self, signal: Dict[str, Any]) -> str:
        haystack = " ".join(
            [
                str(signal.get("subject") or ""),
                str(signal.get("text_note") or ""),
                str(signal.get("raw_text") or ""),
            ]
        ).lower()
        if any(term in haystack for term in ("merger", "acquisition", "acquired", "takeover", "m&a")):
            return "Transaction"
        if any(term in haystack for term in ("regulator", "regulatory", "investigation", "compliance", "legal action")):
            return "Regulatory"
        if any(term in haystack for term in ("appointed", "joined", "new role", "started a new role", "board")):
            return "Leadership"
        return "Market Signal"

    def _summarise_signal_trend(self, signals: List[Dict[str, Any]]) -> str:
        categories: Dict[str, int] = {}
        for signal in signals:
            category = self._classify_signal_category(signal)
            categories[category] = categories.get(category, 0) + 1
        if not categories:
            return "No material trend patterns identified in this window."
        parts = [f"{count} {name.lower()}" for name, count in sorted(categories.items(), key=lambda item: (-item[1], item[0]))]
        return "Observed pattern: " + ", ".join(parts) + "."

    def _signal_corroboration_key(self, signal: Dict[str, Any]) -> str:
        top_match = (signal.get("matches") or [{}])[0]
        return "|".join(
            [
                normalize_lookup(top_match.get("canonical_name") or signal.get("parsed_candidate_name") or signal.get("target_name") or ""),
                normalize_lookup(signal.get("parsed_candidate_employer") or top_match.get("current_employer") or ""),
                normalize_lookup(self._classify_signal_category(signal)),
            ]
        )

    def _score_signal_confidence(self, signal: Dict[str, Any], corroboration_count: int = 1) -> Dict[str, Any]:
        primary_url = str(signal.get("primary_url") or "").strip().lower()
        source_system = str(signal.get("source_system") or "").strip().lower()
        source_type = str(signal.get("source_type") or "").strip().lower()
        confidence_hint = str(signal.get("confidence_hint") or "").strip().lower()
        source_name = str(signal.get("source_name") or "").strip().lower()

        score = 4.5
        band = "Low"
        reason = "unverified or weakly sourced signal"

        if source_system in {"shared_intel", "user_confirmed"} or "verified" in source_type:
            score = 9.5
            band = "User-confirmed"
            reason = "user-confirmed intelligence"
        elif any(token in primary_url for token in ("sec.gov", "asx.com.au", ".gov", ".gov.au", ".edu")) or any(
            token in source_type for token in ("official", "press_release", "regulator")
        ):
            score = 8.6
            band = "High"
            reason = "official or primary-source publication"
        elif primary_url and not any(token in primary_url for token in ("linkedin.com", "twitter.com", "x.com", "facebook.com", "medium.com", "substack.com")):
            score = 6.8
            band = "Medium"
            reason = "third-party publication with attributable source"
        elif primary_url or source_name:
            score = 4.8
            band = "Low"
            reason = "social or secondary-source mention"

        if confidence_hint in {"high", "confirmed"}:
            score = min(10.0, score + 1.0)
        elif confidence_hint in {"low", "weak"}:
            score = max(1.0, score - 1.0)
        elif confidence_hint in {"medium", "probable"}:
            score = min(10.0, score + 0.3)

        if corroboration_count > 1:
            score = min(10.0, score + min(1.2, 0.5 * (corroboration_count - 1)))
            reason += f"; corroborated by {corroboration_count} aligned signals"

        if score >= 8.0 and band != "User-confirmed":
            band = "High"
        elif score >= 5.5 and band not in {"High", "User-confirmed"}:
            band = "Medium"
        elif band != "User-confirmed":
            band = "Low"

        return {
            "confidence_score": round(score, 1),
            "confidence_band": band,
            "confidence_reason": reason,
            "corroboration_count": corroboration_count,
        }

    def _score_signal_actionability(self, signal: Dict[str, Any], relationship_hints: List[str]) -> Dict[str, float]:
        top_match = (signal.get("matches") or [{}])[0]
        confidence_score = float(signal.get("confidence_score") or 0.0)
        category = self._classify_signal_category(signal)
        received_at = _parse_timestamp(signal.get("received_at", "")) or datetime.now(timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - received_at).total_seconds() / 86400.0)

        relevance = 4.0
        if top_match.get("canonical_name"):
            relevance += 2.0
        if signal.get("alumni_hits") or signal.get("profile_alumni_links"):
            relevance += 1.0
        if category in {"Leadership", "Transaction", "Regulatory"}:
            relevance += 1.5

        urgency = 3.0
        if age_days <= 7:
            urgency += 3.0
        elif age_days <= 30:
            urgency += 1.5
        if category in {"Leadership", "Transaction", "Regulatory"}:
            urgency += 2.0

        actionability = 3.5
        if relationship_hints:
            actionability += 2.0
        if signal.get("update_suggestion_ids"):
            actionability += 2.0
        if top_match.get("canonical_name"):
            actionability += 1.0
        if category in {"Leadership", "Transaction"}:
            actionability += 1.0

        overall = (relevance * 0.3) + (urgency * 0.25) + (confidence_score * 0.25) + (actionability * 0.2)
        return {
            "relevance_score": round(min(10.0, relevance), 1),
            "urgency_score": round(min(10.0, urgency), 1),
            "actionability_score": round(min(10.0, actionability), 1),
            "intelligence_score": round(min(10.0, overall), 1),
        }

    def _build_relationship_intelligence(
        self,
        signals: List[Dict[str, Any]],
        profiles_by_key: Dict[str, Dict[str, Any]],
        org_name: str,
        alumni_context: List[str],
    ) -> Dict[str, List[str]]:
        network_lines: List[str] = []
        warm_intro_lines: List[str] = []
        cross_target_lines: List[str] = []
        talent_flow_lines: List[str] = []
        subscriber_node = _stakeholder_graph_node_id("subscriber_org", org_name)

        current_profiles: Dict[str, Dict[str, Any]] = {}
        for signal in signals:
            for profile_key in signal.get("matched_profile_keys") or []:
                if profile_key in profiles_by_key:
                    current_profiles[profile_key] = profiles_by_key[profile_key]

        for profile in current_profiles.values():
            name = str(profile.get("canonical_name") or "").strip()
            if not name:
                continue
            profile_node = _profile_graph_node_id(profile)

            graph_members = self._graph_neighbor_details(profile_node, {"lab_member"})
            if graph_members:
                for member in graph_members[:2]:
                    degree = str((member.get("edge") or {}).get("degree") or "1st").strip() or "1st"
                    warm_intro_lines.append(f"**{name}**: graph shows a {degree}-degree LinkedIn path via {member.get('name')}")

            shared_alumni_nodes = self._graph_common_neighbor_details(profile_node, subscriber_node, {"alumni_group"})
            if shared_alumni_nodes:
                shared_alumni = _normalize_org_name_list(item.get("name") for item in shared_alumni_nodes)
                network_lines.append(f"**{name}** shares graph alumni bridges with the org: {', '.join(shared_alumni[:3])}")

            organization_neighbors = self._graph_neighbor_details(profile_node, {"organization"})
            current_orgs = _normalize_org_name_list(
                item.get("name")
                for item in organization_neighbors
                if str((item.get("edge") or {}).get("relationship") or "").strip() == "works_at"
            )
            prior_orgs = _normalize_org_name_list(
                item.get("name")
                for item in organization_neighbors
                if str((item.get("edge") or {}).get("relationship") or "").strip() != "works_at"
            )
            if current_orgs:
                network_lines.append(f"**{name}** current graph employer node: {', '.join(current_orgs[:2])}")
            shared_with_org = [
                item
                for item in prior_orgs
                if any(_weak_org_link_match(item, alumni) for alumni in alumni_context)
            ]
            if shared_with_org:
                network_lines.append(f"**{name}** has graph-linked prior context overlapping the org alumni set: {', '.join(shared_with_org[:3])}")

            graph_summary = self._graph_shortest_path_summary(
                [profile.get("profile_key") or name, name, *current_orgs, *prior_orgs, *(profile.get("alumni") or [])],
                alumni_context,
            )
            if graph_summary:
                network_lines.append(f"**{name}**: {graph_summary}")

        profile_items = list(current_profiles.values())
        for idx, left in enumerate(profile_items):
            left_node = _profile_graph_node_id(left)
            for right in profile_items[idx + 1:]:
                right_node = _profile_graph_node_id(right)
                left_name = str(left.get("canonical_name") or "").strip()
                right_name = str(right.get("canonical_name") or "").strip()
                if not left_name or not right_name:
                    continue

                graph_manager = self._load_graph_manager()
                if graph_manager is not None and graph_manager.graph.has_edge(left_node, right_node):
                    relationship = str((graph_manager.graph.get_edge_data(left_node, right_node) or {}).get("relationship") or "").strip()
                    if relationship == "co_mentioned":
                        cross_target_lines.append(f"**{left_name}** and **{right_name}** are directly co-mentioned in the stakeholder graph")

                shared_orgs = _normalize_org_name_list(
                    item.get("name")
                    for item in self._graph_common_neighbor_details(left_node, right_node, {"organization"})
                )
                shared_alumni = _normalize_org_name_list(
                    item.get("name")
                    for item in self._graph_common_neighbor_details(left_node, right_node, {"alumni_group"})
                )
                shared_members = _normalize_org_name_list(
                    item.get("name")
                    for item in self._graph_common_neighbor_details(left_node, right_node, {"lab_member"})
                )

                shared_parts: List[str] = []
                if shared_orgs:
                    shared_parts.append(f"organisations: {', '.join(shared_orgs[:2])}")
                if shared_alumni:
                    shared_parts.append(f"alumni groups: {', '.join(shared_alumni[:2])}")
                if shared_members:
                    shared_parts.append(f"lab-member connectors: {', '.join(shared_members[:2])}")
                if shared_parts:
                    cross_target_lines.append(f"**{left_name}** and **{right_name}** are linked in the graph via " + "; ".join(shared_parts))
                    continue

                pair_path = self._graph_path_summary(
                    left_node,
                    right_node,
                    max_hops=3,
                    ignore_entity_types={"subscriber_org"},
                )
                if pair_path:
                    cross_target_lines.append(f"**{left_name}** and **{right_name}**: {pair_path}")

        employer_counts = Counter()
        consulting_inflow = Counter()
        feeder_orgs: Counter[str] = Counter()
        for signal in signals:
            employer = str(signal.get("parsed_candidate_employer") or "").strip()
            if employer:
                employer_counts[employer] += 1
            top_match = (signal.get("matches") or [{}])[0]
            prior_employers = [
                str(item).strip()
                for item in (top_match.get("known_employers") or [])
                if str(item).strip()
            ]
            if employer and any(
                token in normalize_lookup(previous)
                for previous in prior_employers
                for token in ("consult", "mckinsey", "bcg", "bain", "deloitte", "accenture")
            ):
                consulting_inflow[employer] += 1
            for previous in prior_employers:
                if previous and previous != employer:
                    feeder_orgs[previous] += 1

        for employer, count in employer_counts.most_common(3):
            if count >= 2:
                suffix = ""
                if consulting_inflow.get(employer):
                    suffix = f" — {consulting_inflow[employer]} signals suggest consulting-talent inflow"
                talent_flow_lines.append(f"**{employer}** appears in {count} current-window signals{suffix}")
        for feeder_org, count in feeder_orgs.most_common(2):
            if count >= 2:
                talent_flow_lines.append(f"**{feeder_org}** appears as a shared feeder in {count} watched-profile histories")

        return {
            "network_lines": network_lines[:6],
            "warm_intro_lines": warm_intro_lines[:6],
            "cross_target_lines": cross_target_lines[:6],
            "talent_flow_lines": talent_flow_lines[:6],
        }

    def _build_temporal_patterns(
        self,
        historical_signals: List[Dict[str, Any]],
        selected_profile_keys: List[str],
        since_ts: str,
    ) -> List[str]:
        now_dt = datetime.now(timezone.utc)
        since_dt = _parse_timestamp(since_ts)
        if since_dt is None:
            since_dt = now_dt - timedelta(days=7)
        window = max(timedelta(days=3), now_dt - since_dt)
        prev_start = since_dt - window

        selected = [str(item).strip() for item in selected_profile_keys if str(item).strip()]
        if not selected:
            selected = sorted(
                {
                    str(profile_key).strip()
                    for signal in historical_signals
                    for profile_key in signal.get("matched_profile_keys") or []
                    if str(profile_key).strip()
                }
            )

        recent_counts = Counter()
        previous_counts = Counter()
        profile_dates: Dict[str, List[datetime]] = defaultdict(list)

        for signal in historical_signals:
            signal_dt = _parse_timestamp(signal.get("received_at", ""))
            if signal_dt is None:
                continue
            matched = [str(item).strip() for item in signal.get("matched_profile_keys") or [] if str(item).strip()]
            for profile_key in matched:
                if profile_key not in selected:
                    continue
                profile_dates[profile_key].append(signal_dt)
                if signal_dt >= since_dt:
                    recent_counts[profile_key] += 1
                elif prev_start <= signal_dt < since_dt:
                    previous_counts[profile_key] += 1

        lines: List[str] = []
        for profile_key in selected:
            recent = recent_counts.get(profile_key, 0)
            previous = previous_counts.get(profile_key, 0)
            if recent >= 3 and previous > 0 and recent >= previous * 2:
                lines.append(f"Profile `{profile_key}` shows acceleration: {recent} signals vs {previous} in the prior comparable window")
            elif recent == 0 and previous >= 2:
                lines.append(f"Profile `{profile_key}` has gone quiet after {previous} signals in the prior comparable window")

            dates = sorted(profile_dates.get(profile_key) or [])
            if len(dates) >= 4:
                gaps = [
                    (dates[idx + 1] - dates[idx]).days
                    for idx in range(len(dates) - 1)
                    if (dates[idx + 1] - dates[idx]).days > 0
                ]
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    if 25 <= avg_gap <= 40:
                        lines.append(f"Profile `{profile_key}` shows a roughly monthly signal cadence ({avg_gap:.0f}-day average gap)")

        org_recent = sum(recent_counts.values())
        org_previous = sum(previous_counts.values())
        if org_recent >= 6 and org_previous > 0 and org_recent >= org_previous * 2:
            lines.insert(0, f"Organisation-wide signal acceleration: {org_recent} signals vs {org_previous} in the prior comparable window")
        return lines[:8]

    def _build_signal_priority_summary(self, signals: List[Dict[str, Any]]) -> List[str]:
        scored = sorted(
            signals,
            key=lambda item: (
                float(item.get("intelligence_score") or 0.0),
                float(item.get("confidence_score") or 0.0),
                item.get("received_at", ""),
            ),
            reverse=True,
        )
        lines: List[str] = []
        for signal in scored[:5]:
            top_match = (signal.get("matches") or [{}])[0]
            stakeholder_name = top_match.get("canonical_name") or signal.get("parsed_candidate_name") or signal.get("target_name") or signal.get("signal_id")
            lines.append(
                f"- **{stakeholder_name}** | Intelligence {signal.get('intelligence_score', 0):.1f}/10 | "
                f"Confidence {signal.get('confidence_band', 'Low')} ({signal.get('confidence_score', 0):.1f}) | "
                f"Urgency {signal.get('urgency_score', 0):.1f} | Actionability {signal.get('actionability_score', 0):.1f}"
            )
        return lines

    def _enrich_digest_signals(
        self,
        signals: List[Dict[str, Any]],
        state: Dict[str, Any],
        org_name: str,
        profile_keys: List[str],
        alumni_context: List[str],
        since_ts: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        profiles_by_key = {
            str(profile.get("profile_key") or "").strip(): profile
            for profile in state.get("profiles") or []
            if str(profile.get("profile_key") or "").strip()
        }
        historical_signals = [
            signal for signal in state.get("signals") or []
            if _signal_visible_to_org(signal, org_name)
        ]
        corroboration = Counter(self._signal_corroboration_key(signal) for signal in signals)
        relationship_context = self._build_relationship_intelligence(signals, profiles_by_key, org_name, alumni_context)

        enriched_signals: List[Dict[str, Any]] = []
        for signal in signals:
            enriched = dict(signal)
            corroboration_count = corroboration.get(self._signal_corroboration_key(signal), 1)
            confidence = self._score_signal_confidence(enriched, corroboration_count=corroboration_count)
            relationship_hints: List[str] = []
            top_match = (enriched.get("matches") or [{}])[0]
            if top_match.get("canonical_name"):
                profile = profiles_by_key.get(str(top_match.get("profile_key") or "").strip()) or {}
                if profile.get("linkedin_connections"):
                    relationship_hints.append("direct LinkedIn connection available")
                if profile.get("alumni") or profile.get("known_employers"):
                    relationship_hints.append("shared alumni/employer context available")
            enriched.update(confidence)
            enriched.update(self._score_signal_actionability(enriched, relationship_hints))
            enriched["relationship_hints"] = relationship_hints
            enriched_signals.append(enriched)

        temporal_lines = self._build_temporal_patterns(historical_signals, profile_keys, since_ts)
        scoring_lines = self._build_signal_priority_summary(enriched_signals)
        return enriched_signals, {
            "relationship_context": relationship_context,
            "temporal_lines": temporal_lines,
            "scoring_lines": scoring_lines,
        }

    def _prepare_digest_data(
        self,
        signals: List[Dict[str, Any]],
        state: Dict[str, Any],
        org_name: str,
        scope_type: str,
        scope_profile: Dict[str, Any],
        child_org_names: List[str],
        key_themes: List[str],
        regulatory_context: str,
        market_size: str,
        member_alumni: List[str],
        org_alumni: List[str],
        intelligence_context: Dict[str, Any],
        report_depth: str,
        digest_tier: str,
        priority_profile_keys: List[str],
        deep_analysis: bool,
    ) -> str:
        fact_map = {
            str(item.get("fact_id") or "").strip(): item
            for item in state.get("observed_facts") or []
            if str(item.get("fact_id") or "").strip()
        }
        suggestion_map = {
            str(item.get("suggestion_id") or "").strip(): item
            for item in state.get("update_suggestions") or []
            if str(item.get("suggestion_id") or "").strip()
        }
        entries: List[Dict[str, Any]] = []
        for signal in signals:
            top_match = (signal.get("matches") or [{}])[0]
            facts = [
                fact_map[fact_id]
                for fact_id in signal.get("observed_fact_ids") or []
                if fact_id in fact_map
            ]
            suggestions = [
                suggestion_map[suggestion_id]
                for suggestion_id in signal.get("update_suggestion_ids") or []
                if suggestion_id in suggestion_map
            ]
            entry: Dict[str, Any] = {
                "signal_id": signal.get("signal_id", ""),
                "candidate_name": signal.get("parsed_candidate_name", ""),
                "target_type": signal.get("target_type", ""),
                "signal_type": signal.get("signal_type", ""),
                "source_system": signal.get("source_system", ""),
                "subject": signal.get("subject", ""),
                "received_at": signal.get("received_at", ""),
                "matched_profile_keys": signal.get("matched_profile_keys") or [],
                "matched_profile": top_match.get("canonical_name", ""),
                "current_employer": top_match.get("current_employer", ""),
                "current_role": top_match.get("current_role", ""),
                "affiliations": top_match.get("affiliations") or [],
                "match_score": top_match.get("score", 0),
                "match_reasons": top_match.get("reasons", []),
                "text_note": (signal.get("text_note") or signal.get("raw_text") or "")[:800],
                "primary_url": signal.get("primary_url", ""),
                "needs_review": signal.get("needs_review", False),
                "category": self._classify_signal_category(signal),
                "alumni_hits": signal.get("alumni_hits") or [],
                "profile_alumni_links": signal.get("profile_alumni_links") or [],
                "confidence_band": signal.get("confidence_band") or "",
                "confidence_score": signal.get("confidence_score") or 0,
                "confidence_reason": signal.get("confidence_reason") or "",
                "identity_strength": signal.get("digest_identity_strength") or "",
                "relevance_score": signal.get("relevance_score") or 0,
                "urgency_score": signal.get("urgency_score") or 0,
                "actionability_score": signal.get("actionability_score") or 0,
                "intelligence_score": signal.get("intelligence_score") or 0,
                "relationship_hints": signal.get("relationship_hints") or [],
            }
            if facts:
                entry["observed_facts"] = facts
            if suggestions:
                entry["update_suggestions"] = suggestions
            entries.append(entry)
        return json.dumps(
            {
                "org_name": org_name,
                "scope_type": scope_type,
                "scope_profile": {
                    "profile_key": str(scope_profile.get("profile_key") or "").strip(),
                    "canonical_name": str(scope_profile.get("canonical_name") or "").strip(),
                    "description": str(scope_profile.get("description") or "").strip(),
                },
                "child_org_names": child_org_names,
                "key_themes": key_themes,
                "regulatory_context": regulatory_context,
                "market_size": market_size,
                "report_depth": report_depth,
                "digest_tier": digest_tier,
                "priority_profile_keys": priority_profile_keys,
                "deep_analysis": bool(deep_analysis),
                "member_alumni": member_alumni,
                "org_alumni": org_alumni,
                "relationship_context": intelligence_context.get("relationship_context") or {},
                "temporal_patterns": intelligence_context.get("temporal_lines") or [],
                "priority_signals": intelligence_context.get("scoring_lines") or [],
                "signals": entries,
            },
            ensure_ascii=True,
            indent=2,
        )

    def _detect_digest_escalation(self, signals: List[Dict[str, Any]]) -> tuple[bool, List[str], str]:
        flagged_profiles: List[str] = []
        reasons: set[str] = set()
        for signal in signals:
            category = self._classify_signal_category(signal)
            has_updates = bool(signal.get("update_suggestion_ids"))
            if category in {"Leadership", "Regulatory", "Transaction"} or has_updates:
                for profile_key in signal.get("matched_profile_keys") or []:
                    if profile_key and profile_key not in flagged_profiles:
                        flagged_profiles.append(profile_key)
                if has_updates or category == "Leadership":
                    reasons.add("leadership change")
                if category == "Regulatory":
                    reasons.add("regulatory action")
                if category == "Transaction":
                    reasons.add("transaction activity")
        if not flagged_profiles:
            return False, [], ""
        return True, flagged_profiles, f"Potential {', '.join(sorted(reasons)) or 'high-signal events'} detected"

    def _llm_synthesise(
        self,
        raw_data: str,
        org_name: str,
        scope_type: str,
        scope_profile_name: str,
        provider: str,
        model: str,
        report_depth: str,
    ) -> tuple[Optional[str], str, str]:
        provider_name = str(provider or "ollama").strip().lower()
        if report_depth == "summary":
            system_prompt = (
                "You are an intelligence analyst producing a concise markdown digest.\n"
                "Produce a Key Signals section only with one line per stakeholder or organisation.\n"
                "If alumni context is provided, flag alumni-related findings briefly.\n"
                "Use relationship_context and temporal_patterns when they contain material findings.\n"
                "Ignore weakly linked same-name results; if identity is uncertain, omit the signal.\n"
                "Omit any empty sections.\n"
                "Do not include recommendations. Do not fabricate facts."
            )
        elif report_depth == "strategic":
            system_prompt = (
                "You are a senior intelligence analyst producing a strategic markdown brief.\n"
                "Include an executive summary, per-stakeholder sections, competitive implications, follow-up actions, risk and opportunity flags, and market positioning notes.\n"
                "If alumni context is provided, highlight warm-introduction or competitive-context implications.\n"
                "Use relationship_context and temporal_patterns to surface graph-backed network links, shared connectors, and momentum shifts.\n"
                "Exclude weak same-name matches and low-confidence noise.\n"
                "Do not emit placeholder sections such as 'no alumni context' when no such data exists.\n"
                "Do not format confidence scores or labels as markdown links.\n"
                "Use only the provided data. Do not fabricate facts."
            )
        else:
            system_prompt = (
                "You are an intelligence analyst producing a structured markdown digest.\n"
                "Group by stakeholder, include recent signals, trend notes, and concise context.\n"
                "If alumni context is provided, note alumni-relevant findings and weak linkage context.\n"
                "Use relationship_context and temporal_patterns where they add concrete graph-backed insight.\n"
                "Exclude weak same-name matches and low-confidence noise.\n"
                "Only include signals that clearly refer to the matched stakeholder.\n"
                "Do not create empty sections or filler statements when there is no alumni context.\n"
                "Do not format confidence scores or labels as markdown links.\n"
                "Use only the provided data. Do not fabricate facts."
            )
        scope_label = "industry" if scope_type == "industry" else "organisation"
        scope_subject = scope_profile_name or org_name
        user_prompt = f"Generate a {report_depth} {scope_label} digest for organisation: {org_name} and scope: {scope_subject}\n\nSignal data:\n{raw_data}"

        if provider_name == "ollama":
            ollama_text, ollama_model = self._call_ollama(system_prompt, user_prompt, model or _DEFAULT_OLLAMA_WATCH_MODEL)
            if ollama_text:
                return ollama_text, "ollama", ollama_model
            logger.info("Ollama digest synthesis unavailable; falling back to Claude API")
            anthropic_model = str(os.environ.get("CORTEX_WATCH_ANTHROPIC_MODEL") or "").strip() or "claude-haiku-4-5-20251001"
            anthropic_text, actual_model = self._call_anthropic(system_prompt, user_prompt, anthropic_model)
            return anthropic_text, "anthropic", actual_model
        if provider_name == "anthropic":
            anthropic_text, actual_model = self._call_anthropic(system_prompt, user_prompt, model or "claude-haiku-4-5-20251001")
            return anthropic_text, "anthropic", actual_model
        logger.warning("Unsupported digest LLM provider: %s", provider_name)
        return None, provider_name, str(model or "").strip()

    def _preferred_ollama_watch_models(self, requested_model: str) -> List[str]:
        candidates: List[str] = []
        for item in [
            str(os.environ.get("CORTEX_WATCH_OLLAMA_MODEL") or "").strip(),
            str(requested_model or "").strip(),
            *_OLLAMA_WATCH_MODEL_FALLBACKS,
        ]:
            if item and item not in candidates:
                candidates.append(item)
        return candidates

    def _get_ollama_installed_models(self) -> List[str]:
        import requests

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models") or []
            return [
                str(item.get("name") or "").strip()
                for item in models
                if str(item.get("name") or "").strip()
            ]
        except Exception as exc:
            logger.warning("Could not query installed Ollama models: %s", exc)
            return []

    def _resolve_ollama_model(self, requested_model: str) -> str:
        candidates = self._preferred_ollama_watch_models(requested_model)
        installed = set(self._get_ollama_installed_models())
        if installed:
            for candidate in candidates:
                if candidate in installed:
                    return candidate
        return candidates[0] if candidates else _DEFAULT_OLLAMA_WATCH_MODEL

    def _call_ollama(self, system: str, user: str, model: str) -> tuple[Optional[str], str]:
        import requests

        candidates = self._preferred_ollama_watch_models(model)
        resolved_model = self._resolve_ollama_model(model)
        ordered_models = [resolved_model] + [candidate for candidate in candidates if candidate != resolved_model]
        timeout_seconds = _ollama_watch_timeout_seconds()

        for candidate in ordered_models:
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": candidate,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "stream": False,
                    },
                    timeout=timeout_seconds,
                )
                if response.status_code >= 400:
                    body = ""
                    try:
                        body = str(response.json())
                    except Exception:
                        body = response.text
                    if response.status_code == 404 or "not found" in body.lower():
                        logger.warning("Ollama model unavailable for digest synthesis: %s", candidate)
                        continue
                response.raise_for_status()
                logger.info("Using Ollama WATCH digest model: %s", candidate)
                return str(response.json().get("message", {}).get("content") or "").strip() or None, candidate
            except Exception as exc:
                message = str(exc).lower()
                if "not found" in message and candidate != ordered_models[-1]:
                    logger.warning("Ollama model unavailable for digest synthesis: %s", candidate)
                    continue
                logger.warning("Ollama digest synthesis failed with model %s: %s", candidate, exc)
                if candidate != ordered_models[-1]:
                    continue
                return None, candidate
        return None, resolved_model

    def _call_anthropic(self, system: str, user: str, model: str) -> tuple[Optional[str], str]:
        import requests

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set; skipping digest synthesis")
            return None, model
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=60,
            )
            response.raise_for_status()
            content = response.json().get("content") or []
            if not content:
                return None, model
            return str(content[0].get("text") or "").strip() or None, model
        except Exception as exc:
            logger.warning("Anthropic digest synthesis failed: %s", exc)
            return None, model

    def _normalize_profile(
        self,
        raw_profile: Dict[str, Any],
        org_name: str,
        source: str = "",
        trace_id: str = "",
    ) -> Dict[str, Any]:
        canonical_name = str(raw_profile.get("canonical_name") or raw_profile.get("name") or "").strip()
        if not canonical_name:
            raise ValueError("Stakeholder profile requires canonical_name or name")

        target_type = str(raw_profile.get("target_type") or "person").strip().lower() or "person"
        watch_status = _normalize_watch_status(raw_profile.get("watch_status") or "off")
        linkedin_url = str(raw_profile.get("linkedin_url") or "").strip()
        website_url = str(raw_profile.get("website_url") or "").strip()
        current_employer = str(raw_profile.get("current_employer") or raw_profile.get("org_name") or "").strip()
        current_role = str(raw_profile.get("current_role") or "").strip()
        external_profile_id = str(
            raw_profile.get("external_profile_id") or raw_profile.get("website_profile_id") or raw_profile.get("id") or ""
        ).strip()
        aliases = [str(item).strip() for item in raw_profile.get("aliases") or [] if str(item).strip()]
        affiliations = _normalize_affiliations(
            raw_profile.get("affiliations"),
            current_employer=current_employer,
            current_role=current_role,
        )
        primary_affiliation = _primary_affiliation(affiliations)
        if primary_affiliation:
            current_employer = str(primary_affiliation.get("org_name_text") or current_employer).strip()
            current_role = str(primary_affiliation.get("role") or current_role).strip()
        known_employers = [str(item).strip() for item in raw_profile.get("known_employers") or [] if str(item).strip()]
        for employer in _employers_from_affiliations(affiliations):
            if employer not in known_employers:
                known_employers.append(employer)
        if current_employer and current_employer not in known_employers:
            known_employers.append(current_employer)
        industry_affiliations = _normalize_industry_affiliations(raw_profile.get("industry_affiliations"))

        key_basis = external_profile_id or linkedin_url or website_url or canonical_name
        key_material = "|".join([normalize_lookup(org_name), target_type, normalize_lookup(key_basis)])
        profile_key = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:16]

        return {
            "profile_key": profile_key,
            "external_profile_id": external_profile_id,
            "org_name": str(org_name).strip(),
            "target_type": target_type,
            "canonical_name": canonical_name,
            "email": str(raw_profile.get("email") or "").strip(),
            "industry": str(raw_profile.get("industry") or "").strip(),
            "function": str(raw_profile.get("function") or "").strip(),
            "status": _normalize_status(raw_profile.get("status") or "active"),
            "last_verified_at": str(raw_profile.get("last_verified_at") or "").strip(),
            "current_employer": current_employer,
            "current_role": current_role,
            "linkedin_url": linkedin_url,
            "website_url": website_url,
            "description": str(raw_profile.get("description") or "").strip(),
            "key_themes": _normalize_org_name_list(raw_profile.get("key_themes") or raw_profile.get("key_themes_json") or []),
            "regulatory_context": str(raw_profile.get("regulatory_context") or "").strip(),
            "market_size": str(raw_profile.get("market_size") or "").strip(),
            "address": _normalize_address(raw_profile.get("address") or {}),
            "acn_abn": str(raw_profile.get("acn_abn") or "").strip(),
            "phone": str(raw_profile.get("phone") or "").strip(),
            "parent_entity": str(raw_profile.get("parent_entity") or "").strip(),
            "notes": str(raw_profile.get("notes") or "").strip(),
            "watch_status": watch_status,
            "tags": [str(item).strip() for item in raw_profile.get("tags") or [] if str(item).strip()],
            "aliases": aliases,
            "known_employers": known_employers,
            "affiliations": affiliations,
            "industry_affiliations": industry_affiliations,
            "alumni": _normalize_profile_alumni(raw_profile.get("alumni")),
            "linkedin_connections": _normalize_linkedin_connections(raw_profile.get("linkedin_connections")),
            "source": str(source or raw_profile.get("source") or "").strip(),
            "trace_id": str(trace_id or raw_profile.get("trace_id") or "").strip(),
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }

    def _normalize_signal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        org_name = str(payload.get("org_name") or payload.get("tenant_id") or "default").strip()
        raw_text = str(payload.get("raw_text") or payload.get("body") or "").strip()
        subject = str(payload.get("subject") or "").strip()
        primary_url = str(payload.get("primary_url") or payload.get("content") or "").strip()
        text_note = str(payload.get("text_note") or "").strip()
        received_at = str(payload.get("received_at") or _utc_now_iso()).strip()
        message_id = str(payload.get("message_id") or "").strip()

        signal_hash_source = "|".join(
            [
                org_name,
                message_id,
                subject,
                primary_url,
                raw_text,
                received_at,
            ]
        )
        signal_hash = hashlib.sha256(signal_hash_source.encode("utf-8")).hexdigest()
        signal_id = str(payload.get("signal_id") or f"sig_{signal_hash[:16]}")

        return {
            "signal_id": signal_id,
            "signal_hash": signal_hash,
            "trace_id": str(payload.get("trace_id") or "").strip(),
            "source_system": str(payload.get("source_system") or payload.get("source") or "market_radar").strip(),
            "signal_type": str(payload.get("signal_type") or "linkedin_notification").strip(),
            "target_type": str(payload.get("target_type") or "person").strip().lower() or "person",
            "org_name": org_name,
            "submitted_by": str(payload.get("submitted_by") or "").strip(),
            "source_job": str(payload.get("source_job") or "").strip(),
            "source_org_name": str(payload.get("source_org_name") or "").strip(),
            "visible_to_orgs": _normalize_org_name_list(payload.get("visible_to_orgs") or []),
            "shared_with_orgs": _normalize_org_name_list(payload.get("shared_with_orgs") or []),
            "scope_profile_key": str(payload.get("scope_profile_key") or payload.get("industry_profile_key") or "").strip(),
            "child_profile_keys": [str(item).strip() for item in payload.get("child_profile_keys") or [] if str(item).strip()],
            "child_org_names": _normalize_org_name_list(payload.get("child_org_names") or []),
            "key_themes": _normalize_org_name_list(payload.get("key_themes") or []),
            "regulatory_context": str(payload.get("regulatory_context") or "").strip(),
            "market_size": str(payload.get("market_size") or "").strip(),
            "received_at": received_at,
            "message_id": message_id,
            "subject": subject,
            "raw_text": raw_text,
            "primary_url": primary_url,
            "text_note": text_note,
            "target_name": str(payload.get("target_name") or "").strip(),
            "source_name": str(payload.get("source_name") or "").strip(),
            "source_type": str(payload.get("source_type") or "").strip().lower(),
            "confidence_hint": str(payload.get("confidence_hint") or "").strip().lower(),
            "watch_report_meta": dict(payload.get("watch_report_meta") or {}) if isinstance(payload.get("watch_report_meta"), dict) else {},
            "notification_kind": str(payload.get("notification_kind") or "").strip(),
            "parsed_candidate_name": str(payload.get("parsed_candidate_name") or payload.get("stakeholder_name") or "").strip(),
            "parsed_candidate_employer": str(payload.get("parsed_candidate_employer") or payload.get("stakeholder_employer") or "").strip(),
            "tags": [str(item).strip() for item in payload.get("tags") or [] if str(item).strip()],
            "matches": [],
            "matched_profile_keys": [],
            "needs_review": False,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }

    def _write_raw_signal_markdown(self, signal: Dict[str, Any]) -> Path:
        output_path = self.raw_dir / f"{signal['signal_id']}.md"
        matched_names = ", ".join(match["canonical_name"] for match in signal.get("matches") or [])
        lines = [
            "---",
            f"signal_id: {signal['signal_id']}",
            f"trace_id: {signal.get('trace_id', '')}",
            f"org_name: {signal.get('org_name', '')}",
            f"received_at: {signal.get('received_at', '')}",
            f"signal_type: {signal.get('signal_type', '')}",
            f"target_type: {signal.get('target_type', '')}",
            f"primary_url: {signal.get('primary_url', '')}",
            f"matched_stakeholders: {matched_names}",
            "---",
            "",
            f"# {signal.get('subject') or signal['signal_id']}",
            "",
        ]
        if signal.get("text_note"):
            lines.extend(["## Note", signal["text_note"], ""])
        if signal.get("raw_text"):
            lines.extend(["## Raw Email Body", signal["raw_text"], ""])
        if signal.get("matches"):
            lines.append("## Top Matches")
            for match in signal["matches"]:
                lines.append(
                    f"- {match['canonical_name']} ({match['score']:.2f}) — {'; '.join(match.get('reasons') or [])}"
                )
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @staticmethod
    def _merge_unique_items(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]], key_field: str, extra_fields: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        merged = {str(item.get(key_field) or ""): dict(item) for item in existing if item.get(key_field)}
        for item in incoming:
            key = str(item.get(key_field) or "")
            if not key:
                continue
            payload = dict(item)
            for field_name, field_value in (extra_fields or {}).items():
                payload.setdefault(field_name, field_value)
            payload.setdefault("created_at", _utc_now_iso())
            merged[key] = payload
        return sorted(merged.values(), key=lambda item: item.get("created_at", ""), reverse=True)
