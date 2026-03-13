from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

from cortex_engine.stakeholder_signal_matcher import normalize_lookup


def _hash_id(prefix: str, *parts: str) -> str:
    material = "|".join(str(part or "").strip() for part in parts)
    return f"{prefix}_{hashlib.sha1(material.encode('utf-8')).hexdigest()[:16]}"


def _extract_role_and_employer(signal: Dict[str, Any]) -> Dict[str, str]:
    text = " ".join(
        [
            str(signal.get("subject") or ""),
            str(signal.get("raw_text") or ""),
            str(signal.get("text_note") or ""),
        ]
    ).strip()
    employer = str(signal.get("parsed_candidate_employer") or "").strip()
    role = ""

    patterns = [
        re.compile(r"started a new role as\s+(?P<role>.+?)\s+at\s+(?P<org>[A-Z][\w&().,'/\- ]+)", re.IGNORECASE),
        re.compile(r"joined\s+(?P<org>[A-Z][\w&().,'/\- ]+?)\s+as\s+(?P<role>.+?)(?:[.!,]|$)", re.IGNORECASE),
        re.compile(r"appointed\s+(?P<role>.+?)\s+at\s+(?P<org>[A-Z][\w&().,'/\- ]+)", re.IGNORECASE),
        re.compile(r"new role as\s+(?P<role>.+?)(?:[.!,]|$)", re.IGNORECASE),
    ]

    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        role = str(match.groupdict().get("role") or "").strip(" .,-")
        if not employer:
            employer = str(match.groupdict().get("org") or "").strip(" .,-")
        break

    return {
        "current_employer": employer,
        "current_role": role,
    }


def detect_profile_change_artifacts(signal: Dict[str, Any], matches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    top_match = matches[0] if matches else {}
    if not top_match or str(top_match.get("target_type") or "").strip().lower() != "person":
        return {"observed_facts": [], "update_suggestions": []}

    extracted = _extract_role_and_employer(signal)
    observed_facts: List[Dict[str, Any]] = []
    update_suggestions: List[Dict[str, Any]] = []

    signal_id = str(signal.get("signal_id") or "")
    evidence_excerpt = str(signal.get("raw_text") or signal.get("subject") or "")[:500]
    profile_key = str(top_match.get("profile_key") or "")
    canonical_name = str(top_match.get("canonical_name") or "")

    field_candidates = {
        "current_employer": {
            "observed_value": extracted.get("current_employer", ""),
            "current_value": str(top_match.get("current_employer") or "").strip(),
            "confidence": 0.9 if signal.get("parsed_candidate_employer") else 0.75,
        },
        "current_role": {
            "observed_value": extracted.get("current_role", ""),
            "current_value": str(top_match.get("current_role") or "").strip(),
            "confidence": 0.78 if extracted.get("current_role") else 0.0,
        },
    }

    for field_name, meta in field_candidates.items():
        observed_value = str(meta["observed_value"] or "").strip()
        current_value = str(meta["current_value"] or "").strip()
        confidence = float(meta["confidence"] or 0.0)
        if not observed_value:
            continue

        observed_facts.append(
            {
                "fact_id": _hash_id("fact", signal_id, profile_key, field_name, observed_value),
                "signal_id": signal_id,
                "profile_key": profile_key,
                "canonical_name": canonical_name,
                "field_name": field_name,
                "observed_value": observed_value,
                "confidence": round(confidence, 4),
                "evidence_excerpt": evidence_excerpt,
            }
        )

        if normalize_lookup(observed_value) == normalize_lookup(current_value):
            continue

        update_suggestions.append(
            {
                "suggestion_id": _hash_id("sug", profile_key, field_name, observed_value),
                "signal_id": signal_id,
                "profile_key": profile_key,
                "target_type": "person",
                "canonical_name": canonical_name,
                "field_name": field_name,
                "old_value": current_value,
                "proposed_value": observed_value,
                "confidence": round(confidence, 4),
                "evidence_excerpt": evidence_excerpt,
                "status": "pending",
            }
        )

    return {
        "observed_facts": observed_facts,
        "update_suggestions": update_suggestions,
    }
