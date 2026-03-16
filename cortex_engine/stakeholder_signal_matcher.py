from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlparse


def normalize_lookup(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def _text_contains_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return phrase in text


def _normalize_domain(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    host = (parsed.netloc or parsed.path or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _unique_normalized_texts(values: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        normalized = normalize_lookup(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def match_signal_to_profiles(
    signal: Dict[str, Any],
    profiles: List[Dict[str, Any]],
    threshold: float = 0.55,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    primary_url = str(signal.get("primary_url") or "").strip().lower()
    primary_domain = _normalize_domain(primary_url)
    signal_target_type = str(signal.get("target_type") or "").strip().lower()
    candidate_name = normalize_lookup(signal.get("parsed_candidate_name") or "")
    candidate_employer = normalize_lookup(signal.get("parsed_candidate_employer") or "")
    haystack = normalize_lookup(
        " ".join(
            [
                str(signal.get("subject") or ""),
                str(signal.get("raw_text") or ""),
                str(signal.get("text_note") or ""),
            ]
        )
    )

    matches: List[Dict[str, Any]] = []
    for profile in profiles:
        profile_target_type = str(profile.get("target_type") or "person").strip().lower()
        if signal_target_type and profile_target_type and signal_target_type != profile_target_type:
            continue

        score = 0.0
        reasons: List[str] = []

        linkedin_url = str(profile.get("linkedin_url") or "").strip().lower()
        website_url = str(profile.get("website_url") or "").strip().lower()
        website_domain = _normalize_domain(website_url)
        canonical_name = normalize_lookup(profile.get("canonical_name") or "")
        aliases = [normalize_lookup(alias) for alias in profile.get("aliases") or [] if normalize_lookup(alias)]
        employers = [str(emp).strip() for emp in profile.get("known_employers") or [] if str(emp).strip()]
        affiliation_roles = [
            str(item.get("role") or "").strip()
            for item in profile.get("affiliations") or []
            if isinstance(item, dict) and str(item.get("role") or "").strip()
        ]
        employers = _unique_normalized_texts(employers)
        current_employer = normalize_lookup(profile.get("current_employer") or "")
        if current_employer:
            employers.append(current_employer)
        employers = _unique_normalized_texts(employers)
        role = normalize_lookup(profile.get("current_role") or "")
        roles = _unique_normalized_texts([profile.get("current_role") or "", *affiliation_roles])
        parent_entity = normalize_lookup(profile.get("parent_entity") or "")
        industry = normalize_lookup(profile.get("industry") or "")
        function_name = normalize_lookup(profile.get("function") or "")

        if primary_url and linkedin_url and primary_url == linkedin_url:
            score += 0.8
            reasons.append("exact LinkedIn URL match")

        if candidate_name and canonical_name and candidate_name == canonical_name:
            score += 0.7
            reasons.append("exact canonical name match")
        elif candidate_name and candidate_name in aliases:
            score += 0.65
            reasons.append("alias name match")
        elif canonical_name and _text_contains_phrase(haystack, canonical_name):
            score += 0.45
            reasons.append("canonical name found in text")
        else:
            alias_hit = next((alias for alias in aliases if alias and _text_contains_phrase(haystack, alias)), "")
            if alias_hit:
                score += 0.35
                reasons.append(f"alias found in text: {alias_hit}")

        if profile_target_type == "person":
            if candidate_employer:
                if candidate_employer in employers:
                    score += 0.2
                    reasons.append("candidate employer match")
            else:
                employer_hit = next((emp for emp in employers if emp and _text_contains_phrase(haystack, emp)), "")
                if employer_hit:
                    score += 0.12
                    reasons.append(f"employer found in text: {employer_hit}")

            if role and _text_contains_phrase(haystack, role):
                score += 0.08
                reasons.append("current role found in text")
            elif roles:
                role_hit = next((item for item in roles if item and _text_contains_phrase(haystack, item)), "")
                if role_hit:
                    score += 0.06
                    reasons.append(f"affiliation role found in text: {role_hit}")
            if function_name and _text_contains_phrase(haystack, function_name):
                score += 0.06
                reasons.append("function found in text")
        else:
            if primary_domain and website_domain and primary_domain == website_domain:
                score += 0.35
                reasons.append("website domain match")
            tag_hit = next(
                (normalize_lookup(tag) for tag in profile.get("tags") or [] if _text_contains_phrase(haystack, normalize_lookup(tag))),
                "",
            )
            if tag_hit:
                score += 0.1
                reasons.append(f"target tag found in text: {tag_hit}")
            if parent_entity and _text_contains_phrase(haystack, parent_entity):
                score += 0.08
                reasons.append("parent entity found in text")
            if industry and _text_contains_phrase(haystack, industry):
                score += 0.06
                reasons.append("industry found in text")

        final_score = min(1.0, round(score, 4))
        if final_score < threshold:
            continue

        matches.append(
            {
                "profile_key": profile.get("profile_key"),
                "target_type": profile_target_type,
                "canonical_name": profile.get("canonical_name", ""),
                "external_profile_id": profile.get("external_profile_id", ""),
                "email": profile.get("email", ""),
                "industry": profile.get("industry", ""),
                "function": profile.get("function", ""),
                "status": profile.get("status", ""),
                "last_verified_at": profile.get("last_verified_at", ""),
                "current_employer": profile.get("current_employer", ""),
                "current_role": profile.get("current_role", ""),
                "linkedin_url": profile.get("linkedin_url", ""),
                "website_url": profile.get("website_url", ""),
                "parent_entity": profile.get("parent_entity", ""),
                "acn_abn": profile.get("acn_abn", ""),
                "phone": profile.get("phone", ""),
                "address": profile.get("address", {}),
                "affiliations": profile.get("affiliations") or [],
                "score": final_score,
                "reasons": reasons,
            }
        )

    return sorted(matches, key=lambda item: (item["score"], item["canonical_name"]), reverse=True)[:max_results]
