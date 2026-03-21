from __future__ import annotations

from typing import Any, Callable, Dict


def classify_entity_industry(
    entity: Dict[str, Any],
    message: Dict[str, Any],
    strategic_profile: Dict[str, Any],
    org_profile_lookup: Callable[[str], Dict[str, Any]],
) -> str:
    explicit = str(entity.get("industry") or "").strip()
    if explicit:
        return explicit

    org_name = str(
        entity.get("canonical_name")
        or entity.get("name")
        or entity.get("current_employer")
        or ""
    ).strip()
    org_profile = org_profile_lookup(org_name)
    affiliations = org_profile.get("industry_affiliations") or []
    if affiliations:
        first = affiliations[0]
        name = str(first.get("industry_name") or "").strip()
        if name:
            return name

    profile_industry = str(org_profile.get("industry") or "").strip()
    if profile_industry:
        return profile_industry

    strategic_industries = [str(item).strip() for item in strategic_profile.get("industries") or [] if str(item).strip()]
    evidence = " ".join(
        [
            str(message.get("subject") or ""),
            str(message.get("raw_text") or ""),
            str(entity.get("evidence") or ""),
            str(entity.get("current_role") or ""),
            org_name,
        ]
    ).lower()
    for industry in strategic_industries:
        if industry.lower() in evidence:
            return industry
    return ""
