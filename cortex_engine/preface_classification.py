"""
Shared credibility classification helpers for markdown preface generation.
"""

from __future__ import annotations

from typing import Tuple


_CREDIBILITY_TIERS = {
    5: ("peer-reviewed", "Peer-Reviewed"),
    4: ("institutional", "Institutional"),
    3: ("pre-print", "Pre-Print"),
    2: ("editorial", "Editorial"),
    1: ("commentary", "Commentary"),
    0: ("unclassified", "Unclassified"),
}


def classify_credibility_tier(
    text: str,
    source_type: str,
    availability_status: str = "unknown",
) -> Tuple[int, str, str]:
    marker_map = {
        5: ["pubmed", "nlm", "nature", "lancet", "jama", "bmj", "peer-reviewed", "peer reviewed"],
        4: [
            "who", "who.int", "un ", "un.org", "ipcc", "oecd", "world bank", "worldbank.org",
            "government", "department", "ministry", "university", "institute", "centre", "center",
            ".gov", ".edu", "nih.gov", "cdc.gov", "europa.eu",
        ],
        3: ["arxiv", "ssrn", "biorxiv", "researchgate", "preprint", "pre-print"],
        2: [
            "scientific american", "scientificamerican.com", "the conversation", "theconversation.com",
            "hbr", "hbr.org", "harvard business review", "editorial", "op-ed", "opinion",
        ],
        1: [
            "blog", "newsletter", "consulting report", "whitepaper", "white paper",
            "medium.com", "substack.com", "blogspot.", "wordpress.", "linkedin.com",
        ],
    }

    blob = (text or "").lower()
    if source_type == "AI Generated Report":
        tier_value = 0
    else:
        tier_value = 0
        for value in (5, 4, 3, 2, 1):
            if any(marker in blob for marker in marker_map[value]):
                tier_value = value
                break

    # Dead/removed sources are significantly higher poisoning risk.
    if availability_status in {"not_found", "gone"}:
        tier_value = max(0, tier_value - 2)

    key, label = _CREDIBILITY_TIERS[tier_value]
    return tier_value, key, label
