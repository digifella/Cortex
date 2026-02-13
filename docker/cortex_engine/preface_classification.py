"""
Shared credibility classification helpers for markdown preface generation.
"""

from __future__ import annotations

import re
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
    def _has_strong_peer_review_signals(blob: str) -> bool:
        strong_markers = [
            r"\bdoi:\s*10\.\d{4,9}/",
            r"\bdoi\.org/10\.\d{4,9}/",
            r"\bpmid\b",
            r"\bpubmed\b",
            r"\bpmc\d+\b",
            r"\bissn\b",
            r"\bjournal\b",
            r"\bvol(?:ume)?\s*\d+",
            r"\bissue\s*\d+",
        ]
        journal_hosts = [
            "nature.com", "thelancet.com", "jama.com", "bmj.com", "sciencedirect.com",
            "springer.com", "wiley.com", "tandfonline.com", "sagepub.com", "frontiersin.org",
            "plos.org", "acm.org", "ieee.org", "oup.com", "cambridge.org", "science.org",
            "cell.com", "nejm.org", "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
        ]
        if any(host in blob for host in journal_hosts):
            return True
        return any(re.search(pat, blob) for pat in strong_markers)

    marker_map = {
        5: ["pubmed", "nlm", "nature", "lancet", "jama", "bmj"],
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
    commentary_markers = marker_map[1] + ["personal blog", "/blog", "opinion piece"]
    is_commentary_like = any(marker in blob for marker in commentary_markers)
    has_strong_peer = _has_strong_peer_review_signals(blob)

    if source_type == "AI Generated Report":
        tier_value = 0
    else:
        tier_value = 0
        for value in (4, 3, 2, 1):
            if any(marker in blob for marker in marker_map[value]):
                tier_value = value
                break

        # Only promote to peer-reviewed with strong scholarly evidence.
        if has_strong_peer and not is_commentary_like:
            tier_value = max(tier_value, 5)

        # Scraped/general web sources should not become peer-reviewed from weak cues.
        if source_type == "Other" and not has_strong_peer:
            tier_value = min(tier_value, 2) if tier_value else 1

    # Dead/removed sources are significantly higher poisoning risk.
    if availability_status in {"not_found", "gone"}:
        tier_value = max(0, tier_value - 2)

    key, label = _CREDIBILITY_TIERS[tier_value]
    return tier_value, key, label
