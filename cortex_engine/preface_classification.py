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


_TRUSTED_INSTITUTION_HOSTS = {
    "who.int",
    "un.org",
    "oecd.org",
    "worldbank.org",
    "imf.org",
    "europa.eu",
    "ipcc.ch",
    "nih.gov",
    "cdc.gov",
}

_SCHOLARLY_HOSTS = {
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "sciencedirect.com",
    "springer.com",
    "springeropen.com",
    "nature.com",
    "thelancet.com",
    "jama.com",
    "bmj.com",
    "wiley.com",
    "tandfonline.com",
    "sagepub.com",
    "frontiersin.org",
    "plos.org",
    "acm.org",
    "ieee.org",
    "science.org",
    "cell.com",
    "nejm.org",
    "oup.com",
    "cambridge.org",
}

_COMMENTARY_HOSTS = {
    "wikipedia.org",
    "youtube.com",
    "youtu.be",
    "medium.com",
    "substack.com",
    "blogspot.com",
    "wordpress.com",
    "linkedin.com",
    "reddit.com",
    "quora.com",
    "stackexchange.com",
}

_EDITORIAL_HOSTS = {
    "scientificamerican.com",
    "theconversation.com",
    "hbr.org",
}

_GOV_EDU_SUFFIXES = (
    ".gov",
    ".gov.uk",
    ".gov.au",
    ".gc.ca",
    ".gouv.fr",
    ".go.jp",
    ".edu",
    ".ac.uk",
    ".edu.au",
    ".ac.jp",
    ".edu.nz",
    ".ac.nz",
)

_COMMERCIAL_SUFFIXES = (
    ".com",
    ".co",
    ".net",
    ".biz",
    ".info",
    ".io",
    ".ai",
)


def _host_matches(host: str, pattern: str) -> bool:
    p = pattern.lower().strip()
    h = (host or "").lower().strip()
    return h == p or h.endswith(f".{p}")


def _extract_hosts(blob: str) -> list[str]:
    hosts: list[str] = []
    for m in re.finditer(r"https?://([^/\s\"'<>]+)", blob, flags=re.IGNORECASE):
        host = (m.group(1) or "").lower().strip().strip(".,;:!?")
        if host and host not in hosts:
            hosts.append(host)
    for m in re.finditer(r"\b(?:www\.)?[a-z0-9-]+\.[a-z]{2,}(?:\.[a-z]{2,})?\b", blob, flags=re.IGNORECASE):
        host = (m.group(0) or "").lower().strip().strip(".,;:!?")
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _has_strong_peer_review_signals(blob: str, hosts: list[str]) -> bool:
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
    if any(any(_host_matches(h, s) for s in _SCHOLARLY_HOSTS) for h in hosts):
        return True
    return any(re.search(pat, blob) for pat in strong_markers)


def _is_trusted_institutional_host(hosts: list[str]) -> bool:
    for host in hosts:
        if any(_host_matches(host, p) for p in _TRUSTED_INSTITUTION_HOSTS):
            return True
        if host.endswith(_GOV_EDU_SUFFIXES):
            return True
    return False


def _is_commercial_host(hosts: list[str]) -> bool:
    return any(host.endswith(_COMMERCIAL_SUFFIXES) for host in hosts)


def _is_explicit_commentary_host(hosts: list[str]) -> bool:
    return any(any(_host_matches(host, p) for p in _COMMENTARY_HOSTS) for host in hosts)


def _is_editorial_host(hosts: list[str]) -> bool:
    return any(any(_host_matches(host, p) for p in _EDITORIAL_HOSTS) for host in hosts)


def classify_credibility_tier_with_reason(
    text: str,
    source_type: str,
    availability_status: str = "unknown",
) -> Tuple[int, str, str, str]:
    blob = (text or "").lower()
    hosts = _extract_hosts(blob)
    strong_peer = _has_strong_peer_review_signals(blob, hosts)
    trusted_institution = _is_trusted_institutional_host(hosts)
    commercial_host = _is_commercial_host(hosts)
    explicit_commentary_host = _is_explicit_commentary_host(hosts)
    editorial_host = _is_editorial_host(hosts)

    preprint_markers = ["arxiv", "ssrn", "biorxiv", "researchgate", "preprint", "pre-print"]
    editorial_markers = [
        "scientificamerican.com", "theconversation.com", "hbr.org",
        "editorial", "op-ed", "opinion",
    ]
    commentary_markers = [
        "blog", "newsletter", "whitepaper", "white paper",
        "medium.com", "substack.com", "blogspot.", "wordpress.", "linkedin.com",
        "personal blog", "/blog",
    ]

    if source_type == "AI Generated Report":
        tier_value = 0
        reason = "ai_generated_default"
    elif explicit_commentary_host:
        tier_value = 1
        reason = "explicit_commentary_host"
    elif editorial_host:
        tier_value = 2
        reason = "explicit_editorial_host"
    elif strong_peer and not any(m in blob for m in commentary_markers):
        tier_value = 5
        reason = "strong_scholarly_signals"
    elif any(m in blob for m in preprint_markers):
        tier_value = 3
        reason = "preprint_markers"
    elif trusted_institution:
        tier_value = 4
        reason = "trusted_institution_host"
    elif any(m in blob for m in editorial_markers):
        tier_value = 2
        reason = "editorial_markers"
    else:
        tier_value = 1
        reason = "commercial_domain_default" if commercial_host else "commentary_default"

    # For scraped/general web sources, do not over-promote without hard evidence.
    if source_type == "Other" and not strong_peer and not trusted_institution:
        tier_value = min(tier_value, 2)
        if tier_value <= 1 and commercial_host:
            reason = "other_source_commercial_default"

    # Dead/removed sources are significantly higher poisoning risk.
    if availability_status in {"not_found", "gone"}:
        tier_value = max(0, tier_value - 2)
        reason = f"{reason}_downgraded_unavailable"

    key, label = _CREDIBILITY_TIERS[tier_value]
    return tier_value, key, label, reason


def classify_credibility_tier(
    text: str,
    source_type: str,
    availability_status: str = "unknown",
) -> Tuple[int, str, str]:
    tier_value, key, label, _ = classify_credibility_tier_with_reason(
        text=text,
        source_type=source_type,
        availability_status=availability_status,
    )
    return tier_value, key, label
