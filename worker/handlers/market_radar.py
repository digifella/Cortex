"""
Market Radar Handler
Scans the web for market intelligence signals about specified targets,
synthesises a prioritised lead report, and saves it as a Markdown file.

No new dependencies — uses anthropic (existing) and requests (standard).

input_data schema:
    config_id           int
    config_name         str
    targets             list[{name, type, notes, current_employer, profile_url, extra_context}]
    source_sites        {include: [domains], exclude: [domains]}
    intelligence_focus  {topic, industry, themes}
    my_company          {name, description, services, value_prop, target_clients}
    system_prompt       str
    member_email        str
    email_results       bool
    push_to_kb          bool
    kb_category         str
    source_system       str  (lab|cron|admin)
    previous_report     str  markdown content of previous scan (for delta comparison, optional)
"""

import os
import json
import logging
import re
import textwrap
import time
from datetime import date
from pathlib import Path

import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_GEO_FOCUS_KEYS = (
    "geography",
    "geographic_scope",
    "geo_scope",
    "jurisdiction",
    "country",
    "market",
    "region",
    "location",
)

_AUSTRALIA_POSITIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\baustralia\b",
        r"\baustralian\b",
        r"\bvictoria\b",
        r"\bvictorian\b",
        r"\bmelbourne\b",
        r"\bnew south wales\b",
        r"\bnsw\b",
        r"\bsydney\b",
        r"\bqueensland\b",
        r"\bqld\b",
        r"\bbrisbane\b",
        r"\bsouth australia\b",
        r"\badelaide\b",
        r"\bwestern australia\b",
        r"\bperth\b",
        r"\btasmania\b",
        r"\bhobart\b",
        r"\bcanberra\b",
        r"\bnorthern territory\b",
        r"\bdarwin\b",
        r"\bessential services commission\b",
        r"\bipart\b",
        r"\bwater services association of australia\b",
        r"\bwsaa\b",
    )
]

_AUSTRALIA_NEGATIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\buk\b",
        r"\bu\.k\.\b",
        r"\bunited kingdom\b",
        r"\bbritain\b",
        r"\bbritish\b",
        r"\bengland\b",
        r"\bscotland\b",
        r"\bwales\b",
        r"\blondon\b",
        r"\bofwat\b",
        r"\beurope\b",
        r"\beuropean\b",
        r"\beu\b",
        r"\bglobal\b",
        r"\binternational\b",
        r"\bno australian\b",
        r"\bunited states\b",
        r"\busa\b",
        r"\bu\.s\.\b",
        r"\bcanada\b",
        r"\bcanadian\b",
        r"\bnew zealand\b",
        r"\bnz\b",
    )
]

_AUSTRALIA_SUBREGION_TERMS = (
    ("victoria", "Victoria"),
    ("victorian", "Victoria"),
    ("melbourne", "Melbourne"),
    ("new south wales", "New South Wales"),
    ("nsw", "NSW"),
    ("sydney", "Sydney"),
    ("queensland", "Queensland"),
    ("qld", "Queensland"),
    ("brisbane", "Brisbane"),
    ("south australia", "South Australia"),
    ("adelaide", "Adelaide"),
    ("western australia", "Western Australia"),
    ("perth", "Perth"),
    ("tasmania", "Tasmania"),
    ("hobart", "Hobart"),
    ("canberra", "Canberra"),
    ("act", "ACT"),
    ("northern territory", "Northern Territory"),
    ("darwin", "Darwin"),
)


# ── Anthropic client (lazy) ──

def _anthropic_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in config.env")
    return anthropic.Anthropic(api_key=api_key)


def _domain(url: str) -> str:
    """Extract bare domain from a URL."""
    try:
        return urlparse(url).netloc.lstrip("www.") or url
    except Exception:
        return url


def _first_non_empty(*values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _recent_year_terms() -> str:
    this_year = date.today().year
    return f"{this_year - 1} {this_year}"


def _normalise_geography_label(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if any(term in lowered for term in ("australia", "australian", "au")):
        return "Australia"
    return text


def _target_context_text(target: dict | None) -> str:
    if not target:
        return ""
    return " ".join(
        part for part in (
            target.get("name", ""),
            target.get("notes", ""),
            target.get("extra_context", ""),
            target.get("current_employer", ""),
            target.get("address_country", ""),
            target.get("address_state", ""),
            target.get("address_city", ""),
        )
        if str(part).strip()
    )


def _build_geography_policy(focus: dict | None = None, target: dict | None = None, source_sites: dict | None = None) -> dict:
    focus = focus or {}
    source_sites = source_sites or {}
    target = target or {}

    explicit_geo = _first_non_empty(*(focus.get(key, "") for key in _GEO_FOCUS_KEYS))
    include_sites = " ".join(source_sites.get("include", []) or [])
    target_context = _target_context_text(target)
    inferred_geo = "Australia" if any(
        marker in " ".join([include_sites.lower(), target_context.lower()])
        for marker in (".au", "australia", "australian", "victoria", "nsw", "queensland")
    ) else ""
    primary = _normalise_geography_label(explicit_geo or inferred_geo or "Australia")

    extra_terms = []
    if primary == "Australia":
        target_context_lower = target_context.lower()
        for marker, label in _AUSTRALIA_SUBREGION_TERMS:
            if marker in target_context_lower and label not in extra_terms:
                extra_terms.append(label)
        query_terms = ["Australia", "Australian"] + extra_terms[:2]
        instruction = (
            "Australia is the primary market. Strongly prioritise Australian organisations, "
            "regulators, conditions, and sources. If an overseas entity or regulator appears, "
            "exclude it unless it is directly tied to Australian operations or the user explicitly requested it."
        )
        strict = True
    else:
        query_terms = [primary]
        instruction = (
            f"{primary} is the primary market. Prioritise organisations, regulators, and sector conditions "
            f"from {primary}; treat unrelated foreign examples as out of scope."
        )
        strict = bool(primary)

    return {
        "primary": primary,
        "query_terms": query_terms,
        "instruction": instruction,
        "strict": strict,
    }


def _signal_text(signal: dict) -> str:
    return " ".join(
        str(signal.get(field, "") or "")
        for field in ("headline", "snippet", "source", "url")
    )


def _score_australia_signal(signal: dict) -> tuple[int, int]:
    text = _signal_text(signal)
    url = (signal.get("url", "") or "").lower()
    domain = _domain(url).lower()

    positive = sum(1 for pattern in _AUSTRALIA_POSITIVE_PATTERNS if pattern.search(text))
    negative = sum(1 for pattern in _AUSTRALIA_NEGATIVE_PATTERNS if pattern.search(text))

    if domain.endswith(".au") or ".gov.au" in domain or ".com.au" in domain or ".org.au" in domain:
        positive += 2
    if domain.endswith(".uk") or ".co.uk" in domain or ".gov.uk" in domain:
        negative += 2

    return positive, negative


def _filter_signals_for_geography(
    signals: list,
    geo_policy: dict,
    *,
    target_name: str = "",
    require_geo_match: bool = False,
) -> list:
    if not signals or not geo_policy.get("strict"):
        return signals[:10]
    if geo_policy.get("primary") != "Australia":
        return signals[:10]

    filtered = []
    dropped = 0
    target_name_lower = (target_name or "").lower()
    for signal in signals:
        headline = signal.get("headline", "") or ""
        if headline.startswith("[PINNED") or headline.startswith("[SHARED"):
            filtered.append(signal)
            continue

        positive, negative = _score_australia_signal(signal)
        text = _signal_text(signal).lower()
        namesake_match = bool(target_name_lower and target_name_lower in text)

        keep = False
        if positive > negative:
            keep = True
        elif negative > positive:
            keep = False
        elif require_geo_match:
            keep = positive > 0
        else:
            keep = namesake_match or positive > 0

        if keep:
            filtered.append(signal)
        else:
            dropped += 1

    if dropped:
        logger.info(
            "[market_radar] Geography filter dropped %s non-Australian signal(s) for %s",
            dropped,
            target_name or "broad market scan",
        )
    return filtered[:10]


# ── Local LLM helpers (Ollama) ──

def _call_local_llm(system_msg: str, user_msg: str, max_tokens: int = 4096, timeout: int = 180) -> str | None:
    """
    Call the local Ollama LLM for synthesis tasks.
    Returns response text, or None if unavailable/misconfigured/failed.
    Reads LOCAL_LLM_URL and LOCAL_LLM_SYNTHESIS_MODEL from env.
    """
    base_url = os.environ.get("LOCAL_LLM_URL", "").rstrip("/")
    model = os.environ.get("LOCAL_LLM_SYNTHESIS_MODEL", "")
    if not base_url or not model:
        return None
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        logger.warning(f"[market_radar] Local LLM HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        logger.warning(f"[market_radar] Local LLM unavailable ({e}); falling back to Claude")
        return None


def _synthesis_call(client, system_msg: str, user_msg: str, max_tokens: int = 4096) -> str:
    """
    Run a synthesis (JSON-extraction) call.
    Tries local Ollama LLM first; falls back to Claude on failure.
    """
    local_text = _call_local_llm(system_msg, user_msg, max_tokens)
    if local_text is not None:
        logger.info(f"[market_radar] Synthesis via local LLM ({os.environ.get('LOCAL_LLM_SYNTHESIS_MODEL','?')})")
        return local_text
    logger.info("[market_radar] Synthesis via Claude API")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text if response.content else ""


# ── Step 1: Web intelligence gathering per target ──

def _make_web_search_tool(excludes: list, max_uses: int = 3) -> dict:
    """
    Build web_search_20250305 config. Only uses blocked_domains (never allowed_domains)
    so searches are broad across the whole web — domain includes are hints, not hard filters.
    """
    tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}
    if excludes:
        tool["blocked_domains"] = excludes
    return tool


def _run_web_search(client, system_msg: str, user_query: str, web_tool: dict) -> list:
    """
    Run a web search with the proper multi-turn tool use loop.
    The web_search tool may require multiple API round-trips:
      turn 1 → stop_reason='tool_use' (search requested)
      turn 2 → stop_reason='end_turn'  (results delivered + text generated)
    We collect signals from both web_search_tool_result blocks (structured)
    and Claude's final text (prose fallback via extraction call).
    """
    try:
        messages = [{"role": "user", "content": user_query}]
        raw_results = []   # from web_search_tool_result blocks (title + url)
        final_text  = ""
        max_loops   = 6

        for _loop in range(max_loops):
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=3000,
                system=system_msg,
                tools=[web_tool],
                messages=messages,
            )
            block_types = [getattr(b, "type", "?") for b in response.content]
            logger.info(f"[market_radar] Loop {_loop+1}: stop={response.stop_reason} blocks={block_types}")

            # Collect text and web search result blocks from this response turn
            for block in response.content:
                btype = getattr(block, "type", "")

                if btype == "text":
                    final_text += getattr(block, "text", "")

                elif btype == "web_search_tool_result":
                    # Structured search results (title + URL reliably available)
                    for result in getattr(block, "content", []):
                        if getattr(result, "type", "") == "web_search_result":
                            # Debug: log all available fields on first result
                            if not raw_results:
                                logger.debug(f"[market_radar] Search result fields: {[a for a in dir(result) if not a.startswith('_')]}")
                                logger.debug(f"[market_radar] Sample result: title={getattr(result,'title','')!r} url={getattr(result,'url','')!r} page_age={getattr(result,'page_age',None)!r}")
                            raw_results.append({
                                "headline": getattr(result, "title", ""),
                                "url":      getattr(result, "url", ""),
                                "date":     getattr(result, "page_age", None),
                                "snippet":  getattr(result, "page_content", "") or "",
                                "source":   _domain(getattr(result, "url", "")),
                            })

            if response.stop_reason == "end_turn":
                break
            elif response.stop_reason == "tool_use":
                # Pass assistant turn back; Anthropic server handles tool results
                messages.append({"role": "assistant", "content": response.content})
            else:
                logger.warning(f"[market_radar] Unexpected stop_reason: {response.stop_reason}")
                break

        # Prefer structured results from web_search_tool_result blocks
        if raw_results:
            logger.info(f"[market_radar] Got {len(raw_results)} structured results from web_search blocks")
            return raw_results[:10]

        # Fall back: try JSON parse of final text (Claude sometimes follows JSON instructions)
        if final_text:
            signals = _parse_signals_json(final_text)
            if signals:
                return signals

            # Last resort: ask Claude to extract signals from its own prose summary
            logger.info("[market_radar] Extracting signals from prose response")
            extract = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                system=(
                    "Extract market intelligence signals from the research text provided. "
                    "Return a JSON array where each item has: headline (string), "
                    "url (string or null), date (ISO date string or null), "
                    "snippet (1-2 sentence summary), source (domain or publication name). "
                    "Include every distinct finding. Return ONLY the JSON array, no other text."
                ),
                messages=[{"role": "user", "content": f"Extract signals:\n\n{final_text[:4000]}"}],
            )
            extract_text = ""
            for block in extract.content:
                if hasattr(block, "text"):
                    extract_text += block.text
            return _parse_signals_json(extract_text)

        logger.warning("[market_radar] Web search returned no text and no structured results")
        return []

    except Exception as e:
        logger.warning(f"[market_radar] Web search failed: {e}")
        return []


def _gather_broad_market_signals(client, source_sites: dict, focus: dict) -> dict:
    """
    Broad market scan — searches the topic/industry/themes WITHOUT restricting to specific
    target names. Returns a pseudo-target dict labelled 'Market Overview'.
    """
    excludes = source_sites.get("exclude", [])
    topic    = focus.get("topic", "")
    industry = focus.get("industry", "")
    themes   = focus.get("themes", [])
    context_terms = " ".join(filter(None, [topic, industry] + (themes[:3] if themes else [])))
    geo_policy = _build_geography_policy(focus=focus, source_sites=source_sites)

    if not context_terms:
        return {"name": "Market Overview", "type": "broad", "notes": "", "signals": []}

    web_tool   = _make_web_search_tool(excludes, max_uses=4)
    system_msg = (
        f"You are a market intelligence researcher scanning for broad market trends and opportunities "
        f"in: {context_terms}. {geo_policy['instruction']} Search for recent news, industry movements, regulatory changes, "
        f"leadership announcements, digital transformation initiatives, and market activity. "
        f"Return a JSON array of signal objects, each with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array, no other text."
    )
    query = f"Latest news and market intelligence: {' '.join(geo_policy['query_terms'])} {context_terms} {_recent_year_terms()}".strip()
    signals = _run_web_search(client, system_msg, query, web_tool)
    signals = _filter_signals_for_geography(signals, geo_policy, require_geo_match=True)
    logger.info(f"[market_radar] Broad market scan found {len(signals)} signals")
    return {"name": "Market Overview", "type": "broad", "notes": "", "signals": signals}


def _build_shared_intel_signals(shared_intel_items: list) -> list:
    """Convert shared_intel_items (company-level) into signal dicts for synthesis."""
    signals = []
    for item in shared_intel_items:
        itype     = item.get("intel_type", "note")
        content   = item.get("content", "").strip()
        text_note = item.get("text_note", "").strip()
        title     = item.get("title", "") or content[:80]
        added     = (item.get("intel_date") or item.get("submitted_at") or date.today().isoformat())[:10]
        submitter = item.get("submitted_by", "a team member")
        if not content:
            continue
        if itype == "url":
            snippet = f"Shared by {submitter}: {text_note[:1000]}" if text_note else f"Shared by {submitter}: {content[:200]}"
            signals.append({
                "headline": f"[SHARED] {title}",
                "url":      content,
                "date":     added,
                "snippet":  snippet,
                "source":   _domain(content),
            })
        else:
            signals.append({
                "headline": f"[SHARED NOTE] {title}",
                "url":      "",
                "date":     added,
                "snippet":  f"Shared by {submitter}: {content[:1000]}",
                "source":   "shared-intel",
            })
    return signals


def _gather_signals_for_target(client, target: dict, source_sites: dict, focus: dict) -> dict:
    """
    Search for a specific named target (company or person). Searches broadly — no
    allowed_domains restriction. Uses blocked_domains to exclude junk sites.
    Returns {name, type, signals: [{headline, url, date, snippet, source}]}
    """
    target_name = target.get("name", "")
    target_type = target.get("type", "company")
    notes       = target.get("notes", "")
    excludes    = source_sites.get("exclude", [])

    topic    = focus.get("topic", "")
    industry = focus.get("industry", "")
    themes   = focus.get("themes", [])
    context_terms = " ".join(filter(None, [topic, industry] + (themes[:2] if themes else [])))
    geo_policy = _build_geography_policy(focus=focus, target=target, source_sites=source_sites)

    web_tool   = _make_web_search_tool(excludes, max_uses=3)
    system_msg = (
        f"You are a market intelligence researcher. Search for recent news and signals about "
        f"{'company' if target_type == 'company' else 'person'}: {target_name}. "
        f"{geo_policy['instruction']} "
        f"If the name matches entities in multiple countries, prefer the Australian entity and discard foreign namesakes. "
        f"Look for: new leadership, digital initiatives, restructures, funding, challenges, "
        f"regulatory pressures, strategic announcements, and consulting opportunities. "
        f"{'Also look for: ' + context_terms + '.' if context_terms else ''} "
        f"{'Notes: ' + notes if notes else ''} "
        f"Return a JSON array of signal objects, each with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array, no other text."
    )
    query = f"{target_name} {' '.join(geo_policy['query_terms'])} {context_terms} news announcements {_recent_year_terms()}".strip()
    signals = _run_web_search(client, system_msg, query, web_tool)
    signals = _filter_signals_for_geography(signals, geo_policy, target_name=target_name)
    # Prepend shared intel items (contributed by team members via Intel Board)
    shared_signals = _build_shared_intel_signals(target.get("shared_intel_items", []) or [])
    if shared_signals:
        logger.info(f"[market_radar] Injecting {len(shared_signals)} shared intel items for '{target_name}'")
        signals = shared_signals + signals
    logger.info(f"[market_radar] Target '{target_name}' found {len(signals)} signals")
    return {"name": target_name, "type": target_type, "notes": notes, "signals": signals, "target": target}


def _parse_signals_json(text: str) -> list:
    """Extract JSON array from Claude's response, handling markdown fences."""
    text = text.strip()
    # Strip ```json fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Find JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []

    try:
        data = json.loads(match.group())
        if isinstance(data, list):
            return data[:10]  # cap at 10 signals per target
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _gather_competitor_signals(client, target: dict, source_sites: dict, focus: dict) -> dict:
    """Competitive intelligence: wins, losses, talent, pricing, client news."""
    target_name = target.get("name", "")
    excludes    = source_sites.get("exclude", [])
    topic       = focus.get("topic", "")
    industry    = focus.get("industry", "")
    geo_policy  = _build_geography_policy(focus=focus, target=target, source_sites=source_sites)
    web_tool    = _make_web_search_tool(excludes, max_uses=4)
    system_msg  = (
        f"You are a competitive intelligence researcher. Search for recent news about competitor firm: {target_name}. "
        f"{geo_policy['instruction']} "
        f"Look specifically for: client wins or losses, new service offerings, leadership hires or departures, "
        f"pricing changes, public failures or criticisms, strategic announcements, and partnerships. "
        f"Industry context: {topic} {industry}. "
        f"Return a JSON array of signal objects with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array."
    )
    query = f"{target_name} {' '.join(geo_policy['query_terms'])} consulting news wins losses clients strategy {_recent_year_terms()} {industry}".strip()
    signals = _run_web_search(client, system_msg, query, web_tool)
    signals = _filter_signals_for_geography(signals, geo_policy, target_name=target_name)
    # Prepend shared intel items (contributed by team members via Intel Board)
    shared_signals = _build_shared_intel_signals(target.get("shared_intel_items", []) or [])
    if shared_signals:
        logger.info(f"[market_radar] Injecting {len(shared_signals)} shared intel items for competitor '{target_name}'")
        signals = shared_signals + signals
    logger.info(f"[market_radar] Competitor '{target_name}': {len(signals)} signals")
    return {"name": target_name, "type": "company", "notes": target.get("notes",""), "signals": signals, "target": target}


def _proxycurl_lookup(target: dict, linkedin_url: str) -> list:
    """
    Fetch a LinkedIn profile via the Proxycurl API.
    Returns a signal list (0 or 1 items) with structured profile data.
    Requires PROXYCURL_API_KEY in environment.

    Proxycurl Person Profile endpoint:
        GET https://nubela.co/proxycurl/api/v2/linkedin?url=<linkedin_url>
        Authorization: Bearer <api_key>
    """
    api_key = os.environ.get("PROXYCURL_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://nubela.co/proxycurl/api/v2/linkedin",
            params={"url": linkedin_url},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        if resp.status_code == 404:
            logger.info(f"[market_radar] Proxycurl: profile not found for {linkedin_url}")
            return []
        if resp.status_code == 429:
            logger.warning("[market_radar] Proxycurl: rate limit hit")
            return []
        if resp.status_code != 200:
            logger.warning(f"[market_radar] Proxycurl returned {resp.status_code} for {linkedin_url}")
            return []

        data = resp.json()
        name = data.get("full_name") or target.get("name", "")

        # Extract most recent experience
        experiences = data.get("experiences") or []
        current_exp = next((e for e in experiences if not e.get("ends_at")), None)
        if not current_exp and experiences:
            current_exp = experiences[0]

        role_parts = []
        if current_exp:
            if current_exp.get("title"):
                role_parts.append(current_exp["title"])
            if current_exp.get("company"):
                role_parts.append(f"at {current_exp['company']}")
        headline = data.get("headline") or " ".join(role_parts) or "LinkedIn profile"

        # Build a brief snippet from available fields
        summary_parts = []
        if role_parts:
            summary_parts.append("Current role: " + " ".join(role_parts))
        if data.get("summary"):
            summary_parts.append(data["summary"][:300])
        snippet = " | ".join(summary_parts) if summary_parts else ""

        logger.info(f"[market_radar] Proxycurl: got profile for {name} — {headline}")
        return [{
            "headline": f"{name}: {headline}",
            "url":      linkedin_url,
            "date":     date.today().isoformat(),
            "snippet":  snippet,
            "source":   "linkedin.com",
        }]
    except Exception as e:
        logger.warning(f"[market_radar] Proxycurl lookup failed for {linkedin_url}: {e}")
        return []


def _scrape_profile_url(target: dict, url: str) -> list:
    """
    Fetch a target's profile URL (e.g. LinkedIn) for authoritative ground-truth info.
    Returns a signal list (0 or 1 items).
    """
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MarketRadarBot/1.0)"},
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return []
        html = resp.text[:100_000]
        clean = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        clean = re.sub(r"<style[\s\S]*?</style>", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s{3,}", "  ", clean).strip()[:6000]
        if not clean:
            return []
        return [{
            "headline": f"Profile page: {target.get('name')}",
            "url":      url,
            "date":     date.today().isoformat(),
            "snippet":  clean[:2000],
            "source":   _domain(url),
        }]
    except Exception as e:
        logger.debug(f"[market_radar] Profile URL fetch failed for {url}: {e}")
        return []


def _gather_stakeholder_signals(client, target: dict, source_sites: dict, focus: dict) -> dict:
    """
    Individual tracking: career moves, publications, speeches, board roles.
    Uses disambiguation fields (current_employer, profile_url, extra_context)
    to build precise queries and avoid mistaken attribution.
    """
    target_name      = target.get("name", "")
    current_employer = target.get("current_employer", "")
    profile_url      = target.get("profile_url", "")
    notes            = target.get("notes", "")
    extra_context    = target.get("extra_context", "")
    pronouns         = target.get("pronouns", "")
    excludes         = source_sites.get("exclude", [])
    topic            = focus.get("topic", "")
    industry         = focus.get("industry", "")
    geo_policy       = _build_geography_policy(focus=focus, target=target, source_sites=source_sites)

    # Inject user-pinned intel items as pre-seeded high-priority signals
    intel_items = target.get("intel_items", []) or []
    intel_signals = []
    for item in intel_items:
        itype   = item.get("type", "note")
        content = item.get("content", "").strip()
        title   = item.get("title", "") or content[:80]
        added   = item.get("added_at", date.today().isoformat())
        if not content:
            continue
        if itype == "url":
            try:
                resp = requests.get(content, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
                if resp.status_code == 200:
                    html = resp.text[:80000]
                    clean = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
                    clean = re.sub(r"<style[\s\S]*?</style>", " ", clean, flags=re.IGNORECASE)
                    clean = re.sub(r"<[^>]+>", " ", clean)
                    clean = re.sub(r"\s{3,}", "  ", clean).strip()[:500]
                    intel_signals.append({
                        "headline": f"[PINNED] {title}",
                        "url": content,
                        "date": added,
                        "snippet": clean,
                        "source": _domain(content),
                    })
                else:
                    intel_signals.append({
                        "headline": f"[PINNED URL] {title}",
                        "url": content,
                        "date": added,
                        "snippet": f"User-pinned URL (fetch returned {resp.status_code})",
                        "source": _domain(content),
                    })
            except Exception as e:
                logger.warning(f"[market_radar] Intel URL fetch failed for {content}: {e}")
                intel_signals.append({
                    "headline": f"[PINNED URL] {title}",
                    "url": content,
                    "date": added,
                    "snippet": "User-pinned URL (could not fetch content)",
                    "source": _domain(content),
                })
        else:
            intel_signals.append({
                "headline": f"[PINNED NOTE] {title}",
                "url": "",
                "date": added,
                "snippet": content,
                "source": "user-intel",
            })

    if intel_signals:
        logger.info(f"[market_radar] Injecting {len(intel_signals)} pinned intel items for '{target_name}'")

    # Inject org-shared intel items (contributed by team members)
    shared_intel_items = target.get("shared_intel_items", []) or []
    shared_signals = []
    for item in shared_intel_items:
        itype     = item.get("intel_type", "note")
        content   = item.get("content", "").strip()
        text_note = item.get("text_note", "").strip()
        title     = item.get("title", "") or content[:80]
        added     = (item.get("intel_date") or item.get("submitted_at") or date.today().isoformat())[:10]
        submitter = item.get("submitted_by", "a team member")
        if not content:
            continue
        if itype == "url":
            # Use user-provided excerpt as snippet if available; otherwise try to fetch
            if text_note:
                shared_signals.append({
                    "headline": f"[SHARED] {title}",
                    "url": content, "date": added,
                    "snippet": f"Shared by {submitter}: {text_note[:1000]}",
                    "source": _domain(content),
                })
            else:
                try:
                    resp = requests.get(content, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
                    if resp.status_code == 200:
                        html = resp.text[:80000]
                        clean = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
                        clean = re.sub(r"<style[\s\S]*?</style>", " ", clean, flags=re.IGNORECASE)
                        clean = re.sub(r"<[^>]+>", " ", clean)
                        clean = re.sub(r"\s{3,}", "  ", clean).strip()[:500]
                        shared_signals.append({
                            "headline": f"[SHARED] {title}",
                            "url": content, "date": added,
                            "snippet": f"Shared by {submitter}: {clean}",
                            "source": _domain(content),
                        })
                    else:
                        shared_signals.append({
                            "headline": f"[SHARED] {title}", "url": content, "date": added,
                            "snippet": f"Shared by {submitter} (fetch returned {resp.status_code})",
                            "source": _domain(content),
                        })
                except Exception as e:
                    shared_signals.append({
                        "headline": f"[SHARED] {title}", "url": content, "date": added,
                        "snippet": f"Shared by {submitter}",
                        "source": _domain(content),
                    })
        else:
            shared_signals.append({
                "headline": f"[SHARED NOTE] {title}", "url": "", "date": added,
                "snippet": f"Shared by {submitter}: {content}",
                "source": "shared-intel",
            })

    if shared_signals:
        logger.info(f"[market_radar] Injecting {len(shared_signals)} shared intel items for '{target_name}'")

    # Build disambiguation context for system message
    employer_desc = f" (currently at {current_employer})" if current_employer else ""
    pronouns_desc = f" | Pronouns: {pronouns}" if pronouns else ""
    context_parts = [notes, extra_context]
    all_context   = " | ".join(x for x in context_parts if x)

    this_year  = date.today().year
    last_year  = this_year - 1
    web_tool   = _make_web_search_tool(excludes, max_uses=4)
    system_msg = (
        f"You are tracking an EXTERNAL industry contact: {target_name}{employer_desc}. "
        f"This person is NOT affiliated with our company — they are an independent contact we monitor. "
        f"{geo_policy['instruction']} "
        f"Search for their recent PUBLIC activity: new job or role change, promotions, board appointments, "
        f"publications or articles written, conference speeches or podcasts, awards, interviews, and announcements. "
        f"Also search LinkedIn specifically for recent posts where they announced a career move, shared news, "
        f"or made a professional announcement — LinkedIn posts often appear in Google search results. "
        f"Focus on events from the last 6-12 months. Be specific about dates and what changed. "
        + (f"Additional context: {all_context}. " if all_context else "")
        + f"Industry: {industry} {topic}. "
        f"Return a JSON array of signal objects with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array."
    )

    # Pass 1: broad career/activity search with employer for disambiguation
    # Employer helps avoid wrong people (e.g. "Carolyn Bell" at Peoples Bank TX vs Silverchain)
    employer_q = f' "{current_employer}"' if current_employer else (f' {industry}' if industry else "")
    geo_q = " ".join(geo_policy["query_terms"])
    query = f'"{target_name}"{employer_q} {geo_q} role appointment article announcement {last_year} {this_year}'.strip()
    all_signals = _run_web_search(client, system_msg, query, web_tool)

    time.sleep(12)  # Pause between passes to avoid token rate limit

    # Pass 2: transition-focused search — catches role changes, departures, new appointments
    # Uses plain web search (not site:linkedin.com) since LinkedIn feed posts are login-gated
    transition_system = (
        f"Search for recent career transitions or announcements involving {target_name}{employer_desc}. "
        f"{geo_policy['instruction']} "
        f"Look specifically for: leaving or departing a role, joining a new organisation, "
        f"new appointments, board roles, advisory positions, or any announcement of a career change. "
        f"Search broadly — check news sites, company announcements, industry publications, and "
        f"any Google-indexed LinkedIn content. Focus on {this_year} activity. "
        f"Return a JSON array of signal objects with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array."
    )
    transition_query = f'"{target_name}" {geo_q} (joins OR appointed OR "new role" OR "leaving" OR "moving on" OR "departure" OR "delighted to announce" OR "excited to announce") {this_year}'
    transition_signals = _run_web_search(client, transition_system, transition_query, _make_web_search_tool(excludes, max_uses=3))
    all_signals = all_signals + transition_signals
    all_signals = _filter_signals_for_geography(all_signals, geo_policy, target_name=target_name)

    # Filter signals by relevance.
    # Primary: signal explicitly names the person.
    # Secondary: signal mentions their employer + a career/transition keyword
    #   (catches org-restructure articles that name the role but not the person in the title)
    name_lower     = target_name.lower()
    employer_lower = current_employer.lower() if current_employer else ""
    transition_kws = {"restructure", "departure", "resign", "executive", "cdo", "cio",
                      "digital officer", "technology", "senior", "leaving", "appointment"}
    # If target has a known LinkedIn profile URL, use it to reject wrong-person LinkedIn signals
    canonical_linkedin = ""
    if profile_url and "linkedin.com" in profile_url:
        # Normalise: strip trailing slash, lowercase path
        canonical_linkedin = profile_url.rstrip("/").lower()

    # Words from the known employer — used to detect wrong-person signals
    employer_words = set(
        w for w in re.split(r'\W+', employer_lower) if len(w) > 3
    ) if employer_lower else set()
    # Finance keywords: a signal about "Carolyn Bell at Peoples Bank" when the target
    # is at a health/aged-care org is almost certainly a namesake — filter it out.
    finance_terms = {"bank", "credit", "savings", "lending", "mortgage", "brokerage",
                     "insurance", "financial", "wealth", "fund", "invest"}

    def _is_namesake_signal(s: dict) -> bool:
        """Return True if signal is likely about a different person sharing the name."""
        if not employer_words:
            return False
        text = (s.get("headline", "") + " " + s.get("snippet", "")).lower()
        # If signal references ANY word from the known employer, it's probably correct
        if any(w in text for w in employer_words):
            return False
        # If signal is from a financial/banking organisation and target is not, discard
        employer_is_finance = any(t in employer_lower for t in finance_terms)
        if not employer_is_finance and any(t in text for t in finance_terms):
            return True
        return False

    seen_urls = set()
    signals   = []
    for s in all_signals:
        url  = s.get("url", "")
        if url in seen_urls:
            continue
        # Reject LinkedIn signals that don't match the target's known profile URL
        if canonical_linkedin and "linkedin.com" in url.lower():
            if canonical_linkedin not in url.lower():
                continue
        text = (s.get("headline", "") + " " + s.get("snippet", "")).lower()
        if name_lower in text:
            if _is_namesake_signal(s):
                logger.debug(f"[market_radar] Dropping namesake signal: {s.get('headline','')[:80]}")
                continue
            seen_urls.add(url)
            signals.append(s)
        elif employer_lower and employer_lower in text:
            if any(kw in text for kw in transition_kws):
                seen_urls.add(url)
                signals.append(s)
    logger.info(f"[market_radar] Stakeholder '{target_name}': {len(signals)} signals matched (of {len(all_signals)} returned)")
    for _s in signals:
        logger.info(f"[market_radar]   signal: {_s.get('date','?')} | {_s.get('headline','')[:80]} | {_s.get('url','')[:60]}")

    # Profile URL — LinkedIn scraping is blocked; for non-LinkedIn URLs try plain scrape
    if profile_url and profile_url.startswith("http") and "linkedin.com" not in profile_url:
        profile_signals = _scrape_profile_url(target, profile_url)
        if profile_signals:
            signals = profile_signals + signals

    return {"name": target_name, "type": "person", "notes": notes, "signals": (intel_signals + shared_signals + signals)[:10], "target": target}


# ── Step 2: Explicit URL fetches ──

def _fetch_url_signals(target: dict, include_urls: list) -> list:
    """
    For full-URL includes (start with http), fetch page content and look for target mentions.
    Returns additional signals list.
    """
    target_name = target.get("name", "").lower()
    signals = []

    for url in include_urls:
        if not url.startswith("http"):
            continue
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MarketRadarBot/1.0)"},
                allow_redirects=True,
            )
            if resp.status_code != 200:
                continue

            # Strip tags
            html = resp.text[:150_000]
            clean = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
            clean = re.sub(r"<style[\s\S]*?</style>", " ", clean, flags=re.IGNORECASE)
            clean = re.sub(r"<[^>]+>", " ", clean)
            clean = re.sub(r"\s{3,}", "  ", clean).strip()

            # Check if target is mentioned
            if target_name in clean.lower():
                # Extract a brief snippet around the mention
                idx = clean.lower().find(target_name)
                start = max(0, idx - 100)
                end = min(len(clean), idx + 300)
                snippet = clean[start:end].strip()

                signals.append({
                    "headline": f"Mention of {target.get('name')} on {url}",
                    "url":      url,
                    "date":     date.today().isoformat(),
                    "snippet":  snippet,
                    "source":   url,
                })
        except Exception as e:
            logger.debug(f"[market_radar] URL fetch failed for {url}: {e}")

    return signals


# ── Step 3: Lead synthesis via Claude Sonnet ──

def _synthesise_leads(client, all_target_data: list, focus: dict, my_company: dict,
                      system_prompt_template: str, scan_type: str = "opportunity",
                      prev_report: str = "") -> dict:
    """
    Single Sonnet call with all gathered signals. Returns structured lead JSON.
    Framing differs by scan_type to avoid misattribution and improve specificity.
    When prev_report is provided, Claude classifies each finding as new/changed/confirmed.
    """
    my_company_name     = my_company.get("name", "our company")
    my_company_services = my_company.get("services", "consulting services")
    my_company_value    = my_company.get("value_prop", "")
    my_company_clients  = my_company.get("target_clients", "")

    geo_policy = _build_geography_policy(focus=focus)
    system_prompt = system_prompt_template
    system_prompt = system_prompt.replace("{{my_company_name}}", my_company_name)
    system_prompt = system_prompt.replace("{{my_company_services}}", my_company_services)
    system_prompt = system_prompt.replace("{{my_company_value_prop}}", my_company_value)
    system_prompt = system_prompt.replace("{{my_company_target_clients}}", my_company_clients)
    system_prompt += (
        "\n\nIf a signal is marked [PINNED] or [PINNED NOTE], it is user-verified intelligence "
        "that MUST be included in the activity summary regardless of other signals. "
        "If a signal is marked [SHARED] or [SHARED NOTE], it was contributed by a team member "
        "and should be incorporated with the same priority as [PINNED] signals."
    )
    system_prompt += f"\n\nGeography rules: {geo_policy['instruction']}"

    # Build signal blocks — include employer in heading for person targets
    signal_blocks = []
    for td in all_target_data:
        if not td["signals"]:
            continue
        target_info = td.get("target", {})
        employer    = target_info.get("current_employer", "")
        label       = td["name"]
        if employer:
            label += f" — {employer}"
        block = f"### {label} [{td['type']}]\n"
        context_parts = [td.get("notes",""), target_info.get("extra_context","")]
        ctx = " | ".join(x for x in context_parts if x)
        if ctx:
            block += f"*Context:* {ctx}\n"
        for s in td["signals"]:
            block += f"- **{s.get('headline','Signal')}**\n"
            block += f"  Source: {s.get('url','')}\n"
            block += f"  Date: {s.get('date','')}\n"
            block += f"  Snippet: {s.get('snippet','')[:300]}\n"
        signal_blocks.append(block)

    if not signal_blocks:
        return {"high_priority": [], "watch_list": [], "market_context": []}

    focus_desc = (
        f"Topic: {focus.get('topic','')}, Industry: {focus.get('industry','')}, "
        f"Geography: {geo_policy.get('primary','')}, "
        f"Themes: {', '.join(focus.get('themes',[]))}"
    )
    json_schema = textwrap.dedent("""
        {
          "high_priority": [
            {"company": "...", "person": "...", "signal": "...", "opportunity": "...", "source_url": "...", "suggested_action": "...", "delta_status": "new|changed|confirmed"}
          ],
          "watch_list": [
            {"company": "...", "person": "...", "signal": "...", "opportunity": "...", "source_url": "...", "suggested_action": "...", "delta_status": "new|changed|confirmed"}
          ],
          "market_context": [
            {"topic": "...", "insight": "...", "source_url": "...", "delta_status": "new|changed|confirmed"}
          ]
        }
    """).strip()

    # Delta context block — injected when a previous report exists
    delta_block = ""
    delta_instruction = ""
    if prev_report:
        delta_block = textwrap.dedent(f"""
            ## Previous Scan Report (for comparison)
            The following is a summary from the most recent prior scan for this config.
            Use it to classify each finding in the current scan as:
            - "new"       — not mentioned or clearly absent in the previous report
            - "changed"   — was present before but with a different role, status, or signal
            - "confirmed" — same signal still active, no meaningful change

            {prev_report}

            ---
        """).strip()
        delta_instruction = (
            "\nFor every item in high_priority, watch_list, and market_context, "
            "set \"delta_status\" to \"new\", \"changed\", or \"confirmed\" "
            "by comparing against the previous scan above."
        )
    else:
        delta_instruction = (
            "\nSet \"delta_status\" to \"new\" for all items (this is the first scan)."
        )

    if scan_type == "stakeholder":
        user_message = textwrap.dedent(f"""
            Intelligence Focus: {focus_desc}

            {delta_block}

            ## Individual Contact Signals
            These are EXTERNAL industry contacts — they are NOT affiliated with {my_company_name}.
            Each section below contains signals gathered about that individual.

            {chr(10).join(signal_blocks)}

            ---
            Report on what each person has been doing recently based strictly on the signals above.
            Be specific — name the new role, company, publication, or event. Do not speculate.
            If a signal clearly shows a career move or role change, that is HIGH PRIORITY.
            If a signal shows ongoing activity (publications, speeches) without a clear change, that is WATCH LIST.

            For each item use the "person" field (not "company") formatted as "Name (Employer if known)".
            Leave "company" blank for stakeholder items unless a specific company is mentioned in the signal.
            {delta_instruction}

            Return ONLY valid JSON in this exact schema:
            {json_schema}
        """).strip()
    else:
        user_message = textwrap.dedent(f"""
            Intelligence Focus: {focus_desc}

            {delta_block}

            ## Gathered Signals by Target

            {chr(10).join(signal_blocks)}

            ---
            Based on the signals above, produce a structured lead intelligence report.
            Prioritise based on genuine buying signals for {my_company_name}.
            Do not generalise from overseas examples when the geography rules above require Australia-first coverage.
            {delta_instruction}

            Return ONLY valid JSON in this exact schema:
            {json_schema}
        """).strip()

    try:
        text = _synthesis_call(client, system_prompt, user_message, max_tokens=4096)
        return _parse_lead_json(text)
    except Exception as e:
        logger.error(f"[market_radar] Lead synthesis failed: {e}")
        return {"high_priority": [], "watch_list": [], "market_context": []}


def _synthesise_stakeholder_contacts(client, all_target_data: list, my_company: dict,
                                      system_prompt_template: str) -> list:
    """
    Stakeholder-specific synthesis: returns a simple list of verified activity items.
    Only contacts with genuine name-matched signals are included — silence for the rest.
    Returns list of {name, employer, current_role, activity, source_url, activity_type}.
    """
    signal_blocks = []
    for td in all_target_data:
        if not td["signals"]:
            continue
        target_info = td.get("target", {})
        employer  = target_info.get("current_employer", "")
        pronouns  = target_info.get("pronouns", "")
        label = td["name"] + (f" ({employer})" if employer else "")
        block = f"### {label}\n"
        if employer:
            block += f"_Known employer/context: {employer}_\n"
        if pronouns:
            block += f"_Pronouns: {pronouns}_\n"
        for s in td["signals"]:
            date_str = f" [{s['date']}]" if s.get("date") else ""
            headline = s.get("headline", "")
            block += f"- {headline}{date_str}\n"
            if s.get("snippet"):
                # Give SHARED/PINNED signals more snippet space so the LLM has full context
                is_team_intel = "[SHARED" in headline or "[PINNED" in headline
                cutoff = 800 if is_team_intel else 300
                block += f"  {s['snippet'][:cutoff]}\n"
            if s.get("url"):
                block += f"  {s['url']}\n"
        signal_blocks.append(block)

    if not signal_blocks:
        return []

    my_co = my_company.get("name", "our company")
    schema = ('{"contacts": [{'
              '"name": "...", "employer": "current or most recent employer", '
              '"current_role": "current job title or empty string if unknown", '
              '"latest_signal_date": "ISO date of the most recent signal, or null", '
              '"activity": "Plain-English summary leading with the MOST RECENT development. '
              '3-5 sentences when SHARED or PINNED signals are present; 1-2 sentences otherwise.", '
              '"source_url": "URL of the most recent or most relevant signal", '
              '"activity_type": "role_change|news|publication|award|other"'
              '}]}')

    system_msg = (
        f"You extract activity summaries for tracked external contacts. "
        f"These contacts are NOT employees of {my_co} — they are independent industry individuals. "
        f"RULES: "
        f"1. Only include a contact if the provided signals explicitly name that individual. "
        f"2. Generic industry articles with no mention of the person are NOT signals — ignore them. "
        f"3. If a contact has no name-specific signals, omit them entirely. "
        f"3a. DISAMBIGUATION: if a signal refers to a different person who happens to share the same name "
        f"(different employer, different industry, different geography), DISCARD that signal — do NOT use it. "
        f"Use the 'Known employer/context' line as your reference for which person is being tracked. "
        f"4. Depth: use 1-2 sentences for contacts with only web signals. When [SHARED] or [PINNED] "
        f"signals are present, write 3-5 sentences — incorporate the substance of those signals in full. "
        f"5. PRIORITISE RECENCY: always lead with the most recent event. Older signals are background only. "
        f"6. If any signal shows a career transition (leaving a role, new appointment, departure, restructure), "
        f"that MUST be the first sentence. Do not bury it behind older confirmations. "
        f"7. Set employer and current_role to reflect their NEWEST known position, not their old one. "
        f"8. Set latest_signal_date to the date of the most recent signal (not the oldest). "
        f"9. Set source_url to the URL of the most recent or most newsworthy signal. "
        f"10. Return ONLY valid JSON — no markdown fences, no explanation. "
        f"11. If a signal is marked [PINNED] or [PINNED NOTE], it is user-verified intelligence — "
        f"its content MUST be reflected in the activity summary in detail. "
        f"12. If a signal is marked [SHARED] or [SHARED NOTE], it was contributed by another team member "
        f"and carries the same weight as [PINNED] signals — incorporate its content fully, including "
        f"structural details, dates, names of successor roles, and direct quotes where relevant. "
        f"13. Use each person's stated pronouns (shown in the 'Pronouns' line in their context header) "
        f"when referring to them. If no pronouns are listed, use they/them."
    )
    user_msg = (
        "Signals gathered:\n\n" + "\n\n".join(signal_blocks) +
        f"\n\n---\nExtract verified activity only. Omit any contact not explicitly named in the signals above.\n"
        f"Return JSON:\n{schema}"
    )

    try:
        text = _synthesis_call(client, system_msg, user_msg, max_tokens=4096)
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []
        data = json.loads(match.group())
        return data.get("contacts", [])
    except Exception as e:
        logger.error(f"[market_radar] Stakeholder synthesis failed: {e}")
        return []


def _build_stakeholder_report(config_name: str, contacts: list, all_target_data: list,
                               today: str, my_company: dict = None) -> str:
    """
    Clean, minimal report for stakeholder scans.
    Only contacts with verified signals appear. Everyone else is listed as 'no activity'.
    """
    my_company = my_company or {}
    target_count = len(all_target_data)
    names_with_signals = {c.get("name", "").lower() for c in contacts}
    no_activity = [td["name"] for td in all_target_data
                   if td["name"].lower() not in names_with_signals]

    lines = [
        "---",
        f"title: Stakeholder Activity — {config_name}",
        f"date: {today}",
        f"config: {config_name}",
        f"targets_scanned: {target_count}",
        f"contacts_with_activity: {len(contacts)}",
        "scan_type: stakeholder",
        "source_type: market_radar",
        "---",
        "",
        f"# Stakeholder Activity: {config_name}",
        f"*{today} · {target_count} people tracked · {len(contacts)} with signals*",
        "",
    ]
    if my_company.get("name"):
        lines.append(f"**For:** {my_company['name']}")
    lines += ["", "---", ""]

    if contacts:
        lines.append("## Recent Activity")
        lines.append("")
        for c in contacts:
            name         = c.get("name", "")
            employer     = c.get("employer", "")
            role         = c.get("current_role", "")
            activity     = c.get("activity", "")
            source       = c.get("source_url", "")
            signal_date  = c.get("latest_signal_date", "")
            heading = name + (f" — {employer}" if employer else "") + (f" · *{role}*" if role else "")
            if signal_date:
                heading += f"  *(last signal: {signal_date})*"
            lines.append(f"### {heading}")
            if activity:
                lines.append(activity)
            if source:
                lines.append(f"*Source: {source}*")
            lines.append("")
    else:
        lines.append("## Recent Activity")
        lines.append("")
        lines.append("*No verifiable signals found for any tracked contact in this period.*")
        lines.append("")

    if no_activity:
        lines += ["---", "", "## No Recent Activity", ""]
        lines.append("No new signals found for:")
        for name in no_activity:
            lines.append(f"- {name}")
        lines.append("")

    return "\n".join(lines)


def _parse_lead_json(text: str) -> dict:
    """Extract and parse lead JSON from Claude response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {"high_priority": [], "watch_list": [], "market_context": []}
    try:
        data = json.loads(match.group())
        return {
            "high_priority":  data.get("high_priority", []),
            "watch_list":     data.get("watch_list", []),
            "market_context": data.get("market_context", []),
        }
    except Exception:
        return {"high_priority": [], "watch_list": [], "market_context": []}


# ── Step 4: Build Markdown report ──

_DELTA_ICONS = {"new": "🆕", "changed": "🔄", "confirmed": "✓"}


def _build_report(config_name: str, leads: dict, all_target_data: list, today: str,
                  focus: dict = None, source_sites: dict = None, my_company: dict = None,
                  scan_type: str = "opportunity", is_delta_scan: bool = False) -> str:
    target_count = len(all_target_data)
    lead_count   = len(leads.get("high_priority", [])) + len(leads.get("watch_list", []))
    signal_count = sum(len(td["signals"]) for td in all_target_data)
    focus        = focus or {}
    source_sites = source_sites or {}
    my_company   = my_company or {}

    lines = [
        "---",
        f"title: Market Intelligence Report — {config_name}",
        f"date: {today}",
        f"config: {config_name}",
        f"targets_scanned: {target_count}",
        f"lead_count: {lead_count}",
        f"scan_type: {scan_type}",
        f"is_delta: {'true' if is_delta_scan else 'false'}",
        "source_type: market_radar",
        "---",
        "",
        f"# Market Intelligence: {config_name}",
        f"*Generated: {today} · Targets: {target_count} · Signals found: {signal_count}*",
        "",
    ]

    # Delta summary line
    if is_delta_scan:
        all_items = leads.get("high_priority", []) + leads.get("watch_list", []) + leads.get("market_context", [])
        n_new       = sum(1 for x in all_items if x.get("delta_status") == "new")
        n_changed   = sum(1 for x in all_items if x.get("delta_status") == "changed")
        n_confirmed = sum(1 for x in all_items if x.get("delta_status") == "confirmed")
        lines.append(
            f"> **Delta vs previous scan:** 🆕 {n_new} new · 🔄 {n_changed} changed · ✓ {n_confirmed} confirmed"
        )
        lines.append("")

    # ── Scan context preamble ──
    preamble = []
    if my_company.get("name"):
        preamble.append(f"**For:** {my_company['name']}")
    if focus.get("topic") or focus.get("industry"):
        focus_parts = [x for x in [focus.get("topic"), focus.get("industry")] if x]
        preamble.append(f"**Focus:** {' · '.join(focus_parts)}")
    geo_policy = _build_geography_policy(focus=focus, source_sites=source_sites)
    if geo_policy.get("primary"):
        preamble.append(f"**Geography:** {geo_policy['primary']}")
    themes = focus.get("themes", [])
    if isinstance(themes, list) and themes:
        preamble.append(f"**Themes:** {', '.join(themes)}")
    target_names = [td["target"]["name"] for td in all_target_data if td.get("target", {}).get("name")]
    if target_names:
        preamble.append(f"**Targets scanned:** {', '.join(target_names)}")
    include_sites = [s for s in source_sites.get("include", []) if not s.startswith("http")]
    if include_sites:
        preamble.append(f"**Sources:** {', '.join(include_sites[:6])}")
    if preamble:
        lines += preamble + [""]
    lines.append("---")
    lines.append("")

    # ── Priority Leads ──
    hp = leads.get("high_priority", [])
    lines.append("## 🔴 Priority Leads")
    if hp:
        for i, item in enumerate(hp, 1):
            company     = item.get("company", "")
            person      = item.get("person", "")
            who         = company or person
            delta_icon  = _DELTA_ICONS.get(item.get("delta_status", "new"), "🆕")
            lines.append(f"\n### {i}. {delta_icon} {who}")
            lines.append(f"**Signal:** {item.get('signal', '')}")
            lines.append(f"\n**Opportunity:** {item.get('opportunity', '')}")
            if item.get("source_url"):
                lines.append(f"\n**Source:** {item['source_url']}")
            if item.get("suggested_action"):
                lines.append(f"\n**Suggested Action:** {item['suggested_action']}")
            lines.append("\n---")
    else:
        lines.append("*No high priority leads identified in this scan.*\n")

    # ── Watch List ──
    wl = leads.get("watch_list", [])
    lines.append("\n## 🟡 Watch List")
    if wl:
        for i, item in enumerate(wl, 1):
            company     = item.get("company", "")
            person      = item.get("person", "")
            who         = company or person
            delta_icon  = _DELTA_ICONS.get(item.get("delta_status", "new"), "🆕")
            lines.append(f"\n### {i}. {delta_icon} {who}")
            lines.append(f"**Signal:** {item.get('signal', '')}")
            if item.get("opportunity"):
                lines.append(f"\n**Opportunity:** {item['opportunity']}")
            if item.get("source_url"):
                lines.append(f"\n**Source:** {item['source_url']}")
            if item.get("suggested_action"):
                lines.append(f"\n**Suggested Action:** {item['suggested_action']}")
            lines.append("\n---")
    else:
        lines.append("*No watch list items identified.*\n")

    # ── Market Context ──
    mc = leads.get("market_context", [])
    lines.append("\n## 🔵 Market Context")
    if mc:
        for item in mc:
            delta_icon = _DELTA_ICONS.get(item.get("delta_status", "new"), "🆕")
            lines.append(f"\n{delta_icon} **{item.get('topic', 'Signal')}:** {item.get('insight', '')}")
            if item.get("source_url"):
                lines.append(f"*Source: {item['source_url']}*")
    else:
        lines.append("*No market context signals identified.*\n")

    # ── Appendix: Raw Signals ──
    lines.append("\n\n---\n## Appendix: Raw Signals")
    for td in all_target_data:
        if not td["signals"]:
            continue
        lines.append(f"\n### {td['name']} ({td['type']})")
        for s in td["signals"]:
            lines.append(f"- **{s.get('headline', '')}** — {s.get('date', '')}")
            if s.get("url"):
                lines.append(f"  {s['url']}")

    return "\n".join(lines)


# ── Step 5: Push to KB ──

def _push_to_kb(report_md: str, input_data: dict, job: dict, output_dir: Path) -> None:
    """Push the report to the website knowledge base via knowledge_api.php."""
    server_url = os.environ.get("QUEUE_SERVER_URL", "")
    if not server_url:
        logger.warning("[market_radar] QUEUE_SERVER_URL not set — skipping KB push")
        return

    try:
        kb_category = input_data.get("kb_category", "")
        config_name = input_data.get("config_name", "Market Radar")
        today       = date.today().isoformat()
        filename    = f"market-radar-{config_name.lower().replace(' ','-')}-{today}.md"

        resp = requests.post(
            server_url.rstrip("/") + "/admin/knowledge_api.php?action=upload_text",
            data={
                "content":  report_md,
                "filename": filename,
                "category": kb_category or "Market Intelligence",
                "source":   "market_radar",
            },
            timeout=30,
        )
        if resp.status_code == 200:
            logger.info(f"[market_radar] Pushed report to KB as {filename}")
        else:
            logger.warning(f"[market_radar] KB push returned {resp.status_code}")
    except Exception as e:
        logger.error(f"[market_radar] KB push failed: {e}")


# ── Main handler ──

def handle(input_path, input_data: dict, job: dict):
    """
    Main entry point called by the queue worker.
    Returns (output_data_dict, output_file_path).
    """
    config_name   = input_data.get("config_name", "Market Radar Scan")
    scan_type     = input_data.get("scan_type", "opportunity")
    targets       = input_data.get("targets", [])
    source_sites  = input_data.get("source_sites", {"include": [], "exclude": []})
    focus         = input_data.get("intelligence_focus", {})
    my_company    = input_data.get("my_company", {})
    system_prompt = input_data.get("system_prompt", "")
    push_to_kb    = input_data.get("push_to_kb", False)
    prev_report   = input_data.get("previous_report", "")
    is_delta_scan = bool(prev_report)

    client = _anthropic_client()
    today  = date.today().isoformat()

    logger.info(f"[market_radar] Starting {scan_type} scan '{config_name}' — {len(targets)} targets")

    all_target_data = []
    include_urls = [u for u in source_sites.get("include", []) if u.startswith("http")]

    if scan_type == "trends":
        # Trends: broad market scan only — no per-target searches, run multiple angles
        logger.info(f"[market_radar] TRENDS mode — broad sector scanning")
        broad_td = _gather_broad_market_signals(client, source_sites, focus)
        all_target_data.append(broad_td)
        time.sleep(1)
        # Second angle: regulatory/policy signals
        if focus.get("topic") or focus.get("industry"):
            reg_focus = dict(focus)
            reg_focus["themes"] = (focus.get("themes", []) or []) + ["regulation", "policy", "standards"]
            reg_td = _gather_broad_market_signals(client, source_sites, reg_focus)
            reg_td["name"] = "Regulatory & Policy Signals"
            all_target_data.append(reg_td)

    elif scan_type == "competitor":
        # Competitor: focused competitive intelligence per company
        logger.info(f"[market_radar] COMPETITOR mode — competitive intelligence")
        shared_intel_map = input_data.get("shared_intel", {})
        for target in targets:
            logger.info(f"[market_radar] Competitor signals for: {target.get('name')}")
            target = dict(target)
            target["shared_intel_items"] = shared_intel_map.get(target.get("name", ""), [])
            td = _gather_competitor_signals(client, target, source_sites, focus)
            if include_urls:
                td["signals"].extend(_fetch_url_signals(target, include_urls))
            all_target_data.append(td)
            time.sleep(1)
        # No targets? Fall back to broad market scan
        if not all_target_data:
            logger.warning("[market_radar] No targets — falling back to broad market scan")
            all_target_data.append(_gather_broad_market_signals(client, source_sites, focus))

    elif scan_type == "stakeholder":
        # Stakeholder: deep individual tracking — career moves, publications, speeches
        logger.info(f"[market_radar] STAKEHOLDER mode — individual tracking")
        shared_intel_map = input_data.get("shared_intel", {})
        for target in targets:
            logger.info(f"[market_radar] Stakeholder signals for: {target.get('name')}")
            target = dict(target)  # don't mutate input
            target["shared_intel_items"] = shared_intel_map.get(target.get("name", ""), [])
            td = _gather_stakeholder_signals(client, target, source_sites, focus)
            if include_urls:
                td["signals"].extend(_fetch_url_signals(target, include_urls))
            all_target_data.append(td)
            time.sleep(20)  # Pause between targets to avoid token rate limit
        # No targets? Fall back to broad market scan
        if not all_target_data:
            logger.warning("[market_radar] No targets — falling back to broad market scan")
            all_target_data.append(_gather_broad_market_signals(client, source_sites, focus))

    else:
        # Opportunity Radar (default): broad market + per-target buying signals
        logger.info(f"[market_radar] OPPORTUNITY mode — broad market + per-target")
        broad_td = _gather_broad_market_signals(client, source_sites, focus)
        all_target_data.append(broad_td)
        time.sleep(1)
        shared_intel_map = input_data.get("shared_intel", {})
        for target in targets:
            logger.info(f"[market_radar] Opportunity signals for: {target.get('name')}")
            target = dict(target)
            target["shared_intel_items"] = shared_intel_map.get(target.get("name", ""), [])
            td = _gather_signals_for_target(client, target, source_sites, focus)
            if include_urls:
                td["signals"].extend(_fetch_url_signals(target, include_urls))
            all_target_data.append(td)
            time.sleep(1)

    signal_count = sum(len(td["signals"]) for td in all_target_data)
    logger.info(f"[market_radar] Gathered {signal_count} total signals")

    # Step 3 + 4: Synthesise and build report — stakeholder uses its own lean path
    if scan_type == "stakeholder":
        logger.info("[market_radar] Synthesising stakeholder contacts...")
        contacts = _synthesise_stakeholder_contacts(client, all_target_data, my_company, system_prompt)
        lead_count = len(contacts)
        logger.info(f"[market_radar] Stakeholder synthesis complete — {lead_count} contacts with signals")
        report_md = _build_stakeholder_report(config_name, contacts, all_target_data, today, my_company=my_company)
    else:
        logger.info(f"[market_radar] Synthesising leads (delta={is_delta_scan})...")
        leads = _synthesise_leads(client, all_target_data, focus, my_company, system_prompt,
                                  scan_type=scan_type, prev_report=prev_report)
        lead_count = len(leads.get("high_priority", [])) + len(leads.get("watch_list", []))
        logger.info(f"[market_radar] Synthesis complete — {lead_count} leads identified")
        report_md = _build_report(config_name, leads, all_target_data, today,
                                  focus=focus, source_sites=source_sites, my_company=my_company,
                                  scan_type=scan_type, is_delta_scan=is_delta_scan)

    # Save output file
    job_id      = job.get("id", 0)
    safe_name   = re.sub(r"[^a-zA-Z0-9_-]", "-", config_name.lower())
    output_name = f"{job_id}_{safe_name}_{today}.md"

    # Save to a temp directory — worker uploads the file to the website via client.complete()
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    output_path = tmp_dir / output_name
    output_path.write_text(report_md, encoding="utf-8")
    logger.info(f"[market_radar] Report saved to {output_path}")

    # Step 5: Push to KB if requested
    if push_to_kb:
        _push_to_kb(report_md, input_data, job, tmp_dir)

    output_data = {
        "config_id":     input_data.get("config_id"),
        "config_name":   config_name,
        "target_count":  len(targets),
        "signal_count":  signal_count,
        "lead_count":    lead_count,
        "report_date":   today,
    }

    return {"output_data": output_data, "output_file": str(output_path)}
