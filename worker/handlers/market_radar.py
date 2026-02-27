"""
Market Radar Handler
Scans the web for market intelligence signals about specified targets,
synthesises a prioritised lead report, and saves it as a Markdown file.

No new dependencies — uses anthropic (existing) and requests (standard).

input_data schema:
    config_id           int
    config_name         str
    targets             list[{name, type, notes}]
    source_sites        {include: [domains], exclude: [domains]}
    intelligence_focus  {topic, industry, themes}
    my_company          {name, description, services, value_prop, target_clients}
    system_prompt       str
    member_email        str
    email_results       bool
    push_to_kb          bool
    kb_category         str
    source_system       str  (lab|cron|admin)
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
                            raw_results.append({
                                "headline": getattr(result, "title", ""),
                                "url":      getattr(result, "url", ""),
                                "date":     None,
                                "snippet":  "",
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

    if not context_terms:
        return {"name": "Market Overview", "type": "broad", "notes": "", "signals": []}

    web_tool   = _make_web_search_tool(excludes, max_uses=4)
    system_msg = (
        f"You are a market intelligence researcher scanning for broad market trends and opportunities "
        f"in: {context_terms}. Search for recent news, industry movements, regulatory changes, "
        f"leadership announcements, digital transformation initiatives, and market activity. "
        f"Return a JSON array of signal objects, each with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array, no other text."
    )
    query = f"Latest news and market intelligence: {context_terms} 2024 2025"
    signals = _run_web_search(client, system_msg, query, web_tool)
    logger.info(f"[market_radar] Broad market scan found {len(signals)} signals")
    return {"name": "Market Overview", "type": "broad", "notes": "", "signals": signals}


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

    web_tool   = _make_web_search_tool(excludes, max_uses=3)
    system_msg = (
        f"You are a market intelligence researcher. Search for recent news and signals about "
        f"{'company' if target_type == 'company' else 'person'}: {target_name}. "
        f"Look for: new leadership, digital initiatives, restructures, funding, challenges, "
        f"regulatory pressures, strategic announcements, and consulting opportunities. "
        f"{'Also look for: ' + context_terms + '.' if context_terms else ''} "
        f"{'Notes: ' + notes if notes else ''} "
        f"Return a JSON array of signal objects, each with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array, no other text."
    )
    query = f"{target_name} {context_terms} news announcements 2024 2025".strip()
    signals = _run_web_search(client, system_msg, query, web_tool)
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
    web_tool    = _make_web_search_tool(excludes, max_uses=4)
    system_msg  = (
        f"You are a competitive intelligence researcher. Search for recent news about competitor firm: {target_name}. "
        f"Look specifically for: client wins or losses, new service offerings, leadership hires or departures, "
        f"pricing changes, public failures or criticisms, strategic announcements, and partnerships. "
        f"Industry context: {topic} {industry}. "
        f"Return a JSON array of signal objects with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array."
    )
    query = f"{target_name} consulting news wins losses clients strategy 2024 2025 {industry}".strip()
    signals = _run_web_search(client, system_msg, query, web_tool)
    logger.info(f"[market_radar] Competitor '{target_name}': {len(signals)} signals")
    return {"name": target_name, "type": "company", "notes": target.get("notes",""), "signals": signals, "target": target}


def _gather_stakeholder_signals(client, target: dict, source_sites: dict, focus: dict) -> dict:
    """Individual tracking: career moves, publications, speeches, board roles."""
    target_name = target.get("name", "")
    notes       = target.get("notes", "")
    excludes    = source_sites.get("exclude", [])
    topic       = focus.get("topic", "")
    industry    = focus.get("industry", "")
    web_tool    = _make_web_search_tool(excludes, max_uses=4)
    system_msg  = (
        f"You are a relationship intelligence researcher. Search for recent activity by: {target_name}. "
        f"Look for: new job or role change, board appointments, publications or articles written, "
        f"conference speeches or podcasts, LinkedIn posts, strategic announcements, awards, and interviews. "
        f"{'Notes: ' + notes if notes else ''} "
        f"Industry focus: {topic} {industry}. "
        f"Return a JSON array of signal objects with: headline, url, date (ISO), snippet, source. "
        f"Return ONLY the JSON array."
    )
    query = f'"{target_name}" {industry} {topic} announcement role speech article 2024 2025'.strip()
    signals = _run_web_search(client, system_msg, query, web_tool)
    logger.info(f"[market_radar] Stakeholder '{target_name}': {len(signals)} signals")
    return {"name": target_name, "type": "person", "notes": notes, "signals": signals, "target": target}


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

def _synthesise_leads(client, all_target_data: list, focus: dict, my_company: dict, system_prompt_template: str) -> dict:
    """
    Single Sonnet call with all gathered signals. Returns structured lead JSON.
    """
    # Interpolate My Company into system prompt
    my_company_name     = my_company.get("name", "our company")
    my_company_services = my_company.get("services", "consulting services")
    my_company_value    = my_company.get("value_prop", "")
    my_company_clients  = my_company.get("target_clients", "")

    system_prompt = system_prompt_template
    system_prompt = system_prompt.replace("{{my_company_name}}", my_company_name)
    system_prompt = system_prompt.replace("{{my_company_services}}", my_company_services)
    system_prompt = system_prompt.replace("{{my_company_value_prop}}", my_company_value)
    system_prompt = system_prompt.replace("{{my_company_target_clients}}", my_company_clients)

    # Build user message with all signals
    signal_blocks = []
    total_signals = 0
    for td in all_target_data:
        if not td["signals"]:
            continue
        block = f"### {td['name']} ({td['type']})\n"
        if td.get("notes"):
            block += f"*Notes:* {td['notes']}\n"
        for s in td["signals"]:
            block += f"- **{s.get('headline','Signal')}**\n"
            block += f"  Source: {s.get('url','')}\n"
            block += f"  Date: {s.get('date','')}\n"
            block += f"  Snippet: {s.get('snippet','')}\n"
            total_signals += 1
        signal_blocks.append(block)

    if not signal_blocks:
        return {
            "high_priority": [],
            "watch_list": [],
            "market_context": [],
        }

    focus_desc = f"Topic: {focus.get('topic','')}, Industry: {focus.get('industry','')}, Themes: {', '.join(focus.get('themes',[]))}"

    user_message = textwrap.dedent(f"""
        Intelligence Focus: {focus_desc}

        ## Gathered Signals by Target

        {chr(10).join(signal_blocks)}

        ---
        Based on the signals above, produce a structured lead intelligence report.
        Return ONLY valid JSON in this exact schema:
        {{
          "high_priority": [
            {{"company": "...", "person": "...", "signal": "...", "opportunity": "...", "source_url": "...", "suggested_action": "..."}}
          ],
          "watch_list": [
            {{"company": "...", "person": "...", "signal": "...", "opportunity": "...", "source_url": "...", "suggested_action": "..."}}
          ],
          "market_context": [
            {{"topic": "...", "insight": "...", "source_url": "..."}}
          ]
        }}
        Prioritise based on genuine buying signals for {my_company_name}.
    """).strip()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text if response.content else ""
        return _parse_lead_json(text)
    except Exception as e:
        logger.error(f"[market_radar] Lead synthesis failed: {e}")
        return {"high_priority": [], "watch_list": [], "market_context": []}


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

def _build_report(config_name: str, leads: dict, all_target_data: list, today: str,
                  focus: dict = None, source_sites: dict = None, my_company: dict = None,
                  scan_type: str = "opportunity") -> str:
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
        "source_type: market_radar",
        "---",
        "",
        f"# Market Intelligence: {config_name}",
        f"*Generated: {today} · Targets: {target_count} · Signals found: {signal_count}*",
        "",
    ]

    # ── Scan context preamble ──
    preamble = []
    if my_company.get("name"):
        preamble.append(f"**For:** {my_company['name']}")
    if focus.get("topic") or focus.get("industry"):
        focus_parts = [x for x in [focus.get("topic"), focus.get("industry")] if x]
        preamble.append(f"**Focus:** {' · '.join(focus_parts)}")
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
            company = item.get("company", "")
            person  = item.get("person", "")
            who     = company or person
            lines.append(f"\n### {i}. {who}")
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
            company = item.get("company", "")
            person  = item.get("person", "")
            who     = company or person
            lines.append(f"\n### {i}. {who}")
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
            lines.append(f"\n**{item.get('topic', 'Signal')}:** {item.get('insight', '')}")
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
    secret_key = os.environ.get("QUEUE_SECRET_KEY", "")
    if not server_url:
        logger.warning("[market_radar] QUEUE_SERVER_URL not set — skipping KB push")
        return
    if not secret_key:
        logger.warning("[market_radar] QUEUE_SECRET_KEY not set — skipping KB push")
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
                "_secret":  secret_key,
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
        for target in targets:
            logger.info(f"[market_radar] Competitor signals for: {target.get('name')}")
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
        for target in targets:
            logger.info(f"[market_radar] Stakeholder signals for: {target.get('name')}")
            td = _gather_stakeholder_signals(client, target, source_sites, focus)
            if include_urls:
                td["signals"].extend(_fetch_url_signals(target, include_urls))
            all_target_data.append(td)
            time.sleep(1)
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
        for target in targets:
            logger.info(f"[market_radar] Opportunity signals for: {target.get('name')}")
            td = _gather_signals_for_target(client, target, source_sites, focus)
            if include_urls:
                td["signals"].extend(_fetch_url_signals(target, include_urls))
            all_target_data.append(td)
            time.sleep(1)

    signal_count = sum(len(td["signals"]) for td in all_target_data)
    logger.info(f"[market_radar] Gathered {signal_count} total signals")

    # Step 3: Synthesise leads
    logger.info(f"[market_radar] Synthesising leads...")
    leads = _synthesise_leads(client, all_target_data, focus, my_company, system_prompt)

    lead_count = len(leads.get("high_priority", [])) + len(leads.get("watch_list", []))
    logger.info(f"[market_radar] Synthesis complete — {lead_count} leads identified")

    # Step 4: Build report
    report_md = _build_report(config_name, leads, all_target_data, today,
                              focus=focus, source_sites=source_sites, my_company=my_company,
                              scan_type=scan_type)

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
