# Cortex Intelligence Enhancement Spec

## Context

The Market Radar system has two workers:
- **market_radar handler** (longboardfella_website/worker/) — does Claude web search for targets, produces markdown reports
- **Cortex signal_digest handler** (cortex_suite/worker/) — reads from stakeholder signal store, synthesises digests

The current problem: **neither worker leverages the relationship graph or produces cross-referenced intelligence**. The market_radar handler gathers web signals but treats each target independently. The signal_digest handler has access to the graph DB but only filters pre-stored signals by profile key — no analysis.

The user's core need: "Have Haiku add value via Cortex together with its knowledge of relationships from the graph database."

## What the Website Sends to Cortex

### Profile Sync Payload (stakeholder_profile_sync job)
```json
{
  "org_name": "Escient",
  "profiles": [
    {
      "external_profile_id": "42",
      "canonical_name": "Jane Smith",
      "target_type": "person",
      "current_employer": "BigBank",
      "current_role": "VP Digital Transformation",
      "linkedin_url": "https://linkedin.com/in/janesmith",
      "aliases": ["J. Smith"],
      "known_employers": ["BigBank", "OldCorp", "ConsultCo"],
      "affiliations": [
        {"org_name_text": "BigBank", "role": "VP Digital", "affiliation_type": "current", "confidence": "confirmed", "is_primary": 1},
        {"org_name_text": "ConsultCo", "role": "Senior Manager", "affiliation_type": "former"}
      ],
      "tags": ["decision-maker", "digital-transformation"],
      "watch_status": "watch",
      "linkedin_connections": [
        {"member": "paul@escient.com.au", "degree": "1st"}
      ],
      "alumni": ["McKinsey", "University of Melbourne"]
    }
  ],
  "org_alumni": ["McKinsey", "BCG", "Deloitte"]
}
```

### Watch Report Job (market_radar type, source_system: 'watch_report')
```json
{
  "config_name": "Watch Report — Escient (priority)",
  "scan_type": "stakeholder",
  "targets": [
    {"name": "Jane Smith", "type": "person", "current_employer": "BigBank", "profile_url": "...", "extra_context": "VP Digital Transformation"}
  ],
  "intelligence_focus": {"topic": "digital transformation", "industry": "financial services"},
  "my_company": {"name": "Escient", "services": "management consulting", ...},
  "shared_intel": {"Jane Smith": [{"content": "https://article-about-jane...", "text_note": "Mentioned restructure"}]},
  "watch_report_meta": {"org_name": "Escient", "digest_tier": "priority", "report_depth": "detailed"}
}
```

### Signal Digest Job (signal_digest type)
```json
{
  "org_name": "Escient",
  "since_ts": "2026-03-15T00:00:00Z",
  "profile_keys": ["sha1-hash-1", "sha1-hash-2"],
  "digest_tier": "priority",
  "report_depth": "detailed",
  "org_alumni": ["McKinsey", "BCG"]
}
```

## Current Shortcomings

### 1. No Relationship Intelligence
The graph DB stores connections between people, orgs, and affiliations. But signal_digest doesn't query the graph for:
- "Jane Smith moved from OldCorp to BigBank — who else from OldCorp is now at BigBank?" (cluster detection)
- "BigBank is hiring from consulting firms — is this a pattern?" (talent flow analysis)
- "Jane and Bob both worked at McKinsey — alumni connection could be leverage" (relationship path finding)

### 2. No Cross-Target Analysis
Each target is researched independently. The system doesn't look for:
- Connections between watched targets (e.g., two watched people at the same conference)
- Organisational moves affecting multiple targets (e.g., a merger impacting several watched contacts)
- Industry trends affecting the whole watch list

### 3. No Intelligence Compounding
Web research results from market_radar scans are NOT fed back into the Cortex signal store. Each scan starts fresh. There's no cumulative intelligence building — findings from last week's scan aren't available as context for this week's analysis.

### 4. Synthesis Doesn't Leverage Context
The market_radar handler's synthesis prompt doesn't ask Claude to:
- Highlight findings that are NEW vs already known from shared intel
- Identify contradictions between sources
- Rate confidence levels on each finding
- Suggest relationship-based follow-up actions

## Required Enhancements

### Enhancement 1: Signal Ingest from Watch Reports (Close the Loop)

**When a `market_radar` watch report job completes**, Cortex should automatically ingest the gathered signals into the stakeholder signal store so they're available for future analysis.

Flow:
1. market_radar handler produces output with `source_system: 'watch_report'`
2. Website harvests completed job into watch_reports table
3. **NEW**: Cortex also ingests the raw signals (not just the synthesised report) into StakeholderSignalStore
4. Future signal_digest jobs now have richer data to work with

Implementation options:
- A. market_radar handler posts signals back to Cortex via API after completion
- B. Cortex watches for completed market_radar jobs with watch_report source and ingests
- C. Website queues a `signal_ingest` job with the raw signals after harvesting

Option C is simplest — the website already has the output data during harvest.

### Enhancement 2: Graph-Enriched Intelligence Layer

**During signal_digest or as a new `intelligence_analysis` job type**, Cortex should use its graph DB to enrich raw signals with relationship context.

For each watched person:
1. **Career trajectory**: Query graph for employment history → detect patterns (rising star, lateral move, stepping stone)
2. **Network proximity**: Query connections between this person and the subscribing org's team → find warm introduction paths
3. **Alumni clusters**: Find other watched people who share alma maters or former employers → identify potential alliances or competitive dynamics
4. **Org influence map**: For watched organisations, map which watched people are connected to it and how → identify decision-maker clusters

For each watched organisation:
1. **Talent flow**: Who has joined/left recently (from profile change signals)? What does the direction of talent flow suggest?
2. **Shared connections**: Which of the subscriber's team members have connections into this org?
3. **Competitive positioning**: Are competitors also moving people into this org's sector?

Output format (append to existing digest):
```markdown
## Relationship Intelligence

### Network Connections
- **Jane Smith** (BigBank): 1st-degree LinkedIn connection via Paul Cooper
  - Shared alumni: McKinsey (both 2018-2020 cohort)
  - Also connected to Bob Jones (watched, OldCorp) — former colleagues at ConsultCo

### Career Patterns
- **BigBank** has hired 3 people from consulting firms in the last 6 months → likely building internal capability
- **Jane Smith** role change from "Senior Manager" to "VP" in 8 months — fast-track trajectory

### Warm Introduction Paths
- To reach **Tom Lee** (CFO, MegaCorp): Paul → Jane Smith (1st) → Tom Lee (Jane's former board colleague)

### Cross-Target Signals
- Jane Smith and Bob Jones both attended "Digital Banking Summit 2026" → potential joint engagement opportunity
- OldCorp restructure affecting both Sarah Park (watched) and Bob Jones (watched) — coordinate intelligence
```

### Enhancement 3: Confidence-Weighted Signal Ranking

Cortex should rank signals by confidence before synthesis:
- **High confidence**: Direct company announcements, SEC filings, press releases from target org
- **Medium confidence**: News articles, industry publications, conference attendance
- **Low confidence**: Social media mentions, unverified blog posts, secondhand reports
- **User-confirmed**: Shared intel marked as verified by a team member

The synthesis step should explicitly flag confidence levels and highlight where multiple independent sources confirm the same signal (corroboration).

### Enhancement 4: Temporal Pattern Detection

Across multiple watch report cycles, Cortex should detect:
- **Acceleration**: Target org suddenly generating 3x normal signal volume → something brewing
- **Silence**: Previously active target goes quiet → may indicate internal turmoil or stealth mode
- **Cyclical patterns**: Annual budget cycles, quarterly earnings, regulatory reporting periods
- **Trend inflection**: Sentiment shift (positive→negative or vice versa) across multiple signals

This requires the signal store to retain historical data and the digest step to query across time windows.

### Enhancement 5: Actionable Intelligence Scoring

Each insight in the digest should be scored on:
- **Relevance** (0-10): How relevant to the subscriber org's services and target market?
- **Urgency** (0-10): Time-sensitivity — is there a window of opportunity?
- **Confidence** (0-10): How well-sourced is this intelligence?
- **Actionability** (0-10): Can the subscriber org act on this immediately?

Top-scored insights should be flagged for immediate attention. Low-scored items go into "monitoring" bucket.

## Priority Order

1. **Enhancement 1** (Signal Ingest — close the loop) — without this, nothing compounds
2. **Enhancement 2** (Graph-Enriched Intelligence) — this is the core differentiator
3. **Enhancement 4** (Temporal Pattern Detection) — enables trend analysis
4. **Enhancement 3** (Confidence Ranking) — improves signal quality
5. **Enhancement 5** (Actionable Scoring) — improves report usefulness

## Website-Side Changes Needed

### For Enhancement 1 (Signal Ingest)
After harvesting a completed market_radar watch report job, the website should queue a `signal_ingest` job with the raw signals:

```php
// In harvest_digests action, after inserting watch_report:
if ($isWatchScan && !empty($output['raw_signals'])) {
    $ingestPayload = [
        'org_name'    => $jobOrg,
        'signals'     => $output['raw_signals'],  // array of {target, headline, url, date, snippet}
        'source'      => 'market_radar_watch',
        'source_job'  => $jobId,
    ];
    mrQueueCortexJob($qdb, 'signal_ingest', $ingestPayload, 'system', 'anytime', 0);
}
```

This requires the market_radar handler to include `raw_signals` in its output_data (currently it only outputs the synthesised report markdown).

### For Enhancement 2 (Graph Intelligence)
The website's `generate_watch_report` action could optionally queue a second job — `intelligence_analysis` — after the market_radar web search completes. Or Cortex could chain this internally.

### For Enhancements 3-5
These are primarily Cortex-side changes. The website just needs to render the additional metadata (confidence badges, urgency flags, scores) in the report viewer.

## Technical Notes

- The market_radar handler (Python, longboardfella_website/worker/) uses Claude Haiku for web search and Claude Sonnet for synthesis
- It currently outputs `output_data` with: `config_id`, `config_name`, `scan_type`, `signal_count`, `lead_count`, `report_date`
- The `raw_signals` field needs to be added to `output_data` for Enhancement 1
- The handler gathers signals per target in `all_target_data` list (each entry has `name`, `type`, `signals` list)
- Cortex graph DB schema and capabilities should be documented by the Cortex agent

## Success Criteria

A successful implementation means:
1. Running a watch report for "Jane Smith at BigBank" produces intelligence that includes:
   - Recent web findings (news, announcements, publications)
   - Relationship context (who she's connected to, alumni overlap, introduction paths)
   - Cross-references with other watched targets
   - Confidence-rated findings with corroboration notes
   - Delta from previous report (new/changed/confirmed)
2. Each successive watch report is richer than the last (compounding intelligence)
3. Reports clearly distinguish between user-submitted intel and newly discovered intelligence
4. Graph-based insights surface connections the user wouldn't have found manually
