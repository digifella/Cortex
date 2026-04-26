# Cortex Signal Digest — Depth & Tier Specification

**Version:** 1.0
**Date:** 2026-03-13
**For:** Cortex worker agent implementing `signal_digest` job type

---

## Overview

The website queues `signal_digest` jobs to Cortex via the shared queue DB. Cortex performs web search, signal matching, and LLM synthesis, then writes the result back. The website harvests completed jobs and displays the reports.

This spec adds two new dimensions to the job payload:
- **Tier** (`digest_tier`): which profiles are included — already implemented
- **Depth** (`report_depth`): how deep the analysis goes — NEW

---

## Job Input Payload

The `input_data` JSON on a `signal_digest` job contains:

```json
{
  "org_name": "Acme Corp",
  "since_ts": "2026-03-12T08:00:00Z",
  "profile_keys": ["a1b2c3d4e5f6g7h8", "..."],
  "priority_profile_keys": ["a1b2c3d4e5f6g7h8"],
  "digest_tier": "priority|standard",
  "report_depth": "summary|detailed|strategic",
  "deep_analysis": true,
  "max_items": 100,
  "matched_only": true,
  "include_needs_review": true,
  "llm_synthesis": true,
  "llm_provider": "ollama",
  "trace_id": "tr_abc123...",
  "source_system": "market_radar",
  "tenant_id": "default",
  "project_id": "default"
}
```

### New field: `report_depth`

| Value | Label | Description |
|-------|-------|-------------|
| `summary` | Quick Scan | Concise overview. Key signals as short bullets. No recommendations. Fastest/cheapest. |
| `detailed` | Full Analysis | Per-stakeholder breakdown with signal categorisation, trend notes, and context. |
| `strategic` | Strategic Brief | Everything in detailed PLUS competitive implications, suggested follow-up actions, risk/opportunity flags, and market positioning notes. |

**Default:** If `report_depth` is missing or empty, treat as `detailed` (backward compatible).

### Existing field: `digest_tier`

| Value | Meaning |
|-------|---------|
| `priority` | High-value profiles. Deeper web search, more items, higher job priority. |
| `standard` | Regular profiles. Standard search depth and item limits. |

**Typical combinations:**

| Trigger | Tier | Depth | max_items |
|---------|------|-------|-----------|
| Daily cron — standard profiles | standard | summary | 50 |
| Daily cron — priority profiles | priority | detailed | 100 |
| On-demand — user choice | either | user-selected | 50-100 |
| Escalation follow-up | priority | strategic | 100 |

---

## Expected Output

### `output_data` JSON

```json
{
  "signal_count": 12,
  "profiles_covered": 5,
  "digest_id": "dig_abc123",
  "report_depth": "detailed",
  "escalate": false,
  "escalate_profiles": [],
  "escalate_reason": ""
}
```

### New output fields

| Field | Type | Description |
|-------|------|-------------|
| `report_depth` | string | Echo back the depth that was actually used |
| `escalate` | boolean | `true` if Cortex recommends a deeper follow-up scan |
| `escalate_profiles` | string[] | Profile keys that triggered the escalation |
| `escalate_reason` | string | Brief reason (e.g. "Leadership change detected at 2 orgs") |

**Escalation logic:** When processing a `summary` depth job, if the LLM detects high-signal events (leadership changes, major announcements, regulatory actions, M&A activity), set `escalate: true` with the relevant profile keys. The website cron will auto-queue a `strategic` depth follow-up for just those profiles.

### Markdown output file

Written to `queue_files/outputs/{job_id}_output.md` (or as configured).

The markdown structure should vary by depth:

#### `summary` depth
```markdown
# Signal Digest: {org_name}
**Period:** {since_ts} → {now}
**Profiles:** {count} | **Signals:** {count}

## Key Signals
- **{stakeholder_name}**: {one-line signal summary}
- **{stakeholder_name}**: {one-line signal summary}
...
```

#### `detailed` depth
```markdown
# Signal Digest: {org_name}
**Period:** {since_ts} → {now}
**Profiles:** {count} | **Signals:** {count}

## {Stakeholder Name} 🔥
**Role:** {role} at {employer}

### Recent Signals
- **{signal_title}** ({date}) — {category}
  {2-3 sentence analysis}
- ...

### Trend Notes
{paragraph on patterns, trajectory, relevance}

---
## {Next Stakeholder}
...
```

#### `strategic` depth
```markdown
# Strategic Intelligence Brief: {org_name}
**Period:** {since_ts} → {now}
**Profiles:** {count} | **Signals:** {count}

## Executive Summary
{3-5 sentence overview of most significant findings}

## {Stakeholder Name} 🔥
**Role:** {role} at {employer}

### Recent Signals
- **{signal_title}** ({date}) — {category}
  {2-3 sentence analysis}

### Trend Notes
{paragraph on patterns, trajectory}

### Competitive Implications
{how this affects the org's market position}

---
## {Next Stakeholder}
...

---

## Suggested Follow-Up Actions
1. {Specific actionable recommendation}
2. {Specific actionable recommendation}
3. ...

## Risk & Opportunity Flags
| Flag | Type | Stakeholder | Detail |
|------|------|-------------|--------|
| {emoji} | Risk/Opportunity | {name} | {description} |

## Market Positioning Notes
{paragraph on broader market context and how these signals relate}
```

---

## LLM Prompt Guidance

When constructing the LLM prompt for synthesis, adjust based on `report_depth`:

- **summary**: "Produce a concise bullet-point digest. One line per stakeholder with the most important signal. No recommendations or analysis."
- **detailed**: "Produce a structured per-stakeholder analysis. Include signal categorisation, trend notes, and context for each. Group by stakeholder."
- **strategic**: "Produce a strategic intelligence brief. Include per-stakeholder analysis, competitive implications, risk/opportunity flags, and specific follow-up action recommendations. Think like a market intelligence analyst advising a senior executive."

---

## Web Search Guidance

| Depth | Search breadth |
|-------|---------------|
| `summary` | 1-2 searches per stakeholder, recent news only |
| `detailed` | 2-4 searches per stakeholder, include industry context |
| `strategic` | 3-5 searches per stakeholder, include competitors, regulatory, market trends |

---

## Backward Compatibility

- If `report_depth` is missing from `input_data`, default to `detailed`
- If `escalate` fields are missing from output, the website treats it as no escalation
- Existing `deep_analysis` boolean still respected — `true` means at minimum `detailed` depth
- The `digest_tier` and `report_depth` are independent axes (tier = who, depth = how deep)

---

## Testing

To test, queue a job manually:

```sql
INSERT INTO jobs (type, status, input_data, created_at, created_by, trace_id, schedule_type, priority)
VALUES ('signal_digest', 'pending', '{"org_name":"Test Org","profile_keys":["abc123"],"digest_tier":"standard","report_depth":"summary","max_items":10,"llm_synthesis":true,"llm_provider":"ollama","source_system":"market_radar","tenant_id":"default","project_id":"default"}', datetime('now'), 'test', 'tr_test123', 'anytime', 0);
```

Check that the worker:
1. Reads `report_depth` from input
2. Adjusts search breadth and LLM prompt accordingly
3. Writes appropriate markdown structure
4. Returns `escalate` fields in `output_data`
5. Falls back to `detailed` when `report_depth` is missing
