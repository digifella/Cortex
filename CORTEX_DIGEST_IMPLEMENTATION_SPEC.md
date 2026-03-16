# Cortex Signal Digest Implementation Spec

**Version:** 1.0
**Date:** 2026-03-13
**Status:** Working implementation spec for Cortex + website alignment

## Purpose

This document translates the website agent's depth/tier concept into an implementation plan that matches the current Cortex architecture.

It also records two adjacent integration realities discovered during review:

1. `signal_digest` currently operates on already-ingested stakeholder signals held in the external Cortex signal store.
2. Intel mailbox intake does not create queue jobs today; it performs local extraction in Cortex and posts the result directly to the website import endpoint.

This is the spec Cortex should build against until the website and Cortex agents agree on any broader redesign.

## Current Reality

### Digest path today

- Queue job type: `signal_digest`
- Handler: [worker/handlers/signal_digest.py](/home/longboardfella/cortex_suite/worker/handlers/signal_digest.py)
- Store: [cortex_engine/stakeholder_signal_store.py](/home/longboardfella/cortex_suite/cortex_engine/stakeholder_signal_store.py)
- Input currently supported:
  - `org_name`
  - `since_ts`
  - `profile_keys`
  - `max_items`
  - `include_needs_review`
  - `matched_only`
  - `llm_synthesis`
  - `llm_provider`
  - `llm_model`
- Output today:
  - markdown file written under the external DB root: `stakeholder_signals/digests/{digest_id}.md`
  - queue `output_data` with digest metadata
  - uploaded queue output file

### Mailbox path today

- Worker: [worker/intel_mailbox_worker.py](/home/longboardfella/cortex_suite/worker/intel_mailbox_worker.py)
- Processing: [cortex_engine/intel_mailbox.py](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py)
- Website import endpoint:
  [site/admin/queue_worker_api.php](/home/longboardfella/longboardfella_website/site/admin/queue_worker_api.php)
  action `import_cortex_extract`
- This path does **not** create a queue job, so:
  - there is no queue completion row to watch
  - the queue completion email notifier does not run
  - Intel Board visibility depends on website-side import and display logic, not queue status

## Non-Goals For This Phase

- Do not redesign `signal_digest` into a generic web-search orchestrator.
- Do not require LlamaIndex on the search path.
- Do not move digest artifacts into the repo.
- Do not hardcode paths; continue using the configured external DB root.
- Do not assume the website can consume Cortex-local filesystem paths.

## Target Outcome

Implement `signal_digest` as a **multi-stage signal analysis pipeline** over existing stored signals, with optional enrichment stages for deeper reports.

Depth controls **how much analysis Cortex performs** on the already-ingested signal set.

Tier controls **which profile cohort the website asked Cortex to analyse**, but Cortex should still accept explicit `profile_keys` as the final execution filter.

## Required Contract Changes

### Input payload

Extend `validate_signal_digest_input()` to accept and normalize:

- `digest_tier`: `priority|standard|all`
- `report_depth`: `summary|detailed|strategic`
- `deep_analysis`: boolean
- `priority_profile_keys`: string array

Rules:

- If `report_depth` is missing, default to `detailed`.
- If `deep_analysis=true`, effective depth must be at least `detailed`.
- `profile_keys` remains the final filter actually used by Cortex.
- `digest_tier` is metadata unless and until both sides agree Cortex should derive the active cohort itself.
- Unknown `report_depth` values should fail validation.

### Output payload

Extend `output_data` to include:

- `report_depth`
- `digest_tier`
- `analysis_mode`
- `escalate`
- `escalate_profiles`
- `escalate_reason`
- `signal_count`
- `profiles_covered`
- `period_start`
- `period_end`
- `output_filename`

Rules:

- `output_path` should not be treated as a portable website contract.
- The uploaded file is the transport artifact the website should use.
- `escalate_profiles` must contain `profile_key` values, not display names.

## Digest Execution Model

### Shared Stage 1: Collect

Source all candidate items from `StakeholderSignalStore`:

- filter by `org_name`
- filter by `since_ts` when provided
- filter by `profile_keys` when provided
- respect `matched_only`
- respect `include_needs_review`
- cap final working set to `max_items`

This remains the only mandatory retrieval stage for all depths.

### Shared Stage 2: Structure

Before any synthesis, Cortex must build a structured working dataset per profile:

- matched `profile_key`
- canonical stakeholder name
- current employer and current role when available from the matched profile
- signal subject
- signal timestamp
- signal type
- source URL
- extracted facts
- update suggestions
- review state

This stage is required so deeper reports are not a single free-form LLM pass over raw notes.

### Depth: `summary`

Goal:

- fast scan of the current signal window

Steps:

- mechanical grouping by stakeholder
- select highest-signal item per stakeholder
- optional concise synthesis pass to improve wording
- run escalation classifier

Output shape:

- title + period + counts
- `## Key Signals`
- one short bullet per stakeholder

Constraints:

- no recommendations
- no competitive implications section
- no strategic framing unless `escalate=true`

### Depth: `detailed`

Goal:

- per-stakeholder signal analysis with explicit context

Steps:

- complete Stage 1 and Stage 2
- generate deterministic per-stakeholder buckets first
- synthesise each bucket into:
  - recent signals
  - trend notes
  - context
- produce one final editorial pass only after section data exists

Output shape:

- title + period + counts
- one section per stakeholder
- recent signals subsection
- trend notes subsection

Constraints:

- must remain grounded in stored signal data plus matched profile metadata
- do not fabricate employer, role, or prior state if absent

### Depth: `strategic`

Goal:

- deeper brief with second-pass reasoning, not just longer prose

Steps:

- complete Stage 1 and Stage 2
- run first-pass stakeholder synthesis as in `detailed`
- run a second-pass strategic synthesis over the structured stakeholder summaries
- derive:
  - executive summary
  - competitive implications
  - suggested follow-up actions
  - risk/opportunity flags
  - market positioning notes

Allowed optional enrichment in this depth:

- additional local classification over signal themes
- optional external search or enrichment stage, but only after website/Cortex contract is agreed

Output shape:

- strategic title
- executive summary
- per-stakeholder sections
- suggested follow-up actions
- risk & opportunity flags
- market positioning notes

Constraint:

- if no external enrichment stage is implemented, strategic commentary must clearly infer only from stored signals and profile metadata

## Escalation Logic

Escalation is only required for `summary` depth.

Trigger categories:

- leadership changes
- major announcements
- regulatory actions
- M&A activity
- multiple related signals on the same stakeholder within the window

Implementation rule:

- produce escalation from structured data plus a classifier output
- do not attempt to scrape profile keys back out of free-form markdown

Output rule:

- `escalate=true` only when at least one concrete `profile_key` is attached
- `escalate_reason` should be one short sentence

## LLM Design Requirements

### Summary

- one short synthesis pass is acceptable

### Detailed

- must be at least two-step:
  - step 1: structured stakeholder section generation
  - step 2: final markdown assembly or light editorial cleanup

### Strategic

- must be at least two-pass:
  - pass 1: stakeholder analysis
  - pass 2: cross-stakeholder strategic reasoning

This is the minimum change needed to satisfy the requirement that some digests do more than one-pass synthesis.

## Storage and File Contract

- Continue writing the canonical digest artifact under the external DB path managed by `StakeholderSignalStore`.
- Continue returning `output_file` to the queue worker so the website receives the uploaded markdown artifact.
- Add `output_filename` in `output_data` if the website needs a stable reference to the uploaded file.
- Treat the Cortex-local `output_path` as diagnostic only.

## Website Integration Notes

These points need explicit three-way agreement later, but Cortex should code defensively now:

- If the website already resolved the cohort, Cortex should trust `profile_keys`.
- If the website wants separate priority and standard jobs, that is compatible with Cortex.
- If the website wants Cortex to derive priority membership from `digest_tier`, that is a future change and not assumed here.

## Mailbox Integration Notes

The mailbox issue reported on 2026-03-13 is explained by current architecture:

- Cortex mailbox extraction succeeds locally and posts directly to the website import API.
- No queue job is created for these mailbox extracts.
- Therefore no queue completion row appears and no queue completion email is sent.

Observed local evidence:

- latest processed mailbox result:
  [results/4aa3d1295f6276c6.json](/mnt/f/ai_databases/intel_mailbox/results/4aa3d1295f6276c6.json)
- callback recorded as HTTP 200 in:
  [state.json](/mnt/f/ai_databases/intel_mailbox/state.json)

Website-side implication:

- Intel Board currently lists `shared_intel` records, not mailbox extraction results directly, in
  [market_radar_api.php](/home/longboardfella/longboardfella_website/site/lab/market_radar_api.php#L3277)

Follow-up implementation options for later discussion:

1. Keep direct mailbox import, but add a mailbox results view or summary email path.
2. Change mailbox intake to queue an `intel_extract` job so queue status and notifier flows apply.
3. Keep direct import and also write an audit row visible in the Intel Board UI.

## Logging Requirement

Reduce terminal noise without hiding real work:

- queue worker should stay quiet during idle polling
- mailbox worker should not log every empty poll cycle at `INFO`
- keep `INFO` logs for:
  - worker start/stop
  - claimed jobs
  - completed jobs
  - processed mailbox messages
  - failures and warnings
- move routine successful meta-sync and low-value payload dumps to `DEBUG`

## Implementation Checklist

### Cortex

- Update `validate_signal_digest_input()` for new depth/tier fields.
- Extend `signal_digest` handler to pass effective depth/tier metadata into the store.
- Refactor `StakeholderSignalStore.generate_digest()` to support:
  - effective depth resolution
  - structured per-profile working data
  - summary/detailed/strategic output variants
  - escalation metadata
  - multi-pass synthesis for `detailed` and `strategic`
- Add unit tests covering:
  - default depth fallback
  - `deep_analysis` upgrade behaviour
  - summary escalation output
  - strategic multi-pass path
  - output payload fields

### Website

- Confirm whether the website consumes uploaded files, `output_filename`, or local `output_path`.
- Decide whether mailbox processing should remain direct-post or become queue-backed.
- Decide where mailbox extraction results should be visible to analysts.

## Open Decisions For Joint Discussion

- Is `digest_tier` authoritative input or advisory metadata?
- Should mailbox extraction appear in queue UI, Intel Board UI, or both?
- Should completion emails be sent for direct mailbox imports, or only queue jobs?
- Does `strategic` depth require agreed web-search enrichment, or is multi-pass signal reasoning sufficient for phase 1?
