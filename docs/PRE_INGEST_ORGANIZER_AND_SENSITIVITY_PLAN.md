# Pre-Ingest Organizer and Sensitivity Control Plan

Date: 2026-02-19  
Status: Proposal for enhancement planning

## Objective

Improve Cortex Suite ingestion quality and speed by adding a pre-ingestion organization layer that:

- rapidly scans a directory tree and classifies documents before expensive RAG ingestion
- identifies low-value or out-of-scope material (invoices, timesheets, admin contracts, temp files, duplicate versions)
- detects potentially sensitive/proprietary content (client confidential, external IP, restricted source materials)
- enables user-controlled ingest policy decisions (include/exclude/tag-by-class)
- produces structured metadata so search/collections can include or exclude content by policy

## Current State (Observed in Repo)

Existing capabilities already provide a strong base:

- `pages/2_Knowledge_Ingest.py`
  - recursive directory scanning and pre-filters
  - common-folder exclusion
  - pattern exclusion
  - simple dedupe/version preference logic (filename-based)
- `cortex_engine/ingest_cortex.py`
  - staged metadata review/finalization flow
  - rich metadata model (`document_type`, credibility fields)
  - duplicate guards based on `doc_id` hash
- `cortex_engine/document_type_manager.py`
  - keyword-based document type suggestions
- `credibility_spec.md`, `pages/6_Maintenance.py`
  - credibility and source-integrity normalization/retrofit patterns
- Search + synthesizer paths consume metadata filters (credibility and source integrity in places)

## Gap Summary

Current pre-ingest filtering is useful but not enough for messy enterprise repositories:

- document classing is mostly lightweight filename/path heuristics
- version grouping is basic and may miss many real-world naming conventions
- no dedicated sensitivity/proprietary classifier prior to ingestion
- no first-class ingest policy layer for `must_exclude`, `review_required`, `include_with_tag`
- limited metadata model for governance controls such as `external_ip` and `client_related`

## Why This Helps the RAG Pipeline

Adding pre-ingestion triage is likely high leverage:

- reduces embedding/index time by removing non-knowledge noise before processing
- improves retrieval precision by reducing irrelevant chunks
- lowers governance/compliance risk by avoiding accidental ingestion of restricted material
- supports trust and operational control through explicit tags and policy filters
- improves maintainability for large legacy repositories where manual sorting is overwhelming

## Practical Recommendation

Implement a dedicated **Pre-Ingest Organizer** that outputs a reviewable manifest, then let users decide policy outcomes before staging into ingestion.

Do not start with fully automated file moves. Start with scan + classify + decision support, then optional copy/sync workflows.

## Proposed Architecture

### 1) Fast Pre-Scan Index

For each candidate file, collect:

- path, filename, extension, size, modified time
- file hash (`get_file_hash`) for exact dedupe
- lightweight text sample (first N chars/pages/paragraphs)
- header/footer/title tokens where possible (see sensitivity section)

Output to external DB root, not repo:

- `<db_path>/pre_ingest/pre_ingest_manifest.json`
- `<db_path>/pre_ingest/pre_ingest_manifest.csv` (optional for manual review)

### 2) Classification + Policy Engine

Generate preliminary fields:

- `doc_class`: `work_knowledge|admin_finance|legal_contract|draft|duplicate|unknown`
- `ingest_recommendation`: `include|exclude|review`
- `ingest_reason`: concise explanation list
- `version_group_id`, `version_rank`, `is_canonical`

### 3) Sensitivity / Proprietary Detection Layer (New Requirement)

Detect whether material appears client-sensitive or external IP using multiple signals:

- directory signal:
  - path contains known client names or restricted folders
  - folder policy map from user config (for example `do_not_ingest/*`)
- filename signal:
  - keywords like `confidential`, `client`, `internal`, `proprietary`, `NDA`
  - organization names (for example Deloitte, McKinsey) from configurable registry
- document text signal:
  - header/footer extraction and first-page scan for confidentiality legends
  - phrases such as `Confidential`, `For internal use only`, `Client privileged`
  - IP/copyright language
- source ownership signal:
  - known external publisher/org patterns
  - known client identifier matches

Recommended metadata outputs:

- `sensitivity_level`: `public|internal|confidential|restricted|unknown`
- `source_ownership`: `first_party|client_owned|external_ip|third_party_unknown`
- `client_related`: string or list (for example `["ClientA"]`)
- `external_ip_owner`: string (for example `Deloitte`, `McKinsey`) when detected
- `policy_class`: `safe_default|review_required|do_not_ingest`
- `policy_confidence`: float `0..1`
- `policy_reasons`: string list

### 4) Human Governance Controls

Before ingestion, provide user controls to:

- bulk include/exclude by `policy_class`, `sensitivity_level`, `source_ownership`
- confirm ingestion of `review_required` items
- hard-block ingestion of `do_not_ingest` unless explicit override
- assign target collections based on metadata class

### 5) Ingestion Integration

Only selected rows flow into `staging_ingestion.json`.
Persist all governance metadata into document metadata so search/synthesis/collections can filter by policy.

## Specific Answer to the Proprietary/Client Material Question

Your requirement is sensible and practical. It is common in enterprise RAG pipelines.

Most practical approach:

- treat sensitivity detection as a probabilistic classifier with explainable rules
- never rely on a single signal (filename alone is insufficient)
- enforce conservative defaults: uncertain sensitive items should go to `review_required`
- keep a user-managed policy registry for client names, restricted directories, and external IP owners

This balances safety and usability without requiring perfect AI classification.

## Suggested Policy Defaults

- `do_not_ingest`
  - explicit restricted directories
  - high-confidence confidential markers with client identifiers
- `review_required`
  - mixed or uncertain signals
  - external IP likely but not certain
- `safe_default`
  - no sensitivity flags and classed as useful knowledge material

## Versioning and Duplicate Strategy (Improved)

Use both filename conventions and hash/content similarity:

- detect versions using suffix patterns (`v1`, `v2`, `final`, `(001)`, dated variants)
- group by normalized base name + directory context
- prefer canonical candidate by:
  1. explicit latest marker (if reliable)
  2. modified timestamp
  3. content richness (length/pages)
- keep non-canonical variants tagged for optional archival ingestion

## Data Model Extension (Proposed)

Extend metadata with governance fields for every ingested document:

- `ingest_policy_class`
- `ingest_policy_confidence`
- `ingest_policy_reasons`
- `sensitivity_level`
- `source_ownership`
- `client_related`
- `external_ip_owner`
- `version_group_id`
- `is_canonical_version`

This supports downstream collection and search filters without custom ad-hoc logic.

## Rollout Plan (Phased)

### Phase 1 (Low Risk, High Value)

- add pre-ingest scanner manifest generation
- add rule-based classing + version grouping + sensitivity signals
- add review UI table with include/exclude toggles
- no file moves, no destructive actions

### Phase 2

- integrate policy filters into staging/finalization path
- persist governance metadata to Chroma metadata
- add collection presets by policy class

### Phase 3

- optional curated copy/sync output to user-specified destination
- optional advanced classifier model for borderline cases
- maintenance audits and drift reports

## Risks and Mitigations

- False positives on sensitivity:
  - Mitigation: default to `review_required`, expose reasons, allow overrides
- False negatives on proprietary content:
  - Mitigation: hard folder deny lists and conservative keyword rules
- User friction:
  - Mitigation: bulk actions and saved policy profiles
- Performance overhead:
  - Mitigation: lightweight first-pass extraction and batched processing

## Implementation Notes for Cortex Constraints

- Keep all artifacts under user DB path (external storage), not repo root
- Respect path normalization rules (`convert_windows_to_wsl_path` outside Docker)
- Avoid introducing LlamaIndex dependency in search paths
- Reuse existing staged metadata workflow to minimize change surface
- Keep Docker copies synchronized when this moves from plan to implementation

## Recommended Next Build Slice

Build a scan-only organizer prototype that outputs:

- manifest with `doc_class`, version grouping, and sensitivity/proprietary tags
- policy recommendation per file (`include|exclude|review`)
- a simple pre-ingest review table in UI to approve final candidate set

This is the fastest way to validate accuracy and user workflow before deeper automation.
