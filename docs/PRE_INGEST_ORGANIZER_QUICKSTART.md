# Pre-Ingest Organizer Quickstart

## What it does

The pre-ingest organizer runs a fast scan before normal ingestion and writes:

- `<db_path>/pre_ingest/pre_ingest_manifest.json`

The manifest contains per-file recommendations:

- `ingest_policy_class`: `include|exclude|review_required|do_not_ingest`
- `doc_class`
- `sensitivity_level`
- `source_ownership` (`first_party|client_owned|external_ip`)
- `client_related`
- `external_ip_owner`
- version grouping fields (`version_group_id`, `is_canonical_version`)

## How to run (UI)

1. Open `Knowledge Ingest`.
2. Set source and database paths.
3. Select one or more directories.
4. Expand `Pre-Ingest Organizer (Recommended for messy repositories)`.
5. Click `Run Pre-Ingest Organizer ...`.
6. Review the summary and manifest path shown in the UI.

## Current behavior notes

- This is Phase 1 scan-only behavior (no file moves).
- It does not yet auto-apply manifest decisions to staging/finalization.
- `do_not_ingest` is triggered by explicit restricted path markers such as:
  - `do_not_ingest`
  - `restricted`
  - `never_ingest`
  - `blocked_ingest`
- Sensitive/client/external-IP detections default to `review_required` unless stronger exclusion rules apply.

## Next planned integration

- Stage only `include` records by default.
- Allow explicit user override for `review_required`.
- Persist policy metadata through ingestion for collection/search filtering.
