# Pre-Ingest Organizer Quickstart

## What it does

The pre-ingest organizer runs a fast scan before normal ingestion and writes:

- `<db_path>/pre_ingest/pre_ingest_manifest.json` (latest alias)
- `<db_path>/pre_ingest/pre_ingest_manifest_all_<timestamp>.json` (combined snapshot)
- `<db_path>/pre_ingest/pre_ingest_manifest_<directory>_<timestamp>.json` (directory-specific snapshot)

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
6. After scan completion, the manifest is auto-loaded into the editable review table.
7. Edit rows as needed, then click `Save Decisions to Manifest`.
8. Click `Prepare Ingest From Approved Policies` to move approved rows into ingest review.

## Resume/edit without re-scan

You can continue manifest refinement later without selecting directories:

1. Open `Knowledge Ingest`.
2. Ensure DB path is set.
3. Expand `Pre-Ingest Organizer`.
4. Choose a file in `Manifest File` and click `Load Selected Manifest`.
5. Continue edits and save.

Tip: prefer timestamped manifest files over the alias (`pre_ingest_manifest.json`) when resuming detailed work.

## Editing and bulk update behavior

- Table edits are local until `Save Decisions to Manifest` is clicked.
- Bulk actions are available for large manifests:
  - `Apply To Row Range`
  - `Apply To Filtered Rows`
  - `Apply To All Loaded Rows`
- Optional row-hiding filters:
  - Hide `include`
  - Hide `do_not_ingest`
- Row numbers are display indices after filtering, so range selection is continuous (`1..N`).

## Current behavior notes

- This is Phase 1 scan-only behavior (no file moves).
- It does not yet write decisions directly into staging automatically.
- `do_not_ingest` is triggered by explicit restricted path markers such as:
  - `do_not_ingest`
  - `restricted`
  - `never_ingest`
  - `blocked_ingest`
- Internal business-planning markers (for example `sales`, `marketing plan`, `pipeline report`, `quarterly report`, `budget`) are tagged as `sensitivity_level=internal`.
- Sensitive/client/external-IP detections default to `review_required` unless stronger exclusion rules apply.

## Next planned integration

- Stage only `include` records by default.
- Allow explicit user override for `review_required`.
- Persist policy metadata through ingestion for collection/search filtering.
