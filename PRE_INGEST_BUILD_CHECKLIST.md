# Pre-Ingest Organizer Build Checklist

Last Updated: 2026-02-20
Owner: Codex session
Status: In Progress

## Goal

Implement a resumable pre-ingest organizer workflow that scans source directories, classifies documents, identifies sensitivity/proprietary risk, and outputs a manifest the user can review before ingestion.

## Progress Legend

- [ ] Not started
- [~] In progress
- [x] Completed

## Phase 1: Scan-Only Organizer (No file moves)

- [x] Create persistent checklist document in repo root
- [x] Implement organizer engine module for scanning + classification
- [x] Add manifest writer under external DB path (`<db_path>/pre_ingest/`)
- [x] Add sensitivity/proprietary heuristic tagging
- [x] Add version-family grouping + canonical selection
- [x] Add include/exclude/review policy recommendation fields
- [x] Add unit tests for classification and version grouping
- [x] Add minimal UI hook in Knowledge Ingest page to run pre-ingest scan
- [x] Add docs on how to run/use pre-ingest organizer
- [x] Add live pre-ingest scan log output (directories/files/progress)
- [x] Add pause/resume/stop controls for long-running scans
- [x] Add editable manifest review table in UI
- [x] Add bulk row-range/filter update actions for large manifests
- [x] Add directory-specific timestamped manifests + combined snapshots
- [x] Add resume workflow to load and continue editing saved manifests without re-scan

## Phase 2: Staging Integration

- [ ] Add option to stage only `include` docs from manifest
- [ ] Add optional inclusion of `review` docs via explicit user confirmation
- [ ] Persist governance metadata into staged docs for final ingestion

## Phase 3: Policy and Filtering Enhancements

- [ ] Add user policy registry (client names, restricted folders, external IP owners)
- [ ] Add collection filter support for governance metadata in search/synthesis
- [ ] Add maintenance report for policy-tagged documents

## Session Notes

- Initial request confirmed to proceed with implementation and maintain this checklist for crash/session recovery.
- Phase 1 baseline implemented:
  - `cortex_engine/pre_ingest_organizer.py`
  - `pages/2_Knowledge_Ingest.py` pre-ingest organizer UI expander and trigger
  - `tests/unit/test_pre_ingest_organizer.py`
  - `docs/PRE_INGEST_ORGANIZER_QUICKSTART.md`
- Expanded Phase 1 UX completed:
  - live log + progress while scanning
  - pause/resume/stop controls
  - editable review table with controlled field options
  - bulk updates by displayed row range / filter / all rows
  - filtered display mode with continuous display row numbers
  - auto-load manifest on scan completion
  - manifest picker with per-file summary stats
  - directory-specific and timestamped manifest outputs for re-entry
- Targeted test run passed:
  - `pytest -q tests/unit/test_pre_ingest_organizer.py`
