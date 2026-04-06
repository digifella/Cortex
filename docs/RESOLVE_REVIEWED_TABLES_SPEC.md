# Resolve Reviewed Tables Spec

Purpose: define the website workflow for uploading reviewed included-study table files and sending the selected rows into `research_resolve`.

## Goal

Allow a user to:
1. Download and review included-study extraction outputs from Cortex
2. Unzip the result bundle locally
3. Upload one or more reviewed table files back to the website
4. Resolve the selected citations to papers or fallback web/markdown targets
5. Download the resolver outputs as a ZIP

This is a separate workflow from the PDF extraction job. Do not couple it directly to the extractor output pane.

## Page Placement

Recommended location:
- same page as the Included Study Extractor

But keep it as a distinct section:
- `Extract Included Study Tables`
- `Resolve Reviewed Table Files`

These sections must not share submission buttons or state.

## Primary Use Case

Start with:
- `tables/table_3/included_study_selection.csv`

from a successful Included Study extraction bundle.

The website should parse rows where `keep = true` and submit only those rows to `research_resolve`.

## Accepted Upload Types

Preferred upload types:
- `included_study_selection.csv`
- `included_study_selection.xlsx`

Optional advanced/debug upload types:
- `research_resolver_payload.json`
- `research_resolver_queue_job.json`

The main user-facing workflow should be CSV/XLSX upload.

## Expected CSV/XLSX Columns

The upload parser should understand these columns:
- `keep`
- `row_id`
- `table_number`
- `table_title`
- `grouping_basis`
- `group_label`
- `trial_label`
- `combined_group`
- `citation_display`
- `title`
- `authors`
- `year`
- `doi`
- `journal`
- `reference_number`
- `study_design`
- `sample_size`
- `outcome_measure`
- `outcome_result`
- `notes`
- `needs_review`

Minimum practical requirement:
- `title` or `citation_display`

## Selection Logic

For CSV/XLSX uploads:
- include a row if `keep` is truthy
- accepted truthy values:
  - `true`
  - `TRUE`
  - `1`
  - `yes`
  - `y`

If there is no `keep` column:
- default all rows to included
- show a warning to the user

## Upload UI

Section title:
- `Resolve Reviewed Table Files`

UI controls:
1. file upload for one or more CSV/XLSX/JSON files
2. resolver options:
   - `check_open_access`
   - `enrich_sjr`
   - `unpaywall_email`
3. parsed summary per file:
   - table number
   - table title
   - total rows
   - selected rows
4. preview grid of selected citations
5. primary button:
   - `Send Selected Rows to Research Resolver`
6. results area
7. `Download Resolver Results ZIP`

## Queue Submission Contract

### CSV/XLSX Uploads

Convert the selected rows into a `research_resolve` payload.

Build each citation from selected rows:

```json
{
  "row_id": 1,
  "title": "Shah 2021 [38]",
  "authors": "Shah et al.",
  "year": "2021",
  "doi": "",
  "journal": "",
  "notes": "",
  "extra_fields": {
    "table_number": "3",
    "table_title": "Overview of included studies on health state utility values",
    "grouping_basis": "Utility assessment method and region",
    "group_label": "EQ-5D / Global",
    "trial_label": "SADAL",
    "combined_group": "EQ-5D / Global / SADAL",
    "reference_number": "38",
    "citation_display": "Shah 2021 [38]",
    "needs_review": "",
    "study_design": "",
    "sample_size": "",
    "outcome_measure": "",
    "outcome_result": ""
  }
}
```

Submit as a new queue job:

```json
{
  "job_type": "research_resolve",
  "input_data": {
    "citations": [...],
    "options": {
      "check_open_access": true,
      "enrich_sjr": true,
      "unpaywall_email": ""
    },
    "source_workflow": "included_study_extractor",
    "included_study_context": {
      "upload_mode": "reviewed_table_upload"
    }
  }
}
```

### `research_resolver_payload.json`

If the uploaded file is already a resolver payload:
- validate it
- allow optional citation deselection in UI
- submit as a new `research_resolve` queue job

### `research_resolver_queue_job.json`

If the uploaded file is already a queue job:
- use it as the base object
- optionally filter `input_data.citations` to the current selection
- submit as a new queue job

## Result Handling

When the resolver job completes, produce a downloadable ZIP containing:

- `manifest.json`
- `resolver_job.json`
- `resolver_output.json`
- `resolved_citations.csv`
- `resolved_citations.xlsx`
- `unresolved_citations.csv`
- `unresolved_citations.xlsx`
- `url_ingest_candidates.json`

## Result Columns

Resolved outputs should include:
- `row_id`
- `table_number`
- `table_title`
- `combined_group`
- `citation_display`
- `reference_number`
- `resolved_title`
- `resolved_authors`
- `resolved_year`
- `doi`
- `journal`
- `best_url`
- `open_access_url`
- `landing_page_url`
- `resolver_status`
- `notes`

## PDF / Markdown Fallback Requirement

If no open paper PDF is available:
- keep the best landing-page / publisher / registry / abstract URL
- include it in `url_ingest_candidates.json`
- do not fail the whole resolver run

This allows the next workflow step to ingest:
- PDF URL when available
- web/markdown fallback when PDF is unavailable

## Recommended UX Copy

Use labels like:
- `Upload reviewed table files for paper resolution`
- `Send selected rows to Research Resolver`

Avoid implying this is tied only to the extractor page state.

## Acceptance Criteria

1. User can upload `tables/table_3/included_study_selection.csv` from a recent Included Study bundle.
2. Website parses `keep = true` rows only.
3. Website submits a valid `research_resolve` queue job.
4. Website returns a ZIP with resolved and unresolved outputs.
5. Multiple reviewed table files can be uploaded together.
6. No dependency on the previous extractor session state.

## Current Recommended Test Fixture

Use the latest successful Table 3 reviewed-table file from the current bundle:
- `tables/table_3/included_study_selection.csv`

This file currently contains:
- 14 total rows
- 10 selected rows (`keep = true`)

It is the best first integration target for the website-side reviewed-table upload flow.
