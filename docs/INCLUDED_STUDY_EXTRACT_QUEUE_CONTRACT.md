# Included Study Extract Queue Contract

Purpose: define the planned queue contract for website-driven systematic review PDF extraction into grouped included-study table outputs.

Status:
- Proposed contract
- Not yet implemented as a queue worker handler
- Intended queue job type: `included_study_extract`

## Goal

Allow the website to:
1. Upload a systematic review PDF
2. Choose extraction options such as `All trials/studies` vs `RCT/clinical trials only`
3. Submit a Cortex queue job
4. Receive structured grouped table outputs plus downloadable artifacts
5. Optionally pass selected citations into the existing `research_resolve` workflow

## Queue Registration

Required new queue job type:

```text
included_study_extract
```

Expected new Cortex worker handler:

```text
worker/handlers/included_study_extract.py
```

Expected worker registration points:
- `worker/handlers/__init__.py`
- `worker/config.env.example` `SUPPORTED_TYPES`
- if applicable, website queue type allowlist such as `queue_api_shared.php`

## Input Mode

This job should be file-backed.

- `input_path`: uploaded systematic review PDF
- `input_data`: JSON options and metadata only

The worker should treat the downloaded queue input file as the source review PDF.

## Proposed Input Contract

```json
{
  "trace_id": "trace-optional",
  "idempotency_key": "optional",
  "source_system": "website",
  "tenant_id": "default",
  "project_id": "included_study_extract",
  "review_title": "optional display title",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "fallback_provider": "gemini",
  "fallback_model": "gemini-2.5-flash",
  "extraction_scope": "rct_or_clinical",
  "output_detail": "reference_map",
  "include_low_value_tables": false,
  "download_formats": ["json", "xlsx"],
  "resolver_defaults": {
    "check_open_access": true,
    "enrich_sjr": true,
    "unpaywall_email": ""
  }
}
```

## Input Field Semantics

- `provider`
  - optional
  - expected values: `anthropic`, `gemini`
  - recommended default: `anthropic`

- `model`
  - optional
  - recommended default: `claude-sonnet-4-6`

- `fallback_provider`, `fallback_model`
  - optional
  - use only if the primary provider fails or is unavailable

- `extraction_scope`
  - required
  - allowed values:
    - `all_trials`
    - `rct_or_clinical`

- `output_detail`
  - required
  - allowed values:
    - `reference_map`
    - `detailed_fields`
  - recommended website default: `reference_map`

- `include_low_value_tables`
  - optional boolean
  - if `false`, low-value/reference-free tables such as some HTA summary tables may be excluded from primary outputs

- `download_formats`
  - optional string array
  - allowed values:
    - `json`
    - `xlsx`
    - `csv`
  - recommended default: `["json", "xlsx"]`

- `resolver_defaults`
  - optional
  - becomes the default options embedded in resolver payloads and resolver queue jobs generated from the extraction

## Expected Worker Behavior

The new `included_study_extract` worker handler should:

1. Validate the queue input PDF exists and is a PDF
2. Run the existing included-study slicer
3. Extract bibliography text/entries
4. Run per-table extraction using the selected provider/model
5. Produce grouped included-study outputs for each detected table
6. Build per-table:
   - selection CSV
   - selection XLSX
   - resolver payload JSON
   - resolver queue job JSON
   - website handoff JSON
7. Build a single ZIP bundle containing all table artifacts
8. Return both:
   - structured `output_data`
   - `output_file` pointing to the ZIP bundle

## Proposed output_data Contract

```json
{
  "status": "completed",
  "source_workflow": "included_study_extractor",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "review_title": "A systematic literature review...",
  "input_filename": "review.pdf",
  "extraction_scope": "rct_or_clinical",
  "output_detail": "reference_map",
  "bibliography_entry_count": 78,
  "table_count": 3,
  "tables": [
    {
      "table_label": "table 2",
      "table_number": "2",
      "table_title": "Overview of included studies on HRQOL measures",
      "kind": "included_studies",
      "group_count": 12,
      "citation_count": 23,
      "needs_review_count": 0,
      "page_numbers": [6, 7, 8, 9],
      "artifacts": {
        "selection_csv": "tables/table_2/included_study_selection.csv",
        "selection_xlsx": "tables/table_2/included_study_selection.xlsx",
        "resolver_payload_json": "tables/table_2/research_resolver_payload.json",
        "resolver_queue_job_json": "tables/table_2/research_resolver_queue_job.json",
        "website_handoff_json": "tables/table_2/included_study_website_handoff.json",
        "table_pdf": "tables/table_2/table_2.pdf"
      }
    }
  ],
  "warnings": [],
  "result_bundle_name": "included_study_extract_bundle.zip"
}
```

## output_file Contract

The worker should return:

```text
output_file = path to a ZIP bundle
```

The queue framework already supports this:
- `output_data` is stored in the completed job record
- `output_file` is uploaded as the queue result file

Recommended ZIP layout:

```text
manifest.json
bibliography/bibliography.txt
bibliography/bibliography.csv
tables/table_2/table_2.pdf
tables/table_2/included_study_selection.csv
tables/table_2/included_study_selection.xlsx
tables/table_2/research_resolver_payload.json
tables/table_2/research_resolver_queue_job.json
tables/table_2/included_study_website_handoff.json
tables/table_3/...
tables/table_4/...
combined/included_study_website_handoff.json
combined/research_resolver_payload.json
combined/research_resolver_queue_job.json
```

## Website Consumption Model

The website should:

1. Submit the PDF as a queue job of type `included_study_extract`
2. Poll until complete
3. Read `output_data` for summary and per-table metadata
4. Offer `output_file` ZIP for download
5. Optionally render per-table grouped outputs directly from:
   - `website_handoff_json` inside the ZIP, or
   - mirrored grouped JSON inside `output_data` if later added

## Resolver Follow-On Contract

Each per-table output should include:
- `research_resolver_payload.json`
- `research_resolver_queue_job.json`

Website follow-on behavior:
- preferred: use `resolver_queue_job`
- fallback: build a `research_resolve` job from `resolver_payload`

## Validation Rules

The future handler should reject:
- non-PDF queue inputs
- missing `extraction_scope`
- invalid `extraction_scope`
- invalid `output_detail`
- invalid `provider` if specified

## Recommended Defaults

Website defaults:

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "extraction_scope": "rct_or_clinical",
  "output_detail": "reference_map",
  "include_low_value_tables": false,
  "download_formats": ["json", "xlsx"]
}
```

## Implementation Notes

- The extraction logic already exists in Streamlit under:
  - `cortex_engine/included_study_slicer.py`
  - `cortex_engine/included_study_extractor.py`
  - `pages/7_Document_Extract.py`
- What does not yet exist is the queue worker wrapper for this flow.
- This contract is designed so the website can be built now while the Cortex queue handler is implemented next.
