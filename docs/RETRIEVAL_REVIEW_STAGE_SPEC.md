## Retrieval Review Stage Spec

Purpose: define the website stage that sits after `Resolve + Retrieve Papers` and lets a human finish the retrieval loop for blocked publisher pages.

### Goal

The system is now good at:
- extracting included-study citations
- resolving them to likely papers
- attempting automatic retrieval
- capturing markdown fallbacks when PDF retrieval fails

The weak point is publisher access. This stage adds a human-in-the-loop review workflow so the user can:
- see which papers were retrieved automatically
- open the remaining resolved URLs manually
- upload PDFs that are openly available but blocked to automation
- mark rows as web-only, unavailable, or wrong match
- build one final researcher package

### Entry Point

This should be a separate stage on the same website page as:
- `Extract Included Study Tables`
- `Resolve Reviewed Table Files`

Recommended new section title:
- `Review Retrieval Results`

### Inputs

The stage should accept either:
- a completed `research_resolve_bundle.zip`
- or a completed `research_resolve_result.json` plus companion CSVs

Preferred input artifact:
- `research_resolve_bundle.zip`

Expected bundle contents:
- `research_resolve_result.json`
- `resolved_citations.csv`
- `unresolved_citations.csv`
- `preferred_urls.txt`
- `url_ingest_candidates.json`
- optional `retrieval/` subtree with:
  - `reports/*.csv`
  - `reports/*.json`
  - `pdfs/*`
  - `markdown/*`

### Core UX

For each resolved citation row, show:
- `table_number`
- `combined_group`
- `reference_number`
- `citation_display`
- `matched_title`
- `resolved_doi`
- `resolved_url`
- `open_access_pdf_url`
- retrieval status
- whether a PDF file exists in the bundle
- whether a markdown/web artifact exists in the bundle

Add filters:
- `All`
- `PDF retrieved`
- `Resolved but no PDF`
- `Web only`
- `Unresolved`
- `Needs manual review`

### Row Actions

Each row should support:
- `Open best URL`
- `Open DOI`
- `Open PDF URL` when present
- `Open web markdown` when present
- `Upload PDF`
- `Mark web only`
- `Mark unavailable`
- `Mark wrong match`
- `Exclude from final package`

### Row Status Model

Do not use a single checkbox only.

Use a row status field with these values:
- `retrieved_auto_pdf`
- `retrieved_auto_web`
- `retrieved_manual_pdf`
- `web_only_confirmed`
- `unavailable`
- `wrong_match`
- `excluded`

Optional companion fields:
- `review_notes`
- `manual_url`
- `manual_pdf_filename`
- `reviewed_by`
- `reviewed_at`

### Website Behavior

Automatic detection on upload:
- if a row has a downloaded PDF in the bundle, prefill status as `retrieved_auto_pdf`
- if a row has markdown but no PDF, prefill status as `retrieved_auto_web`
- otherwise leave it blank and show it in the manual review queue

Manual review behavior:
- user can click out to the publisher/registry page
- if they obtain a PDF manually, upload it against that row
- uploaded PDF changes row status to `retrieved_manual_pdf`
- if no PDF exists but the landing page is useful, set `web_only_confirmed`
- if the resolution is incorrect, set `wrong_match`
- rows marked `excluded` are omitted from the final researcher package

### Upload Requirements

For manual PDF upload:
- allow one PDF per resolved row
- keep the upload associated with:
  - `row_id`
  - `reference_number`
  - `citation_display`
  - `matched_title`

Recommended storage in the review session:
- `manual_uploads/<row_id>_<safe_slug>.pdf`

### Final Researcher Package

Add button:
- `Build Final Researcher Package`

The final package should include:
- all automatically retrieved PDFs
- all manually uploaded PDFs
- markdown/web captures
- final reviewed CSV
- unresolved/missing CSV
- manifest JSON

Recommended output structure:

```text
researcher_package.zip
  manifest.json
  final_reviewed_citations.csv
  final_reviewed_citations.xlsx
  missing_or_unavailable.csv
  resolved/
    pdfs/
    markdown/
    manual_pdfs/
  originals/
    research_resolve_result.json
    resolved_citations.csv
    unresolved_citations.csv
    preferred_urls.txt
    url_ingest_candidates.json
```

### Final Reviewed CSV Columns

Include at least:
- `row_id`
- `table_number`
- `combined_group`
- `reference_number`
- `citation_display`
- `matched_title`
- `resolved_doi`
- `resolved_url`
- `open_access_pdf_url`
- `auto_pdf_present`
- `auto_markdown_present`
- `manual_pdf_uploaded`
- `final_status`
- `review_notes`
- `final_package_path`

### Suggested Workflow

1. User uploads reviewed table CSV/XLSX files.
2. User clicks `Resolve + Retrieve Papers`.
3. Website shows resolver/retrieval results.
4. User moves to `Review Retrieval Results`.
5. User filters to `Resolved but no PDF`.
6. User manually opens likely publisher pages.
7. User uploads any PDFs they can access.
8. User marks remaining rows as `web_only_confirmed`, `unavailable`, or `wrong_match`.
9. User clicks `Build Final Researcher Package`.

### Acceptance Criteria

1. Website can ingest a `research_resolve_bundle.zip`.
2. Automatically retrieved PDFs are visible as completed rows.
3. Rows with only markdown are visible as review-needed rows.
4. User can upload a PDF against a specific citation row.
5. User can set a final row status.
6. Final package includes both auto and manual artifacts.
7. Excluded or wrong-match rows do not appear as completed researcher papers.

### Important Product Note

This stage should be presented as a retrieval review queue, not as a failure screen.

The framing should be:
- `Cortex resolved the references and retrieved what it could automatically. Review the remaining items and add any accessible papers manually.`

### Current Backend Status

Already implemented in Cortex:
- `research_resolve` resolution
- preferred URL generation
- automatic retrieval
- fallback markdown capture
- combined retrieval bundle output

Not yet implemented on the website:
- row-level retrieval review state
- manual PDF upload binding to resolved citation rows
- final curated researcher package builder
