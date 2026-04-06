## Streamlit Retrieval Review Workflow

Purpose: document the local Cortex workflow for reviewing `research_resolve_bundle.zip` outputs without using the website.

### Entry Point

Open:

- `pages/7_Document_Extract.py`
- tab: `Research Resolver`
- section: `Review Retrieval Results`

This stage mirrors the website retrieval-review flow and is intended for local human-in-the-loop finishing of paper retrieval.

### What It Accepts

The Streamlit stage can use either:

- the current local resolver run directory, if `research_resolve_bundle.zip` exists there
- an uploaded external `research_resolve_bundle.zip`

Preferred artifact:

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

### What The Review Stage Shows

For each citation row, Streamlit derives:

- table metadata
- citation display text
- resolved DOI and URL
- retrieval status/reason
- whether an automatic PDF is present
- whether an automatic markdown/web artifact is present
- final review status
- review notes
- manual PDF upload state

Headline metrics include:

- resolved rows
- automatic PDFs
- automatic web captures
- manual PDFs
- unresolved rows

### Filters

Available filters:

- `All`
- `PDF retrieved`
- `Resolved but no PDF`
- `Web only`
- `Unresolved`
- `Needs manual review`

### Row-Level Actions

For the selected row, Streamlit supports:

- `Open Best URL`
- `Open Resolved URL`
- `Open PDF URL`
- `Open DOI`
- download the automatic PDF if present
- download the automatic markdown artifact if present
- upload a manual PDF
- remove a manual PDF

### Final Status Values

The review table supports these statuses:

- `retrieved_auto_pdf`
- `retrieved_auto_web`
- `retrieved_manual_pdf`
- `web_only_confirmed`
- `unavailable`
- `wrong_match`
- `excluded`

### Final Researcher Package

Use:

- `Download Final Researcher Package`

The package includes:

- `manifest.json`
- `final_reviewed_citations.csv`
- `final_reviewed_citations.xlsx`
- `missing_or_unavailable.csv`
- `missing_or_unavailable.xlsx`
- `resolved/pdfs/`
- `resolved/markdown/`
- `resolved/manual_pdfs/`
- `originals/`

### Recommended Workflow

1. Run `Resolve + Retrieve Papers` or load an existing resolver bundle.
2. Open `Review Retrieval Results`.
3. Filter to `Resolved but no PDF`.
4. Open likely publisher pages manually from the row actions.
5. Upload any PDFs you can obtain.
6. Mark remaining rows as `web_only_confirmed`, `unavailable`, `wrong_match`, or `excluded`.
7. Download the final researcher package.

### Why This Exists

Automatic retrieval is limited by publisher access controls. This stage turns retrieval misses into a review queue instead of a dead end, so the researcher can still leave with:

- automatically downloaded PDFs
- automatically captured web fallbacks
- manually uploaded PDFs
- a curated final package with row-level status tracking
