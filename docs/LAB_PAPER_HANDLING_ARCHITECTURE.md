# Lab Paper Handling Architecture

Purpose: explain, in plain language, how the Lab paper workflow now works from systematic review PDF upload through table extraction, paper resolution, retrieval, and final researcher packaging.

## What This System Does

The Lab paper workflow is built to help a researcher move from:

- a systematic review PDF

to:

- extracted included-study tables
- resolved references
- attempted paper retrieval
- human review of retrieval gaps
- one final curated researcher package

This is not a single model call. It is a staged workflow.

## The End-to-End Flow

At a high level, the system now works like this:

1. upload a review PDF
2. slice the PDF into bibliography plus candidate included-study tables
3. extract grouped trial/reference mappings per table
4. export reviewed table files
5. resolve selected citations to likely papers
6. attempt URL/PDF retrieval
7. let a human finish the blocked cases
8. build a final package for the researcher

## The Main Components

### 1. Included Study Slicer

Primary role:

- split the source review into usable per-table slices
- isolate bibliography text and CSV

Core file:

- [included_study_slicer.py](/home/longboardfella/cortex_suite/cortex_engine/included_study_slicer.py)

This is the structural pre-processing stage.

It answers:

- where are the included-study tables?
- which pages belong to each table?
- what is the bibliography text we can use later?

### 2. Included Study Extractor

Primary role:

- turn each sliced table into structured grouped output

Core file:

- [included_study_extractor.py](/home/longboardfella/cortex_suite/cortex_engine/included_study_extractor.py)

This is the model-driven extraction stage.

It supports choices such as:

- `All trials/studies`
- `RCT/clinical trials only`

and output detail modes such as:

- compact reference map
- more detailed study fields

The practical goal is not to reproduce every table cell. It is to produce useful table-aware citation groupings such as:

- table
- instrument/group
- trial label
- citation display
- bibliography reference number

### 3. Included Study Handoff Builder

Primary role:

- turn extracted table rows into downstream artifacts

Core file:

- [included_study_handoff.py](/home/longboardfella/cortex_suite/cortex_engine/included_study_handoff.py)

This stage creates:

- reviewed selection CSV/XLSX
- resolver payload JSON
- resolver queue job JSON
- website handoff JSON

It also enriches rows from the review bibliography so downstream resolution is stronger.

### 4. Included Study Queue Worker

Primary role:

- run the PDF-to-table workflow asynchronously for the website

Core file:

- [included_study_extract.py](/home/longboardfella/cortex_suite/worker/handlers/included_study_extract.py)

This worker:

- validates the PDF input
- runs slicing
- runs per-table extraction
- writes per-table artifacts
- returns a bundle ZIP

This is the website/backend bridge for the extraction stage.

### 5. Research Resolver

Primary role:

- take selected citations and resolve them to likely papers

Core file:

- [research_resolve.py](/home/longboardfella/cortex_suite/cortex_engine/research_resolve.py)

This is the reference resolution stage.

It now does more than simple Crossref search. It can:

- use bibliography reference numbers from the source review
- enrich rows from matched bibliography entries
- carry forward URLs already present in bibliography text
- backfill DOI URLs where plausible
- preserve review context such as:
  - table number
  - combined group
  - trial label
  - reference number

This is why the workflow can resolve difficult rows like:

- conference abstracts
- registry records
- bibliography-listed papers with sparse metadata

### 6. URL Ingestor / Retriever

Primary role:

- take preferred URLs and try to fetch the actual paper or a useful fallback

Core file:

- [url_ingestor.py](/home/longboardfella/cortex_suite/cortex_engine/url_ingestor.py)

This is the retrieval stage.

It tries to:

- download open PDFs
- capture useful web markdown when PDF retrieval fails
- preserve landing pages or registry pages instead of returning nothing
- use browser-assisted fallback for harder publisher cases

The retriever is deliberately practical:

- if it gets a PDF, great
- if it cannot, it should still try to keep a useful website artifact

### 7. Retrieval Review Stage

Primary role:

- let a human finish the last mile

Implemented in:

- website review stage, documented in [RETRIEVAL_REVIEW_STAGE_SPEC.md](/home/longboardfella/cortex_suite/docs/RETRIEVAL_REVIEW_STAGE_SPEC.md)
- local Streamlit stage in [7_Document_Extract.py](/home/longboardfella/cortex_suite/pages/7_Document_Extract.py), documented in [STREAMLIT_RETRIEVAL_REVIEW_WORKFLOW.md](/home/longboardfella/cortex_suite/docs/STREAMLIT_RETRIEVAL_REVIEW_WORKFLOW.md)

This exists because publisher access is the weakest part of automation.

The system can now:

- show what was retrieved automatically
- show what only has web fallback
- let the user upload missing PDFs manually
- let the user mark rows as:
  - web only
  - unavailable
  - wrong match
  - excluded

This turns retrieval misses into a review queue instead of a failure screen.

## The Workflow in More Detail

## Stage 1. Extract Included Study Tables

Entry points:

- Streamlit `Document Extract`
- website queue path

Input:

- one systematic review PDF

Output:

- bibliography files
- per-table PDFs
- per-table extraction JSON
- reviewed selection CSV/XLSX
- resolver payloads
- handoff JSON

The extraction bundle is the working package for the next stages.

## Stage 2. Review Table Files

The user can now inspect per-table files manually.

Typical files:

- `included_study_selection.csv`
- `included_study_selection.xlsx`

The key editable field is:

- `keep`

This is the user’s control point:

- keep rows to send to resolution
- turn off rows that should not proceed
- optionally override review-only defaults

This is especially important for economic tables under `RCT/clinical only`, where rows may sensibly default to `FALSE`.

## Stage 3. Resolve Reviewed Table Files

Input:

- one or more reviewed table files

Accepted forms:

- CSV
- XLSX
- resolver payload JSON
- resolver queue job JSON

The system merges all selected `keep = true` rows into one resolver job if multiple files are uploaded together.

That is what gives the user the simpler workflow:

- inspect tables locally
- edit them
- upload them together
- resolve and retrieve in one combined run

## Stage 4. Resolve Citations to Papers

This stage produces:

- `resolved_citations.csv`
- `unresolved_citations.csv`
- `research_resolve_result.json`
- `preferred_urls.txt`
- `url_ingest_candidates.json`

The resolver now prefers:

1. source review bibliography evidence
2. bibliography-derived DOI/URL carry-through
3. plausible DOI backfill
4. generic external search only when needed

That is why it now behaves much more like a human who uses the bibliography first.

## Stage 5. Retrieve Papers

If retrieval is enabled, the resolver continues into URL ingest.

Result bundle includes:

- resolver outputs
- retrieval reports
- PDFs where found
- markdown/web fallbacks where PDFs are blocked

The retrieval step is intentionally layered:

- first try the strongest direct URL
- then try landing-page and publisher cues
- then keep useful web markdown instead of dropping the case

This is important because many publisher sites are open to a human but awkward for simple automation.

## Stage 6. Human Retrieval Review

This is the final curation step.

The user can:

- filter to `Resolved but no PDF`
- open the likely landing pages
- upload manually obtained PDFs
- mark rows with final statuses

The system then builds one final researcher package.

## The Two Main User Surfaces

## A. Website

The website is best when the user wants:

- PDF upload
- async queue processing
- reviewed table upload
- combined resolve and retrieve
- retrieval review in browser

The website now supports a multi-step but coherent workflow.

## B. Streamlit

Streamlit is best when the user wants:

- local iteration
- direct inspection of artifacts
- local resolver runs
- local retrieval review without re-uploading through the website

The new local review stage means the website is no longer the only place to finish retrieval.

## Why the Workflow Is Split Into Stages

This split is intentional.

The system has learned that these are different problems:

1. extract the table structure
2. decide which rows to keep
3. resolve citations
4. retrieve papers
5. handle publisher edge cases with a human in the loop

Trying to collapse that into one opaque AI action produces brittle results.

Staging makes the workflow:

- more inspectable
- more debuggable
- more trustworthy

## What the System Is Now Good At

- slicing systematic review PDFs into usable table artifacts
- extracting table-aware grouped citation maps
- preserving bibliography reference numbers
- turning table rows into resolver-ready payloads
- resolving citations with bibliography-first logic
- retrieving some open PDFs automatically
- keeping useful website markdown when PDFs are blocked
- letting a human finish the last mile cleanly

## What Still Limits Full Automation

- publisher anti-bot or browser-gated access
- inconsistent landing-page structures
- sparse metadata for some abstracts or supplements
- cases where a human can click through an open site more easily than generic HTTP automation can

That is why the human review stage is part of the architecture, not a temporary hack.

## Recommended Mental Model

Think of the Lab paper workflow as four layers:

1. Structural extraction
   - slice PDF
   - detect tables
   - extract grouped citations

2. Selection and handoff
   - review rows
   - choose `keep`
   - build resolver payloads

3. Resolution and retrieval
   - match citations to likely papers
   - generate preferred URLs
   - fetch PDFs or web fallbacks

4. Human finishing
   - inspect blocked cases
   - upload missing PDFs
   - export final curated package

That is the architecture we built today.
