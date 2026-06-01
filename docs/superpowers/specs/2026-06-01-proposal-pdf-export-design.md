# Proposal PDF Export — Design

**Date:** 2026-06-01
**Status:** Approved (pending spec review)

## Purpose

Add PDF as an output format for completed proposals and idea-generation reports.
The system currently exports proposals as Markdown and DOCX (via
`ProposalExportEngine`) and idea sessions as JSON + Markdown (via `IdeaExporter`).
PDF gives users a paginated, print-ready document without manual conversion.

PDF is an **export/render** capability (HTML/CSS → PDF). It does not touch the
ingest/extract path (`textifier.py`, `article_markdown_extractor.py`) and does
not involve any LLM — WeasyPrint is model-free layout code.

## Approach

**One shared Markdown → PDF helper, with thin call sites at each export point.**

All three export points already produce Markdown (or Markdown-like) text. A single
helper converts Markdown → HTML → PDF with one shared stylesheet, so styling lives
in exactly one place. Each call site is a few lines that pass already-assembled
Markdown to the helper.

Rejected alternatives:
- **Per-site PDF code** — styling would drift across three locations.
- **Jinja HTML templates** — more layout control than needed for "clean & minimal",
  and all sites already emit Markdown.

## Components

### New module: `cortex_engine/pdf_export.py`

```python
def markdown_to_pdf_bytes(markdown_text: str, *, title: str = "") -> Optional[bytes]
```

- Lazy-imports `markdown` and `weasyprint`. Returns `None` if either is
  unavailable (mirrors the existing DOCX `None`-degrades pattern in
  `ProposalExportEngine.generate_export_docx`).
- Markdown → HTML using the `markdown` package with `tables` and `fenced_code`
  extensions (proposal content can contain tables).
- Wraps the HTML in a full document with one embedded `<style>` block:
  - readable body typography, sensible page margins
  - styled headings (h1/h2/h3)
  - `@page` rule with bottom-centre page numbers
  - the metadata header (the existing `**bold**` / `---` lines) renders naturally
- Returns PDF bytes.

### Call site 1: `ProposalExportEngine.generate_export_pdf()`

```python
def generate_export_pdf(workspace_id, include_citations=False, flag_incomplete=True) -> Optional[bytes]
```

Wraps the existing `generate_export_markdown(...)` and passes its output to
`markdown_to_pdf_bytes(md, title=tender_name)`. No content-assembly duplication.

### Call site 2: `pages/13_Proposal_Manager.py`

- Add `"PDF"` to the format selectbox (currently `["Markdown", "DOCX"]`, line ~1298).
- Add a PDF download branch alongside the Markdown/DOCX branches:
  `mime="application/pdf"`, filename `proposal_<name>_<YYYYMMDD>.pdf`.
- If the helper returns `None`, show a warning (same pattern as the DOCX branch).

### Call site 3: `pages/Proposal_Intelligent_Completion.py`

- Add a "Download PDF" button next to the existing "Export All" button.
- Feeds the same assembled `export_text` to `markdown_to_pdf_bytes`.
- The per-field `.txt` "Export" buttons are unchanged (PDF only at aggregate level).

### Call site 4: `cortex_engine/idea_generator/export.py`

- In `IdeaExporter.export_results`, after writing JSON + Markdown, also write
  `<prefix>_<ts>.pdf` **to disk** (this module writes files, not download buttons)
  when the helper is available.
- Add the path to the returned `exported_files` dict under key `"pdf"`.
- If the helper returns `None`, skip silently (JSON/MD still written).

## Dependencies

- Add `markdown` to `requirements.txt` and `docker/requirements.txt`.
- WeasyPrint is already installed (v68.1, native libs present). It stays a
  runtime-optional lazy import; no requirements change strictly required, but it
  is the rendering engine.

## Scope guards (YAGNI)

- No logo, cover page, or branding — "clean & minimal" only.
- No per-field PDF in the IC page; aggregate export only.
- No changes to the ingest/extract path.

## Error handling

- Missing `markdown` or `weasyprint` → helper returns `None`.
- UI call sites (Proposal Manager, IC page) show a warning when `None`.
- File-writing call site (`IdeaExporter`) silently skips the PDF.

## Testing

Unit test for `markdown_to_pdf_bytes`:
- Returns bytes starting with `%PDF` for sample Markdown that includes a heading,
  a paragraph, and a table.
- Returns `None` when `markdown` or `weasyprint` import is monkeypatched to fail.
