# Review Study Miner Status

Date: 2026-04-03

## Current State

The `Study Miner` workflow in [pages/7_Document_Extract.py](/home/longboardfella/cortex_suite/pages/7_Document_Extract.py) now supports:

- batch review screening for likely systematic reviews
- human confirmation of which reviews to mine
- local table-first extraction of candidate included studies
- bibliography linking from table labels such as `Author 2020 [19]`
- mismatch warnings when a table row does not agree with the referenced bibliography entry
- provenance carry-through into `Research Resolver`
- parse evidence view showing:
  - detected PDF table snapshots when available
  - extracted markdown table blocks
  - parsed candidate rows
- optional Claude Sonnet cloud rescue for complex tables

Core logic currently lives in:

- [cortex_engine/review_study_miner.py](/home/longboardfella/cortex_suite/cortex_engine/review_study_miner.py)
- [cortex_engine/review_table_rescue.py](/home/longboardfella/cortex_suite/cortex_engine/review_table_rescue.py)
- [cortex_engine/research_resolve.py](/home/longboardfella/cortex_suite/cortex_engine/research_resolve.py)

## What Works

- straightforward markdown-style included-studies tables
- numbered bibliography links such as `[19]`, `(19)`, and `ref 19`
- author-year fallback matching when the table omits the explicit reference number
- filtering obvious non-study rows such as `Region`, `Study design`, and `Source of utility values`
- `Needs review` warnings when a table row and bibliography entry disagree
- Streamlit can now find `ANTHROPIC_API_KEY` from:
  - `.env`
  - process environment
  - `worker/config.env` fallback

## Current Limitation

The hard failure case is large rotated multi-page systematic-review tables.

What is happening now:

- Docling/PDF extraction often reconstructs these as several partial markdown tables instead of one logical study table.
- The local miner therefore sees fragmented row-label/value blocks rather than clean study rows.
- The current Claude rescue path does **not** send the whole PDF. It sends:
  - up to 4 detected table snapshots
  - up to 4 extracted markdown table blocks
  - a reference-section excerpt
- If the upstream table reconstruction is already wrong or incomplete, Claude inherits that confusion.

This means the current cloud fallback is useful for messy but still localized tables, but it is not yet robust for very large rotated tables that continue across multiple pages.

## Current UX

`Study Miner` now has:

- `Vision assist` with `Auto`, `On`, and `Off`
- `Auto-run Claude rescue when the local parse looks confused`
- `Cloud Table Rescue` section with explicit Anthropic key-source reporting
- `Parse Evidence` section for human cross-checking

Recommended current setting for difficult review PDFs:

- `Vision assist = Auto (Recommended)`

## Next Likely Step

The next real improvement is not another regex tweak. It is table reconstruction.

Most likely next engineering slice:

1. detect rotated review table pages
2. render them upright
3. merge multi-page continuations into one logical table
4. send that reconstructed table evidence to the parser / Claude rescue
5. keep explicit human-review warnings when uncertain

If Claude is used again for this class of table, the likely better path is to send a larger contiguous evidence bundle for the full table region or table-run, rather than isolated partial tables.
