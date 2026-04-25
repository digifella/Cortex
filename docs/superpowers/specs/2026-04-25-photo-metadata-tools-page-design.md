# Photo & Metadata Tools Page — Design

**Date:** 2026-04-25
**Status:** Approved

---

## Overview

Create a new Streamlit page (`20_Photo_Metadata_Tools.py`) dedicated to image and metadata workflows. This page consolidates two tools:

1. **Photo Processor** — the existing batch resize + AI keyword + EXIF ownership tool, moved from `7_Document_Extract.py`.
2. **LLM Metadata Sync** — a new tool that propagates LLM-generated keywords and descriptions from a flat JPG directory back into matching RAW source files and embedded derivatives via ExifTool, implementing `docs/llm_metadata_sync_spec.md`.

`7_Document_Extract.py` reverts to a pure text/PDF page with three tabs: PDF Ingestor, PDF Image Extractor, Anonymizer.

---

## New Page: `pages/20_Photo_Metadata_Tools.py`

- **Title:** "Photo & Metadata Tools"
- **Page icon:** 📷
- **Tabs:** `["Photo Processor", "LLM Metadata Sync"]`

### Tab 1 — Photo Processor

Verbatim extraction of `_render_photo_keywords_tab()` and all its helpers from `7_Document_Extract.py`. These functions are entirely self-contained (not called by any other tab):

- `_read_photo_metadata_preview()`
- `_write_photo_metadata_quick_edit()`
- `_photo_description_issue()`
- `_photokw_temp_dir()`, `_photokw_manifest_path()`, `_save_photokw_manifest()`, `_load_photokw_manifest()`
- `_render_photo_keywords_tab()`

The tab renders identically to today — no behaviour changes.

### Tab 2 — LLM Metadata Sync

Implements the spec in `docs/llm_metadata_sync_spec.md`. UI layout (top to bottom):

1. Title + one-paragraph description + Lightroom pre-run guidance (always visible).
2. Two `st.text_input` path fields: RAW root directory, JPG source directory. Validated on change (exists + readable).
3. ExifTool availability banner — if not on PATH, show install instructions and disable Scan/Apply.
4. `st.expander("Advanced options")` (collapsed):
   - Filter keywords (text area, comma-separated, default `nogps`)
   - ExifTool backups toggle (default on)
   - Rating suffix range (two number inputs, 1–5)
   - Derivative patterns (text area, defaults populated)
5. **Scan** button. Builds index + resolves matches. Writes nothing.
6. Scan results: `st.dataframe` with columns (JPG filename, matched targets count, target types, action summary). Orphaned JPGs in a collapsed expander. Summary counts below.
7. Dry run / Live run radio. **Default: Live run** (`dry_run: bool = False`).
8. **Apply** button — disabled until scan completes. Shows `st.status` progress block with live log stream. On completion: summary (counts, errors, elapsed time) + log download button.
9. Post-run Lightroom guidance block (Read Metadata from File instructions).

Session state keys: `lms_report` (last `SyncReport`), `lms_config` (config used for scan), `lms_scan_clean` (bool). Config-change-before-Apply triggers a warning and re-enables Scan.

---

## New Backend Module: `cortex_engine/llm_metadata_sync/`

All business logic lives in `cortex_engine/` following the repo pattern.

### `models.py`

```python
class TargetType(Enum): SIDECAR | EMBEDDED
class SidecarAction(Enum): CREATE | MERGE | NONE
@dataclass SyncAction: jpg_path, target_path, target_type, sidecar_action, raw_path
@dataclass SyncResult: action, success, keywords_written, description_written, error
@dataclass SyncReport: actions, results, orphaned_jpgs
@dataclass SyncConfig: raw_root, jpg_dir, filter_keywords, keep_backups,
                        rating_suffix_range, raw_extensions, embed_extensions,
                        deriv_patterns, dry_run=False
```

### `matcher.py` — pure functions, no I/O beyond filename reads

- `build_raw_index(raw_root: Path) -> dict[str, list[Path]]` — single `os.walk` pass, keys are case-folded stems
- `resolve_jpg(jpg_path: Path, index: dict, config: SyncConfig) -> list[SyncAction]` — strips rating suffix, looks up stem, returns `SyncAction` list (sidecar and/or embedded per match)
- ACR files skipped at index-build time

### `merger.py` — pure functions

- `strip_rating_suffix(stem: str, suffix_range: tuple) -> str`
- `read_jpg_metadata(jpg: Path) -> tuple[list[str], str]` — reads via exiftool JSON, returns (keywords, description)
- `read_existing_keywords(target: Path) -> list[str]` — reads via exiftool JSON
- `build_keyword_union(existing, new, filter_list) -> list[str]` — union, dedup, case-sensitive, first-seen order; filter applied case-insensitively

### `exiftool_runner.py`

- Validates exiftool on PATH at module import; raises `ExifToolNotFoundError` if absent
- `clear_keyword_lists(target: Path, target_type: TargetType, keep_backups: bool) -> RunResult`
- `write_metadata(jpg: Path, target: Path, target_type: TargetType, keywords: list[str], description: str, keep_backups: bool) -> RunResult`
- Filters `IPTCDigest is not current` from stderr before surfacing
- `RunResult`: returncode, stdout, stderr, command

### `sync.py` — orchestrator

- `run_sync(config: SyncConfig) -> Generator[SyncResult, None, SyncReport]` — generator; caller iterates for progress
- Builds index once, resolves all JPGs, then for each action: reads existing keywords → merges → calls runner (or logs dry-run intent)
- Failures on one file do not abort the run; captured in `SyncResult.error`

---

## `7_Document_Extract.py` Changes

- Remove `tab_photo` from `st.tabs(...)` call
- Remove `with tab_photo: _render_photo_keywords_tab()` block
- Remove helper functions: `_read_photo_metadata_preview`, `_write_photo_metadata_quick_edit`, `_photo_description_issue`, `_photokw_temp_dir`, `_photokw_manifest_path`, `_save_photokw_manifest`, `_load_photokw_manifest`, `_render_photo_keywords_tab`
- Update title to "Document Processing" and caption to match

---

## Tests

`tests/unit/test_llm_metadata_sync/`

- `test_matcher.py` — stem stripping with/without rating suffix; index building with `tmp_path` fixtures; ACR exclusion; multi-directory same-stem handling
- `test_merger.py` — keyword union overlap, dedup, filter (case-insensitive), order preservation, empty inputs
- `test_sync.py` — orchestration with mocked `exiftool_runner`; correct calls for sidecar vs embedded; dry-run produces no exiftool calls; single-file failure doesn't abort run

---

## Docker Sync

After implementation, copy to docker mirror:
- `docker/pages/` ← `pages/20_Photo_Metadata_Tools.py`, updated `7_Document_Extract.py`
- `docker/cortex_engine/llm_metadata_sync/` ← new module

---

## Out of Scope

Per spec §10: no filesystem watching, no LRC catalog modification, no folder picker, no LLM calls, no recursive JPG scanning, no reverse sync.
