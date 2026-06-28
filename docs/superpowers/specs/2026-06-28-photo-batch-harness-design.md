# Photo Batch Harness — Design

**Date:** 2026-06-28
**Status:** Approved (design)
**Author:** Paul + Claude

## Problem

5000+ JPGs sit in a "to be tagged" directory. They need AI metadata
(description, keywords, location, ownership) written into them, and that
metadata then reconciled onto the original RAW / TIF / PSD / catalog-JPG files in
the Lightroom Classic catalog so Lightroom can re-import it.

Both capabilities already exist in `cortex_suite` but only run through the
Streamlit page `pages/20_Photo_Metadata_Tools.py`, which is browser/session bound
and capped at 1 GB per upload batch — unworkable for 5000+ photos.

## Existing engines (reused unchanged)

1. **Tagging** — `cortex_engine.textifier.DocumentTextifier(use_vision=True).keyword_image(path, ...)`.
   Runs the local Ollama vision model (`qwen3-vl:8b`, first in `VISION_MODELS`,
   confirmed installed), then writes description + keywords + location +
   ownership into the file in place via exiftool. No Streamlit import at module
   level — fully headless.

2. **Reconciliation / sync** — `cortex_engine.llm_metadata_sync`:
   - `build_raw_index(raw_root, config)` walks `raw_root` once, indexing targets
     by case-folded filename stem.
   - `resolve_jpg(jpg, index, config)` maps a tagged JPG to its targets:
     XMP sidecar for RAW (`.RAF/.NEF/.CR2/.CR3/.ARW/.DNG/...`), embedded write for
     `tif/tiff/psd/psb/dng/png` derivatives, `.old` backup + replace for catalog
     JPGs.
   - `run_sync(SyncConfig)` is a generator yielding one `SyncResult` per action.

The Streamlit page wires these together (`_render_photo_keywords_tab`,
`_render_lms_tab`). The harness replicates that wiring headlessly.

## Solution

One CLI script: **`cortex_suite/scripts/photo_batch.py`**, run under the
cortex_suite venv (`venv/bin/python`), with two subcommands: `tag` and `sync`.
No changes to any engine code — pure reuse. Target ~150–200 lines.

Workflow is two-phase with a human review checkpoint between them, because the
sync step is destructive (it renames each matched original to `.old`/`.bak`
before writing).

### Phase 1 — `tag <to_tag_dir>`

```
venv/bin/python scripts/photo_batch.py tag <to_tag_dir> [options]
```

- Selects **top-level** `*.jpg` / `*.JPG` files in `<to_tag_dir>`. This matches
  `run_sync`'s top-level glob so both phases operate on the same file set.
  (Nested subfolders are intentionally not processed; sync would not see them.)
- For each photo not already done, calls:
  ```python
  DocumentTextifier(use_vision=True).keyword_image(
      path,
      generate_description=True,
      populate_location=True,
      anonymize_keywords=False,
      ownership_notice=OWNERSHIP_NOTICE,   # non-empty => ownership written
  )
  ```
  Tagging options per approved scope: description+keywords, location+GPS,
  ownership. Anonymize off. No resize/halftone.
- **Checkpoint** at `<to_tag_dir>/.photo_batch_tag.json`:
  - Records, per processed file, a key of `name + size + mtime` and a one-line
    result summary (ok flag, short description, keyword count, error).
  - Written atomically (temp file + `os.replace`) after **every** photo.
  - On startup the checkpoint is loaded; files whose `name+size+mtime` key is
    already present and `ok` are skipped. A file that changed on disk (different
    size/mtime) is re-processed.
- Per-photo stdout line: `[i/N] OK name: description…` or `[i/N] FAIL name: error`.
- `--cooldown SECONDS` (default 0): pause between photos.
- `--ownership "text"` overrides the default notice; `--no-ownership` disables.
- Final summary: tagged / failed / skipped counts.

Default ownership notice (matches the Streamlit default):
`All rights (c) Longboardfella. Contact longboardfella.com for info on use of photos.`

### Human review checkpoint

Operator inspects tagged JPGs, the stdout log, and/or the checkpoint JSON before
running the destructive sync. Not automated.

### Phase 2 — `sync <to_tag_dir> <raw_root>`

```
venv/bin/python scripts/photo_batch.py sync <to_tag_dir> <raw_root> [--apply] [options]
```

- Builds `SyncConfig` with UI-matching defaults:
  ```python
  SyncConfig(
      raw_root=<raw_root>,
      jpg_dir=<to_tag_dir>,
      filter_keywords=["nogps"],          # overridable via --filter-keywords
      keep_backups=True,                  # --no-backups to disable
      rating_suffix_range=(1, 5),
      timestamp_tolerance_seconds=0,      # --timestamp-tolerance N
      deriv_patterns=<SyncConfig default>,
      dry_run=<not --apply>,
  )
  ```
- **Dry-run by default.** Without `--apply`: builds the index, resolves every
  JPG, prints a matched-actions table (JPG → target, type, sidecar action) plus
  matched/orphaned counts, and exits. Nothing is written.
- **`--apply`** runs the live, destructive sync: iterates `run_sync(cfg)`, prints
  one line per `SyncResult`, and prints a final summary (actions succeeded /
  failed, keywords / descriptions / location fields written). Failed actions are
  listed.
- Single `raw_root` walked recursively (matcher behaviour); matches by filename
  stem regardless of subfolder depth.

### Flags summary

| Flag | Phase | Default | Effect |
|------|-------|---------|--------|
| `--cooldown N` | tag | 0 | Seconds between photos |
| `--ownership "…"` | tag | standard notice | Override ownership text |
| `--no-ownership` | tag | off | Skip ownership write |
| `--apply` | sync | off (dry-run) | Perform destructive sync |
| `--no-backups` | sync | backups on | Don't keep `.old`/`.bak` |
| `--filter-keywords a,b` | sync | `nogps` | Keywords to drop on sync |
| `--timestamp-tolerance N` | sync | 0 | Fuzzy RAW/JPG time match (panoramas) |

## Non-goals

- No resize, halftone repair, or keyword anonymization (outside approved scope).
- No multi-root sync orchestration — one `raw_root` per invocation.
- No changes to `textifier.py`, `llm_metadata_sync/`, or the Streamlit page.
- No GUI; this is operator-run CLI.

## Error handling

- Tag: each photo wrapped in try/except; failures recorded in the checkpoint and
  reported, the run continues. A re-run retries files not marked `ok`.
- Sync: `run_sync` already returns a `SyncResult` (with `error`) per action and
  the sync engine restores the original on copy failure during JPG-replace. The
  harness surfaces these; it does not abort the batch on a single failure.

## Testing

TDD on the harness logic that does not require a GPU/VLM:

1. **Checkpoint resume** — given a checkpoint marking file A as `ok`, a tag run
   over {A, B} skips A and only processes B; a changed A (different mtime/size)
   is re-processed.
2. **Sync dry-run wiring** — synthetic `raw_root` with a `.NEF` (or `.tif`) whose
   stem matches a tagged JPG fixture; assert the dry-run reports the expected
   matched action and that no files were modified.
3. **Arg/config mapping** — flags produce the expected `SyncConfig`
   (`dry_run`, `keep_backups`, `filter_keywords`, `timestamp_tolerance_seconds`).

The matcher/merger/sync engines already have unit tests under
`tests/unit/test_llm_metadata_sync/` — not re-tested here.

Manual smoke test before the full 5000 run: `tag` ~3 real photos, eyeball the
written metadata, `sync --dry-run`, then `sync --apply` on those 3, confirm
sidecars/`.old` backups appear, then run the full batch.

## Operational notes

- Runtime: vision model takes up to ~20 s/photo (timeout) plus enrich, so 5000
  photos is multi-hour. The atomic checkpoint makes the run interruptible and
  resumable.
- Backend: tagging uses local Ollama (`qwen3-vl:8b`). Ensure Ollama is running
  (`sudo systemctl start ollama`) before a tag run.
- Both phases operate on the top level of `<to_tag_dir>` only.
