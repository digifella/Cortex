# Vault Ingest — Single-Document Upload — Design

**Date:** 2026-08-01
**Status:** Approved (pending spec review)

## Purpose

The Vault GraphRAG page's ingest panel (v6.4.0) takes a folder path and ingests
every supported document inside it. That is the right shape for a batch of
manuals, but heavy when there is exactly one document to file. This adds a
second input mode: pick a single document with a browser file dialog.

Scope is the **private** vault only. The folder-path mode is unchanged.

## Approach

**`st.file_uploader` with content-aware staging, driven through the ingest
script's existing `--file-list` flag.**

The user chose a true upload dialog over a path-based picker, with the
provenance tradeoff stated and accepted (see "Accepted losses" below).

Streamlit's uploader hands the server file *bytes* and a filename — never the
original path. So the bytes must be written somewhere before the ingest script
can read them. Two consequences drive the whole design:

1. **The manifest keys on the absolute source path** (`f"{branch_name}::{source_path}"`),
   so the staging location determines de-duplication identity.
2. **`is_unchanged` compares `mtime_ns` and `size`** — not a content hash. A
   naive "write the upload to a temp file each time" therefore re-ingests the
   same document on every upload, because each write produces a fresh mtime.

The fix is to make the staging write content-aware: stage to a *stable* path and
skip the write entirely when the bytes already match.

### Staging

Uploads are written to a persistent directory:

```
~/.nemoclaw/vault-ingest-uploads/<original-filename>
```

Before writing, take a SHA-256 of the uploaded bytes and compare it against a
SHA-256 of the existing staged file. If identical, **leave the file untouched** so
its `mtime_ns` is preserved.
The manifest then reports the file as unchanged and the ingest skips it — a
re-upload of the same document is a correct no-op, matching the folder flow.

If the bytes differ (an edited document under the same name), overwrite; the new
mtime correctly triggers re-ingestion.

### Why `--file-list` rather than a per-upload directory

The staging directory accumulates past uploads, so pointing `--source-root` at
it would re-ingest everything on every run. Instead:

```
--source-root <staging dir>  --file-list <one-line list naming the staged file>
```

`--source-root` must stay an ancestor of every listed file because the script
calls `source_path.relative_to(source_root)` in four places (lines 134, 155, 233,
318) even on the explicit-file path. The staging directory satisfies that.

The alternative — a fresh `<staging>/<timestamp>/<filename>` subdirectory per
upload, with no file list — was rejected: a new path every time defeats the
mtime-preservation trick and permanently breaks de-duplication.

The file-list format the script expects (verified at lines 371-379): one path per
line, `#` comments and blank lines ignored, each line `expanduser().resolve()`d.
The list file is written to `~/.nemoclaw/vault-ingest-uploads/.filelist.txt`, a
single stable name overwritten per run rather than a timestamped series — the
`flock` in `start_vault_ingest` already prevents concurrent starts, so one file
suffices and the staging directory stays free of litter.

### Accepted losses

The vault note's provenance will record the staging path
(`~/.nemoclaw/vault-ingest-uploads/manual.pdf`), not the document's real
location. This is irreducible — the browser never sends the path — and the user
accepted it when choosing the upload dialog. The original **filename** is
preserved, so the note remains identifiable. The external ingest script cannot be
modified to improve this: it is shared with the cron queue runner.

## Components

One optional parameter threaded through the existing chain, plus the UI.

### `cortex_engine/vault_ingest.py`

Add `file_list: Path | None = None` to `build_ingest_command` and
`run_ingest_then_index`, and `--file-list` to the argparse in `main()`. When
present, append `--file-list <path>` to the phase-1 command; when absent, the
command is byte-identical to today's.

### `cortex_engine/private_vault_rag.py`

Add `file_list: Path | None = None` to `start_vault_ingest`, passed through to
the spawned wrapper as `--file-list`. Everything else — the flock, the atomic
state write, the detached spawn, status resolution, cancel — is unchanged.

This module also gains the staging constant and helper, alongside the existing
`VAULT_INGEST_STATE` / `VAULT_INGEST_LOG_DIR`:

```python
VAULT_INGEST_UPLOAD_DIR = HOME / ".nemoclaw" / "vault-ingest-uploads"

def stage_upload(data: bytes, filename: str, staging_dir: Path | None = None) -> Path
```

`stage_upload` takes raw bytes rather than a Streamlit `UploadedFile` so it has
no Streamlit dependency and is directly unit-testable — the page calls
`stage_upload(uploaded.getvalue(), uploaded.name)`. It writes only when the
bytes differ from what is already staged, and returns the staged path.

### `pages/18_Private_Vault_GraphRAG.py`

A mode radio at the top of the "Ingest documents" expander:

- **Folder** — today's inputs, unchanged.
- **Single document** — `st.file_uploader(type=["pdf", "docx", "pptx", "txt"])`,
  matching the `type=`-constrained uploader convention already used in
  `pages/8_Document_Summarizer.py:291` and `pages/10_Visual_Analysis.py:128`.

In upload mode the branch name defaults to the uploaded file's stem, the
destination default follows the same `30 Resources/Imported Knowledge/<branch>`
pattern, and **"File limit" is hidden** (meaningless for one file). PDF strategy,
describe-images, destination and dry-run apply to both modes.

The uploader's allowed types mirror the ingest script's `SUPPORTED_EXTS`
(`.pdf`, `.docx`, `.pptx`, `.txt`). These two lists must not drift.

The page itself gains no new testable logic: it calls `stage_upload` (defined in
`private_vault_rag.py`, above) and passes the result to `start_vault_ingest`. The
staging directory path is surfaced in the UI so it can be cleared manually.

## Data flow

```
browser file dialog
   │  bytes + filename (no path)
   ▼
stage_upload()  ──► ~/.nemoclaw/vault-ingest-uploads/<filename>
   │                  (write skipped when bytes are unchanged → mtime preserved)
   ├──────────────► .filelist-<stamp>.txt   (one line: the staged path)
   ▼
start_vault_ingest(source_root=<staging dir>, file_list=<list>, ...)
   ▼
python -m cortex_engine.vault_ingest --source-root … --file-list …
   ├─ phase 1: textify just that one file → vault markdown
   └─ phase 2: index the private vault
```

## Error handling

- **Unsupported extension** — blocked by the uploader's `type=` constraint.
- **Zero-byte upload** — rejected inline before staging; nothing is spawned.
- **Staging write fails** (permissions, disk) — error surfaced inline, no spawn.
- **Filename collision with different content** — overwrites by design; the new
  mtime triggers re-ingestion, which is the wanted behaviour for an edited file.
- **Upload larger than the cap** — Streamlit rejects it in the widget. There is
  no `maxUploadSize` in `.streamlit/config.toml`, so the limit is the 200MB
  default. Not changed here; noted so a future large-scan failure is diagnosable.
- Existing guards are untouched: the already-running check, the destination
  vault-containment check, detached-run status, and cancel.

## Testing

Unit tests for `stage_upload` (the dedup guarantee is the point):

- new file → written, path returned
- identical bytes re-staged → **`mtime_ns` unchanged** (the guarantee that makes
  re-upload a no-op)
- changed bytes, same name → rewritten, `mtime_ns` advances
- staging directory created when absent

Unit tests for the command threading:

- `build_ingest_command` emits `--file-list <path>` when given one
- `build_ingest_command` emits **no** `--file-list` when not given one, and the
  resulting command is unchanged from today's

An AppTest check that the mode radio swaps the inputs and that "File limit" is
absent in upload mode.

All existing tests must keep passing; the folder path must be provably unchanged.

## Version management

Feature increment 6.4.0 → 6.5.0, following the project workflow: update
`cortex_engine/version_config.py`, run `scripts/version_manager.py --sync-all`,
`--update-changelog`, `--check`, update the page header, and sync
`docker/cortex_engine/` and `docker/pages/`.

## Out of scope

- Multiple-file upload. `accept_multiple_files=True` is a one-argument change if
  wanted later; the request was explicitly for a single document.
- Changing the folder-path mode.
- Modifying the external ingest or indexer scripts.
- Raising the Streamlit upload size cap.
- Pruning the staging directory. Staged files persist deliberately — that is what
  makes re-upload de-duplication work. The path is surfaced in the UI so it can
  be cleared manually.
