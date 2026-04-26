# LLM Metadata Sync — Streamlit App Spec

A Streamlit page/module that copies LLM-generated keywords and descriptions from a directory of curated JPGs back into matching source files (RAW + XMP sidecars, or embedded into TIF/PSD/DNG derivatives) in a separate, possibly-nested RAW directory tree.

This spec assumes integration into an existing Streamlit app repo as a new page or module. ExifTool is invoked as a subprocess — do **not** reimplement XMP parsing in Python.

---

## 1. Goal & context

The user runs a local LLM that adds keywords and descriptions to JPGs exported from Lightroom Classic. Those edits live only in the JPGs. This tool propagates the LLM's metadata back to the original source files in a separate RAW library, so when Lightroom re-reads metadata, the catalog absorbs the LLM's contributions while preserving existing develop settings and edit work.

The tool runs once per batch — it is not a daemon, watcher, or scheduled job.

## 2. Inputs

Two directory paths supplied via the UI:

- **RAW root directory** — top-level folder containing source files. May be deeply nested. Contains a mix of: `.RAF`, `.NEF`, `.CR2`, `.CR3`, `.ARW`, `.DNG`, `.RW2`, `.ORF`, `.PEF`, `.SRW`, `.IIQ`, `.3FR` (raw originals); `.xmp` (sidecars); `.tif`, `.tiff`, `.psd`, `.psb`, `.dng` (derivative edits); `.acr` (Adobe AI Denoise enhancement caches — must be ignored).
- **JPG source directory** — flat directory of LLM-tagged JPGs (no recursion needed; assume flat). Filenames may or may not have a `-[1-5]` rating suffix appended.

Both paths are typed/pasted as plain strings (Streamlit has no real folder picker; a `st.text_input` with validation is sufficient).

## 3. Filename matching rules

For each JPG in the JPG directory:

1. Strip extension. If the stem ends in `-1`, `-2`, `-3`, `-4`, or `-5`, strip that too. The remainder is `<stem>`.
2. Search the RAW tree (recursively) for matches against `<stem>`:
   - **Raw original** — `<stem>.<RAW_EXT>` for any extension in the raw list above. Match → write metadata to `<stem>.xmp` in the same directory (create the sidecar if it doesn't exist; merge if it does).
   - **Adobe derivative** — `<stem><suffix>.<EMBED_EXT>` where suffix matches one of: `-Edit`, `-Edit-2`, `-Edit-3`, …, `-Enhanced`, `-Enhanced-NR`, `-HDR`, `-HDR-2`, …, `-Pano`, `-Pano-2`, …; and `EMBED_EXT` is `tif`, `tiff`, `psd`, `psb`, or `dng`. Match → write metadata directly into the file.
   - **Skip** any `.acr` file regardless of stem match — these are binary AI Denoise caches and contain no editable metadata.

Multiple matches per JPG are allowed and expected — a single stem may have a RAW + sidecar AND one or more `-Edit*` derivatives. All matched sources receive the metadata.

A JPG with zero matches is reported as orphaned (warning logged, no error).

### Performance note

Do not glob the RAW tree once per JPG. **Build an index up front:**

```python
# Pseudo: walk RAW_ROOT once, build dict[stem -> list[matched_path]]
# Key by case-folded stem to handle case mismatches
```

Then look up each JPG's stem against the index. For libraries with tens of thousands of files, this is the difference between minutes and hours.

## 4. Metadata merge rules

Already decided. Do not parameterise these unless the spec explicitly says so.

### Keywords

- **Read source**: `IPTC:Keywords` from the JPG (equivalent to `XMP-dc:Subject` — pick one consistently).
- **Read existing target**: `XMP-dc:Subject` from the target.
- **Merge**: union, deduplicated, case-sensitive, **first-seen order preserved** (existing target keywords appear first, then new JPG keywords appended in their original order).
- **Filter**: drop any keyword in the configurable filter list (default: `["nogps"]`, case-insensitive).
- **Write**: see §5 below.

### Description

- **Read source**: `IPTC:Caption-Abstract` from the JPG.
- **Action**: always overwrite the target's existing description with the JPG's version. (User decision — do not preserve existing.)
- If the JPG has no caption (empty/missing), do not write — leave target as-is.

## 5. Write rules per target type

### XMP sidecar (raw original case)

Write only to the XMP namespace. Two-step process — **mixing `=` (clear) with `+=` (add) on a list tag in a single ExifTool call does not work**; it appends to the existing list instead of replacing.

```bash
# Step 1: clear the keyword list
exiftool -overwrite_original "-xmp-dc:subject=" "<target.xmp>"

# Step 2: populate keywords + description in one call
exiftool -overwrite_original \
    "-xmp-dc:subject+=keyword1" \
    "-xmp-dc:subject+=keyword2" \
    ... \
    -tagsfromfile "<jpg>" \
    "-xmp-dc:description<iptc:Caption-Abstract" \
    "<target.xmp>"
```

If the target XMP doesn't exist yet, skip Step 1 and let Step 2 create it (omit `-overwrite_original` for creation, or use `-o <target.xmp>` form — verify in implementation).

### Embedded (TIF/PSD/DNG/PSB)

Write to **both** IPTC and XMP-dc namespaces, kept in sync. Lightroom can read either; if they drift, you get "Metadata conflict" badges in the catalog.

```bash
# Step 1: clear both keyword lists
exiftool -overwrite_original \
    "-xmp-dc:subject=" "-iptc:Keywords=" \
    "<target.tif>"

# Step 2: populate both keyword lists + both description fields
exiftool -overwrite_original \
    "-xmp-dc:subject+=keyword1" "-iptc:Keywords+=keyword1" \
    "-xmp-dc:subject+=keyword2" "-iptc:Keywords+=keyword2" \
    ... \
    -tagsfromfile "<jpg>" \
    "-xmp-dc:description<iptc:Caption-Abstract" \
    "-iptc:Caption-Abstract<iptc:Caption-Abstract" \
    "<target.tif>"
```

### Backup behaviour

ExifTool writes a `<filename>_original` backup by default. Configurable in the UI:

- **Keep backups** (safe default for first runs): use `-overwrite_original` on each call (this still creates `_original` once, then doesn't again — actually verify; may need to track this).
- **No backups** (faster, less disk): use `-overwrite_original_in_place`.

Be aware: `_original` files accumulate. For a 100-GB library, this matters.

## 6. Architecture

Modular, testable. Suggested file layout inside the existing repo:

```
your_streamlit_app/
├── pages/
│   └── llm_metadata_sync.py        # Streamlit page — UI only
├── llm_metadata_sync/
│   ├── __init__.py
│   ├── matcher.py                  # filename matching, index building
│   ├── exiftool_runner.py          # subprocess wrapper, error handling
│   ├── merger.py                   # keyword union, description handling
│   ├── sync.py                     # orchestrator: ties matcher + merger + runner
│   └── models.py                   # dataclasses: MatchResult, SyncAction, SyncReport
└── tests/
    └── test_llm_metadata_sync/
        ├── test_matcher.py
        ├── test_merger.py
        └── test_sync.py
```

### Key data structures

```python
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

class TargetType(Enum):
    SIDECAR = "sidecar"      # write to .xmp file
    EMBEDDED = "embedded"    # write into TIF/PSD/DNG

class SidecarAction(Enum):
    CREATE = "create"
    MERGE = "merge"
    NONE = "none"            # for embedded, no sidecar action

@dataclass
class SyncAction:
    jpg_path: Path
    target_path: Path        # the XMP for sidecar, or the TIF/PSD/etc. for embedded
    target_type: TargetType
    sidecar_action: SidecarAction
    raw_path: Path | None    # the original RAW for sidecar mode (informational)
    
@dataclass
class SyncResult:
    action: SyncAction
    success: bool
    keywords_written: int
    description_written: bool
    error: str | None = None

@dataclass
class SyncReport:
    actions: list[SyncAction]
    results: list[SyncResult]
    orphaned_jpgs: list[Path]    # JPGs with no matched source
```

### Module responsibilities

- **`matcher.py`** — pure functions. Builds the stem index from a directory walk; resolves a JPG path to a list of `SyncAction` objects. No I/O beyond reading filenames. Easily unit-testable with `tmp_path` fixtures.
- **`exiftool_runner.py`** — wraps `subprocess.run` for exiftool calls. Two functions: `clear_keyword_lists(target, target_type)` and `write_metadata(jpg, target, target_type, keywords, description, keep_backups)`. Returns a result object with stdout/stderr captured. Validates exiftool is on PATH at startup.
- **`merger.py`** — pure functions. `build_keyword_union(existing, new, filter_list)` returns the deduplicated, filtered, ordered list. `read_existing_keywords(target)` reads via exiftool. `read_jpg_metadata(jpg)` reads JPG keywords + description.
- **`sync.py`** — orchestrates a full run. Accepts config + the two paths, produces a `SyncReport`. Generator-style API so the UI can show progress per file.
- **`pages/llm_metadata_sync.py`** — Streamlit only. Imports from the module. Handles UI state, progress bars, result display.

### Configuration

Use a single `SyncConfig` dataclass with sensible defaults:

```python
@dataclass
class SyncConfig:
    raw_root: Path
    jpg_dir: Path
    filter_keywords: list[str] = field(default_factory=lambda: ["nogps"])
    keep_backups: bool = True
    rating_suffix_range: tuple[int, int] = (1, 5)
    raw_extensions: tuple[str, ...] = (
        "RAF", "NEF", "CR2", "CR3", "ARW", "DNG",
        "RW2", "ORF", "PEF", "SRW", "IIQ", "3FR",
    )
    embed_extensions: tuple[str, ...] = ("tif", "tiff", "psd", "psb", "dng")
    deriv_patterns: tuple[str, ...] = (
        r"-Edit", r"-Edit-\d+",
        r"-Enhanced", r"-Enhanced-NR",
        r"-HDR", r"-HDR-\d+",
        r"-Pano", r"-Pano-\d+",
    )
    dry_run: bool = True
```

## 7. UI flow

### Page layout (top to bottom)

1. **Title + brief description** (one paragraph).
2. **Path inputs** (two `st.text_input`s with placeholder examples). Validate on entry: directory exists, is readable.
3. **Configuration expander** (`st.expander("Advanced options")`, collapsed by default):
   - Filter keywords (text area, comma-separated, default `nogps`)
   - Keep ExifTool backups (toggle, default on)
   - Rating suffix range (two number inputs, defaults 1 and 5)
   - Derivative patterns (text area, default values populated)
4. **"Scan" button**. Triggers indexing + matching but writes nothing.
5. **Scan results table** (`st.dataframe`):
   - Columns: JPG filename, matched targets (count), target types, action summary
   - Shown only after a successful scan
   - Below: counts of total JPGs, matched, orphaned
   - Expander: "Show orphaned JPGs (N)" listing them
6. **Dry run / Live run toggle** (radio button, default Dry run).
7. **"Apply" button**, disabled until a scan has been run. On click:
   - Show progress bar (one tick per JPG processed).
   - Stream a live log to a scrollable text area showing per-file actions.
   - On completion: summary block (counts, errors, elapsed time).
   - Offer a download button for the full log as `.txt`.
8. **Post-run guidance**: a small markdown block explaining the Lightroom Read-Metadata-from-File step (see §9).

### State management

Use `st.session_state` to hold:
- The most recent `SyncReport` from a scan (so re-rendering doesn't re-scan)
- The config used for the scan (warn if changed before Apply)
- A flag for whether scan completed cleanly

### Progress & responsiveness

Streamlit reruns the script on every interaction. For long-running operations, use a generator + `st.status` block so progress updates without blocking. ExifTool subprocess calls should not be threaded — they're already external processes and adding threads creates pickling pain in Streamlit.

For very large batches (1000+ files), consider batching exiftool calls — but only if simple per-file processing is too slow in practice. **Start simple**; optimise if the user reports it's too slow.

## 8. Edge cases & error handling

| Case | Handling |
|---|---|
| RAW root or JPG dir doesn't exist | Show inline error before scan can run |
| ExifTool not on PATH | Detect at app startup; show installation instructions |
| JPG has no keywords AND no description | Skip silently with note in log |
| JPG has keywords but no description | Process normally (description simply not written) |
| Target file is read-only | Catch exiftool error, log, mark result as failed |
| Multiple JPGs match same source (e.g. two different ratings of same shot) | Process all — last write wins; order them consistently (alphabetical by JPG name) |
| Same stem appears in multiple subdirectories | Process each independently — each has its own sidecar/embed |
| Stem index collision (rare but possible if user has flat name reuse) | Handle as multiple matches — the index value is a list, not a single path |
| User changes config between Scan and Apply | Warn and force re-scan |
| Scan finds zero matches across all JPGs | Show clear message, don't enable Apply |

### ExifTool error capture

Exiftool returns 0 on success, non-zero on errors. Capture both stdout and stderr per call. The "IPTCDigest is not current" warning is benign and should be filtered from displayed output. Real errors (file not found, permission denied, malformed XMP) should be surfaced per-file in the result.

## 9. Lightroom Classic workflow

Critical pre- and post-steps the UI must surface to the user.

### Before running

In LRC, select the photos that will be affected → **Metadata → Save Metadata to File** (Ctrl+S). This flushes any pending catalog-only edits to the XMP sidecars (and into TIFs) before the script overwrites them. Skipping this step can lose recent in-LR keyword work that hasn't been written out yet.

### After running

Same selection → **Metadata → Read Metadata from File**. LRC pulls the merged keywords + descriptions into the catalog. Any new keywords appear as flat keywords (not under a hierarchy) — the user can drag them into their hierarchy later if desired.

LRC does **not** auto-detect external XMP changes. There is no "auto-resync" — the Read Metadata step is required.

For embedded-edit files (TIFs/PSDs), LRC's catalog also stores keywords; these are pulled in via the same Read Metadata step.

The UI should display this guidance block after a successful Apply, and ideally also as an info message on initial load.

## 10. Out of scope

Explicitly **do not** build:

- Live filesystem watching / auto-sync on JPG changes
- Lightroom catalog (`.lrcat`) database modification — too risky, and the Read Metadata step achieves the same end
- A folder picker dialog (Streamlit doesn't have a real one; text input is sufficient)
- Multi-user support, authentication, or remote deployment
- Image content analysis or any LLM calls — all that happens upstream
- Recursive scanning of the JPG directory — assume flat
- Handling JPGs that aren't tagged (no keywords or description) — these should still be detected as orphaned-by-content (zero metadata to copy) and logged
- Reverse sync (source → JPG)

## 11. Testing

### Unit tests (pytest)

- `test_matcher.py`: stem stripping with/without rating suffix; index building from a fake directory tree (use `tmp_path`); edge cases like stems that end in legit `-N` not meant as ratings (acceptable false strip — document the assumption).
- `test_merger.py`: keyword union with overlap, dedup behaviour, filter list application (case-insensitive), order preservation, empty inputs on either side.
- `test_sync.py`: orchestration with mocked `exiftool_runner`. Verify the right ExifTool calls are issued for sidecar vs. embedded targets.

### Integration tests

- One with real ExifTool against fixture files (small synthetic XMP and TIF). Skip if exiftool not on PATH.
- Provide one canonical fixture: a tiny TIF with known keywords + description, and an XMP sidecar similarly, plus a JPG with overlapping + new keywords. Assert the post-sync state matches expectations.

### Manual acceptance

Provide a checklist in the PR description:

- [ ] Scan a directory tree of ~10 known JPGs against a known RAW tree. Verify match count and reported targets are exactly right.
- [ ] Dry run produces no file changes (verify modification times unchanged).
- [ ] Live run on a copy of real data, then open in LRC and confirm Read Metadata from File pulls in the new keywords/description.
- [ ] Verify `.acr` files are untouched after a run.
- [ ] Verify orphaned JPGs are listed and don't crash the run.
- [ ] Verify `_original` backups are created (with backup mode on) and not (with backup mode off).

## 12. Notes on existing prototype

A bash prototype exists with verified ExifTool commands and merge logic. The Python implementation should follow the same approach (subprocess calls to exiftool, two-step list write, dedup-in-Python rather than relying on `-api NoDups`). Specifically:

- `-api NoDups=1` was tested and **does not deduplicate** when used with `-tagsfromfile` source operations. Do not rely on it.
- The two-step (clear, then populate) sequence is required for list tags — single-call mixing of `=` and `+=` does not clear.
- For embedded targets, write to both IPTC and XMP-dc namespaces in the same exiftool call to avoid sync drift.
- Filter `IPTCDigest is not current` warnings from displayed output; they're benign.

## 13. Acceptance criteria

The feature is complete when:

1. A user can specify two directory paths in the UI, run a scan, and see an accurate preview of what would be written.
2. A user can apply the sync and see real-time progress.
3. After applying, opening LRC and running Read Metadata from File on the affected photos pulls in the new keywords and descriptions correctly.
4. Orphaned JPGs are clearly reported and don't block the run.
5. A failed write on one file doesn't abort the run for other files.
6. `.acr` files are never modified.
7. Tests pass; coverage on `matcher.py` and `merger.py` is meaningful (>80%).

---

## Appendix A — Reference exiftool commands (verified working)

These were verified against real Fujifilm RAF/XMP and Topaz-processed TIF files.

**Read keywords from JPG:**
```bash
exiftool -s -s -s -iptc:Keywords <jpg>
# Output: comma-separated string
```

**Read description from JPG:**
```bash
exiftool -s -s -s -iptc:Caption-Abstract <jpg>
```

**Read existing keywords from XMP sidecar:**
```bash
exiftool -s -s -s -xmp-dc:subject <target.xmp>
```

**Clear keyword list on XMP sidecar:**
```bash
exiftool -overwrite_original "-xmp-dc:subject=" <target.xmp>
```

**Write merged metadata to XMP sidecar (after clearing):**
```bash
exiftool -overwrite_original \
    "-xmp-dc:subject+=Melbourne" \
    "-xmp-dc:subject+=moon_eclipse" \
    "-xmp-dc:subject+=red moon" \
    -tagsfromfile <jpg> \
    "-xmp-dc:description<iptc:Caption-Abstract" \
    <target.xmp>
```

**Write merged metadata to embedded TIF (both namespaces):**
```bash
# Clear
exiftool -overwrite_original \
    "-xmp-dc:subject=" "-iptc:Keywords=" \
    <target.tif>
# Populate
exiftool -overwrite_original \
    "-xmp-dc:subject+=Melbourne" "-iptc:Keywords+=Melbourne" \
    "-xmp-dc:subject+=westgate_park" "-iptc:Keywords+=westgate_park" \
    -tagsfromfile <jpg> \
    "-xmp-dc:description<iptc:Caption-Abstract" \
    "-iptc:Caption-Abstract<iptc:Caption-Abstract" \
    <target.tif>
```

## Appendix B — Filename pattern reference

Real examples confirmed in the user's library:

| Type | Example filename |
|---|---|
| LLM-tagged JPG, no rating | `2026-03-03_22-19-41-X-T5.jpg` |
| LLM-tagged JPG, rated 5 | `2026-02-22_11-02-58-X-T5-5.jpg` |
| RAW original | `2026-03-03_22-19-41-X-T5.RAF` |
| XMP sidecar | `2026-03-03_22-19-41-X-T5.xmp` |
| Adobe Camera Raw enhancement cache | `2026-03-03_22-19-41-X-T5.acr` (skip) |
| Photoshop edit derivative | `2026-02-22_11-02-58-X-T5-Edit.tif` |

The shared stem is everything up to the rating suffix (JPG side) or up to the derivative suffix (source side). Matching is on stem equality, not exact filename equality.
