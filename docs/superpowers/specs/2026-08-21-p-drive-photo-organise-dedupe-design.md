# P: drive photo organise + dedupe — design

**Date:** 2026-08-21
**Status:** design approved in outline; awaiting spec review
**Target:** `P:` — "8TBVOL Photos Backup", 7.45 TB, 365 GB free

## 1. Problem

`P:` holds roughly 840,000 files across 7.5 TB, accumulated from many
imports with inconsistent names, nested and overlapping folder trees, and
substantial duplication. It is an **independent archive**: it is not to be
reconciled against the Lightroom catalog on `L:` or the OneDrive tree on
`E:`. Everything is decided from `P:` alone.

Goals, in priority order:

1. Remove duplicate files, freeing space, without ever destroying a unique image.
2. Rename surviving files to the established convention.
3. File them by capture year, from EXIF rather than filename.

## 2. Constraints

- **No room for a safety copy.** 365 GB free against 7.5 TB used. Same-volume
  moves are free renames; cross-volume moves are slow copies.
- **Duplicate quarantine goes to `L:`** (2.96 TB free), which the user then
  reviews and either relocates or discards.
- **`L:` has a cable fault confirmed 2026-08-20.** Any sustained write to `L:`
  must be resumable and hash-verified, and must halt rather than continue on I/O error.
- **`P:` is USB and can drop out.** Every stage is resumable from persisted state.
- **Windows long paths.** `LongPathsEnabled=1`. Every *filesystem* call goes
  through `win_long()`, which applies the `\\?\` prefix.

  ⚠ **But `\\?\` paths must NEVER be handed to exiftool.** exiftool 12.85
  parses the `?` as a wildcard and refuses the path — *even a short one* —
  answering "Wildcards don't work in the directory specification". Since
  `walk.py` stores every Windows path prefixed, passing stored paths straight
  through made every EXIF read fail **silently**: `read_many` returned `{}`,
  every photo was marked undated, and the whole archive routed to `_UNDATED`
  with no error at all. `_exiftool_args()` strips the prefix. This was
  invisible from WSL, where `win_long` is a no-op, and only surfaced when the
  pipeline was run under real Windows Python (2026-08-21).

  **Correction (2026-08-21):** an earlier draft justified the prefix by
  claiming a PowerShell pass undercounted `3+ Star_RAW_Backups` by 47 GB
  through long-path truncation. That diagnosis was wrong. A full census of the
  in-scope tree found **zero** paths at or over 260 characters across 746,266
  files; the longest is 194. The scan discrepancy is real but has another cause
  (most likely PowerShell's suppressed access errors). `win_long()` is retained
  as cheap insurance against future deep paths, not because P: has any today.

  Silent skipping remains the primary data-loss risk in this project — the
  exiftool bug above is exactly that class, and it reached working, fully
  green-tested code.
- **Execution is Windows-native.** `C:\Python311\python.exe` (3.11.9) and
  `C:\WINDOWS\exiftool.exe` (12.85). Running ~764k files through WSL's 9p
  drvfs mount would be prohibitively slow. `numpy` is present; `PIL` and
  `imagehash` are **not** and must be installed for tiers 2–3.
- **No GPU or LLM involvement.** This is pure I/O; it does not compete with the
  LM Studio / Hermes stack and is not bound by the 6am–2pm power window.

## 3. Scope

### In scope — 12 folders, ~764k files, ~5.4 TB

| Folder | Files | GB |
|---|---|---|
| `0 and 1 star photos originals and dupes` | 175,842 | 3,078 |
| `Backup Consolidated Photos` | 425,988 | 2,076 |
| `3+ Star_RAW_Backups` | 9,841 | 740 |
| `Catalogued Photos_Predig and Preraw backups_19 Aug 2026` | 22,453 | 284 |
| `family_Randoms` | 118,975 | 164 |
| `2024-RAW` | 7,837 | 134 |
| `Fully Tagged Photos` | 23,645 | 38 |
| `Google Drive Photos` | 1,058 | 1 |
| `SMS_Photos` | 80 | 0.1 |
| `Carey 50 yr reunion` | 27 | ~0 |
| `Cremorne` | 28 | ~0 |
| `250922 Bahnreise Schweiz (Glacier+Zermatt)` | 5 | ~0 |

Counts above are from a non-long-path census and are **indicative only**. The
authoritative inventory is produced by stage 1.

### Out of scope

**Hard exclusions — touching these breaks things:**

- `New LR Catalog` — live `.lrcat` files and `.lrdata` preview pyramids
  (226,921 files). Renaming inside a Lightroom catalog corrupts it.
- `LR Backups` — catalog backup archives.
- `Music DRM Quarantine 2026-08-12` — nested inside the `0 and 1 star` folder, non-photo.
- `$RECYCLE.BIN`, `System Volume Information`.

**Excluded by user decision:**

- Other people's folders: `Scott`, `Heather`, `Chris Booth`, `LR Tim and Marisa`,
  `LR Tim and Marisa Friends`, `Mai and Jacob to share`, `marisaandtimsifnos-photo-download-1of1`
- Non-personal imagery: `NASA`, `Wikipedia_Images`, `Wildflowers`, `Professional Images`
- Derived products and admin: `Momento`, `Momento Photo Books`,
  `Imageworks Proof Folders`, `Camera Stuff`, `Office Lens`

**`*_original` files are excluded entirely.** The user is relocating these off
`P:` separately. Any file whose name ends in `_original` is skipped at walk
time, so the pipeline is correct whether or not that relocation has finished.

For the record, the `_original` population measured 2026-08-21 was 21,679 files
/ 1,131.6 GB, of which **17,136 files / 1,125.6 GB were orphaned** — no live
sibling, meaning the `_original` was the only copy at that path. This is why
they are not treated as redundant backups.

`.old` files (3,270 / 9.8 GB, from earlier `photo_batch sync` JPG replaces) get
**no special case**. They pass through the normal hash pipeline: caught if
genuinely duplicate, kept if not.

## 4. Architecture

**Index-first.** One read-only pass builds a SQLite index. Every subsequent
decision is a query against that index producing a reviewable CSV plan. A
separate executor applies an approved plan and journals every operation.

This matters because the expensive work — reading ~5.4 TB — happens once.
Re-planning after a rule change costs seconds, not hours. It also finds
**cross-folder** duplicates, which is the dominant duplicate class here:
`Backup Consolidated Photos` (426k files) and `0 and 1 star photos originals
and dupes` (176k files) are near-certainly holding copies of one another, and
no folder-local scheme can see that.

Rejected alternatives:

- *Folder-at-a-time* — simpler and yields results sooner, but structurally
  blind to cross-folder duplicates, which is where the reclaimable space is.
- *Streaming single-pass* — fastest in wall-clock, but unreviewable,
  unresumable and irreversible against 7 TB.

**Execution order is dedupe first, organise second** — the reverse of the
initial request. Renaming and moving several hundred thousand files that are
about to be quarantined wastes hours of USB I/O, and hashing is name-independent
so dedupe loses nothing by running first. The single scan feeds both stages;
only execution order changes.

## 5. Data model

SQLite database on `C:` — never on `P:`, so a drive dropout cannot cost the index.

```sql
CREATE TABLE files (
    id             INTEGER PRIMARY KEY,
    path           TEXT UNIQUE NOT NULL,  -- full \\?\ path
    top_folder     TEXT NOT NULL,         -- in-scope root it came from
    rel_dir        TEXT NOT NULL,         -- path below top_folder, for provenance
    filename       TEXT NOT NULL,
    ext            TEXT NOT NULL,         -- lowercased
    size           INTEGER NOT NULL,
    mtime          REAL NOT NULL,
    kind           TEXT NOT NULL,         -- image | raw | video | sidecar | other
    partial_hash   TEXT,                  -- first+last 64KB, size-collision groups only
    sha256         TEXT,                  -- full, only when partial_hash collides
    pixel_hash     TEXT,                  -- tier 2, decoded content
    percept_hash   TEXT,                  -- tier 3
    exif_dt        TEXT,                  -- ISO capture datetime
    exif_dt_source TEXT,                  -- which tag supplied it
    camera_model   TEXT,
    sidecar_of     INTEGER REFERENCES files(id),
    state          TEXT NOT NULL DEFAULT 'walked'
);
CREATE INDEX idx_size    ON files(size);
CREATE INDEX idx_sha     ON files(sha256);
CREATE INDEX idx_dt      ON files(exif_dt);
CREATE INDEX idx_state   ON files(state);
```

## 6. Stages

Every stage is independently runnable, resumable, and **dry-run by default**;
writing requires `--apply`.

### Stage 1 — walk and stat

`os.scandir` over the 12 in-scope roots with `\\?\` paths. No file contents are
read. Skips excluded folders and any `*_original`. Classifies `kind` by
extension. Links `.xmp` and `.aae` sidecars to their parent image by stem.

**Halt-on-error:** an `OSError` on a scope root aborts the stage. A run that
cannot see a folder must fail loudly, never record it as empty — that is exactly
the failure that left the 1990/1991 photo index at zero rows without an error.

### Stage 2 — EXIF read

`exiftool -stay_open` batch mode over image/raw/video rows, requesting
`-DateTimeOriginal -CreateDate -SubSecDateTimeOriginal -Model -Make -FileType`
with `-fast2` (header only, no full scan). Date precedence:
`SubSecDateTimeOriginal` → `DateTimeOriginal` → `CreateDate`. The winning tag is
recorded in `exif_dt_source`.

**File mtime is never used as a capture date.** Per user decision, EXIF only;
everything else is undated.

### Stage 3 — hash

The efficiency core. A file whose `size` is globally unique cannot be a
byte-duplicate and is **never read**. For the remainder:

1. `partial_hash` = SHA-256 of first 64 KB + last 64 KB + size.
2. Full `sha256` only where `partial_hash` collides.

This should eliminate the large majority of those 5.4 TB of reads, and is the
difference between a run measured in hours and one measured in days.

### Stage 4 — duplicate planning

- **Tier 1 — byte-identical.** Equal `size` and `sha256`. Applied automatically
  per user decision.
- **Tier 2 — same pixels, different encoding.** Equal `pixel_hash` over decoded
  image data. Catches PNG/JPG re-encodes and re-saves. **Reviewable report; not
  auto-applied.**
- **Tier 3 — visually near-identical.** Perceptual hash. Catches resizes,
  recompression, minor crops. **Reviewable report; not auto-applied.**

**RAW and JPG are never paired.** A JPG is never removed because a RAW of the
same scene exists, regardless of tier.

**Tiers 2 and 3 cover decodable stills only** — `.jpg`, `.jpeg`, `.png`,
`.tif`, `.tiff`, `.heic`, `.bmp`, `.gif`. RAW files (`.raf`, `.nef`, `.cr2`,
`.dng`) are compared by byte hash alone: decoding 70,671 RAF files to compare
pixels would cost more than the whole rest of the pipeline, and two RAWs that
differ byte-wise are in practice genuinely different exposures. Video is
excluded from tiers 2 and 3 entirely.

**Keeper rule**, deterministic and printed in every report — first
discriminator wins:

1. Prefer a copy whose path does **not** contain `dupes`, `Backup`, or `_DUPES`.
2. Prefer richer EXIF (has `exif_dt`, then has `camera_model`).
3. Prefer the shallower path.
4. Prefer the older `mtime`.
5. Tie-break on `path` lexically, so runs are reproducible.

A sidecar is never an independent duplicate; it follows its parent's fate.

### Stage 5 — quarantine

Approved duplicates are moved to `P:\_DUPES\<top_folder>\<rel_dir>\`, preserving
provenance in the path. Same-volume rename: fast and free. Nothing is deleted.

### Stage 6 — evacuate to `L:`

Separate, resumable, hash-verified: copy to `L:`, re-hash the destination,
compare, and only then remove the `P:` source. Any mismatch or I/O error halts
the stage. Deliberately decoupled from stage 5 so a multi-hour cross-drive copy
over a link with a known cable fault never blocks the pipeline.

### Stage 7 — provenance stamp

Before anything moves, the original location is written into the file so context
survives the flattening:

- `XMP-dc:Source` = `<top_folder>/<rel_dir>` — machine-readable provenance.
- For folders whose name carries meaning, the folder name is appended as a
  keyword (`XMP-dc:Subject+=`) so it is findable in Lightroom and in
  `kb-photos` search. "Meaningful" is **an explicit list in config**, not a
  heuristic — seeded with `Carey 50 yr reunion`, `Cremorne`,
  `250922 Bahnreise Schweiz (Glacier+Zermatt)`, `SMS_Photos`,
  `Google Drive Photos`. The stage prints any in-scope folder not on the list
  so the user can extend it before running with `--apply`.

**Never write XMP into a proprietary RAW.** For `.raf`/`.nef`/`.cr2`, the stamp
goes to the `.xmp` sidecar, creating it if absent. `.dng` and `.tif` embed
directly, consistent with the existing `llm_metadata_sync` behaviour.

Applies only to files that are actually moving and whose source folder name is
not already generic or year-like — stamping all ~764k files would be wasteful.

### Stage 8 — organise

Target: `P:\<YYYY>\<YYYY-MM>\<YYYY-MM-DD HH-MM-SS-Model>.<ext>`

Matching the `L:` `Catalogued Post-Raw/<YEAR>/<YYYY-MM>` convention. Month
subfolders are structural, not cosmetic: several years here will exceed 50k
files, past which Explorer becomes unusable.

Naming rules:

- Base: `YYYY-MM-DD HH-MM-SS-<CameraModel>` — matching the existing exports
  (`2019-01-01 15-41-12-X-M1.jpg`). **No star-rating suffix**; `P:` material is
  largely unrated.
- No camera model → the trailing hyphen is dropped, giving
  `YYYY-MM-DD HH-MM-SS.ext`, not the dangling `1994-01-01 21-03-33--4.jpg` form.
- Model string sanitised for filesystem safety; internal spaces preserved as in
  the existing `HP pstc5200` names.
- Collisions append `-2`, `-3`, … matching the existing convention. Ordering is
  by `path` so re-runs are deterministic.
- `.xmp` and `.aae` sidecars are renamed to the parent's new stem and moved **in
  the same transaction**. A sidecar is never moved independently — detaching one
  loses the Lightroom edit it carries.

No `exif_dt` → `P:\_UNDATED\<top_folder>\<rel_dir>\<original filename>`,
preserving the original structure for later manual handling.

## 7. Journal and undo

Append-only CSV on `C:`, one per stage run:

```
timestamp,stage,operation,src,dst,size,sha256,status
```

The undo tool replays a journal in reverse. Before each reverse move it
verifies the destination still exists at the recorded size and hash, and
**refuses** to act on anything that changed since. Nothing in this design ever
deletes a file; the only irreversible act is the user emptying `_DUPES` on `L:`
by hand.

## 8. Failure handling

| Failure | Response |
|---|---|
| Path > 260 chars | `\\?\` prefix everywhere; a path that still fails is logged and skipped **loudly** |
| `P:` or `L:` dropout | Halt the stage immediately. Never skip-and-continue. Resume from `state`. |
| exiftool crash on a file | Record `exif_dt = NULL`, continue; the file lands in `_UNDATED` |
| Corrupt/unreadable image in tier 2/3 | Skip that comparison, keep the file |
| Rename collision at target | Suffix `-2`, `-3`; never overwrite |
| Interrupted run | Per-row `state` makes every stage idempotent on re-run |
| Two runs at once | Lock file; second run refuses to start |

## 9. Testing

TDD against a synthetic fixture tree carrying every hazard: known duplicate
sets, paths over 260 characters, image+sidecar pairs, undated files, unicode
and space-bearing names, RAW/JPG pairs of the same scene, colliding target
names, and a simulated mid-run abort.

Assertions cover: the planner's output for each tier, keeper selection
determinism, sidecars never separated from parents, `_UNDATED` routing, and
undo restoring the tree byte-for-byte.

**Live rehearsal before anything large**: `SMS_Photos` (80 files), then
`Google Drive Photos` (1,058), verified by hand, then `family_Randoms`
(118,975) as the first at-scale run.

## 10. Code location

`cortex_suite/scripts/photo_archive/` — sibling to `photo_batch.py` and
`photo_dedup.py`, keeping photo tooling together.

**Stdlib-only, with no `cortex_engine` imports**, because it executes under
Windows `C:\Python311\python.exe` rather than the WSL cortex venv. `PIL` and
`imagehash` are required for tiers 2–3 and must be installed into that
interpreter; tiers 0–1 and the organise stage need nothing beyond stdlib plus
`exiftool.exe`.

## 11. Open items

- `PIL` and `imagehash` need installing into `C:\Python311` before stages
  requiring tier 2/3 planning can run. Tier 1 and organise are unblocked.
- Video files (2,601 / 108 GB) are carried through naming and foldering on
  `CreateDate`, but are excluded from tiers 2 and 3, which are image-only.
- The true in-scope file count is unknown until stage 1 completes with
  `\\?\` paths; the non-long-path census undercounted by at least 47 GB in one
  folder alone.
