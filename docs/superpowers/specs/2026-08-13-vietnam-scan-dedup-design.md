# 1996 Vietnam scan dedup + enrichment — design

**Date:** 2026-08-13
**Status:** approved, implementation in progress
**Target data:** `/mnt/photos/1996 - Vietnam` (198 JPGs) → `/mnt/l/Catalogued Pre-Dig/1996` (284 files)

## Problem

A folder of 1996 Vietnam holiday scans contains many redundant shots of the same
subjects (Cham towers, pagodas, markets, Mekong river scenes). They are poor
scans with almost no metadata: 1 of 198 has a description, 0 have a city, all
198 carry only generic keywords. Ratings are 161×3-star and 37×4-star.

The goal is a memories cleanup, not an archival-quality curation: keep one photo
per scene, flag the rest with the keyword `Duplicate` so they can be moved out of
the Lightroom catalog by hand, and enrich what survives with caption, keywords
and location.

### Why the existing `photo_batch.py sync` cannot do this

Verified by dry-run: **9 of 198 matched, 189 orphaned.** The Pre-Dig catalog uses
a different filename convention from the export:

```
catalog:  1996-04-14 18-38-12_Vietnam_1126 x 1692_.jpg
export:   1996-04-14 18-38-12--3.jpg
```

The stem-based matcher has nothing to join on. Capture timestamps cannot
substitute: the export side has 35 distinct timestamps across 198 files, the
catalog side 101 across 284, with single timestamps covering 54 and 40 files.

### Why filenames are NOT regularised first (considered and rejected)

1. **The current names carry human categorisation that would be destroyed.**
   Catalog filenames embed a subject token: Vietnam 100, Holiday 58, Market 18,
   Mekong 14, SMS 8, Family 4, Cuchi 4, Cham 4, Birthday 4, Cao Dai 3,
   Graduation 1. Mekong/Cu Chi/Cham/Cao Dai identify *specific sites* and are
   used by this design as a hard grouping constraint.
2. **Renaming cannot resolve the ambiguity, only disguise it.** A regularised
   `timestamp-camera-rating-N` name must assign `N` by directory walk order, not
   by image identity. Both sides would then present stems that look matchable
   while the copy-indices correspond only by coincidence — turning honest
   orphaning into silent, confident mis-attribution.
3. Renaming on disk would also orphan every file in Lightroom.

Regularisation remains available as a **follow-up**, driven by the verified
identity mapping this pass produces, and is the only safe order.

## Decisions

| Question | Decision |
|---|---|
| Dedup aggressiveness | **Same scene** — same subject at the same moment/viewpoint collapses; different sites or clearly different compositions stay separate |
| Location inference | **Landmark-only city** — `Country=Vietnam` always; city/site only when a named landmark is identifiable; never an invented city |
| Enrichment scope | **All survivors**, 3-star included |
| Filename regularisation | **Not before**; optional follow-up afterwards |

## Architecture

A single script, `scripts/photo_dedup.py`, run under the cortex 3.11 venv, with
one subcommand per stage so each is independently runnable and resumable.

### Stage 0 — `link` (export ↔ catalog identity)

Perceptual hash (64-bit dHash) of all exports and all catalog files; nearest
neighbour within a tight Hamming radius, filtered by aspect ratio and
tie-broken on capture timestamp. Emits a link manifest.

Catalog `-Edit.tif` derivatives (92 files) may be cropped beyond hash tolerance.
Those are reported **unlinked** rather than guessed.

**Aspect ratio is a second, independent join key.** Catalog filenames encode the
original scan dimensions and the exports are clean rescales — verified on three
samples at 1.00x, 2.00x and 2.50x, with aspect ratios agreeing to four decimal
places. Ratio must agree within tolerance for a link to be accepted.

**Validation (revised).** An earlier draft proposed checking against the 9
photos the existing stem matcher resolves. That check is invalid: those are not
verified pairs but a cartesian blowup on shared timestamps — a single export
maps to 9 different catalog photos including a portrait-orientation frame and
three different site tokens. There is no ground truth available.

Instead, linkage accuracy is estimated by **site-token agreement**: Haiku never
sees the filename, so a correct linkage puts `_Market_` files on photos it
independently called market scenes at a high rate, while a random linkage
collapses that agreement to chance. Report the rate; investigate before writing
if it is low.

> **Trap:** never run `photo_batch.py sync --apply` against a Pre-Dig folder
> without inspecting the dry run first. On the 1996 set it would write one
> photo's caption onto nine unrelated photos. The 2000 folder is far healthier
> (119 distinct timestamps across 138 files) but has two colliding clusters
> (10 files on `2000-03-01 11-46-23`, 5 on `2000-08-01 14-37-01`) that would
> mis-attribute the same way.

### Stage 1 — `describe`, part A (perceptual dedup, no LLM)

Exports within a very tight Hamming radius are the same frame re-scanned or
re-exported. Certain duplicates, decided arithmetically, no API cost.

### Stage 2 — `describe`, part B (one Haiku vision call per image)

`claude-haiku-4-5-20251001`, matching `textifier.CLAUDE_VISION_MODEL`. Returns
strict JSON per image: `caption`, `keywords`, `subject`, `scene_signature`,
`landmark`, `city`. One pass serves both grouping and enrichment.

Checkpointed to `.photo_dedup.json` in the export folder after every image, so a
WSL restart cannot lose the work.

### Stage 3 — `group` (same-scene clustering)

A group requires **both** a semantic match (subject + scene signature) **and** a
proximity signal (shared capture-time cluster or near hash). The conjunction is
what prevents two different Cham sites merging. Additionally, differing catalog
site tokens (Cuchi vs Cham vs Mekong vs Cao Dai) **veto** a merge outright.

Keeper selection, deterministic: highest star rating → largest pixel dimensions
→ earliest filename.

### Stage 4 — `sheet` (review gate)

A self-contained HTML contact sheet: one row per group, thumbnails, keeper
marked, proposed caption and location. Reviewed before any write. This is the
backstop for Stage 3 being judgment rather than arithmetic.

### Stage 5 — `apply` (write)

- Non-keepers: keyword `Duplicate` added.
- All survivors: caption, keywords, `Country=Vietnam`, city where a landmark was
  identified.
- Written to the export JPGs and, via the Stage 0 manifest, to the catalog
  originals — `exiftool -m`, `XMP:MetadataDate` bumped so Lightroom registers the
  external change, backups retained.

User then runs Read Metadata from File on 1996, filters on `Duplicate`, and moves
those photos out of the catalog.

## Safety

- Nothing is deleted. `Duplicate` is a keyword; removal stays a human action.
- Dry-run is the default; `--apply` required to write.
- Stage 0 aborts on ground-truth mismatch.
- Per-image checkpointing; interrupted runs resume for free.
- exiftool backups retained on catalog writes.

## Verification

- Stage 0: site-token agreement rate (see above); one-to-one assignment.
- Stage 3: contact-sheet review before any write.
- Stage 5: re-read a sample from the catalog and confirm keywords, description
  and a bumped `MetadataDate`.

## Risks

- Heavily cropped `-Edit.tif` derivatives may not hash-match; reported unlinked.
- Haiku may over-merge visually similar but distinct scenes; mitigated by the
  site-token veto, the proximity requirement, and the review gate.
- 69 catalog files carry no site token, so the veto does not apply to them.
