---
name: photo-enrich
description: Enrich Lightroom catalog photos (TIF/JPG) in place with AI descriptions, keywords and reverse-geocoded location — Claude Haiku vision + Nominatim, written via exiftool. Use whenever Paul asks to enrich, tag, caption, describe or geo-label photos, or mentions the Photo Processor, catalog masters, or the 4+ star export folder.
---

# Photo Enrichment (this laptop)

Enrich Lightroom catalog masters in place so LRC can read the metadata back. Paul's laptop, WSL2, `/home/longboardfella/projects/Cortex`.

## The workflow (current — use this)

```
LRC: Ctrl+S (Save Metadata to File) on the photos to enrich
  └─► enrich catalog masters IN PLACE (TIF + JPG)
        └─► LRC: Read Metadata from File
              └─► re-export 4+ star JPGs to OneDrive, overwriting the previous export
```

**Key principle:** the catalog TIF/JPG are the masters. RAWs are converted to TIF via export/reimport and then rated 1 star and excluded. Metadata flows catalog → export, one direction. Do not resurrect the old export→match→splice round trip.

### Standard paths

| What | Path |
|---|---|
| Catalog masters (enrich these) | `C:\Users\paul\Pictures\2026\Catalog_Sources` |
| 4+ star export (backup only, downstream) | `C:\Users\paul\OneDrive\Master_Pictures\2026\...` |

WSL form: `/mnt/c/Users/paul/...`. Always run paths through `convert_windows_to_wsl_path`.

## Before running — check these

1. **Paul has done Ctrl+S in LRC.** Ask if unstated. Enriching a photo whose catalog edits weren't flushed means Read Metadata will overwrite them. He usually saves only 3+ star — if the folder contains lower-rated photos, say so.
2. **Dependencies resolve.** All three fail *silently*:
   ```bash
   venv/bin/python -c "import anthropic, geopy; print('ok')"
   which exiftool
   grep -c ANTHROPIC_API_KEY .env
   ```
   - `anthropic` missing → Claude silently falls back to local Ollama, captions look fine but are worse
   - `geopy` missing → location comes back empty with only a log warning; GPS still reads fine
   - `ANTHROPIC_API_KEY` missing/invalid/no-credit → same silent Ollama fallback
3. **Confirm Haiku is actually being used** — watch for `Claude vision (claude-haiku-4-5-20251001) returned N chars`. Its absence, or `Claude vision returned empty — falling back to local Ollama model`, means it is not.

## Offline / travelling mode

Three switches make a run fully network-free (see `docs/photo_processor_spec.md` §1b):

| Setting | Value |
|---|---|
| Use local vision model only | on (skips Claude even with the key set) |
| Location lookup | Offline only |
| Person tags | `Paul_C=Paul, Jacqui_C=Jacqui` (always offline) |

In code: `DocumentTextifier(geocode_mode="offline", prefer_local_vision=True, name_tags=parse_name_tags(...))`.

Offline geocoding returns the nearest **suburb** (Toowong) rather than the metro name (Brisbane). Paul has accepted this — he adds the city tag manually when it matters. State and country are identical to online.

**Local vision quality caveat:** `llava:7b` follows the format instructions perfectly but makes confident content errors (called a jalapeño "a slice of lime and a pickle"). Paul reviewed the local descriptions and preferred them, so this is his call — but flag accuracy, not formatting, as the risk when he asks about local models. `qwen3-vl:8b` is downloaded and unbenchmarked; the harness is at `scratchpad/vlm_bench.py`.

## Running it

Either the Streamlit page (**Photo & Metadata Tools → Photo Processor → source: Folder on disk**) or headless. Headless is better for large batches — no browser tab to keep alive:

```python
from dotenv import load_dotenv; load_dotenv()
from cortex_engine.textifier import DocumentTextifier
t = DocumentTextifier()
r = t.keyword_image(path, generate_description=True, populate_location=True,
                    clear_keywords=False, apply_ownership=True,
                    ownership_notice="All rights (c) Longboardfella. Contact longboardfella.com for info on use of photos.")
```

Collect files with `_collect_photo_dir()` from `pages/20_Photo_Metadata_Tools.py` — it recurses, filters to supported extensions, and skips `*_original`.

Budget ~10-15s per photo (Haiku ~8-15s on large TIFs, plus ~1s Nominatim pacing). Run in the background with a log file and report progress.

### Settings that matter

- **`clear_keywords` MUST stay False.** Paul's Lightroom keywords are the thing worth protecting; the pipeline merges them and feeds them to the vision model as hints. Clearing destroys them.
- `clear_location` False — the catalog files have no City fields, so nothing blocks geocoding. Only set True if re-processing photos with wrong existing location.
- **No exiftool backups.** Paul has accepted this (writes use `-overwrite_original`). Source files live on another drive; losing keywords is "not a huge drama". **Losing AI edits is the real concern** — so never re-run enrichment over already-enriched files without asking.

## How the pipeline works

`DocumentTextifier.keyword_image()` (`cortex_engine/textifier.py`), in order:

1. Optional resize → 2. optional clears → 3. **location resolution** → 4. vision description → 5. keyword extraction → 6. **location appended as keywords** → 7. optional anonymisation → 8. exiftool write

Stage 3 before stage 4 is deliberate: the resolved location is fed to the model as a hint so it doesn't guess the wrong country.

Stage 6 is what puts `melbourne`, `victoria`, `australia` into the tags. **It only runs when `generate_description=True`** — with descriptions off, location goes to EXIF fields only, not keywords. No GPS → tagged `nogps` instead.

Full detail: `docs/photo_processor_spec.md`.

## Sharp edges

- **Silent degradation** is the dominant failure mode — see dependency checks above. Output always looks plausible.
- **Existing location wins over geocoding** (`_merge_location_fields`, first-non-empty). An existing City is never overwritten unless `clear_location=True`.
- **Derived GPS is written as though measured.** When a photo has no GPS but has location hints, the geocoded city centroid is written to the GPS fields, indistinguishable from real coordinates afterwards.
- **Nominatim expects ~1 req/sec.** Keep a cooldown on large batches.
- **HEIC and RAW cannot enter the pipeline** — supported extensions are png/jpg/jpeg/tif/tiff/webp/gif/bmp. Convert first.

## Tab 2 (LLM Metadata Sync) — mostly obsolete here

Tab 2 propagates metadata from exported JPGs back to catalog sources via filename stem matching (`docs/llm_metadata_sync_spec.md`). **It is broken for this library shape**: TIF masters without a `-Edit`/`-Enhanced`/`-HDR`/`-Pano` derivative suffix are routed to *non-existent* XMP sidecars, which LRC ignores for non-RAW files. It silently no-ops.

Only relevant if a library still contains actual RAW files with sidecars. For Paul's current catalog, don't use it.
