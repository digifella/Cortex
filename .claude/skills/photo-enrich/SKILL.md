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

## ⚠️ Check VRAM before any local-model run — this dominates everything

Paul's laptop: RTX 4060 Laptop, **8188 MiB VRAM**, WSL2. **Lightroom Classic holds ~3.5 GB of it while open.** He has confirmed he cannot use LRC during enrichment runs.

**Always do this first, before estimating time or diagnosing slowness:**

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
tasklist.exe 2>/dev/null | grep -i lightroom   # warn him to close it if present
```

Measured 2026-07-28: LRC open → 3510 MiB used / 4450 free. LRC closed → **0 used / 7959 free**.

After the model loads, `ollama ps` is authoritative — it prints the CPU/GPU split:

| Model | Resident | With LRC open | With LRC closed |
|---|---|---|---|
| `qwen3-vl:8b` | **7.4 GB** (not the 6.1 GB download) | 75% CPU / 25% GPU, or crashes | 100% GPU |
| `llava:7b` | 4.9 GB | 19% CPU / 81% GPU | 100% GPU |

Anything other than `100% GPU` means it is spilling and will run 10–100× slower. A crash shows in the server log as `llama runner terminated, exit status 2`, and surfaces to the caller as an empty description or HTTP 500 `model runner has unexpectedly stopped`.

## Model choice is automatic — don't hardcode one

`cortex_engine/vision_model_selector.py` picks the best installed model that fits **current** free VRAM. It adapts to the machine, so on Paul's RTX 8000 it will select a larger model than on the laptop with no config change. Check what it will do:

```python
from cortex_engine.vision_model_selector import describe_selection
print(describe_selection())   # "7867MB free VRAM → gemma4:e2b-it-qat (…)"
```

**Ranking is by content accuracy, not style.** Style problems are fixable by re-running; a wrong noun becomes a permanent catalog keyword. On a big-VRAM machine expect `qwen3-vl:8b` or `:32b` — qwen3-vl was the only model that read a jalapeño garnish correctly where llava said "a slice of lime and a pickle" and Haiku said "cucumber".

**To add a model:** append a `VisionModelProfile`. Measure `vram_mb` from `ollama ps` *during inference* — `ollama list` reports the download size and understates `qwen3-vl:8b` by 1.3 GB, the difference between fitting and crashing.

**Beating Haiku is the bar Paul cares about.** Haiku got the cocktail wrong too (cucumber, and the wrong rim), so a local model that reads fine detail correctly is a genuine upgrade, not just a cost saving.

## Reasoning models: the trap that corrupted 44 captions

Reasoning models emit chain-of-thought *before* the answer. If `num_predict` is too small the answer never arrives — `content` is empty, `thinking` is full — and the pipeline used to write that reasoning into photo metadata as the caption.

**Naming does not tell you which is which.** `gemma4:e2b-it-qat` reads as instruction-tuned and thinks anyway; Gemma 3 doesn't.

| Family | Reasoning | num_predict |
|---|---|---|
| `qwen3-vl`, `gemma4` | yes | 640 |
| `gemma3`, `llava`, `minicpm-v`, `qwen2.5vl` | no | 160 |

`/no_think` and Ollama's `think: false` **do not work** on qwen3-vl (verified on 0.32.5 — `think: false` produced 3× more reasoning).

**When a new model returns empty descriptions,** the log says exactly what to do: `produced N chars of reasoning but no answer — raise CORTEX_VLM_NUM_PREDICT`. Add it to the profile table as a reasoning model rather than re-enabling the fallback.

## Ollama version matters

Gemma 4 needs Ollama ≥ 0.32; on 0.20.0 the pull fails instantly with HTTP 412 `requires a newer version`. If a model won't pull, check `ollama --version` before blaming the network. Upgrade: `curl -fsSL https://ollama.com/install.sh | sudo sh` (keeps existing models).

## Timing — what actually costs time

Measured on the 122-photo rated re-run (2026-07-28):

| Stage | Cost | Notes |
|---|---|---|
| Read 72 MB TIF from `/mnt/c` | 0.61 s | negligible |
| Decode + downscale + JPEG encode | 1.07 s | negligible |
| **Model inference** | **14 s – 120 s** | dominates entirely |

**Do not "optimise" the TIF path.** Extracting an embedded JPEG preview via ffmpeg/exiftool was considered and measured: file prep is 1.68 s of a ~20 s cycle, under 10%. The apparent "TIFs are slower" signal in the run logs was confounded — the TIF-heavy folders happened to be processed during a period of VRAM pressure. TIF median 152 s vs JPG 127 s in that window; when VRAM freed up, both dropped to ~20 s.

**Estimate from a warm, uncontended measurement of the real file mix** — not from a small JPG-only benchmark. A 6-photo benchmark predicted ~15 s/photo; the real 122-photo run averaged 2.3 min/photo because of VRAM contention.

**Use the headless script for bulk work**, not the Streamlit page. At minutes-per-photo, the one-photo-per-rerun UI loop means an unusable browser session.

## Running it — use the two-pass batch script

`scripts/photo_enrich_batch.py` is the tool for bulk work:

```bash
python scripts/photo_enrich_batch.py "C:\Users\paul\Pictures\2026\Catalog_Sources" \
    --only-empty --state /path/to/resume.json
```

- **Pass 1** uses the VRAM-appropriate model (fast, handles most photos).
- **Pass 2** automatically retries anything that came back as a `[Image: ...]` placeholder, using the strongest *installed* model regardless of VRAM fit — slow-but-correct beats fast-but-empty.
- `--only-empty` treats a placeholder as "not yet captioned", so re-running never redoes good work and always sweeps up past failures.
- Resumable via `--state`; an interrupted run continues rather than re-describing.
- `--no-retry` skips pass 2, `--pass1` / `--pass2` override model choice, `--after` filters by capture time.

**A placeholder is a failure, not a caption.** `DocumentTextifier.is_placeholder_description()` is the check. 26 photos once carried `[Image: vision model returned empty description]` as their permanent caption because a runner treated a placeholder as success — never write one and move on, always collect it for retry.

## Provenance

Every real caption records its author in `IPTC:Writer-Editor` and `XMP-photoshop:CaptionWriter` as `Cortex <version> / <model>` — Lightroom shows it in the metadata panel. Paul chose the dedicated field over a caption suffix so descriptions stay clean and searchable and attribution doesn't travel into exports as visible text.

**Provenance is written only for genuine captions.** Stamping it on a placeholder would assert authorship of a failure and make the file look processed.

## Verify a run properly

Checking for reasoning leakage is not sufficient — that check once reported "0 leak patterns" while 26 photos held placeholder captions. Check **both**:

```bash
# placeholders (failures written as captions)
exiftool -q -m -r -if '$IPTC:Caption-Abstract =~ /^\[Image:/' -p '$FileName' "$DIR"
# reasoning leakage
exiftool -q -m -r -p '$IPTC:Caption-Abstract' "$DIR" | grep -iE "mention the|let's check|^The photo is (of|a)|at [0-9]{1,2}:[0-9]{2}"
```

## Running it (details)

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
