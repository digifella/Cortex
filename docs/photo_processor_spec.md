# Photo Processor — Enrichment Stage Spec

Tab 1 of `pages/20_Photo_Metadata_Tools.py`. Enriches Lightroom-exported JPGs in place with resolved location metadata, an AI-generated description, and keywords — producing the tagged JPGs that Tab 2 (LLM Metadata Sync) later splices back into the catalog sources.

This document was written retrospectively. The tab was moved verbatim out of `7_Document_Extract.py` (see `docs/superpowers/specs/2026-04-25-photo-metadata-tools-page-design.md`), and that migration spec deliberately recorded no behaviour — "renders identically to today". This fills that gap.

ExifTool is invoked as a subprocess throughout. Do **not** reimplement EXIF/IPTC/XMP parsing in Python.

---

## 1. Position in the overall workflow

```
Lightroom Classic
  └─ export rated JPGs ──► [Tab 1: Photo Processor]  ◄── this document
                              enrich: location → description → keywords
                                            │
                                            ▼
                           [Tab 2: LLM Metadata Sync]
                              splice back to XMP sidecars + TIF/PSD/DNG
                                            │
                                            ▼
                           Lightroom: Read Metadata from File
```

The enriched JPGs are the **source of truth** for the sync stage. Anything not written here cannot propagate back to the catalog.

Tab 2's contract is specified separately and in full in `docs/llm_metadata_sync_spec.md`. The two documents meet at one point: the keywords and description this stage writes into the JPG are exactly what the sync stage reads back out (`IPTC:Keywords` and `IPTC:Caption-Abstract`).

## 1a. Input modes

**Upload files** — `st.file_uploader`, capped at 1 GB per batch. Streamlit hands the app *bytes*, not paths, so uploads are always materialised into a temp folder and enriched there. The originals on disk are never touched; the enriched copies are downloaded from the Results panel.

**Folder on disk** — a validated directory path (Windows or WSL form). Supported images are collected recursively (`PHOTO_DIR_EXTENSIONS`, skipping exiftool `*_original` backups) and passed to the pipeline as **real paths, enriched in place**. This is the mode for enriching a Lightroom catalog's TIF/JPG masters directly, so LRC can then Read Metadata from File.

Folder mode has no single-photo preview or quick-edit panel — those depend on in-memory upload bytes.

> **Historical note.** Before this mode existed, the tab carried a "Write to original files" toggle that was never wired into `run_settings` and had no effect. It has been removed; folder mode is the real implementation of that intent.

## 1b. Offline mode (travelling)

The pipeline can run with no network access at all. Three independent switches:

| Concern | Online default | Offline setting | Module |
|---|---|---|---|
| Vision | Claude Haiku when `ANTHROPIC_API_KEY` is set | **Use local vision model only** → local Ollama model | `textifier.prefer_local_vision` |
| Location | Nominatim (~1 req/sec) | **Location lookup: Offline only** → local GeoNames dataset | `cortex_engine/offline_geocoder.py` |
| Person names | — | always post-hoc, no network | `cortex_engine/photo_name_tags.py` |

Set the first two and a run makes zero network calls. Offline geocoding is also *faster* than online, because it removes the Nominatim rate-limit pacing.

**Granularity trade-off.** The offline dataset returns the nearest populated place — a suburb rather than the metro name. Measured on real coordinates:

| Coordinates | Offline | Online (Nominatim) |
|---|---|---|
| −27.4846, 152.9948 | Toowong | Brisbane |
| −27.44, 153.06 | Ascot | Brisbane |
| −37.67, 144.88 | Attwood | Melbourne |

**State and country are identical either way**, so keyword search by state/country is unaffected; only the city keyword differs. `Auto` mode (the default) tries Nominatim first and falls back offline, giving metro names at home and uninterrupted enrichment on the road.

Requires `reverse_geocoder` and `pycountry`. When absent, the UI shows a warning and `describe_mode()` explains the fix — it does **not** silently return empty location fields.

## 1c. Person name substitution

When a photo carries a person keyword, the description names them instead of saying "a man":

```
"A man smiles at the camera."              + Paul_C            -> "Paul smiles at the camera."
"Two smiling adults sit at a table."       + Paul_C, Jacqui_C  -> "Paul and Jacqui sit at a table."
"A sandy path where a solitary figure walks" + Paul_C          -> "...where Paul walks"
```

Configured in the UI as `Tag=Name` pairs (default `Paul_C=Paul, Jacqui_C=Jacqui`); matching is case-insensitive, so a library tagged `Paul_C` works with the lowercase key.

**This runs after the model, not via the prompt.** Asking a small local model to use a supplied name reliably is precisely the instruction-following task they fail at; a post-hoc rewrite is deterministic and unit-tested.

**Deliberately conservative** — it declines to guess rather than risk a wrong name in a permanent archive:
- one name + a plural subject ("Two women...") → unchanged, since which woman is unknowable
- no generic person-phrase in the text → unchanged
- placeholder descriptions (`[Image: ...]`) → unchanged

Only the *first* person-reference is replaced, so a photo containing the tagged person plus strangers names only the primary subject.

## 1d. Local vision model selection (VRAM-adaptive)

`cortex_engine/vision_model_selector.py` picks the best **installed** model that fits the VRAM **free right now**, so one install adapts from an 8 GB laptop to a 48 GB workstation without configuration. `config.VLM_MODEL` is the floor — what `model_checker` asks users to install — not the ceiling.

Selection = highest `quality` among profiles whose `vram_mb + 1024 MB headroom` fits free VRAM. The headroom exists because running at the limit produced `llama runner terminated, exit status 2` in testing.

### Measured profiles

RTX 4060 Laptop (8188 MiB), six photos, identical prompts, 2026-07-28:

| Model | Resident | Time/photo | Clean | Notes |
|---|---|---|---|---|
| `gemma4:e2b-it-qat` | **1.6 GB** | 80 s | **5/6** | Most practical. Fits *alongside Lightroom*. |
| `qwen3-vl:8b` | 7.4 GB | 127 s | 4/6 | **Best content accuracy** — only model to read a jalapeño garnish correctly. |
| `llava:7b` | 4.9 GB | 90 s | 2/6 | Fast, weakest accuracy ("a slice of lime and a pickle"). |
| `qwen3-vl:4b` | 4.7 GB | 159 s | 3/6 | Worst of both — slow *and* 3 empty outputs. |

`vram_mb` is the **resident** size from `ollama ps` after a real inference, **not** the download size from `ollama list` — the latter understates `qwen3-vl:8b` by 1.3 GB, which is the difference between fitting and crashing.

### Adding a model

Add a `VisionModelProfile` to `VISION_MODEL_PROFILES`. Nothing else changes — the selector, the token budget and the UI status line all read from it. Measure `vram_mb` with `ollama ps` during inference and rank `quality` on **content accuracy**, not style: style is fixable by re-running, a wrong noun becomes a permanent catalog keyword.

### Reasoning vs instruct models

Reasoning models emit chain-of-thought before answering and need a large `num_predict` to reach the answer at all. **Naming is not a reliable signal** — `gemma4:e2b-it-qat` reads as instruction-tuned but thinks; Gemma 3 does not. Verified behaviour:

| Family | Reasoning | `num_predict` | Evidence |
|---|---|---|---|
| `qwen3-vl` | yes | 640 | at 140 → 610 chars reasoning, **zero** answer |
| `gemma4` | yes | 640 | at 160 → 635 chars reasoning, **zero** answer |
| `gemma3`, `llava`, `minicpm-v`, `qwen2.5vl` | no | 160 | at 512 llava overran to 36 words (25 at 140) |

`/no_think` and Ollama's `think: false` parameter **do not suppress reasoning** on qwen3-vl — verified on Ollama 0.32.5; `think: false` produced *three times more* reasoning.

### The failure this prevents

When a reasoning model exhausts its budget, `content` is empty and `thinking` is full. The pipeline used to fall back to the reasoning text as the caption — silently writing *"First, the beach: sandy, people are around."* into 44 photos' permanent metadata. That fallback is now **off by default** (`CORTEX_VLM_USE_THINKING_FALLBACK` to re-enable) and logs the exact remedy instead.

## 2. Prerequisites

| Requirement | Purpose | Absence behaviour |
|---|---|---|
| `exiftool` on PATH | All metadata read/write | Feature cannot run |
| `ANTHROPIC_API_KEY` | Claude Haiku vision (preferred) | **Silent** fallback to local Ollama |
| Ollama + a `qwen3-vl` model | Local vision fallback | `[Image: could not be described — vision model unavailable]` |
| Network access | Nominatim reverse geocoding | Location fields stay empty |

The Claude fallback is silent by design (`textifier.py:1018`) — a missing, invalid, or zero-credit key produces plausible captions from the local model with no UI warning. The only signal is the log line `Claude vision (claude-haiku-4-5-20251001) returned N chars`. Its absence, or the line `Claude vision returned empty — falling back to local Ollama model`, means Haiku is not being used.

## 3. Per-photo pipeline

Implemented by `DocumentTextifier.keyword_image()` (`cortex_engine/textifier.py:3705`). Ordering matters — each stage feeds the next.

### Stage 1 — Optional resize

When a resize profile is selected, downsample first so the vision model and all metadata writes operate on the final image. Metadata is preserved across the resize.

### Stage 2 — Optional clears

`clear_keywords` strips existing XMP Subject / IPTC Keywords; `clear_location` strips existing Country/State/City. Both default off. Clears happen **before** the existing-keyword read, so cleared values do not survive into the merge.

### Stage 3 — Location resolution

`resolve_photo_location()` (`textifier.py:3016`). Three paths:

1. **GPS present** → reverse-geocode via Nominatim → merge into city/state/country.
2. **No GPS, but location fields or fallbacks present** → forward-geocode the hint to derive GPS, then reverse-geocode that to normalise the fields. This writes *derived* GPS back to the photo.
3. **Neither** → location stays empty; the photo is tagged `nogps` at stage 6.

Merge precedence is **first non-empty wins** (`_merge_location_fields`, `textifier.py:2929`), and existing embedded fields are seeded ahead of geocoded values. Consequence: **an existing City in the file is never overwritten by the geocoder.** To force geocoded values to win, enable `clear_location`.

Fallback city/country apply only when the photo has neither GPS nor any embedded location field.

### Stage 4 — Vision description

Two hints are assembled and passed to the vision model as `context_hint`:

- **Time hint** — local time and sun phase derived from capture timestamp + GPS, so the model can distinguish dawn from dusk.
- **Keyword hint** — existing EXIF keywords plus the resolved location. Location is placed first deliberately, so it overrides visual guesswork (`_build_keyword_hint`, `textifier.py:2645`).

This is why stage 3 precedes stage 4: a photo whose location resolves correctly produces a better-grounded description.

Description output is constrained to 1–2 declarative sentences, max ~35 words, no structural labels or self-directives. Images detected as logos/icons return `[Image: logo/icon omitted]`.

### Stage 5 — Keyword extraction

Keywords are extracted from the generated description, anchored on existing EXIF keywords.

### Stage 6 — Location keywords

The decisive step for the "tags with locations" requirement (`textifier.py:3808-3815`):

- If location resolved → append `city`, `state`, `country` as **lower-cased** keywords, skipping case-insensitive duplicates.
- If no GPS and no location → append the literal keyword `nogps`.

`nogps` is a deliberate marker, not noise. Tab 2's default filter list drops it on write-back (`docs/llm_metadata_sync_spec.md` §4), so it flags un-located photos for review in the JPGs without polluting the catalog.

Both branches are gated on `generate_description` being on. **With descriptions off, no location keywords are written** even when `populate_location` is on — location goes to the EXIF fields only.

### Stage 7 — Anonymisation (optional)

When enabled, sensitive keywords are filtered from both generated and existing keyword sets using a hybrid filter plus the user's blocked list (UI default: `friends,family,paul,paul_c,jacqui`).

### Stage 8 — Write

Keywords, description, resolved location fields, derived GPS, and the optional ownership/copyright notice are written to the JPG via exiftool.

## 4. UI options reference

All controls live in the left "Input" column.

| Control | Default | Notes |
|---|---|---|
| Photo source | Upload files | `Upload files` = drag-and-drop, processed as temp copies. `Folder on disk` = enrich real files **in place**. |
| Photo folder | — | Folder mode only. Accepts Windows or WSL paths; recurses subfolders; skips `*_original` backups. |
| City location radius | 5 km | Reverse-geocode radius. Larger helps rural locations. |
| Clear existing keywords/tags first | off | See stage 2. |
| Clear existing location fields first | off | Required to let the geocoder override embedded City/State/Country. |
| Generate AI description + keywords | **on** | Off ⇒ no location keywords either (stage 6). |
| Fill location and GPS metadata | **on** | Enables stage 3. |
| Location lookup | Auto | Auto / Online only / Offline only — see §1b. |
| Use local vision model only | off | Skips Claude even when the API key is set — see §1b. |
| Person tags | `Paul_C=Paul, Jacqui_C=Jacqui` | `Tag=Name` pairs — see §1c. |
| Fallback city / country | empty | Only used when no GPS *and* no embedded location. |
| Resize profile | Keep original | Low 1920×1080, Medium 2560×1440. |
| Convert resized output to JPG | off | Non-JPG sources only. |
| JPG quality | 90 | Only when converting. |
| Halftone repair strength | 42 | For scanned/printed sources. |
| Preserve colour during halftone repair | on | Repairs luminance channel only. |
| Anonymize sensitive keywords | off | Applies blocked list. |
| Blocked keywords | `friends,family,paul,paul_c,jacqui` | |
| Insert ownership info | **on** | Writes copyright/ownership to EXIF/IPTC/XMP. |
| Ownership notice | Longboardfella notice | |
| Cooldown between photos | 2.0 s | Batch responsiveness. Also paces Nominatim (~1 req/s policy). |

## 5. Batch execution, pause and resume

Specified in `docs/superpowers/specs/2026-04-27-photo-pause-resume-design.md`. Summary of the contract:

- One photo per Streamlit rerun; state persisted to a manifest between photos.
- Progress and live log render **before** the vision call, so the page is never blank mid-batch.
- Per-photo CLI progress line printed to the Streamlit terminal: `[n/total] ✅ filename: description…`.
- On fresh page load with a `running` manifest, status downgrades to `paused` and a Resume / Cancel banner is shown rather than auto-resuming.

## 6. Known sharp edges

1. **Silent Haiku fallback** — §2. The highest-impact failure mode: output looks fine but quality silently degrades.
2. **Existing location wins over geocoding** — §3. Surprising when re-processing photos that already carry partial or wrong location data.
3. **Location keywords require descriptions on** — §3 stage 6.
4. **Derived GPS is written as though measured.** Path 2 of stage 3 writes geocoded coordinates into the photo's GPS fields. These are a city-centroid approximation, not where the shutter fired, and nothing in the file distinguishes them from real GPS afterwards.
5. **Nominatim rate limits.** Public endpoint expects ~1 request/second. The cooldown slider is the pacing mechanism; setting it to 0 for a large batch risks throttling or empty geocode results. Offline geocoding has no such limit.
6. **Offline city names are suburbs.** See §1b — `Toowong` rather than `Brisbane`. Fine for archival, but a library enriched in mixed online/offline sessions will have inconsistent city keywords.
7. **Local vision models trade instruction-following for accuracy.** `_normalize_vlm_text` (`textifier.py:446`) scrubs self-directives and chain-of-thought, so local output *reads* clean — but a small model can be confidently wrong about content (in testing, `llava:7b` described a jalapeño garnish as "a slice of lime and a pickle"). Regex cannot catch that, and the keywords derived from it are written into the catalog.

## 7. Out of scope

- Writing to catalog sources — that is Tab 2's job (`docs/llm_metadata_sync_spec.md`).
- Lightroom catalog (`.lrcat`) modification.
- Recursive scanning of the input folder.
- Reverse sync (source → JPG).
