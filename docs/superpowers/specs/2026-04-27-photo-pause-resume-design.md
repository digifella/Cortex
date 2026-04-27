# Photo Batch Pause/Resume + EXIF Time Fix — Design Spec
**Date:** 2026-04-27  
**Status:** Approved

---

## Problem

Two independent issues in the Photo Processor tab (`20_Photo_Metadata_Tools.py` + `cortex_engine/textifier.py`):

1. **No pause/resume.** Large batches run as a single synchronous for-loop. There is no way to stop processing mid-batch without losing all progress beyond what the manifest has already saved. A tab close or page refresh abandons the run entirely.

2. **EXIF overrides filename incorrectly for time-of-day context.** `_read_photo_capture_datetime` prefers EXIF `DateTimeOriginal` over the filename timestamp. When the camera stores time in UTC (no tz offset in EXIF) but the filename encodes local time, the sun-phase hint passed to the VLM is wrong — e.g., a photo taken at 12:51 local time reports "night" because the raw EXIF hour is 22 UTC.

---

## Scope

- `pages/20_Photo_Metadata_Tools.py` — batch loop, manifest, UI controls
- `cortex_engine/textifier.py` — `_read_photo_capture_datetime`

No changes to other pages, the knowledge ingest pipeline, or the download/results UI.

---

## Design

### 1. Manifest schema (version 2)

The existing `_last_batch.json` grows two new fields and a `status` discriminator.

```json
{
  "version": 2,
  "status": "running | paused | done",
  "timestamp": "2026-04-27T14:03:00",
  "mode": "keyword_metadata",
  "all_file_paths": ["...", "..."],   // full ordered list, written once at run start
  "current_idx": 7,                   // index of next photo to process
  "results": [...],                   // completed results so far
  "file_paths": [...],                // completed file paths so far
  "run_settings": {                   // frozen snapshot of all UI options at start
    "generate_description": true,
    "populate_location": true,
    "clear_keywords": false,
    "clear_location": false,
    "anonymize_keywords": false,
    "blocked_keywords": [],
    "apply_ownership": false,
    "ownership_notice": "",
    "city_radius": 5,
    "fallback_city": "",
    "fallback_country": "",
    "max_width": null,
    "max_height": null,
    "convert_to_jpg": false,
    "jpg_quality": 88,
    "halftone_strength": 50,
    "halftone_preserve_color": true,
    "batch_cooldown_seconds": 3.0
  },
  "resume_after": null                // ISO timestamp for cooldown delay, or null
}
```

`status="done"` is the same shape as today's completed manifest — existing recovery logic continues to work unchanged.

`_save_photokw_manifest` is updated to accept and write these fields. `_load_photokw_manifest` still filters missing files and returns `None` when no usable manifest exists.

A new helper `_is_active_run(manifest)` returns `True` when `status in ("running", "paused")`.

---

### 2. Processing flow

#### Run initialisation (replaces lines ~858–1066)

When a Process button is clicked:
1. Write a fresh manifest: `status="running"`, `current_idx=0`, `all_file_paths=[all uploaded paths]`, `run_settings={frozen options}`, `results=[]`, `file_paths=[]`.
2. Call `st.rerun()` immediately — no photo processing in the button handler.

#### Single-photo dispatch (new block at top of `_render_photo_keywords_tab`)

Runs before any UI widgets are rendered:

```
manifest = _load_photokw_manifest()
if manifest and _is_active_run(manifest) and status == "running":
    if resume_after is set and now < resume_after:
        sleep(remaining seconds)
        st.rerun()
    process photo at all_file_paths[current_idx]
    append result, increment current_idx
    if current_idx >= len(all_file_paths):
        write status="done"
    else:
        write updated manifest (status="running", new current_idx, cooldown resume_after if applicable)
        st.rerun()
```

If `status == "paused"`, this block does nothing — the UI renders normally with Resume/Cancel controls.

#### Cooldown

When `batch_cooldown_seconds > 0` and there are more photos remaining, write `resume_after = now + cooldown` to the manifest instead of calling `time.sleep()` directly. The dispatch block above handles the sleep on the next rerun. This keeps the cooldown transparent to the user and survives a tab close during the delay.

---

### 3. UI controls

**During an active run (`status="running"`):**
- Progress bar as today, updated per photo
- Live log as today
- **Pause** button rendered above the progress bar. On click: write `status="paused"` to manifest + set `photokw_paused=True` in session state. Takes effect after the current photo finishes.

**When paused (`status="paused"`):**
- Info banner: "⏸ Batch paused — N of M photos done."
- **Resume** button: clears `photokw_paused`, writes `status="running"`, calls `st.rerun()`.
- **Cancel** button: calls `_clear_photokw_manifest()`, clears session state.

**Recovery banner (page load, no active session):**

Two mutually exclusive cases read from the same manifest file:

| Manifest status | Banner shown |
|---|---|
| `"running"` or `"paused"` | "▶ Resume batch — N of M done. Last run: [timestamp]." → **Resume** + **Cancel** |
| `"done"` | Existing recovery banner (unchanged) |

---

### 4. EXIF time-of-day fix (`textifier.py`)

In `_read_photo_capture_datetime`:

```
exif = _read_exif_capture_datetime(file_path)
filename_dt = _parse_filename_capture_datetime(Path(file_path).name)

if exif is None:
    return filename_dt  # unchanged from today

if exif["offset_minutes"] is not None:
    return exif  # explicit tz offset — trust it fully

if filename_dt is not None:
    exif_hour = exif["datetime_naive"].hour
    fname_hour = filename_dt["datetime_naive"].hour
    if abs(exif_hour - fname_hour) > 3 and abs(abs(exif_hour - fname_hour) - 24) > 3:
        log warning: "EXIF time ({exif_hour}h) and filename time ({fname_hour}h) differ by >3h; using filename"
        return filename_dt

return exif  # EXIF has no offset but filename agrees — keep EXIF (has full date)
```

The `abs(...- 24) > 3` guard handles midnight-crossing differences correctly (e.g. 23h vs 1h = 2h apart, not 22h).

---

## Files changed

| File | Change |
|---|---|
| `pages/20_Photo_Metadata_Tools.py` | Manifest schema v2, dispatch block, Pause/Resume UI, run init refactor |
| `cortex_engine/textifier.py` | `_read_photo_capture_datetime` cross-check logic |
| `docker/cortex_engine/textifier.py` | Same fix synced to Docker distribution |

---

## Out of scope

- Mid-photo interruption (pause takes effect between photos only)
- Parallel photo processing
- Progress persistence across different machines / users
