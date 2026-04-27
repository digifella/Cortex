# Photo Batch Pause/Resume + EXIF Time Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full pause/resume (surviving tab close) to the photo keywording batch job, and fix incorrect "night" time-of-day hints caused by EXIF time overriding filename time when no timezone offset is present.

**Architecture:** Replace the synchronous batch for-loop with a one-photo-per-rerun dispatch model. An extended on-disk manifest (`_last_batch.json`) tracks the full run state; `st.rerun()` drives the loop. The EXIF fix is a targeted cross-check in `_read_photo_capture_datetime`.

**Tech Stack:** Streamlit session state + `st.rerun()`, JSON manifest on disk (`/tmp/cortex_photokw/`), Python `shutil.copy2`, `datetime.timedelta`.

---

## File Map

| File | What changes |
|---|---|
| `pages/20_Photo_Metadata_Tools.py` | Manifest v2 helpers, dispatch block at top of render, run-init refactor (replaces for-loop), Pause/Resume UI, recovery banner update |
| `cortex_engine/textifier.py` | `_read_photo_capture_datetime` cross-check: prefer filename when EXIF has no tz offset and times diverge >3h |
| `docker/cortex_engine/textifier.py` | Same EXIF fix synced to Docker distribution |
| `tests/unit/test_textifier_photo_time_hint.py` | New tests for the cross-check logic |

---

## Task 1: EXIF time cross-check fix

**Files:**
- Modify: `cortex_engine/textifier.py` — `_read_photo_capture_datetime` (~line 2199)
- Modify: `docker/cortex_engine/textifier.py` — same function
- Modify: `tests/unit/test_textifier_photo_time_hint.py` — add cross-check tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_textifier_photo_time_hint.py`:

```python
# ------------------------------------------------------------------
# _read_photo_capture_datetime cross-check: prefer filename when
# EXIF has no tz offset and times diverge by more than 3 hours
# ------------------------------------------------------------------

def test_read_capture_datetime_prefers_filename_when_exif_time_diverges(tmp_path):
    """When EXIF has no tz offset and differs from filename by >3h, filename wins."""
    fake_jpg = tmp_path / "2025-09-27 12-51-13--5.jpg"
    fake_jpg.touch()

    # EXIF says 22:51 (UTC, camera was wrong), filename says 12:51
    exif_result = {
        "datetime_naive": datetime.datetime(2025, 9, 27, 22, 51, 13),
        "offset_minutes": None,
        "source": "exif",
    }
    with patch(
        "cortex_engine.textifier.DocumentTextifier._read_exif_capture_datetime",
        return_value=exif_result,
    ):
        result = DocumentTextifier._read_photo_capture_datetime(str(fake_jpg))

    assert result is not None
    assert result["datetime_naive"].hour == 12
    assert result["source"].startswith("filename:")


def test_read_capture_datetime_keeps_exif_when_times_agree(tmp_path):
    """When EXIF and filename agree (within 3h), EXIF is kept (it has higher precision)."""
    fake_jpg = tmp_path / "2025-09-27 12-51-13--5.jpg"
    fake_jpg.touch()

    exif_result = {
        "datetime_naive": datetime.datetime(2025, 9, 27, 12, 55, 0),
        "offset_minutes": None,
        "source": "exif",
    }
    with patch(
        "cortex_engine.textifier.DocumentTextifier._read_exif_capture_datetime",
        return_value=exif_result,
    ):
        result = DocumentTextifier._read_photo_capture_datetime(str(fake_jpg))

    assert result["source"] == "exif"
    assert result["datetime_naive"].hour == 12


def test_read_capture_datetime_trusts_exif_with_tz_offset(tmp_path):
    """When EXIF has an explicit tz offset, it is always used regardless of filename."""
    fake_jpg = tmp_path / "2025-09-27 06-00-00.jpg"
    fake_jpg.touch()

    # EXIF offset present: +10:00, so 22:00 UTC = 08:00 local — trust it
    exif_result = {
        "datetime_naive": datetime.datetime(2025, 9, 27, 22, 0, 0),
        "offset_minutes": 600,
        "source": "exif",
    }
    with patch(
        "cortex_engine.textifier.DocumentTextifier._read_exif_capture_datetime",
        return_value=exif_result,
    ):
        result = DocumentTextifier._read_photo_capture_datetime(str(fake_jpg))

    assert result["source"] == "exif"
    assert result["offset_minutes"] == 600
```

Add `from unittest.mock import patch` to the imports at the top of the test file.

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
source venv/bin/activate
python -m pytest tests/unit/test_textifier_photo_time_hint.py::test_read_capture_datetime_prefers_filename_when_exif_time_diverges tests/unit/test_textifier_photo_time_hint.py::test_read_capture_datetime_keeps_exif_when_times_agree tests/unit/test_textifier_photo_time_hint.py::test_read_capture_datetime_trusts_exif_with_tz_offset -v
```

Expected: all 3 FAIL (function does not yet do the cross-check).

- [ ] **Step 3: Implement the cross-check in `cortex_engine/textifier.py`**

Replace the current `_read_photo_capture_datetime` (~line 2199):

```python
@staticmethod
def _read_photo_capture_datetime(file_path: str) -> Optional[Dict[str, Any]]:
    """Return capture datetime info preferring EXIF, falling back to filename.

    When EXIF has no explicit timezone offset, the filename time is cross-checked.
    If they differ by more than 3 hours (modulo 24) the filename is preferred —
    cameras often store UTC without an offset tag, while filenames encode local time.
    """
    exif_result = DocumentTextifier._read_exif_capture_datetime(file_path)
    filename_result = DocumentTextifier._parse_filename_capture_datetime(
        Path(file_path).name
    )

    if exif_result is None:
        return filename_result

    if exif_result.get("offset_minutes") is not None:
        # Explicit timezone — trust EXIF unconditionally
        return exif_result

    if filename_result is not None:
        exif_hour = exif_result["datetime_naive"].hour
        fname_hour = filename_result["datetime_naive"].hour
        raw_diff = abs(exif_hour - fname_hour)
        diff = min(raw_diff, 24 - raw_diff)  # shortest arc around the clock
        if diff > 3:
            logger.warning(
                f"EXIF time ({exif_hour:02d}h) and filename time ({fname_hour:02d}h) "
                f"differ by {diff}h for {Path(file_path).name}; "
                "no tz offset in EXIF — using filename time"
            )
            return filename_result

    return exif_result
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
source venv/bin/activate
python -m pytest tests/unit/test_textifier_photo_time_hint.py -v
```

Expected: all 34 PASS.

- [ ] **Step 5: Sync the fix to the Docker distribution**

The function body is identical; copy just this function from `cortex_engine/textifier.py` to `docker/cortex_engine/textifier.py`. The function starts at the `@staticmethod` decorator before `def _read_photo_capture_datetime` and ends at `return exif_result`.

Find the old function in `docker/cortex_engine/textifier.py` (same line range as in the main file, typically within 10 lines) and replace it with the new version above.

- [ ] **Step 6: Commit**

```bash
git add cortex_engine/textifier.py docker/cortex_engine/textifier.py \
        tests/unit/test_textifier_photo_time_hint.py
git commit -m "fix: prefer filename time over EXIF when no tz offset and times diverge >3h

Cameras often store DateTimeOriginal in UTC without an OffsetTimeOriginal
tag. When the filename encodes local time and it differs from the raw EXIF
hour by more than 3h, the filename is the more reliable source for
sun-phase context passed to the VLM.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Manifest v2 helpers

**Files:**
- Modify: `pages/20_Photo_Metadata_Tools.py` — `_save_photokw_manifest`, `_load_photokw_manifest`, add `_is_active_run`
- Modify: `pages/20_Photo_Metadata_Tools.py` — add `timedelta` to datetime import

- [ ] **Step 1: Add `timedelta` to the datetime import**

Find line 10 in `pages/20_Photo_Metadata_Tools.py`:

```python
from datetime import datetime
```

Replace with:

```python
from datetime import datetime, timedelta
```

- [ ] **Step 2: Replace `_save_photokw_manifest` with a dict-accepting version**

Find and replace the current `_save_photokw_manifest` function (lines ~354–373):

```python
def _save_photokw_manifest(payload: dict) -> None:
    """Write a manifest dict to disk atomically. Adds/updates 'timestamp' on every write."""
    try:
        temp_dir = _photokw_temp_dir()
        temp_dir.mkdir(exist_ok=True, mode=0o755)
        to_write = dict(payload)
        to_write["timestamp"] = datetime.now().isoformat(timespec="seconds")
        manifest_path = _photokw_manifest_path()
        tmp_path = manifest_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(to_write, f, default=str)
        os.replace(tmp_path, manifest_path)
    except Exception as e:
        logger.warning(f"Could not save photo batch manifest: {e}")
```

- [ ] **Step 3: Add `_is_active_run` helper after `_save_photokw_manifest`**

```python
def _is_active_run(manifest: dict) -> bool:
    """Return True when the manifest represents an in-progress or paused batch."""
    return manifest.get("status") in ("running", "paused")
```

- [ ] **Step 4: Update `_load_photokw_manifest` to handle v2 active-run manifests**

Find and replace the current `_load_photokw_manifest` function (lines ~376–405):

```python
def _load_photokw_manifest() -> Optional[dict]:
    """Load the last-batch manifest from disk.

    For completed batches (status='done' or v1 manifests): filters out entries
    whose temp files no longer exist and returns None if nothing is left.
    For active runs (status='running'/'paused'): returns the manifest as-is so
    the dispatch block can resume; the caller handles missing files.
    Returns None if no usable manifest is present.
    """
    manifest_path = _photokw_manifest_path()
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read photo batch manifest: {e}")
        return None

    # Active runs (v2) are returned as-is — dispatch block handles skipping
    if _is_active_run(payload):
        return payload

    # Completed / legacy v1 manifests: filter missing file paths
    results = payload.get("results") or []
    file_paths = payload.get("file_paths") or []
    kept_results: list = []
    kept_paths: list = []
    for idx, path in enumerate(file_paths):
        if path and Path(path).exists():
            kept_paths.append(path)
            if idx < len(results):
                kept_results.append(results[idx])

    if not kept_paths:
        return None

    payload["file_paths"] = kept_paths
    payload["results"] = kept_results
    payload["missing_count"] = len(file_paths) - len(kept_paths)
    return payload
```

- [ ] **Step 5: Update the two existing `_save_photokw_manifest` call sites that use the old signature**

Search for all calls to `_save_photokw_manifest(` in `pages/20_Photo_Metadata_Tools.py`. There are two inside the old batch for-loop (lines ~1041 and ~1076). Both will be removed entirely in Task 3 (they're inside the loop that gets deleted), so no change needed here.

Verify no other calls exist:

```bash
grep -n "_save_photokw_manifest" pages/20_Photo_Metadata_Tools.py
```

Expected: only lines inside the `if do_resize_only or do_halftone_repair or do_keywords:` block (~1041, ~1076). Task 3 replaces that entire block.

- [ ] **Step 6: Commit**

```bash
git add pages/20_Photo_Metadata_Tools.py
git commit -m "refactor: upgrade photo batch manifest to v2 schema with status/active-run support

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Run initialisation refactor (replace the batch for-loop)

**Files:**
- Modify: `pages/20_Photo_Metadata_Tools.py` — lines ~858–1076 (the `if do_resize_only or do_halftone_repair or do_keywords:` block)

The current block saves uploads to temp, then runs the entire batch in a for-loop. Replace it with a block that saves uploads and writes the initial manifest, then calls `st.rerun()`.

- [ ] **Step 1: Locate the block to replace**

The block starts at (approximately):
```python
if do_resize_only or do_halftone_repair or do_keywords:
    if do_keywords and not any([generate_description, populate_location, apply_ownership]):
```

And ends after (approximately line 1076):
```python
    _save_photokw_manifest(results, file_paths, mode)
```

- [ ] **Step 2: Replace the entire block**

Replace everything from `if do_resize_only or do_halftone_repair or do_keywords:` through the end of the `if results:` block (~line 1076) with:

```python
if do_resize_only or do_halftone_repair or do_keywords:
    if do_keywords and not any([generate_description, populate_location, apply_ownership]):
        st.warning("Select at least one metadata action before processing.")
        return

    from cortex_engine.textifier import DocumentTextifier  # noqa: F401 (validates import)

    # Save uploads to temp dir
    temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw"
    temp_dir.mkdir(exist_ok=True, mode=0o755)
    all_file_paths: list[str] = []
    if total == 1 and st.session_state.get("photokw_single_working_path"):
        working_path = st.session_state.get("photokw_single_working_path")
        if working_path and Path(working_path).exists():
            dest = temp_dir / uploaded[0].name
            shutil.copy2(working_path, dest)
            os.chmod(str(dest), 0o644)
            all_file_paths.append(str(dest))
    if not all_file_paths:
        if total == 1:
            uf = uploaded[0]
            dest = temp_dir / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getvalue())
            os.chmod(str(dest), 0o644)
            all_file_paths.append(str(dest))
        else:
            for uf in uploaded:
                dest = temp_dir / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getvalue())
                os.chmod(str(dest), 0o644)
                all_file_paths.append(str(dest))

    if not all_file_paths:
        st.error("No files to process.")
        return

    blocked_keywords = [k.strip().lower() for k in blocked_keywords_text.split(",") if k.strip()]
    mode = "resize_only" if do_resize_only else ("halftone_repair" if do_halftone_repair else "keyword_metadata")

    _save_photokw_manifest({
        "version": 2,
        "status": "running",
        "mode": mode,
        "all_file_paths": all_file_paths,
        "current_idx": 0,
        "results": [],
        "file_paths": [],
        "run_settings": {
            "generate_description": generate_description,
            "populate_location": populate_location,
            "clear_keywords": clear_keywords,
            "clear_location": clear_location,
            "anonymize_keywords": anonymize_keywords,
            "blocked_keywords": blocked_keywords,
            "apply_ownership": apply_ownership,
            "ownership_notice": ownership_notice,
            "city_radius": city_radius,
            "fallback_city": fallback_city,
            "fallback_country": fallback_country,
            "max_width": max_width,
            "max_height": max_height,
            "convert_to_jpg": convert_to_jpg,
            "jpg_quality": jpg_quality,
            "halftone_strength": halftone_strength,
            "halftone_preserve_color": halftone_preserve_color,
            "batch_cooldown_seconds": photokw_batch_cooldown_seconds,
        },
        "resume_after": None,
    })
    st.rerun()
```

- [ ] **Step 3: Verify the page loads without errors after this change**

```bash
source venv/bin/activate
python -c "import ast, sys; ast.parse(open('pages/20_Photo_Metadata_Tools.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add pages/20_Photo_Metadata_Tools.py
git commit -m "refactor: replace batch for-loop with manifest-init + st.rerun() run start

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Single-photo dispatch block

**Files:**
- Modify: `pages/20_Photo_Metadata_Tools.py` — add dispatch block at the very top of `_render_photo_keywords_tab`, before any UI widgets

The dispatch block runs on every page rerun. If the manifest says `status="running"`, it processes one photo and calls `st.rerun()`. If `status="paused"`, it does nothing.

- [ ] **Step 1: Extract `_render_live_log` to module level**

The current `_render_live_log` is a closure inside the old for-loop (now removed). It needs to be a standalone function so it can be called from the results section.

Add this as a module-level function near the other helper functions (after `_clear_photokw_manifest`, before `_render_photo_keywords_tab`):

```python
def _render_live_log(placeholder: Any, entries: list, mode: str) -> None:
    """Render the processing log into a Streamlit placeholder."""
    if not entries:
        return
    lines = ["**Processing log**"]
    lines.append(
        '<div style="max-height:320px;overflow-y:auto;'
        'border:1px solid #333;border-radius:6px;padding:10px 14px;'
        'background:#111;font-size:0.82em;font-family:monospace;">'
    )
    for e in reversed(entries):
        icon = "✅" if e.get("ok") else "❌"
        lines.append(
            f'<div style="margin-bottom:10px;padding-bottom:8px;'
            f'border-bottom:1px solid #2a2a2a;">'
        )
        lines.append(
            f'<span style="color:#ccc;">{icon} <strong style="color:#e8e8e8;">'
            f'{e["fname"]}</strong></span>'
        )
        if e.get("error"):
            lines.append(
                f'<br><span style="color:#f87171;">Error: {e["error"]}</span>'
            )
        else:
            if e.get("description"):
                desc_preview = e["description"][:120].replace("<", "&lt;").replace(">", "&gt;")
                if len(e["description"]) > 120:
                    desc_preview += "…"
                lines.append(
                    f'<br><span style="color:#9ca3af;font-style:italic;">'
                    f'"{desc_preview}"</span>'
                )
            if e.get("new_keywords"):
                kw_html = ", ".join(
                    f'<span style="color:#6ee7b7;">{k}</span>'
                    for k in e["new_keywords"][:30]
                )
                if len(e["new_keywords"]) > 30:
                    kw_html += f', <span style="color:#6b7280;">+{len(e["new_keywords"])-30} more</span>'
                lines.append(f'<br><span style="color:#6b7280;">New keywords: </span>{kw_html}')
            elif e.get("mode") == "keyword_metadata":
                lines.append('<br><span style="color:#6b7280;font-style:italic;">No new keywords</span>')
            if e.get("location_str"):
                lines.append(
                    f'<br><span style="color:#93c5fd;">Location: {e["location_str"]}</span>'
                )
            if e.get("mode") == "resize_only":
                resize_info = e.get("resize_info", {})
                if resize_info.get("resized"):
                    lines.append(
                        f'<br><span style="color:#fde68a;">Resized: '
                        f'{resize_info.get("original_size","?")} → {resize_info.get("new_size","?")}</span>'
                    )
                else:
                    lines.append('<br><span style="color:#6b7280;">No resize needed</span>')
        lines.append("</div>")
    lines.append("</div>")
    placeholder.markdown("\n".join(lines), unsafe_allow_html=True)
```

Note: `Any` is already imported in the file. If not, add it: `from typing import Any, List, Optional`.

- [ ] **Step 2: Add the dispatch block at the top of `_render_photo_keywords_tab`**

The function currently starts at line ~417. Insert the dispatch block as the very first thing inside the function, before `st.markdown(...)`:

```python
def _render_photo_keywords_tab():
    """Render the Photo Processor tool for batch resize and photo metadata workflows."""

    # ── Active-run dispatch ──────────────────────────────────────────────────
    # Runs before any widgets so it can call st.rerun() cleanly.
    _dispatch_manifest = _load_photokw_manifest()
    if _dispatch_manifest and _dispatch_manifest.get("status") == "running":
        # Handle cooldown: if resume_after is set and still in the future, sleep briefly
        resume_after_str = _dispatch_manifest.get("resume_after")
        if resume_after_str:
            try:
                resume_after_dt = datetime.fromisoformat(resume_after_str)
                remaining = (resume_after_dt - datetime.now()).total_seconds()
                if remaining > 0:
                    time.sleep(min(remaining, 0.5))
                    st.rerun()
            except Exception:
                pass  # malformed timestamp — ignore and proceed

        all_paths = _dispatch_manifest.get("all_file_paths") or []
        current_idx = int(_dispatch_manifest.get("current_idx") or 0)
        settings = _dispatch_manifest.get("run_settings") or {}
        mode = _dispatch_manifest.get("mode", "keyword_metadata")

        if current_idx < len(all_paths):
            fpath = all_paths[current_idx]
            fname = Path(fpath).name

            if not Path(fpath).exists():
                # File was lost (e.g. /tmp cleared) — skip and continue
                logger.warning(f"Dispatch: file no longer exists, skipping: {fpath}")
                result = {"file_name": fname, "error": "file no longer exists"}
            else:
                from cortex_engine.textifier import DocumentTextifier
                textifier = DocumentTextifier(use_vision=True)

                def _dispatch_progress(frac, msg, _name=fname, _idx=current_idx, _total=len(all_paths)):
                    st.session_state["photokw_dispatch_progress"] = (frac, msg, _name, _idx, _total)

                textifier.on_progress = _dispatch_progress
                try:
                    if mode == "resize_only":
                        max_width = settings.get("max_width")
                        max_height = settings.get("max_height")
                        if max_width is None or max_height is None:
                            result = {
                                "file_name": fname,
                                "output_path": fpath,
                                "resize_info": {"resized": False, "metadata_preserved": True, "skipped_resize": True},
                            }
                        else:
                            result = textifier.resize_image_only(
                                fpath,
                                max_width=max_width,
                                max_height=max_height,
                                convert_to_jpg=settings.get("convert_to_jpg", False),
                                jpg_quality=settings.get("jpg_quality", 90),
                            )
                        output_path = str(result.get("output_path", fpath))
                        all_paths[current_idx] = output_path
                        if settings.get("anonymize_keywords"):
                            result["keyword_anonymize_result"] = textifier.anonymize_existing_photo_keywords(
                                output_path, blocked_keywords=settings.get("blocked_keywords", [])
                            )
                        if settings.get("apply_ownership") and settings.get("ownership_notice", "").strip():
                            result["ownership_result"] = textifier.write_ownership_metadata(
                                output_path, settings["ownership_notice"].strip()
                            )
                    elif mode == "halftone_repair":
                        result = textifier.repair_halftone_image(
                            fpath,
                            strength=settings.get("halftone_strength", 42),
                            preserve_color=settings.get("halftone_preserve_color", True),
                            convert_to_jpg=settings.get("convert_to_jpg", False),
                            jpg_quality=settings.get("jpg_quality", 90),
                        )
                        output_path = str(result.get("output_path", fpath))
                        all_paths[current_idx] = output_path
                        if settings.get("apply_ownership") and settings.get("ownership_notice", "").strip():
                            result["ownership_result"] = textifier.write_ownership_metadata(
                                output_path, settings["ownership_notice"].strip()
                            )
                    else:
                        result = textifier.keyword_image(
                            fpath,
                            city_radius_km=settings.get("city_radius", 5),
                            clear_keywords=(settings.get("clear_keywords", False) if settings.get("generate_description", True) else False),
                            clear_location=(settings.get("clear_location", False) if settings.get("populate_location", True) else False),
                            generate_description=settings.get("generate_description", True),
                            populate_location=settings.get("populate_location", True),
                            anonymize_keywords=settings.get("anonymize_keywords", False),
                            blocked_keywords=settings.get("blocked_keywords", []),
                            fallback_city=settings.get("fallback_city", ""),
                            fallback_country=settings.get("fallback_country", ""),
                            ownership_notice=(settings.get("ownership_notice", "").strip() if settings.get("apply_ownership") else ""),
                        )
                except Exception as exc:
                    logger.error(f"Dispatch error for {fpath}: {exc}", exc_info=True)
                    result = {"file_name": fname, "error": str(exc)}

            # Update cumulative results in session state
            done_results = list(_dispatch_manifest.get("results") or [])
            done_paths = list(_dispatch_manifest.get("file_paths") or [])
            done_results.append(result)
            done_paths.append(all_paths[current_idx])
            new_idx = current_idx + 1

            st.session_state["photokw_results"] = done_results
            st.session_state["photokw_paths"] = done_paths
            st.session_state["photokw_mode"] = mode

            # Build live log entry
            _loc = result.get("location") or {}
            _loc_parts = [v for v in (_loc.get("city"), _loc.get("state"), _loc.get("country")) if v]
            _live_log = list(st.session_state.get("photokw_live_log") or [])
            _live_log.append({
                "fname": fname,
                "ok": not bool(result.get("error")),
                "mode": mode,
                "description": result.get("description", ""),
                "new_keywords": result.get("new_keywords", []),
                "location_str": ", ".join(_loc_parts) if _loc_parts else "",
                "resize_info": result.get("resize_info", {}),
                "error": result.get("error"),
            })
            st.session_state["photokw_live_log"] = _live_log

            # Cooldown
            cooldown = float(settings.get("batch_cooldown_seconds", 0))
            if new_idx < len(all_paths) and cooldown > 0:
                resume_after_ts = (datetime.now() + timedelta(seconds=cooldown)).isoformat()
            else:
                resume_after_ts = None

            new_status = "done" if new_idx >= len(all_paths) else "running"
            _save_photokw_manifest({
                **_dispatch_manifest,
                "status": new_status,
                "current_idx": new_idx,
                "results": done_results,
                "file_paths": done_paths,
                "all_file_paths": all_paths,
                "resume_after": resume_after_ts,
            })

            if new_status == "running":
                st.rerun()
            # If done: fall through — normal render will show results

    st.markdown(
```

The last line (`st.markdown(`) is the existing first line of the render function's body — attach the dispatch block above it.

- [ ] **Step 3: Verify syntax**

```bash
source venv/bin/activate
python -c "import ast; ast.parse(open('pages/20_Photo_Metadata_Tools.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add pages/20_Photo_Metadata_Tools.py
git commit -m "feat: add single-photo dispatch block for pause/resume batch processing

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Pause/Resume UI + recovery banner

**Files:**
- Modify: `pages/20_Photo_Metadata_Tools.py` — recovery banner section (~lines 427–445), `with col2:` results section

- [ ] **Step 1: Update the recovery banner to handle active runs**

Find the current recovery banner block (~lines 427–445):

```python
if not st.session_state.get("photokw_results"):
    _manifest = _load_photokw_manifest()
    if _manifest:
        _ts = _manifest.get("timestamp", "unknown")
        ...
        if rc1.button("Recover", ...):
```

Replace with:

```python
if not st.session_state.get("photokw_results"):
    _manifest = _load_photokw_manifest()
    if _manifest:
        _ts = _manifest.get("timestamp", "unknown")
        if _is_active_run(_manifest):
            # Paused run — show resume/cancel (running runs auto-resume via dispatch)
            _done = int(_manifest.get("current_idx") or 0)
            _total_count = len(_manifest.get("all_file_paths") or [])
            st.info(f"⏸ **Paused batch** — {_done} of {_total_count} photos done. Last run: {_ts}")
            _rc1, _rc2, _ = st.columns([1, 1, 4])
            if _rc1.button("▶ Resume", key="photokw_banner_resume", type="primary"):
                _save_photokw_manifest({**_manifest, "status": "running", "resume_after": None})
                st.rerun()
            if _rc2.button("✕ Cancel", key="photokw_banner_cancel"):
                _clear_photokw_manifest()
                st.rerun()
        else:
            _count = len(_manifest.get("results") or [])
            _missing = int(_manifest.get("missing_count") or 0)
            _msg = f"📦 **Recover last batch** — {_count} processed photo(s) from {_ts} are still on disk."
            if _missing:
                _msg += f" ({_missing} file(s) no longer present and will be skipped.)"
            st.info(_msg)
            rc1, rc2, _ = st.columns([1, 1, 4])
            if rc1.button("Recover", key="photokw_recover_last_batch", type="primary"):
                st.session_state["photokw_results"] = _manifest.get("results") or []
                st.session_state["photokw_paths"] = _manifest.get("file_paths") or []
                st.session_state["photokw_mode"] = _manifest.get("mode", "keyword_metadata")
                st.rerun()
            if rc2.button("Discard", key="photokw_discard_last_batch"):
                _clear_photokw_manifest()
                st.rerun()
```

- [ ] **Step 2: Add active-run progress + Pause/Resume controls in `with col2:`**

Inside `with col2:`, immediately after `st.header("Results")` (and before the `if uploaded:` block), add:

```python
# Active-run progress display
_run_manifest = _load_photokw_manifest()
if _run_manifest and _is_active_run(_run_manifest):
    _run_idx = int(_run_manifest.get("current_idx") or 0)
    _run_total = len(_run_manifest.get("all_file_paths") or [])
    _run_status = _run_manifest.get("status", "running")
    _run_frac = _run_idx / _run_total if _run_total > 0 else 0

    if _run_status == "running":
        _prog_info = st.session_state.get("photokw_dispatch_progress")
        if _prog_info:
            _pfrac, _pmsg, _pname, _pidx, _ptotal = _prog_info
            _overall = min((_pidx + _pfrac) / _ptotal, 1.0)
            st.progress(_overall, f"[{_pname}] {_pmsg}")
        else:
            st.progress(_run_frac, f"Processing photo {_run_idx} of {_run_total}...")
        if st.button("⏸ Pause after this photo", key="photokw_pause_btn"):
            _save_photokw_manifest({**_run_manifest, "status": "paused"})
            st.rerun()
    else:  # paused
        st.progress(_run_frac, f"Paused — {_run_idx} of {_run_total} done")
        st.info(f"⏸ Batch paused at photo {_run_idx} of {_run_total}.")
        _pr1, _pr2, _ = st.columns([1, 1, 4])
        if _pr1.button("▶ Resume", key="photokw_resume_btn", type="primary"):
            _save_photokw_manifest({**_run_manifest, "status": "running", "resume_after": None})
            st.session_state.pop("photokw_dispatch_progress", None)
            st.rerun()
        if _pr2.button("✕ Cancel", key="photokw_cancel_btn"):
            _clear_photokw_manifest()
            st.session_state.pop("photokw_results", None)
            st.session_state.pop("photokw_paths", None)
            st.session_state.pop("photokw_mode", None)
            st.session_state.pop("photokw_live_log", None)
            st.session_state.pop("photokw_dispatch_progress", None)
            st.rerun()

    # Live log from session state
    _live_entries = st.session_state.get("photokw_live_log") or []
    if _live_entries:
        _log_ph = st.empty()
        _render_live_log(_log_ph, _live_entries, _run_manifest.get("mode", "keyword_metadata"))
```

- [ ] **Step 3: Clear `photokw_live_log` in the "Clear All Photos" button handler**

Find the "Clear All Photos" button handler (~line 611) and add `photokw_live_log` and `photokw_dispatch_progress` to the keys that get cleared:

The existing loop already deletes all `photokw_*` keys:
```python
for key in list(st.session_state.keys()):
    if key.startswith("photokw_") and key != "photokw_upload_version":
        del st.session_state[key]
```
This already covers the new keys — no change needed.

- [ ] **Step 4: Verify syntax**

```bash
source venv/bin/activate
python -c "import ast; ast.parse(open('pages/20_Photo_Metadata_Tools.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 5: Commit**

```bash
git add pages/20_Photo_Metadata_Tools.py
git commit -m "feat: add Pause/Resume UI and recovery banner for photo batch jobs

Pause takes effect after the current photo completes. Resume and Cancel
are available both during the run and on fresh page load when a paused
manifest is found on disk. Tab close and refresh resume automatically
for running batches; paused batches require a manual Resume.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage check:**
- ✅ Manifest v2 schema — Task 2
- ✅ Run init → manifest + rerun — Task 3
- ✅ Dispatch block — Task 4
- ✅ Pause button during run — Task 5
- ✅ Resume/Cancel when paused — Task 5
- ✅ Recovery banner for paused runs — Task 5
- ✅ Cooldown via `resume_after` timestamp — Task 4
- ✅ EXIF cross-check fix — Task 1
- ✅ Docker sync — Task 1 Step 5

**Placeholder scan:** No TBDs or vague steps — all code is explicit.

**Type consistency:**
- `_save_photokw_manifest(payload: dict)` — called with dict literal in Task 3, spread in Task 4/5 ✅
- `_is_active_run(manifest)` — defined in Task 2, used in Task 4/5 ✅
- `_render_live_log(placeholder, entries, mode)` — defined in Task 4, called in Task 5 ✅
- `photokw_live_log` session state key — written in Task 4, read in Task 5, cleared by existing Clear handler ✅
- `resume_after` is a str (ISO) in manifest, parsed with `datetime.fromisoformat` in Task 4 ✅
