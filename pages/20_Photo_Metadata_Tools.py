import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import hashlib
import streamlit as st
from PIL import Image, ImageOps

from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger, resolve_db_root_path
from cortex_engine.version_config import VERSION_STRING

from cortex_engine.llm_metadata_sync import exiftool_runner
from cortex_engine.llm_metadata_sync.matcher import build_raw_index, resolve_jpg
from cortex_engine.llm_metadata_sync.models import SyncConfig, SyncReport
from cortex_engine.llm_metadata_sync.sync import run_sync

PAGE_VERSION = VERSION_STRING
MAX_BATCH_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB

st.set_page_config(
    page_title="Photo & Metadata Tools", layout="wide", page_icon="📷"
)

logger = get_logger(__name__)


# ── Helper class ─────────────────────────────────────────────────────────────

class _SessionUpload:
    """Persist uploaded file bytes across Streamlit reruns."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data or b""
        self.size = len(self._data)

    def getvalue(self) -> bytes:
        return self._data


# ── Photo helper functions ────────────────────────────────────────────────────

def _read_photo_metadata_preview(file_path: str) -> dict:
    """Read existing photo metadata fields for preview (keywords/description/location)."""
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        return {"available": False, "reason": "exiftool not found on PATH"}
    try:
        result = subprocess.run(
            [
                exiftool_path,
                "-json",
                "-XMP-dc:Subject",
                "-IPTC:Keywords",
                "-XMP-dc:Description",
                "-IPTC:Caption-Abstract",
                "-EXIF:ImageDescription",
                "-XMP-photoshop:City",
                "-IPTC:City",
                "-XMP-photoshop:State",
                "-IPTC:Province-State",
                "-XMP-photoshop:Country",
                "-IPTC:Country-PrimaryLocationName",
                "-GPSLatitude",
                "-GPSLongitude",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "reason": result.stderr.strip() or "exiftool read failed"}

        payload = json.loads(result.stdout)
        if not payload:
            return {"available": False, "reason": "No metadata found"}
        row = payload[0]

        keywords = []
        for field in ("Subject", "Keywords"):
            val = row.get(field, [])
            if isinstance(val, str):
                val = [val]
            for item in val:
                v = (item or "").strip()
                if v:
                    keywords.append(v)
        keywords = list(dict.fromkeys(keywords))

        description = (
            (row.get("Description") or "").strip()
            or (row.get("Caption-Abstract") or "").strip()
            or (row.get("ImageDescription") or "").strip()
        )
        city = (row.get("City") or "").strip()
        state = (row.get("State") or row.get("Province-State") or "").strip()
        country = (row.get("Country") or row.get("Country-PrimaryLocationName") or "").strip()

        gps_lat = row.get("GPSLatitude")
        gps_lon = row.get("GPSLongitude")
        gps = None
        if gps_lat not in (None, "") and gps_lon not in (None, ""):
            gps = f"{gps_lat}, {gps_lon}"

        return {
            "available": True,
            "keywords": keywords,
            "description": description,
            "city": city,
            "state": state,
            "country": country,
            "gps": gps,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _write_photo_metadata_quick_edit(
    file_path: str,
    keywords: List[str],
    description: str,
    city: str,
    state: str,
    country: str,
) -> dict:
    """Apply quick metadata edits (replace keywords/description/location fields)."""
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        return {"success": False, "message": "exiftool not found on PATH"}
    try:
        cmd = [
            exiftool_path,
            "-overwrite_original",
            # Clear existing keyword/caption/location fields so this acts as an explicit edit.
            "-XMP-dc:Subject=",
            "-IPTC:Keywords=",
            "-XMP-dc:Description=",
            "-IPTC:Caption-Abstract=",
            "-EXIF:ImageDescription=",
            "-IPTC:Country-PrimaryLocationName=",
            "-XMP-photoshop:Country=",
            "-IPTC:Province-State=",
            "-XMP-photoshop:State=",
            "-IPTC:City=",
            "-XMP-photoshop:City=",
        ]
        for kw in keywords:
            cmd.append(f"-XMP-dc:Subject+={kw}")
            cmd.append(f"-IPTC:Keywords+={kw}")
        desc = (description or "").strip()
        if desc:
            cmd.append(f"-XMP-dc:Description={desc}")
            cmd.append(f"-IPTC:Caption-Abstract={desc}")
            cmd.append(f"-EXIF:ImageDescription={desc}")
        if country.strip():
            cmd.append(f"-IPTC:Country-PrimaryLocationName={country.strip()}")
            cmd.append(f"-XMP-photoshop:Country={country.strip()}")
        if state.strip():
            cmd.append(f"-IPTC:Province-State={state.strip()}")
            cmd.append(f"-XMP-photoshop:State={state.strip()}")
        if city.strip():
            cmd.append(f"-IPTC:City={city.strip()}")
            cmd.append(f"-XMP-photoshop:City={city.strip()}")
        cmd.append(file_path)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return {"success": True, "message": result.stdout.strip()}
        return {"success": False, "message": result.stderr.strip() or "metadata write failed"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def _halftone_strength_label(strength: float) -> str:
    value = float(strength or 0.0)
    if value < 34:
        return "Light"
    if value < 67:
        return "Medium"
    return "Strong"


def _zoom_crop_image(image_path: str, zoom: float, focus_x: int, focus_y: int) -> Optional[Image.Image]:
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in {"RGB", "RGBA", "L"}:
                img = img.convert("RGB")
            if float(zoom) <= 1.05:
                return img.copy()

            width, height = img.size
            crop_width = max(1, int(round(width / float(zoom))))
            crop_height = max(1, int(round(height / float(zoom))))

            center_x = int(round((max(0, min(100, int(focus_x))) / 100.0) * width))
            center_y = int(round((max(0, min(100, int(focus_y))) / 100.0) * height))

            left = max(0, min(width - crop_width, center_x - crop_width // 2))
            top = max(0, min(height - crop_height, center_y - crop_height // 2))
            box = (left, top, left + crop_width, top + crop_height)
            return img.crop(box).copy()
    except Exception:
        return None


def _resize_preview_image(image: Image.Image, target_width: int = 1400) -> Image.Image:
    if image.width <= 0 or image.height <= 0:
        return image
    if image.width >= target_width:
        return image
    scale = float(target_width) / float(image.width)
    target_height = max(1, int(round(image.height * scale)))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _build_ab_window_image(
    original_image: Image.Image,
    repaired_image: Image.Image,
    split_pct: int,
    target_width: int = 1400,
) -> Optional[Image.Image]:
    try:
        left = original_image.convert("RGB")
        right = repaired_image.convert("RGB")
        if left.size != right.size:
            right = right.resize(left.size, Image.Resampling.LANCZOS)

        if left.width < target_width:
            left = _resize_preview_image(left, target_width=target_width)
            right = right.resize(left.size, Image.Resampling.LANCZOS)

        split_x = int(round((max(0, min(100, int(split_pct))) / 100.0) * left.width))
        merged = Image.new("RGB", left.size)
        if split_x > 0:
            merged.paste(left.crop((0, 0, split_x, left.height)), (0, 0))
        if split_x < left.width:
            merged.paste(right.crop((split_x, 0, right.width, right.height)), (split_x, 0))

        band_half = 2
        for offset in range(-band_half, band_half + 1):
            x = split_x + offset
            if 0 <= x < merged.width:
                color = (255, 255, 255) if offset == 0 else (0, 0, 0)
                for y in range(merged.height):
                    merged.putpixel((x, y), color)
        return merged
    except Exception:
        return None


def _render_halftone_ab_compare(
    original_path: str,
    repaired_path: str,
    strength: float,
    widget_prefix: str,
    heading: str = "A/B Window",
) -> None:
    if not original_path or not repaired_path:
        return
    if not Path(original_path).exists() or not Path(repaired_path).exists():
        return

    zoom = st.slider(
        f"{heading} zoom",
        min_value=1.0,
        max_value=12.0,
        value=3.0,
        step=0.25,
        key=f"{widget_prefix}_zoom",
    )
    focus_cols = st.columns(3)
    with focus_cols[0]:
        focus_x = st.slider("Focus X (%)", 0, 100, 50, 1, key=f"{widget_prefix}_focus_x")
    with focus_cols[1]:
        focus_y = st.slider("Focus Y (%)", 0, 100, 50, 1, key=f"{widget_prefix}_focus_y")
    with focus_cols[2]:
        split_position = st.slider("A/B split (%)", 0, 100, 50, 1, key=f"{widget_prefix}_split")

    original_zoom = _zoom_crop_image(original_path, zoom, focus_x, focus_y)
    repaired_zoom = _zoom_crop_image(repaired_path, zoom, focus_x, focus_y)

    if original_zoom and repaired_zoom:
        ab_window = _build_ab_window_image(
            original_zoom,
            repaired_zoom,
            split_pct=split_position,
            target_width=1600,
        )
        if ab_window:
            st.markdown(
                f"**{heading}**  \nZoom: {zoom:.2f}x · Strength: {int(round(float(strength)))} · {_halftone_strength_label(strength)}"
            )
            st.image(ab_window, use_column_width=True)
            st.caption("Left of the divider is original. Right is repaired.")

    detail_tabs = st.tabs(["Zoomed Original", "Zoomed Repair", "Full Image A/B"])
    with detail_tabs[0]:
        if original_zoom:
            st.image(_resize_preview_image(original_zoom, target_width=1400), use_column_width=True)
    with detail_tabs[1]:
        if repaired_zoom:
            st.image(_resize_preview_image(repaired_zoom, target_width=1400), use_column_width=True)
    with detail_tabs[2]:
        full_original = _zoom_crop_image(original_path, 1.0, 50, 50)
        full_repaired = _zoom_crop_image(repaired_path, 1.0, 50, 50)
        if full_original and full_repaired:
            full_ab = _build_ab_window_image(
                full_original,
                full_repaired,
                split_pct=split_position,
                target_width=1400,
            )
            if full_ab:
                st.image(full_ab, use_column_width=True)


def _photo_description_issue(description: str) -> Optional[str]:
    """Return a user-facing issue for placeholder image descriptions."""
    desc = (description or "").strip()
    if not desc.startswith("[Image:"):
        return None
    low = desc.lower()
    if "timed out" in low:
        return "Image description timed out. The vision model may be overloaded."
    if "unavailable" in low:
        return "Image description was skipped because the vision model was unavailable."
    if "error" in low:
        return "Image description failed due to a vision model error."
    if "logo/icon omitted" in low:
        return "Image looked like a logo or icon, so description was intentionally skipped."
    return desc


def _photokw_temp_dir() -> Path:
    return Path(tempfile.gettempdir()) / "cortex_photokw"


def _photokw_manifest_path() -> Path:
    return _photokw_temp_dir() / "_last_batch.json"


def _save_photokw_manifest(results: list, file_paths: list, mode: str) -> None:
    """Persist the latest batch summary to disk so Results can be recovered
    if Streamlit session state is wiped (file-watcher rerun, WS drop, sleep)."""
    try:
        temp_dir = _photokw_temp_dir()
        temp_dir.mkdir(exist_ok=True, mode=0o755)
        payload = {
            "version": 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "file_paths": list(file_paths),
            "results": results,
        }
        manifest_path = _photokw_manifest_path()
        tmp_path = manifest_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, default=str)
        os.replace(tmp_path, manifest_path)
    except Exception as e:
        logger.warning(f"Could not save photo batch manifest: {e}")


def _load_photokw_manifest() -> Optional[dict]:
    """Load the last-batch manifest, filtering out entries whose files no longer exist.
    Returns None if no usable manifest is present."""
    manifest_path = _photokw_manifest_path()
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read photo batch manifest: {e}")
        return None

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


def _clear_photokw_manifest() -> None:
    manifest_path = _photokw_manifest_path()
    if manifest_path.exists():
        try:
            manifest_path.unlink()
        except Exception as e:
            logger.warning(f"Could not remove photo batch manifest: {e}")


def _render_photo_keywords_tab():
    """Render the Photo Processor tool for batch resize and photo metadata workflows."""
    st.markdown(
        "Process photos in batch: resize for gallery use, generate AI keywords, "
        "clean sensitive tags, and write EXIF/XMP ownership metadata."
    )

    # Recovery banner — if the browser/session was reset after a batch finished,
    # the processed files and their summary survive on disk in /tmp/cortex_photokw/.
    # Offer to restore them into the Results panel without re-processing.
    if not st.session_state.get("photokw_results"):
        _manifest = _load_photokw_manifest()
        if _manifest:
            _ts = _manifest.get("timestamp", "unknown")
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

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")

        st.session_state["photokw_batch"] = True  # always batch-capable

        # Upload version counter for clear button
        if "photokw_upload_version" not in st.session_state:
            st.session_state["photokw_upload_version"] = 0
        ver = st.session_state["photokw_upload_version"]

        uploaded_input = st.file_uploader(
            "Drop photos here:",
            type=["png", "jpg", "jpeg", "tiff", "webp", "gif", "bmp"],
            accept_multiple_files=True,
            key=f"photokw_upload_v{ver}",
        )
        upload_cache_key = "photokw_uploaded_cache"
        if uploaded_input:
            uploaded = [_SessionUpload(uf.name, uf.getvalue()) for uf in uploaded_input]
            st.session_state[upload_cache_key] = [
                {"name": uf.name, "data": uf.getvalue()}
                for uf in uploaded_input
            ]
        else:
            cached_uploads = st.session_state.get(upload_cache_key) or []
            uploaded = [
                _SessionUpload(str(item.get("name") or ""), item.get("data") or b"")
                for item in cached_uploads
                if isinstance(item, dict) and item.get("name")
            ]

        write_to_original = st.toggle(
            "Write to original files",
            value=False,
            key="photokw_write_original",
            help="When OFF, keywords are written to copies in a temp folder (originals untouched). "
                 "When ON, keywords are written directly to the uploaded files.",
        )

        city_radius = st.slider(
            "City location radius",
            min_value=1, max_value=50, value=5, step=1,
            key="photokw_city_radius",
            help="Radius (km) for city-level reverse geocoding of GPS coordinates. "
                 "Larger values may match broader city names for rural locations.",
        )

        clear_keywords = st.checkbox(
            "Clear existing keywords/tags first",
            value=False,
            key="photokw_clear_keywords",
            help="Remove all existing XMP Subject and IPTC Keywords before writing new ones.",
        )
        clear_location = st.checkbox(
            "Clear existing location fields first",
            value=False,
            key="photokw_clear_location",
            help="Remove existing Country, State, and City EXIF fields and rebuild from GPS/location hints.",
        )
        generate_description = st.checkbox(
            "Generate AI description + keywords",
            value=True,
            key="photokw_generate_description",
            help="Writes a fresh description and any new keywords. Turn this off to update only location/GPS metadata.",
        )
        populate_location = st.checkbox(
            "Fill location and GPS metadata",
            value=True,
            key="photokw_populate_location",
            help="Completes City/State/Country from GPS, or derives GPS from City/Country hints when GPS is missing.",
        )
        fallback_city = st.text_input(
            "Fallback city (optional)",
            value="",
            key="photokw_fallback_city",
            disabled=not populate_location,
            help="Used only when a photo has no GPS and no embedded location fields.",
        )
        fallback_country = st.text_input(
            "Fallback country (optional)",
            value="",
            key="photokw_fallback_country",
            disabled=not populate_location,
            help="If only a country is provided, Cortex will try that country's capital city.",
        )
        resize_profile = st.selectbox(
            "Resize profile",
            options=["Keep original dimensions", "Low (1920 x 1080)", "Medium (2560 x 1440)"],
            index=0,
            key="photokw_resize_profile",
            help="Maximum output dimensions. Photos already below the selected profile are not resized.",
        )
        no_resize_selected = resize_profile == "Keep original dimensions"
        convert_to_jpg = st.checkbox(
            "Convert resized output to JPG",
            value=False,
            key="photokw_convert_to_jpg",
            disabled=no_resize_selected,
            help="Only non-JPG sources are converted. Existing JPG files stay JPG and are only resized when needed.",
        )
        jpg_quality = st.slider(
            "JPG quality",
            min_value=60,
            max_value=100,
            value=90,
            step=1,
            key="photokw_jpg_quality",
            disabled=(not convert_to_jpg) or no_resize_selected,
            help="Only applies when JPG conversion is enabled.",
        )
        halftone_strength = st.slider(
            "Halftone repair strength",
            min_value=0,
            max_value=100,
            value=42,
            step=1,
            key="photokw_halftone_strength",
            help="Lower values are gentler. Higher values remove more screen pattern but can soften detail.",
        )
        st.caption(f"Current repair profile: {_halftone_strength_label(halftone_strength)}")
        halftone_preserve_color = st.checkbox(
            "Preserve colour during halftone repair",
            value=True,
            key="photokw_halftone_preserve_color",
            help="When enabled, Cortex repairs the luminance channel and keeps colour information where possible.",
        )

        anonymize_keywords = st.checkbox(
            "Anonymize sensitive keywords",
            value=False,
            key="photokw_anonymize_keywords",
            help="Remove personal/sensitive tags from generated keywords using the blocked list below.",
        )
        blocked_keywords_text = st.text_input(
            "Blocked keywords (comma-separated)",
            value="friends,family,paul,paul_c,jacqui",
            key="photokw_blocked_keywords",
            help="These keywords are removed when anonymization is enabled.",
        )

        apply_ownership = st.checkbox(
            "Insert ownership info",
            value=True,
            key="photokw_apply_ownership",
            help="Write ownership/copyright metadata fields in EXIF/IPTC/XMP.",
        )
        ownership_notice = st.text_area(
            "Ownership notice",
            value="All rights (c) Longboardfella. Contact longboardfella.com for info on use of photos.",
            key="photokw_ownership_notice",
            height=90,
        )
        photokw_batch_cooldown_seconds = st.slider(
            "Cooldown between photos (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="photokw_batch_cooldown_s",
            help="Adds a short pause between photos to keep batch processing more responsive.",
        )

        if st.button("Clear All Photos", key="photokw_clear", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("photokw_") and key != "photokw_upload_version":
                    del st.session_state[key]
            st.session_state["photokw_upload_version"] = ver + 1
            if "photokw_results" in st.session_state:
                del st.session_state["photokw_results"]
            temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw"
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            st.rerun()

    with col2:
        st.header("Results")

        if uploaded:
            accepted_uploads = []
            accepted_bytes = 0
            for uf in uploaded:
                size_bytes = int(getattr(uf, "size", 0) or 0)
                if size_bytes <= 0:
                    try:
                        size_bytes = len(uf.getvalue())
                    except Exception:
                        size_bytes = 0
                if accepted_bytes + size_bytes > MAX_BATCH_UPLOAD_BYTES:
                    break
                accepted_uploads.append(uf)
                accepted_bytes += size_bytes
            if not accepted_uploads:
                st.error("Selected photos exceed the 1GB total upload limit.")
                return
            if len(accepted_uploads) < len(uploaded):
                st.warning(
                    f"Maximum 1GB total upload per photo batch — only the first "
                    f"{len(accepted_uploads)} of {len(uploaded)} photos will be processed."
                )
            uploaded = accepted_uploads
            st.session_state[upload_cache_key] = [
                {"name": uf.name, "data": uf.getvalue()}
                for uf in uploaded
            ]
            total = len(uploaded)
            st.info(f"{total} photo(s) selected ({accepted_bytes / (1024 * 1024):.1f} MB total)")

            # Single-photo metadata preview for quick testing before processing.
            if total == 1:
                preview_photo = uploaded[0]
                preview_bytes = preview_photo.getvalue()
                preview_temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw_preview"
                preview_temp_dir.mkdir(exist_ok=True, mode=0o755)

                # Keep a stable working copy path for this uploaded file across reruns,
                # so quick metadata edits are not lost.
                preview_sig = f"{preview_photo.name}:{len(preview_bytes)}:{hashlib.md5(preview_bytes).hexdigest()}"
                existing_sig = st.session_state.get("photokw_single_upload_sig")
                existing_path = st.session_state.get("photokw_single_working_path")
                if preview_sig != existing_sig or not existing_path or not Path(existing_path).exists():
                    preview_path = preview_temp_dir / preview_photo.name
                    with open(preview_path, "wb") as f:
                        f.write(preview_bytes)
                    os.chmod(str(preview_path), 0o644)
                    st.session_state["photokw_single_upload_sig"] = preview_sig
                    st.session_state["photokw_single_working_path"] = str(preview_path)
                    st.session_state.pop("photokw_vlm_probe", None)
                else:
                    preview_path = Path(existing_path)

                probe_col, refresh_col = st.columns(2)
                run_probe = probe_col.button("Run VLM Diagnostic Probe", key="photokw_vlm_probe_btn", use_container_width=True)
                if refresh_col.button("Refresh Preview", key="photokw_preview_refresh", use_container_width=True):
                    st.rerun()

                if run_probe:
                    from cortex_engine.textifier import DocumentTextifier

                    probe_result = DocumentTextifier(use_vision=True).probe_image_vlm(preview_bytes, simple_prompt=True)
                    st.session_state["photokw_vlm_probe"] = probe_result

                preview_meta = _read_photo_metadata_preview(str(preview_path))
                probe_result = st.session_state.get("photokw_vlm_probe")

                with st.expander("Single Photo Metadata Preview", expanded=True):
                    st.image(preview_bytes, caption=preview_photo.name, width=420)
                    if probe_result:
                        st.markdown("**VLM Diagnostic Probe**")
                        st.json(probe_result)
                        st.caption(
                            "This shows the raw response shape from Ollama for the current photo using the simple prompt."
                        )
                        st.divider()
                    if preview_meta.get("available"):
                        description = preview_meta.get("description", "")
                        keywords = preview_meta.get("keywords", [])
                        city = preview_meta.get("city", "")
                        state = preview_meta.get("state", "")
                        country = preview_meta.get("country", "")
                        gps = preview_meta.get("gps")

                        if description:
                            st.markdown(f"**Description:** {description}")
                        else:
                            st.caption("No existing description found in metadata.")

                        if keywords:
                            st.markdown(f"**Keywords ({len(keywords)}):** {', '.join(keywords)}")
                        else:
                            st.caption("No existing keywords found in metadata.")

                        location_parts = [v for v in [city, state, country] if v]
                        if location_parts:
                            st.markdown(f"**Location fields:** {', '.join(location_parts)}")
                        else:
                            st.caption("No existing City/State/Country metadata found.")
                        if gps:
                            st.caption(f"GPS: {gps}")

                        st.divider()
                        st.markdown("**Quick Edit Metadata**")
                        edit_keywords = st.text_area(
                            "Edit keywords (comma-separated)",
                            value=", ".join(keywords),
                            key="photokw_edit_keywords",
                            height=90,
                        )
                        edit_description = st.text_area(
                            "Edit description",
                            value=description,
                            key="photokw_edit_description",
                            height=90,
                        )
                        ec1, ec2, ec3 = st.columns(3)
                        with ec1:
                            edit_city = st.text_input("City", value=city, key="photokw_edit_city")
                        with ec2:
                            edit_state = st.text_input("State", value=state, key="photokw_edit_state")
                        with ec3:
                            edit_country = st.text_input("Country", value=country, key="photokw_edit_country")

                        if st.button("Apply Quick Metadata Edits", key="photokw_apply_quick_edit", use_container_width=True):
                            edited_keywords = [k.strip() for k in edit_keywords.split(",") if k.strip()]
                            write_result = _write_photo_metadata_quick_edit(
                                str(preview_path),
                                keywords=edited_keywords,
                                description=edit_description,
                                city=edit_city,
                                state=edit_state,
                                country=edit_country,
                            )
                            if write_result.get("success"):
                                st.success("Metadata edits applied.")
                                st.rerun()
                            else:
                                st.error(f"Could not apply metadata edits: {write_result.get('message', 'Unknown error')}")
                    else:
                        st.info(f"Metadata preview unavailable: {preview_meta.get('reason', 'Unknown reason')}")

                halftone_preview_state = st.session_state.get("photokw_halftone_preview") or {}
                halftone_preview_matches = (
                    halftone_preview_state.get("source_sig") == preview_sig
                    and float(halftone_preview_state.get("strength", -1)) == float(halftone_strength)
                    and bool(halftone_preview_state.get("preserve_color")) == bool(halftone_preserve_color)
                    and Path(str(halftone_preview_state.get("output_path") or "")).exists()
                )

                with st.expander("Halftone Repair Preview", expanded=True):
                    st.caption(
                        "Generate a repaired preview for the current strength, then inspect the full image and a zoomed crop before batch processing."
                    )
                    preview_button_cols = st.columns(2)
                    generate_halftone_preview = preview_button_cols[0].button(
                        "Generate Halftone Preview",
                        key="photokw_generate_halftone_preview",
                        use_container_width=True,
                    )
                    clear_halftone_preview = preview_button_cols[1].button(
                        "Clear Preview",
                        key="photokw_clear_halftone_preview",
                        use_container_width=True,
                    )

                    if clear_halftone_preview:
                        existing_preview = Path(str(halftone_preview_state.get("output_path") or ""))
                        if existing_preview.exists():
                            try:
                                existing_preview.unlink()
                            except Exception:
                                pass
                        st.session_state.pop("photokw_halftone_preview", None)
                        halftone_preview_state = {}
                        halftone_preview_matches = False

                    if generate_halftone_preview:
                        from cortex_engine.textifier import DocumentTextifier

                        preview_output_path = preview_temp_dir / (
                            f"{preview_path.stem}_halftone_preview_{int(halftone_strength)}"
                            f"{'_color' if halftone_preserve_color else '_mono'}{preview_path.suffix}"
                        )
                        shutil.copy2(preview_path, preview_output_path)
                        os.chmod(str(preview_output_path), 0o644)

                        with st.spinner("Generating halftone preview..."):
                            preview_result = DocumentTextifier(use_vision=False).repair_halftone_image(
                                str(preview_output_path),
                                strength=halftone_strength,
                                preserve_color=halftone_preserve_color,
                                convert_to_jpg=False,
                            )
                        preview_info = preview_result.get("halftone_repair_info", {})
                        if preview_info.get("repaired"):
                            st.session_state["photokw_halftone_preview"] = {
                                "source_sig": preview_sig,
                                "strength": float(halftone_strength),
                                "preserve_color": bool(halftone_preserve_color),
                                "output_path": str(preview_result.get("output_path") or preview_output_path),
                            }
                            halftone_preview_state = st.session_state["photokw_halftone_preview"]
                            halftone_preview_matches = True
                        else:
                            st.error(f"Preview generation failed: {preview_info.get('error', 'Unknown error')}")

                    if halftone_preview_state and not halftone_preview_matches:
                        st.info("Preview settings changed. Generate a new preview to match the current strength and colour options.")

                    if halftone_preview_matches:
                        preview_output_path = str(halftone_preview_state["output_path"])
                        _render_halftone_ab_compare(
                            str(preview_path),
                            preview_output_path,
                            strength=float(halftone_strength),
                            widget_prefix="photokw_halftone_preview_compare",
                            heading="Preview A/B Window",
                        )

            resolution_map = {
                "Keep original dimensions": (None, None),
                "Low (1920 x 1080)": (1920, 1080),
                "Medium (2560 x 1440)": (2560, 1440),
            }
            max_width, max_height = resolution_map.get(resize_profile, (None, None))

            action_cols = st.columns(3)
            do_resize_only = action_cols[0].button("Resize Photos Only", use_container_width=True)
            do_halftone_repair = action_cols[1].button("Repair Halftone Artefacts", use_container_width=True)
            do_keywords = action_cols[2].button("Process Selected Metadata", type="primary", use_container_width=True)

            if do_resize_only or do_halftone_repair or do_keywords:
                if do_keywords and not any([generate_description, populate_location, apply_ownership]):
                    st.warning("Select at least one metadata action before processing.")
                    return

                from cortex_engine.textifier import DocumentTextifier

                # New batch starting — invalidate any stale recovery manifest
                # so the one we write reflects only the current run.
                _clear_photokw_manifest()

                # Save uploads to temp dir
                temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw"
                temp_dir.mkdir(exist_ok=True, mode=0o755)
                file_paths = []
                if total == 1 and st.session_state.get("photokw_single_working_path"):
                    working_path = st.session_state.get("photokw_single_working_path")
                    if working_path and Path(working_path).exists():
                        dest = temp_dir / uploaded[0].name
                        shutil.copy2(working_path, dest)
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))
                    else:
                        uf = uploaded[0]
                        dest = temp_dir / uf.name
                        with open(dest, "wb") as f:
                            f.write(uf.getvalue())
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))
                else:
                    for uf in uploaded:
                        dest = temp_dir / uf.name
                        with open(dest, "wb") as f:
                            f.write(uf.getvalue())
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))

                textifier = DocumentTextifier(use_vision=True)
                results = []
                if do_resize_only:
                    mode = "resize_only"
                    progress_message = "Starting resize..."
                elif do_halftone_repair:
                    mode = "halftone_repair"
                    progress_message = "Starting halftone repair..."
                else:
                    mode = "keyword_metadata"
                    progress_message = "Starting metadata processing..."
                progress = st.progress(0.0, progress_message)
                blocked_keywords = [k.strip().lower() for k in blocked_keywords_text.split(",") if k.strip()]

                # Live processing log — updated after each photo completes
                log_placeholder = st.empty()
                _live_log: list[dict] = []

                def _render_live_log(entries: list[dict]) -> None:
                    """Render the processing log inside the placeholder."""
                    if not entries:
                        return
                    lines = ["**Processing log**"]
                    lines.append(
                        '<div style="max-height:320px;overflow-y:auto;'
                        'border:1px solid #333;border-radius:6px;padding:10px 14px;'
                        'background:#111;font-size:0.82em;font-family:monospace;">'
                    )
                    for e in reversed(entries):
                        icon = "✅" if e["ok"] else "❌"
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
                    log_placeholder.markdown("\n".join(lines), unsafe_allow_html=True)

                for idx, fpath in enumerate(file_paths):
                    fname = Path(fpath).name

                    def _on_progress(frac, msg, _idx=idx, _total=total, _name=fname):
                        overall = min((_idx + frac) / _total, 1.0)
                        progress.progress(overall, f"[{_name}] {msg}")

                    textifier.on_progress = _on_progress
                    try:
                        if do_resize_only:
                            if max_width is None or max_height is None:
                                result = {
                                    "file_name": Path(fpath).name,
                                    "output_path": fpath,
                                    "resize_info": {
                                        "resized": False,
                                        "metadata_preserved": True,
                                        "skipped_resize": True,
                                    },
                                }
                            else:
                                result = textifier.resize_image_only(
                                    fpath,
                                    max_width=max_width,
                                    max_height=max_height,
                                    convert_to_jpg=convert_to_jpg,
                                    jpg_quality=jpg_quality,
                                )
                            output_path = str(result.get("output_path", fpath))
                            file_paths[idx] = output_path
                            if anonymize_keywords:
                                result["keyword_anonymize_result"] = textifier.anonymize_existing_photo_keywords(
                                    output_path, blocked_keywords=blocked_keywords
                                )
                            if apply_ownership and ownership_notice.strip():
                                result["ownership_result"] = textifier.write_ownership_metadata(
                                    output_path, ownership_notice.strip()
                                )
                        elif do_halftone_repair:
                            result = textifier.repair_halftone_image(
                                fpath,
                                strength=halftone_strength,
                                preserve_color=halftone_preserve_color,
                                convert_to_jpg=convert_to_jpg,
                                jpg_quality=jpg_quality,
                            )
                            output_path = str(result.get("output_path", fpath))
                            file_paths[idx] = output_path
                            if apply_ownership and ownership_notice.strip():
                                result["ownership_result"] = textifier.write_ownership_metadata(
                                    output_path, ownership_notice.strip()
                                )
                        else:
                            result = textifier.keyword_image(
                                fpath, city_radius_km=city_radius,
                                clear_keywords=(clear_keywords if generate_description else False),
                                clear_location=(clear_location if populate_location else False),
                                generate_description=generate_description,
                                populate_location=populate_location,
                                anonymize_keywords=anonymize_keywords,
                                blocked_keywords=blocked_keywords,
                                fallback_city=fallback_city,
                                fallback_country=fallback_country,
                                ownership_notice=(ownership_notice.strip() if apply_ownership else ""),
                            )
                        results.append(result)
                        # Persist incremental progress so a mid-batch session wipe
                        # (rerun / WS drop / PC sleep) can still recover what's done.
                        _save_photokw_manifest(results, file_paths[: idx + 1], mode)
                        # Build log entry from result
                        _loc = result.get("location") or {}
                        _loc_parts = [v for v in (_loc.get("city"), _loc.get("state"), _loc.get("country")) if v]
                        _live_log.append({
                            "fname": fname,
                            "ok": True,
                            "mode": mode,
                            "description": result.get("description", ""),
                            "new_keywords": result.get("new_keywords", []),
                            "location_str": ", ".join(_loc_parts) if _loc_parts else "",
                            "resize_info": result.get("resize_info", {}),
                        })
                        _render_live_log(_live_log)
                    except Exception as e:
                        st.error(f"Failed: {fname}: {e}")
                        logger.error(f"Photo keyword error for {fpath}: {e}", exc_info=True)
                        _live_log.append({"fname": fname, "ok": False, "mode": mode, "error": str(e)})
                        _render_live_log(_live_log)
                    if photokw_batch_cooldown_seconds > 0 and total > 1 and idx < total - 1:
                        progress.progress(
                            min((idx + 1) / total, 1.0),
                            f"Cooling down for {photokw_batch_cooldown_seconds:.1f}s before next photo..."
                        )
                        time.sleep(photokw_batch_cooldown_seconds)

                progress.progress(1.0, "Done!")

                # If writing to originals, user needs to copy back — but since
                # we're working on uploaded copies in temp, the writes already happened.
                # The user downloads the processed files.
                if results:
                    st.session_state["photokw_results"] = results
                    st.session_state["photokw_paths"] = file_paths
                    st.session_state["photokw_mode"] = mode
                    _save_photokw_manifest(results, file_paths, mode)

        # Display results
        results = st.session_state.get("photokw_results")
        file_paths = st.session_state.get("photokw_paths", [])
        photokw_mode = st.session_state.get("photokw_mode", "keyword_metadata")

        if results:
            st.divider()

            if photokw_mode == "resize_only":
                resized_count = sum(1 for r in results if r.get("resize_info", {}).get("resized"))
                total_removed = sum(len(r.get("keyword_anonymize_result", {}).get("removed_keywords", [])) for r in results)
                ownership_ok = sum(
                    1 for r in results
                    if not r.get("ownership_result") or r.get("ownership_result", {}).get("success")
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Photos Processed", len(results))
                with mc2:
                    st.metric("Resized", f"{resized_count}/{len(results)}")
                with mc3:
                    st.metric("Sensitive Tags Removed", total_removed)
                with mc4:
                    st.metric("Ownership Written", f"{ownership_ok}/{len(results)}")
            elif photokw_mode == "halftone_repair":
                repaired_count = sum(1 for r in results if r.get("halftone_repair_info", {}).get("repaired"))
                converted_count = sum(1 for r in results if r.get("halftone_repair_info", {}).get("converted_to_jpg"))
                ownership_ok = sum(
                    1 for r in results
                    if not r.get("ownership_result") or r.get("ownership_result", {}).get("success")
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Photos Processed", len(results))
                with mc2:
                    st.metric("Repaired", f"{repaired_count}/{len(results)}")
                with mc3:
                    st.metric("Converted To JPG", converted_count)
                with mc4:
                    st.metric("Ownership Written", f"{ownership_ok}/{len(results)}")
            else:
                # Summary metrics
                total_new = sum(len(r.get("new_keywords", [])) for r in results)
                total_existing = sum(len(r.get("existing_keywords", [])) for r in results)
                total_removed = sum(len(r.get("removed_sensitive_keywords", [])) for r in results)
                successful = sum(1 for r in results if r["exif_result"]["success"])
                location_written = sum(
                    1 for r in results if r.get("location_result", {}).get("location_written")
                )
                gps_written = sum(
                    1 for r in results if r.get("location_result", {}).get("gps_written")
                )
                ownership_ok = sum(
                    1 for r in results
                    if not r.get("ownership_result") or r.get("ownership_result", {}).get("success")
                )
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                with mc1:
                    st.metric("Photos Processed", len(results))
                with mc2:
                    st.metric("Existing Tags", total_existing)
                with mc3:
                    st.metric("New Tags Added", total_new)
                with mc4:
                    st.metric("Sensitive Tags Removed", total_removed)
                with mc5:
                    st.metric("Metadata Written", f"{successful}/{len(results)}")
                st.caption(
                    f"Location fields written: {location_written}/{len(results)} | "
                    f"GPS written: {gps_written}/{len(results)} | "
                    f"Ownership metadata written: {ownership_ok}/{len(results)}"
                )

            # Download — single file direct, multiple as zip
            if file_paths:
                if len(file_paths) == 1:
                    fpath = file_paths[0]
                    fname = Path(fpath).name
                    mime = "image/jpeg" if fname.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    with open(fpath, "rb") as dl_f:
                        st.download_button(
                            f"Download {fname}",
                            dl_f.read(),
                            file_name=fname,
                            mime=mime,
                            use_container_width=True,
                        )
                else:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fpath in file_paths:
                            zf.write(fpath, Path(fpath).name)
                    buf.seek(0)
                    st.download_button(
                        f"Download All {len(file_paths)} Photos",
                        buf.getvalue(),
                        file_name="processed_photos.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

            # Per-image details
            if len(results) == 1:
                # Single photo — show inline preview (like Textifier)
                r = results[0]
                resize_info = r.get("resize_info", {})
                repair_info = r.get("halftone_repair_info", {})
                ownership_result = r.get("ownership_result")
                if photokw_mode == "resize_only":
                    if resize_info.get("skipped_resize"):
                        st.info(f"Left dimensions unchanged for {r['file_name']}")
                    elif resize_info.get("resized"):
                        st.success(
                            f"Resized {r['file_name']}: "
                            f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                            f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                        )
                    elif resize_info.get("converted_to_jpg"):
                        st.success(f"Converted to JPG: {r['file_name']}")
                    else:
                        st.info(f"No resize needed for {r['file_name']}")
                    if ownership_result:
                        if ownership_result.get("success"):
                            st.success("Ownership metadata written")
                        else:
                            st.warning(f"Ownership metadata write failed: {ownership_result.get('message', '')}")
                elif photokw_mode == "halftone_repair":
                    if repair_info.get("repaired"):
                        st.success(
                            f"Applied halftone repair strength {int(round(float(repair_info.get('strength', 0))))} "
                            f"to {r['file_name']}"
                        )
                    else:
                        st.error(f"Halftone repair failed: {repair_info.get('error', 'Unknown error')}")
                    if ownership_result:
                        if ownership_result.get("success"):
                            st.success("Ownership metadata written")
                        else:
                            st.warning(f"Ownership metadata write failed: {ownership_result.get('message', '')}")
                else:
                    exif = r["exif_result"]
                    desc_issue = _photo_description_issue(r.get("description", ""))
                    if exif.get("message") == "Keyword generation skipped":
                        st.info("Description/keyword generation skipped")
                    elif exif["success"]:
                        st.success(f"EXIF written: {exif['keywords_written']} keywords to {r['file_name']}")
                    else:
                        st.error(f"EXIF write failed: {exif['message']}")
                    if desc_issue:
                        st.warning(desc_issue)
                    if ownership_result:
                        if ownership_result.get("success"):
                            st.success("Ownership metadata written")
                        else:
                            st.warning(f"Ownership metadata write failed: {ownership_result.get('message', '')}")

                    # GPS / location feedback
                    location_result = r.get("location_result", {})
                    if not location_result.get("enabled"):
                        st.caption("Location/GPS processing was skipped.")
                    elif r.get("location") and any((r.get("location") or {}).values()):
                        loc = r["location"]
                        parts = [v for v in [loc.get("city"), loc.get("state"), loc.get("country")] if v]
                        if parts:
                            st.info(f"Location: **{', '.join(parts)}**")
                        if r.get("gps"):
                            st.caption(f"GPS: {r['gps'][0]:.5f}, {r['gps'][1]:.5f}")
                        if location_result.get("gps_written"):
                            st.success("GPS coordinates were derived and written.")
                    else:
                        st.warning(
                            f"No GPS or usable location hint was found for **{r['file_name']}**. "
                            "Add fallback City/Country to auto-fill empty photos."
                        )

                with st.expander("Preview", expanded=True):
                    # Show thumbnail of the photo
                    if file_paths and Path(file_paths[0]).exists():
                        st.image(file_paths[0], caption=r["file_name"], width=400)
                    if resize_info.get("resized"):
                        st.caption(
                            "Resized: "
                            f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                            f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                        )
                    if photokw_mode == "resize_only":
                        st.markdown(
                            f"**Metadata preserved after resize:** "
                            f"{'Yes' if resize_info.get('metadata_preserved') else 'Partial/Unknown'}"
                        )
                        if resize_info.get("skipped_resize"):
                            st.caption("Dimensions were left unchanged by request.")
                    elif photokw_mode == "halftone_repair":
                        strength_value = float(repair_info.get("strength", 0))
                        st.markdown(
                            f"**Repair strength:** {int(round(strength_value))} ({_halftone_strength_label(strength_value)})  \n"
                            f"**Preserve colour:** {'Yes' if repair_info.get('preserve_color') else 'No'}  \n"
                            f"**Metadata preserved after repair:** {'Yes' if repair_info.get('metadata_preserved') else 'Partial/Unknown'}"
                        )
                        if repair_info.get("converted_to_jpg"):
                            st.caption("Converted repaired output to JPG.")
                        original_compare_path = st.session_state.get("photokw_single_working_path")
                        if original_compare_path and Path(str(original_compare_path)).exists() and file_paths:
                            st.divider()
                            _render_halftone_ab_compare(
                                str(original_compare_path),
                                str(file_paths[0]),
                                strength=strength_value,
                                widget_prefix="photokw_halftone_result_compare",
                                heading="Result A/B Window",
                            )
                    else:
                        desc = r["description"] or "(no description generated)"
                        st.markdown(f"**Description:**\n\n{desc}")
                        desc_issue = _photo_description_issue(desc)
                        if desc_issue:
                            st.warning(desc_issue)
                        st.divider()
                        # Location fields
                        if r.get("location") and any(r["location"].values()):
                            loc = r["location"]
                            st.markdown(
                                f"**Location:** {loc.get('city', '')} · "
                                f"{loc.get('state', '')} · {loc.get('country', '')}"
                            )
                            if r.get("gps"):
                                st.caption(f"GPS: {r['gps'][0]:.5f}, {r['gps'][1]:.5f}")
                            st.divider()
                        existing = r.get("existing_keywords", [])
                        new_kw = r.get("new_keywords", [])
                        removed_kw = r.get("removed_sensitive_keywords", [])
                        if existing:
                            st.markdown(f"**Existing tags ({len(existing)}):** {', '.join(existing)}")
                        if new_kw:
                            st.markdown(f"**New tags added ({len(new_kw)}):** {', '.join(new_kw)}")
                        elif not existing:
                            st.warning("No keywords generated — the vision model may have failed to describe this image.")
                        if removed_kw:
                            st.caption(f"Removed sensitive tags: {', '.join(removed_kw)}")
                        st.markdown(f"**Combined keywords ({len(r['keywords'])}):**")
                        if r["keywords"]:
                            st.markdown(", ".join(r["keywords"]))
            else:
                # Batch mode
                if photokw_mode != "resize_only":
                    no_gps = [
                        r for r in results
                        if "nogps" in [kw.lower() for kw in r.get("keywords", [])]
                    ]
                    if no_gps:
                        st.warning(
                            f"**{len(no_gps)} photo(s) have no GPS data** — tagged with "
                            f"'nogps' for easy filtering: "
                            f"{', '.join(r['file_name'] for r in no_gps)}"
                        )
                for r in results:
                    resize_info = r.get("resize_info", {})
                    loc = r.get("location")
                    loc_label = ""
                    if loc and any(loc.values()):
                        parts = [v for v in [loc.get("city"), loc.get("state"), loc.get("country")] if v]
                        loc_label = f"  —  {', '.join(parts)}"
                    with st.expander(f"{r['file_name']}{loc_label}", expanded=False):
                        # Show thumbnail in batch mode too
                        idx = next((i for i, fp in enumerate(file_paths) if Path(fp).name == r["file_name"]), None)
                        if idx is not None and Path(file_paths[idx]).exists():
                            st.image(file_paths[idx], caption=r["file_name"], width=300)
                        if resize_info.get("resized"):
                            st.caption(
                                "Resized: "
                                f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                                f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                            )
                        if photokw_mode == "resize_only":
                            st.caption(
                                "Metadata preserved after resize: "
                                f"{'Yes' if resize_info.get('metadata_preserved') else 'Partial/Unknown'}"
                            )
                            if resize_info.get("converted_to_jpg"):
                                st.caption("Converted to JPG")
                            anon_result = r.get("keyword_anonymize_result")
                            if anon_result:
                                if anon_result.get("success"):
                                    removed = anon_result.get("removed_keywords", [])
                                    if removed:
                                        st.caption(f"Removed sensitive tags: {', '.join(removed)}")
                                    else:
                                        st.caption("No sensitive tags removed.")
                                else:
                                    st.warning(
                                        f"Keyword anonymization failed: {anon_result.get('message', 'Unknown error')}"
                                    )
                        elif photokw_mode == "halftone_repair":
                            repair_info = r.get("halftone_repair_info", {})
                            if repair_info.get("repaired"):
                                strength_value = float(repair_info.get("strength", 0))
                                st.caption(
                                    f"Repair strength: {int(round(strength_value))} ({_halftone_strength_label(strength_value)}) | "
                                    f"Preserve colour: {'Yes' if repair_info.get('preserve_color') else 'No'} | "
                                    f"Metadata preserved: {'Yes' if repair_info.get('metadata_preserved') else 'Partial/Unknown'}"
                                )
                                if repair_info.get("converted_to_jpg"):
                                    st.caption("Converted repaired output to JPG")
                            else:
                                st.error(f"Repair failed: {repair_info.get('error', 'Unknown error')}")
                        else:
                            desc = r.get('description') or '(no description)'
                            st.markdown(f"**Description:** {desc}")
                            desc_issue = _photo_description_issue(desc)
                            if desc_issue:
                                st.warning(desc_issue)
                            if loc and any(loc.values()):
                                st.markdown(
                                    f"**Location:** {loc.get('city', '')} · "
                                    f"{loc.get('state', '')} · {loc.get('country', '')}"
                                )
                            elif not r.get("has_gps"):
                                st.caption("No GPS data — tagged 'nogps'")
                            existing = r.get("existing_keywords", [])
                            new_kw = r.get("new_keywords", [])
                            removed_kw = r.get("removed_sensitive_keywords", [])
                            if existing:
                                st.caption(f"Existing: {', '.join(existing)}")
                            if new_kw:
                                st.caption(f"Added: {', '.join(new_kw)}")
                            if removed_kw:
                                st.caption(f"Removed: {', '.join(removed_kw)}")
                            st.markdown(f"**Keywords ({len(r['keywords'])}):** {', '.join(r['keywords'])}")
                            exif = r["exif_result"]
                            if exif.get("message") == "Keyword generation skipped":
                                st.info("Description/keyword generation skipped")
                            elif exif["success"]:
                                st.success(f"EXIF written: {exif['keywords_written']} new keywords")
                            else:
                                st.error(f"EXIF write failed: {exif['message']}")

        elif uploaded:
            st.info("Choose an action: **Resize Photos Only**, **Repair Halftone Artefacts**, or **Process Selected Metadata**")
        else:
            st.info("Upload photos from the left panel to get started")


# ── LLM Metadata Sync tab ────────────────────────────────────────────────────

def _render_lms_tab() -> None:
    st.info("LLM Metadata Sync — coming soon.")


# ── Page footer ──────────────────────────────────────────────────────────────

def _render_page_footer() -> None:
    try:
        from cortex_engine.ui_components import render_version_footer
        render_version_footer()
    except Exception:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Photo & Metadata Tools")
    st.caption(
        f"Version: {PAGE_VERSION} • Photo processing, batch keywords, and LLM metadata sync"
    )

    tab_photo, tab_lms = st.tabs(["Photo Processor", "LLM Metadata Sync"])

    with tab_photo:
        _render_photo_keywords_tab()

    with tab_lms:
        _render_lms_tab()

    _render_page_footer()


if __name__ == "__main__":
    main()
