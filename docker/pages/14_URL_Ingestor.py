# ## File: pages/14_URL_Ingestor.py
# Version: v6.0.8
# Date: 2026-02-13
# Purpose: Download open-access PDFs from URL lists and optionally convert to Markdown in one pass.

import io
import os
import sys
import time
from pathlib import Path
from typing import List

import streamlit as st

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.config_manager import ConfigManager
from cortex_engine.url_ingestor import URLIngestor, URLIngestResult, normalize_url_list
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger, resolve_db_root_path
from cortex_engine.version_config import VERSION_STRING

logger = get_logger(__name__)

st.set_page_config(page_title="URL PDF Ingestor", layout="wide", page_icon="🌐")


def _resolve_db_root() -> Path:
    config = ConfigManager().get_config()
    raw_db = (config.get("ai_database_path") or "").strip()
    if not raw_db:
        raise ValueError("No ai_database_path configured. Set database path first in Knowledge Ingest or Maintenance.")
    resolved = resolve_db_root_path(raw_db)
    if resolved:
        return Path(str(resolved))
    normalized = raw_db if os.path.exists("/.dockerenv") else convert_windows_to_wsl_path(raw_db)
    return Path(normalized)


def _results_table_rows(results: List[URLIngestResult]) -> List[dict]:
    rows = []
    for r in results:
        rows.append(
            {
                "status": r.status,
                "input_url": r.input_url,
                "final_url": r.final_url or r.input_url,
                "pdf_found": r.open_access_pdf_found,
                "web_captured": r.web_captured,
                "converted_to_md": r.converted_to_md,
                "http_code": r.http_code,
                "reason": r.reason,
                "pdf_file": Path(r.pdf_path).name if r.pdf_path else "",
                "md_file": Path(r.md_path).name if r.md_path else "",
                "elapsed_s": r.elapsed_seconds,
            }
        )
    return rows


st.title("URL PDF Ingestor")
st.caption(f"Version: {VERSION_STRING} • Download open-access PDFs and optionally convert to Markdown in one pass.")

st.markdown(
    "Paste URLs or upload a source list (.md/.txt/.csv/.json). The ingestor extracts valid links, then attempts to find open-access PDFs."
)

with st.container(border=True):
    url_text = st.text_area(
        "Paste URL List (optional)",
        height=220,
        placeholder="https://example.org/article\nhttps://doi.org/10.xxxx/yyyy\nhttps://example.org/report.pdf",
        key="url_ingestor_input",
    )
    path_text = ""
    paste_path_lines = [ln.strip().strip("\"'") for ln in (url_text or "").splitlines() if ln.strip()]
    loaded_paths = []
    for line in paste_path_lines:
        try:
            p = Path(line).expanduser()
            if p.exists() and p.is_file():
                try:
                    file_text = p.read_text(encoding="utf-8", errors="ignore")
                    path_text += ("\n" + file_text)
                    loaded_paths.append(p.name)
                except Exception as e:
                    st.warning(f"Could not read path from pasted text `{line}`: {e}")
        except (OSError, ValueError):
            # Ignore non-path content pasted into the URL box.
            continue

    paste_urls = normalize_url_list("\n".join([url_text, path_text]))
    if url_text.strip():
        extra_note = f" (including {len(loaded_paths)} loaded file path(s))" if loaded_paths else ""
        st.caption(f"Pasted text extracted {len(paste_urls)} URL(s){extra_note}")

    uploaded_source = st.file_uploader(
        "Upload URL source file (optional)",
        type=["md", "txt", "csv", "json"],
        key="url_ingestor_file",
        help="Useful for Perplexity/Deep Research markdown exports with mixed text and links.",
    )
    uploaded_text = ""
    uploaded_urls: List[str] = []
    if uploaded_source is not None:
        uploaded_text = uploaded_source.getvalue().decode("utf-8", errors="ignore")
        uploaded_urls = normalize_url_list(uploaded_text)
        st.caption(f"Loaded `{uploaded_source.name}` • extracted {len(uploaded_urls)} URL(s)")

    combined_input = "\n".join([url_text.strip(), path_text.strip(), uploaded_text.strip()]).strip()
    preview_urls = normalize_url_list(combined_input)
    with st.expander("URL Extraction Preview", expanded=False):
        st.caption(
            f"Combined extracted URLs: {len(preview_urls)} "
            f"(paste={len(paste_urls)}, upload={len(uploaded_urls)})"
        )
        if preview_urls:
            st.text_area(
                "Normalized URLs to be processed",
                value="\n".join(preview_urls),
                height=200,
                disabled=True,
            )
        else:
            st.info("No valid URLs detected yet.")

    c1, c2, c3 = st.columns(3)
    convert_to_md = c1.checkbox("Convert downloaded PDFs to Markdown", value=True)
    use_vision_for_md = c2.checkbox("Use vision during PDF->MD conversion", value=False)
    capture_web_md_on_no_pdf = c3.checkbox(
        "Capture web page as Markdown when PDF unavailable",
        value=True,
    )
    timeout_seconds = st.number_input("Request timeout (seconds)", min_value=5, max_value=120, value=25, step=5)

    try:
        db_root = _resolve_db_root()
        st.caption(f"Output root: {db_root / 'url_ingest'}")
    except Exception as e:
        db_root = None
        st.error(str(e))

    run_btn = st.button("Process URLs", type="primary", use_container_width=True, disabled=(db_root is None))

if run_btn and db_root is not None:
    urls = preview_urls
    if not urls:
        st.warning("No valid http/https URLs found in pasted text or uploaded file.")
    else:
        run_dir = db_root / "url_ingest" / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        ingestor = URLIngestor(run_dir, timeout=int(timeout_seconds))

        progress = st.progress(0.0, text="Starting URL processing...")
        log_lines: List[str] = []
        log_box = st.empty()

        def _progress(done: int, total: int, message: str):
            frac = 0.0 if total <= 0 else min(1.0, max(0.0, done / float(total)))
            progress.progress(frac, text=message)

        def _event(message: str):
            stamp = time.strftime("%H:%M:%S")
            log_lines.append(f"{stamp} {message}")
            if len(log_lines) > 400:
                del log_lines[:-400]
            log_box.text_area("Processing Log", value="\n".join(log_lines), height=220, disabled=True)

        results = ingestor.process_urls(
            urls=urls,
            convert_to_md=convert_to_md,
            use_vision_for_md=use_vision_for_md,
            capture_web_md_on_no_pdf=capture_web_md_on_no_pdf,
            progress_cb=_progress,
            event_cb=_event,
        )
        csv_path, json_path = ingestor.build_reports(results)
        zip_bytes = ingestor.build_zip_bytes(results, csv_path, json_path)
        progress.progress(1.0, text=f"Completed {len(results)} URL(s)")

        st.session_state["url_ingestor_results"] = results
        st.session_state["url_ingestor_csv_path"] = str(csv_path)
        st.session_state["url_ingestor_json_path"] = str(json_path)
        st.session_state["url_ingestor_zip_bytes"] = zip_bytes
        st.session_state["url_ingestor_run_dir"] = str(run_dir)
        st.session_state["url_ingestor_event_log"] = log_lines

results: List[URLIngestResult] = st.session_state.get("url_ingestor_results", [])
if results:
    with st.container(border=True):
        total = len(results)
        downloaded = sum(1 for r in results if r.status == "downloaded")
        web_captured = sum(1 for r in results if r.web_captured)
        converted = sum(1 for r in results if r.converted_to_md)
        failed = total - downloaded - web_captured
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total URLs", total)
        m2.metric("PDF Downloaded", downloaded)
        m3.metric("Web MD Captured", web_captured)
        m4.metric("Failed", failed)
        st.caption(f"Markdown outputs (PDF->MD + Web capture): {converted}")

        st.dataframe(
            _results_table_rows(results),
            use_container_width=True,
            hide_index=True,
            column_config={
                "input_url": st.column_config.LinkColumn("input_url", display_text="open"),
                "final_url": st.column_config.LinkColumn("final_url", display_text="resolved"),
            },
        )

        event_log: List[str] = st.session_state.get("url_ingestor_event_log", [])
        if event_log:
            with st.expander("Processing Event Log", expanded=False):
                st.text_area("Event log", value="\n".join(event_log), height=260, disabled=True)

        csv_path = Path(st.session_state.get("url_ingestor_csv_path", ""))
        json_path = Path(st.session_state.get("url_ingestor_json_path", ""))
        zip_bytes = st.session_state.get("url_ingestor_zip_bytes", b"")
        run_dir = st.session_state.get("url_ingestor_run_dir", "")

        c1, c2, c3 = st.columns(3)
        if csv_path.exists():
            c1.download_button("Download CSV Report", csv_path.read_bytes(), file_name=csv_path.name, mime="text/csv")
        if json_path.exists():
            c2.download_button("Download JSON Report", json_path.read_bytes(), file_name=json_path.name, mime="application/json")
        if zip_bytes:
            c3.download_button("Download PDFs + Markdown + Reports (ZIP)", zip_bytes, file_name="url_ingest_bundle.zip", mime="application/zip")

        st.caption(f"Artifacts saved under: {run_dir}")
