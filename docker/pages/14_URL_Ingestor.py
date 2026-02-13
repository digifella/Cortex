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
                "pdf_found": r.open_access_pdf_found,
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
    "Paste URLs (one per line). The ingestor will attempt to find an open-access PDF and report what succeeded or failed."
)

with st.container(border=True):
    url_text = st.text_area(
        "URL List",
        height=220,
        placeholder="https://example.org/article\nhttps://doi.org/10.xxxx/yyyy\nhttps://example.org/report.pdf",
        key="url_ingestor_input",
    )
    c1, c2, c3 = st.columns(3)
    convert_to_md = c1.checkbox("Convert downloaded PDFs to Markdown", value=True)
    use_vision_for_md = c2.checkbox("Use vision during PDF->MD conversion", value=False)
    timeout_seconds = c3.number_input("Request timeout (seconds)", min_value=5, max_value=120, value=25, step=5)

    try:
        db_root = _resolve_db_root()
        st.caption(f"Output root: {db_root / 'url_ingest'}")
    except Exception as e:
        db_root = None
        st.error(str(e))

    run_btn = st.button("Process URLs", type="primary", use_container_width=True, disabled=(db_root is None))

if run_btn and db_root is not None:
    urls = normalize_url_list(url_text)
    if not urls:
        st.warning("No valid http/https URLs found.")
    else:
        run_dir = db_root / "url_ingest" / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        ingestor = URLIngestor(run_dir, timeout=int(timeout_seconds))

        progress = st.progress(0.0, text="Starting URL processing...")

        def _progress(done: int, total: int, message: str):
            frac = 0.0 if total <= 0 else min(1.0, max(0.0, done / float(total)))
            progress.progress(frac, text=message)

        results = ingestor.process_urls(
            urls=urls,
            convert_to_md=convert_to_md,
            use_vision_for_md=use_vision_for_md,
            progress_cb=_progress,
        )
        csv_path, json_path = ingestor.build_reports(results)
        zip_bytes = ingestor.build_zip_bytes(results, csv_path, json_path)
        progress.progress(1.0, text=f"Completed {len(results)} URL(s)")

        st.session_state["url_ingestor_results"] = results
        st.session_state["url_ingestor_csv_path"] = str(csv_path)
        st.session_state["url_ingestor_json_path"] = str(json_path)
        st.session_state["url_ingestor_zip_bytes"] = zip_bytes
        st.session_state["url_ingestor_run_dir"] = str(run_dir)

results: List[URLIngestResult] = st.session_state.get("url_ingestor_results", [])
if results:
    with st.container(border=True):
        total = len(results)
        downloaded = sum(1 for r in results if r.status == "downloaded")
        converted = sum(1 for r in results if r.converted_to_md)
        failed = total - downloaded
        m1, m2, m3 = st.columns(3)
        m1.metric("Total URLs", total)
        m2.metric("PDF Downloaded", downloaded)
        m3.metric("Failed", failed)
        st.caption(f"PDF->MD converted: {converted}")

        st.dataframe(_results_table_rows(results), use_container_width=True, hide_index=True)

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
