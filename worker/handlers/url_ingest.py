from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

from cortex_engine.handoff_contract import validate_url_ingest_input
from cortex_engine.url_ingestor import URLIngestor, URLIngestResult, normalize_url_list


TEXTIFY_OPTION_KEYS = {
    "pdf_strategy",
    "cleanup_provider",
    "cleanup_model",
    "docling_timeout_seconds",
    "image_description_timeout_seconds",
    "image_enrich_max_seconds",
}

VAULT_LAB_NOTES_DIR = Path(os.environ.get("NEMOCLAW_LAB_NOTES_DIR", "/mnt/c/Users/paul/Documents/AI-Vault/lab-notes"))
VAULT_NOTE_FILENAME_MAX = int(os.environ.get("NEMOCLAW_NOTE_FILENAME_MAX", "56"))


def _extract_urls(input_data: dict) -> List[str]:
    urls = input_data.get("urls")
    if isinstance(urls, list):
        clean = [str(u).strip() for u in urls if str(u).strip()]
        if clean:
            return clean

    raw = str(input_data.get("url_text") or input_data.get("url_list") or "").strip()
    if raw:
        return normalize_url_list(raw)
    return []


def _extract_textify_options(input_data: dict) -> Dict[str, object]:
    # Contract: website sends textify options as top-level keys.
    opts = {k: input_data[k] for k in TEXTIFY_OPTION_KEYS if k in input_data}
    # Backward compatibility: allow nested textify_options if present.
    nested = dict(input_data.get("textify_options") or {})
    for k in TEXTIFY_OPTION_KEYS:
        if k in nested and k not in opts:
            opts[k] = nested[k]
    return opts


def _build_markdown_report(results: List[URLIngestResult], output_data: dict) -> str:
    """Build a combined markdown report from URL ingest results for inline result display."""
    today = date.today().isoformat()
    total = output_data.get("total_urls", len(results))
    downloaded = output_data.get("downloaded_pdfs", 0)
    web_captured = output_data.get("web_markdown_captured", 0)
    converted = output_data.get("converted_to_md", 0)
    failed = output_data.get("failed", 0)

    lines = [
        "---",
        "title: URL Ingest Summary",
        f"date: {today}",
        "source_type: url_ingest",
        "---",
        "",
        "# URL Ingest Summary",
        f"Generated: {today}",
        "",
        "## Overview",
        f"- **Total URLs processed:** {total}",
        f"- **Web pages captured:** {web_captured}",
        f"- **PDFs downloaded:** {downloaded}",
        f"- **Converted to Markdown:** {converted}",
        f"- **Failed:** {failed}",
        "",
    ]

    for r in results:
        url = r.input_url
        page_title = r.page_title or ""
        status = r.status
        reason = r.reason or ""

        lines.append("---")
        lines.append("")
        heading = page_title if page_title else url
        lines.append(f"## {heading}")
        lines.append(f"**URL:** {url}")
        lines.append(f"**Status:** {status}")
        if reason:
            lines.append(f"**Note:** {reason}")
        lines.append("")

        if r.md_path and Path(r.md_path).exists():
            try:
                md_content = Path(r.md_path).read_text(encoding="utf-8", errors="replace").strip()
                if md_content:
                    lines.append(md_content)
            except Exception:
                lines.append("*[Could not read captured content]*")
        elif r.pdf_path and Path(r.pdf_path).exists():
            lines.append(
                f"*PDF downloaded: {Path(r.pdf_path).name}"
                " — enable 'Convert to Markdown' to see content inline.*"
            )
        elif status == "failed":
            lines.append(f"*Processing failed: {reason}*")

        lines.append("")

    return "\n".join(lines)


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^\w\s.-]+", "", str(value or ""), flags=re.UNICODE).strip()
    value = re.sub(r"\s+", "-", value)
    return value[:VAULT_NOTE_FILENAME_MAX].strip(".-") or "url-ingest-summary"


def _report_title(results: List[URLIngestResult]) -> str:
    if len(results) == 1:
        result = results[0]
        return result.page_title or result.input_url or "URL Ingest Summary"
    return f"URL Ingest Summary - {len(results)} URLs"


def _write_vault_lab_note(content: str, title: str, job: dict) -> None:
    try:
        today = date.today().isoformat()
        VAULT_LAB_NOTES_DIR.mkdir(parents=True, exist_ok=True)
        path = VAULT_LAB_NOTES_DIR / f"{today}-{_safe_filename(title)}.md"
        if path.exists():
            return
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
    except Exception:
        # Queue completion should not fail solely because the vault write failed.
        return


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    URL ingest handler for open-access PDF discovery/download and optional PDF->MD conversion.

    Input contract:
    - input_data.urls: List[str]
    - input_data.ingest_options: {convert_to_md, use_vision, capture_web_md_on_no_pdf}
    - input_data.timeout_seconds: int (optional)
    - input_data.<textify options>: top-level advanced options forwarded to DocumentTextifier
    """
    payload = validate_url_ingest_input(input_data or {})
    ingest_options = dict(payload.get("ingest_options") or {})

    urls = _extract_urls(payload)
    if not urls:
        raise ValueError("url_ingest requires non-empty input_data.urls (or url_text/url_list)")

    convert_to_md = bool(ingest_options.get("convert_to_md", False))
    use_vision_for_md = bool(ingest_options.get("use_vision", False))
    capture_web_md_on_no_pdf = bool(ingest_options.get("capture_web_md_on_no_pdf", True))
    timeout_seconds = int(payload.get("timeout_seconds") or 25)
    textify_options = _extract_textify_options(payload)

    base_dir = input_path.parent if input_path else Path(tempfile.gettempdir())
    run_dir = base_dir / f"url_ingest_job_{job.get('id', 'unknown')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ingestor = URLIngestor(run_dir, timeout=timeout_seconds)

    def _worker_progress(done: int, total: int, message: str) -> None:
        if not progress_cb:
            return
        frac = 0.0 if total <= 0 else min(1.0, max(0.0, done / float(total)))
        pct = 10 + int(frac * 80)
        progress_cb(pct, message, "url_ingest_processing")

    def _worker_event(message: str) -> None:
        if is_cancelled_cb and is_cancelled_cb():
            raise RuntimeError("Cancelled by operator")
        if progress_cb:
            progress_cb(10, message, "url_ingest_event")

    if progress_cb:
        progress_cb(5, "Starting URL ingest", "url_ingest_start")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before URL ingest started")

    results = ingestor.process_urls(
        urls=urls,
        convert_to_md=convert_to_md,
        use_vision_for_md=use_vision_for_md,
        textify_options=textify_options,
        capture_web_md_on_no_pdf=capture_web_md_on_no_pdf,
        progress_cb=_worker_progress,
        event_cb=_worker_event,
    )
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled after URL ingest processing")

    total = len(results)
    downloaded = sum(1 for r in results if r.status == "downloaded")
    web_captured = sum(1 for r in results if r.web_captured)
    converted = sum(1 for r in results if r.converted_to_md)
    failed = sum(1 for r in results if r.status == "failed")

    output_data = {
        "summary": "Processed via cortex_engine.url_ingestor.URLIngestor",
        "total_urls": total,
        "downloaded_pdfs": downloaded,
        "web_markdown_captured": web_captured,
        "converted_to_md": converted,
        "failed": failed,
        "convert_to_md": convert_to_md,
        "use_vision_for_md": use_vision_for_md,
        "capture_web_md_on_no_pdf": capture_web_md_on_no_pdf,
        "timeout_seconds": timeout_seconds,
        "textify_options": textify_options,
        "results": [r.to_dict() for r in results],
    }
    if progress_cb:
        progress_cb(100, "URL ingest complete", "done")

    # Keep output_data JSON-safe for queue transport.
    json.loads(json.dumps(output_data))

    # Build a combined markdown report (inline-viewable on the result page).
    report_md = _build_markdown_report(results, output_data)
    if str(input_data.get("source_system", "")).lower() == "email":
        _write_vault_lab_note(report_md, _report_title(results), job)

    md_suffix = f"_url_ingest_{date.today().isoformat()}.md"
    with tempfile.NamedTemporaryFile(mode="w", suffix=md_suffix, delete=False, encoding="utf-8") as mf:
        mf.write(report_md)
        md_output_path = Path(mf.name)

    return {"output_data": output_data, "output_file": md_output_path}
