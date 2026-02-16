from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

from cortex_engine.url_ingestor import URLIngestor, normalize_url_list


TEXTIFY_OPTION_KEYS = {
    "pdf_strategy",
    "cleanup_provider",
    "cleanup_model",
    "docling_timeout_seconds",
    "image_description_timeout_seconds",
    "image_enrich_max_seconds",
}


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
    payload = dict(input_data or {})
    ingest_options = dict(payload.get("ingest_options") or {})

    urls = _extract_urls(payload)
    if not urls:
        raise ValueError("url_ingest requires non-empty input_data.urls (or url_text/url_list)")

    convert_to_md = bool(ingest_options.get("convert_to_md", False))
    use_vision_for_md = bool(ingest_options.get("use_vision", False))
    capture_web_md_on_no_pdf = bool(ingest_options.get("capture_web_md_on_no_pdf", True))
    timeout_seconds = int(payload.get("timeout_seconds") or ingest_options.get("timeout_seconds") or 25)
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

    csv_path, json_path = ingestor.build_reports(results)
    zip_bytes = ingestor.build_zip_bytes(results, csv_path, json_path)
    output_path = run_dir / "url_ingest_bundle.zip"
    output_path.write_bytes(zip_bytes)

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
        "report_csv": csv_path.name,
        "report_json": json_path.name,
        "results": [r.to_dict() for r in results],
    }
    if progress_cb:
        progress_cb(100, "URL ingest complete", "done")

    # Keep output_data JSON-safe for queue transport.
    json.loads(json.dumps(output_data))
    return {"output_data": output_data, "output_file": output_path}
