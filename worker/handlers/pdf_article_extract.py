from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from cortex_engine.article_markdown_extractor import extract_pdf_articles_to_bundle
from cortex_engine.handoff_contract import validate_pdf_article_extract_input

logger = logging.getLogger(__name__)


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    if input_path is None:
        raise ValueError("pdf_article_extract requires an input file")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError("pdf_article_extract requires a PDF input file")

    payload = validate_pdf_article_extract_input(input_data or {})
    options = dict(payload.get("article_extract_options") or {})
    segmentation_strategy = str(options.get("segmentation_strategy") or "pdf_bookmarks")

    logger.info(
        "Article extracting %s (strategy=%s)",
        input_path.name,
        segmentation_strategy,
    )
    if progress_cb:
        progress_cb(10, "Starting article extraction", "article_extract_start")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before article extraction started")

    result = extract_pdf_articles_to_bundle(
        input_path,
        output_dir=input_path.parent / "article_extract_bundle",
        progress_cb=(lambda frac, msg: progress_cb(10 + max(0, min(80, int(frac * 80))), msg, "article_extract_processing"))
        if progress_cb
        else None,
    )

    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled after article extraction")
    if progress_cb:
        progress_cb(95, "Preparing article bundle", "article_extract_bundle")

    zip_path = Path(result["zip_path"])
    manifest = dict(result.get("manifest") or {})
    output_data = {
        "summary": "Converted PDF bookmarks/pages into separate article markdown files",
        "source_filename": input_path.name,
        "bundle_filename": zip_path.name,
        "article_count": int(manifest.get("article_count") or 0),
        "articles": list(manifest.get("articles") or []),
        "segmentation_strategy": segmentation_strategy,
    }

    if progress_cb:
        progress_cb(100, "Article extraction complete", "done")

    return {"output_data": output_data, "output_file": zip_path}
