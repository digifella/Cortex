from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_pdf_textify_input
from cortex_engine.textifier import DocumentTextifier

logger = logging.getLogger(__name__)


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Document-to-Markdown handler.

    Accepts PDF, DOCX, PPTX, and image files. Uses Docling for layout-aware
    PDF parsing and optional vision model for image descriptions.

    Returns:
        {
            "output_data": dict,
            "output_file": Path to .md file
        }
    """
    if input_path is None:
        raise ValueError("pdf_textify requires an input file")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = validate_pdf_textify_input(input_data or {})
    options = dict(payload.get("textify_options") or {})
    mode = str(options.get("pdf_strategy", "hybrid")).strip().lower() or "hybrid"
    use_vision = bool(options.get("use_vision", True))

    logger.info(
        "Textifying %s (mode=%s, use_vision=%s)",
        input_path.name,
        mode,
        use_vision,
    )
    if progress_cb:
        progress_cb(15, "Starting textification", "textify_start")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before textification started")

    textifier = DocumentTextifier.from_options(
        options,
        on_progress=(lambda frac, msg: progress_cb(15 + max(0, min(70, int(frac * 70))), msg, "textify_processing"))
        if progress_cb
        else None,
    )
    markdown_text = textifier.textify_file(str(input_path))
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled after textification")
    if progress_cb:
        progress_cb(90, "Writing markdown output", "write_output")

    if not markdown_text or not markdown_text.strip():
        raise RuntimeError(f"Textifier returned empty output for {input_path.name}")

    # Pick output format based on email mode (when invoked from the lab mailbox).
    email_mode = str((input_data or {}).get("email_textify_mode") or "").strip().lower()

    if email_mode == "text":
        body_text = DocumentTextifier.markdown_to_plaintext(markdown_text, width=80)
        output_filename = input_path.stem + ".txt"
    else:
        body_text = markdown_text
        output_filename = input_path.stem + ".md"

    output_path = input_path.parent / output_filename
    output_path.write_text(body_text, encoding="utf-8")

    # Stats computed against the markdown (more meaningful than against plain text)
    line_count = len(markdown_text.splitlines())
    table_count = markdown_text.count("|---")
    heading_count = sum(1 for line in markdown_text.splitlines() if line.startswith("#"))

    output_data = {
        "summary": "Converted via cortex_engine.textifier.DocumentTextifier",
        "source_filename": input_path.name,
        "output_filename": output_filename,
        "markdown_length": len(markdown_text),
        "line_count": line_count,
        "headings_found": heading_count,
        "tables_found": table_count,
        "use_vision": use_vision,
        "pdf_strategy": mode,
        "email_textify_mode": email_mode or "",
    }

    logger.info(
        "Textified %s -> %s (%d lines, %d headings, %d tables, mode=%s)",
        input_path.name,
        output_filename,
        line_count,
        heading_count,
        table_count,
        email_mode or "default",
    )
    if progress_cb:
        progress_cb(100, "Textification complete", "done")

    return {"output_data": output_data, "output_file": output_path}
