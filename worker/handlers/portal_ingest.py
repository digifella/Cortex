from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_portal_ingest_input
from cortex_engine.textifier import DocumentTextifier

logger = logging.getLogger(__name__)

# Default target ~500 tokens per chunk (~2000 chars)
DEFAULT_CHUNK_TARGET_CHARS = 2000
DEFAULT_CHUNK_MIN_CHARS = 200
DEFAULT_MAX_CHUNKS = 250


def chunk_text(
    text: str,
    chunk_target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    chunk_min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """Split text into chunks by paragraphs/sections, targeting ~500 tokens each."""
    if not text or not text.strip():
        return []

    # Split on double newlines (paragraph boundaries) or markdown headings.
    blocks = re.split(r"\n{2,}", text.strip())
    chunks = []
    current = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        is_heading = block.startswith("#")

        # If adding this block exceeds target, flush current.
        if is_cancelled_cb and is_cancelled_cb():
            raise RuntimeError("Cancelled during chunking")

        if current and (len(current) + len(block) > chunk_target_chars or is_heading):
            if len(current.strip()) >= chunk_min_chars:
                chunks.append(current.strip())
                if len(chunks) >= max_chunks:
                    break
            current = ""

        current += ("\n\n" if current else "") + block

    # Flush remaining.
    if current.strip() and len(current.strip()) >= chunk_min_chars and len(chunks) < max_chunks:
        chunks.append(current.strip())
    elif current.strip() and chunks:
        # Append short trailing text to last chunk.
        chunks[-1] += "\n\n" + current.strip()
    elif current.strip():
        chunks.append(current.strip())

    return [
        {"text": c, "metadata": {"chunk_index": i, "char_count": len(c)}}
        for i, c in enumerate(chunks)
    ]


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Portal document ingest handler.

    Parses a document (PDF, DOCX, PPTX, MD, TXT) into text via Docling/Textifier,
    then chunks the text for storage in the portal knowledge base.
    """
    payload = validate_portal_ingest_input(input_data or {})
    chunk_target_chars = int(payload.get("chunk_target_chars", DEFAULT_CHUNK_TARGET_CHARS))
    chunk_min_chars = int(payload.get("chunk_min_chars", DEFAULT_CHUNK_MIN_CHARS))
    max_chunks = int(payload.get("max_chunks", DEFAULT_MAX_CHUNKS))

    if input_path is None:
        raise ValueError("portal_ingest requires an input file")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    logger.info("Portal ingest: %s (type=%s)", input_path.name, suffix)

    if progress_cb:
        progress_cb(10, "Starting document parsing", "parse_start")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before processing")

    # For plain text / markdown, read directly.
    if suffix in (".md", ".txt"):
        markdown_text = input_path.read_text(encoding="utf-8", errors="replace")
        if progress_cb:
            progress_cb(50, "Text file loaded", "text_loaded")
    else:
        # Use Textifier with Docling-first strategy for rich document formats.
        options = {
            "use_vision": False,
            "pdf_strategy": "docling",
            "cleanup_provider": "none",  # skip semantic cleanup for speed/stability
        }
        textifier = DocumentTextifier.from_options(
            options,
            on_progress=(
                lambda frac, msg: progress_cb(
                    10 + max(0, min(50, int(frac * 50))), msg, "textify_processing"
                )
            )
            if progress_cb
            else None,
        )
        markdown_text = textifier.textify_file(str(input_path))

    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled after parsing")

    if not markdown_text or not markdown_text.strip():
        raise RuntimeError(f"No text extracted from {input_path.name}")

    if progress_cb:
        progress_cb(70, "Chunking text", "chunking")

    chunks = chunk_text(
        markdown_text,
        chunk_target_chars=chunk_target_chars,
        chunk_min_chars=chunk_min_chars,
        max_chunks=max_chunks,
        is_cancelled_cb=is_cancelled_cb,
    )

    if progress_cb:
        progress_cb(90, f"Generated {len(chunks)} chunks", "chunks_ready")

    logger.info(
        "Portal ingest: %s -> %d chunks (%d chars total)",
        input_path.name,
        len(chunks),
        len(markdown_text),
    )

    output_data = {
        "chunks": chunks,
        "summary": f"Parsed and chunked {input_path.name}",
        "source_filename": input_path.name,
        "total_chars": len(markdown_text),
        "chunk_count": len(chunks),
        "chunk_target_chars": chunk_target_chars,
        "chunk_min_chars": chunk_min_chars,
        "max_chunks": max_chunks,
        "portal_document_id": payload.get("portal_document_id"),
        "project_id": payload.get("project_id"),
        "tenant_id": payload.get("tenant_id"),
    }

    if progress_cb:
        progress_cb(100, "Portal ingest complete", "done")

    return {"output_data": output_data}
