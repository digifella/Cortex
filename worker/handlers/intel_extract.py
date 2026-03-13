from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_intel_extract_input
from cortex_engine.intel_extractor import extract_intel


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    del input_path, job

    payload = validate_intel_extract_input(input_data or {})
    if progress_cb:
        progress_cb(10, "Preparing extraction payload", "prepare")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before extraction started")

    if progress_cb:
        progress_cb(35, "Extracting entities from email and attachments", "extract")
    result, output_path = extract_intel(payload)

    if progress_cb:
        progress_cb(100, f"Extracted {result.get('entity_count', 0)} entities", "done")

    return {"output_data": result, "output_file": output_path}
