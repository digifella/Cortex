from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_research_resolve_input
from cortex_engine.research_resolve import run_research_resolve


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    payload = validate_research_resolve_input(input_data or {})

    base_dir = input_path.parent if input_path else Path(tempfile.gettempdir())
    run_dir = base_dir / f"research_resolve_job_{job.get('id', 'unknown')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_data = run_research_resolve(
        payload=payload,
        run_dir=run_dir,
        progress_cb=progress_cb,
        is_cancelled_cb=is_cancelled_cb,
    )
    bundle_path = run_dir / "research_resolve_bundle.zip"
    return {"output_data": output_data, "output_file": bundle_path if bundle_path.exists() else None}
