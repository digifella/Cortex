from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_org_profile_refresh_input
from cortex_engine.org_profile_refresh import run_org_profile_refresh
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    payload = validate_org_profile_refresh_input(input_data or {})

    base_dir = input_path.parent if input_path else Path(tempfile.gettempdir())
    run_dir = base_dir / f"org_profile_refresh_job_{job.get('id', 'unknown')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_data = run_org_profile_refresh(
        payload=payload,
        run_dir=run_dir,
        progress_cb=progress_cb,
        is_cancelled_cb=is_cancelled_cb,
        signal_store=StakeholderSignalStore(),
    )
    return {"output_data": output_data, "output_file": None}
