from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_stakeholder_profile_sync_input
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    del input_path, job

    payload = validate_stakeholder_profile_sync_input(input_data or {})
    org_name = payload["org_name"]
    profiles = list(payload["profiles"])
    if not org_name:
        raise ValueError("stakeholder_profile_sync requires input_data.org_name")
    if not profiles:
        raise ValueError("stakeholder_profile_sync requires input_data.profiles")

    if progress_cb:
        progress_cb(10, "Validating stakeholder profiles", "validate")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before profile sync")

    store = StakeholderSignalStore()
    result = store.upsert_profiles(
        org_name=org_name,
        profiles=profiles,
        source=str(payload.get("source_system") or "market_radar").strip(),
        trace_id=str(payload.get("trace_id") or "").strip(),
        replace_org_scope=True,
    )

    if progress_cb:
        progress_cb(100, f"Synchronized {len(profiles)} stakeholder profiles", "done")

    return {
        "output_data": {
            "status": "synced",
            "org_name": org_name,
            "profiles_received": len(profiles),
            **result,
        },
        "output_file": None,
    }
