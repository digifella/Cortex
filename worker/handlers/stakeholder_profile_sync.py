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

    raw_input = dict(input_data or {})
    payload = validate_stakeholder_profile_sync_input(raw_input)
    org_name = payload["org_name"]
    profiles = list(payload["profiles"])
    if not org_name:
        raise ValueError("stakeholder_profile_sync requires input_data.org_name")
    raw_strategic_profile = raw_input.get("org_strategic_profile") if "org_strategic_profile" in raw_input else None
    strategic_profile = payload.get("org_strategic_profile") if raw_strategic_profile is not None else None

    if progress_cb:
        progress_cb(10, "Validating stakeholder/org sync payload", "validate")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before profile sync")

    store = StakeholderSignalStore()
    result = store.upsert_profiles(
        org_name=org_name,
        profiles=profiles,
        org_alumni=payload.get("org_alumni") or [],
        org_strategic_profile=strategic_profile,
        source=str(payload.get("source_system") or "market_radar").strip(),
        trace_id=str(payload.get("trace_id") or "").strip(),
        replace_org_scope=True,
    )

    if progress_cb:
        progress_cb(100, f"Synchronized {len(profiles)} stakeholder profiles and organisation context", "done")

    return {
        "output_data": {
            "status": "synced",
            "org_name": org_name,
            "profiles_received": len(profiles),
            "org_alumni_count": len(payload.get("org_alumni") or []),
            "org_strategic_industry_count": len((result.get("org_strategic_profile") or {}).get("industries") or []),
            **result,
        },
        "output_file": None,
    }
