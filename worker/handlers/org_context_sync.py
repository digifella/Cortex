"""Handler for org_context_sync jobs.

Syncs org strategic profile data (description, industries, key themes,
strategic objectives) to the Cortex signal store without touching profiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from cortex_engine.stakeholder_signal_store import StakeholderSignalStore


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    payload = dict(input_data or {})
    org_name = str(payload.get("org_name") or "").strip()
    if not org_name:
        raise ValueError("org_context_sync requires org_name")

    strategic_profile = payload.get("org_strategic_profile") or {}
    if not isinstance(strategic_profile, dict):
        strategic_profile = {}

    if progress_cb:
        progress_cb(0.1, f"Syncing context for {org_name}", None)

    store = StakeholderSignalStore()
    result = store.upsert_profiles(
        org_name=org_name,
        profiles=[],
        org_strategic_profile=strategic_profile,
        source="market_radar_org_context_sync",
        trace_id=str(job.get("trace_id") or ""),
    )

    if progress_cb:
        progress_cb(1.0, "Done", None)

    return {
        "output_data": {
            "org_name": org_name,
            "industries_synced": len(strategic_profile.get("industries") or []),
            "themes_synced": len(strategic_profile.get("key_themes") or []),
            "objectives_synced": len(strategic_profile.get("strategic_objectives") or []),
            "org_alumni_count": result.get("org_alumni_count", 0),
            "org_strategic_industry_count": result.get("org_strategic_industry_count", 0),
        },
        "output_file": None,
    }
