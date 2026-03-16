from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_signal_digest_input
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    del input_path, job

    payload = validate_signal_digest_input(input_data or {})
    if progress_cb:
        progress_cb(15, "Collecting matched signals", "collect")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before digest generation")

    store = StakeholderSignalStore()
    digest = store.generate_digest(
        org_name=payload["org_name"],
        since_ts=payload.get("since_ts", ""),
        profile_keys=payload.get("profile_keys") or [],
        member_alumni=payload.get("member_alumni") or [],
        org_alumni=payload.get("org_alumni") or [],
        report_depth=str(payload.get("report_depth") or "detailed"),
        digest_tier=str(payload.get("digest_tier") or "standard"),
        priority_profile_keys=payload.get("priority_profile_keys") or [],
        deep_analysis=bool(payload.get("deep_analysis", False)),
        max_items=int(payload.get("max_items", 25)),
        include_needs_review=bool(payload.get("include_needs_review", True)),
        matched_only=bool(payload.get("matched_only", True)),
        llm_synthesis=bool(payload.get("llm_synthesis", False)),
        llm_provider=str(payload.get("llm_provider", "ollama")),
        llm_model=str(payload.get("llm_model", "")),
    )

    if progress_cb:
        progress_cb(100, f"Generated digest with {digest['signal_count']} signals", "done")

    return {
        "output_data": {
            "status": "generated",
            "digest_id": digest["digest_id"],
            "org_name": digest["org_name"],
            "signal_count": digest["signal_count"],
            "output_path": digest["output_path"],
            "llm_synthesised": digest.get("llm_synthesised", False),
            "profiles_covered": digest.get("profiles_covered", 0),
            "period_start": digest.get("period_start", ""),
            "period_end": digest.get("period_end", ""),
            "report_depth": digest.get("report_depth", "detailed"),
            "digest_tier": digest.get("digest_tier", "standard"),
            "deep_analysis": digest.get("deep_analysis", False),
            "escalate": digest.get("escalate", False),
            "escalate_profiles": digest.get("escalate_profiles", []),
            "escalate_reason": digest.get("escalate_reason", ""),
            "member_alumni": digest.get("member_alumni", []),
            "org_alumni": digest.get("org_alumni", []),
            "llm_provider": digest.get("llm_provider", ""),
            "llm_model": digest.get("llm_model", ""),
        },
        "output_file": Path(digest["output_path"]),
    }
