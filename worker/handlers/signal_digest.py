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
            "llm_provider": digest.get("llm_provider", ""),
            "llm_model": digest.get("llm_model", ""),
        },
        "output_file": Path(digest["output_path"]),
    }
