from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_signal_ingest_input
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    del input_path, job

    payload = validate_signal_ingest_input(input_data or {})
    org_name = payload["org_name"]
    raw_text = payload["raw_text"]
    subject = payload["subject"]
    if not org_name:
        raise ValueError("signal_ingest requires input_data.org_name")
    if not raw_text and not subject:
        raise ValueError("signal_ingest requires input_data.raw_text or input_data.subject")

    if progress_cb:
        progress_cb(15, "Normalizing stakeholder signal", "normalize")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before signal ingest")

    store = StakeholderSignalStore()
    signal = store.ingest_signal(payload)

    if progress_cb:
        progress_cb(100, f"Ingested signal {signal['signal_id']}", "done")

    return {
        "output_data": {
            "status": "ingested",
            "signal_id": signal["signal_id"],
            "org_name": signal["org_name"],
            "match_count": len(signal.get("matches") or []),
            "observed_fact_count": len(signal.get("observed_fact_ids") or []),
            "update_suggestion_count": len(signal.get("update_suggestion_ids") or []),
            "needs_review": bool(signal.get("needs_review")),
            "top_matches": signal.get("matches", []),
            "raw_file": signal.get("raw_file", ""),
        },
        "output_file": None,
    }
