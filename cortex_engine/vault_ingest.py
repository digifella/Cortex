# ## File: cortex_engine/vault_ingest.py
# Version: v6.4.0
# Date: 2026-07-31
# Purpose: Two-phase private vault ingest -- textify documents, then index them.

from __future__ import annotations

import re
from dataclasses import dataclass

# Emitted as the final line of nemoclaw-private-knowledge-ingest.py.
SUMMARY_RE = re.compile(
    r"done changed=(\d+) skipped=(\d+) failures=(\d+) dry_run=(\w+)"
)


@dataclass(frozen=True)
class IngestSummary:
    changed: int
    skipped: int
    failures: int
    dry_run: bool


def parse_ingest_summary(output: str) -> IngestSummary | None:
    """Pull the ingest script's summary line out of its stdout, or None."""
    match = SUMMARY_RE.search(output or "")
    if not match:
        return None
    return IngestSummary(
        changed=int(match.group(1)),
        skipped=int(match.group(2)),
        failures=int(match.group(3)),
        dry_run=match.group(4).lower() == "true",
    )


def should_index(summary: IngestSummary | None) -> bool:
    """Decide whether phase 2 should run.

    A missing summary means the ingest crashed or was killed: state is unknown,
    so we do not index a partial branch as if it were complete. Partial failures
    (exit code 2) still index -- `changed` may be 97 of 100.
    """
    if summary is None:
        return False
    if summary.dry_run:
        return False
    return summary.changed > 0
