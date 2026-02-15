"""
Citation formatting helpers for claim mapper output.
"""

from __future__ import annotations

from typing import Dict, List


def build_numeric_citations(claim_map_result: Dict) -> List[Dict]:
    """
    Build compact numeric citation payload:
    claim_id -> [source_file #rank]
    """
    rows: List[Dict] = []
    for claim in claim_map_result.get("claims", []):
        refs = []
        for idx, ev in enumerate(claim.get("evidence", []), start=1):
            source = ev.get("source_file") or ev.get("doc_id") or "source"
            refs.append(f"[{idx}] {source}")
        rows.append(
            {
                "claim_id": claim.get("claim_id"),
                "status": claim.get("status"),
                "citations": refs,
            }
        )
    return rows
