"""
Citation formatting helpers for claim mapper output.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


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


def build_reference_index(claim_map_result: Dict) -> Tuple[Dict[str, int], List[str]]:
    """
    Build a stable numeric reference index for sources present in evidence.
    Returns mapping and ordered reference labels.
    """
    refs: Dict[str, int] = {}
    ordered: List[str] = []
    for claim in claim_map_result.get("claims", []):
        for ev in claim.get("evidence", []):
            label = (ev.get("source_file") or ev.get("doc_id") or "source").strip()
            if label and label not in refs:
                refs[label] = len(refs) + 1
                ordered.append(label)
    return refs, ordered


def annotate_text_with_citations(
    draft_text: str,
    claim_map_result: Dict,
    include_statuses: Tuple[str, ...] = ("supported", "weak"),
    max_refs_per_claim: int = 2,
) -> Tuple[str, List[str]]:
    """
    Insert numeric citations into the first occurrence of each claim sentence.
    Returns (annotated_text, reference_lines).
    """
    text = draft_text or ""
    ref_map, ref_labels = build_reference_index(claim_map_result)
    if not text or not ref_map:
        return text, []

    for claim in claim_map_result.get("claims", []):
        status = claim.get("status")
        claim_text = (claim.get("claim_text") or "").strip()
        if status not in include_statuses or not claim_text:
            continue

        nums: List[int] = []
        for ev in claim.get("evidence", [])[: max(1, int(max_refs_per_claim))]:
            label = (ev.get("source_file") or ev.get("doc_id") or "source").strip()
            num = ref_map.get(label)
            if num is not None and num not in nums:
                nums.append(num)
        if not nums:
            continue

        cite = "".join(f"[{n}]" for n in nums)
        pattern = re.escape(claim_text)
        text, replaced = re.subn(pattern, f"{claim_text} {cite}", text, count=1)
        if replaced == 0:
            # fallback to whitespace-normalized lookup
            compact_claim = re.sub(r"\s+", " ", claim_text).strip()
            compact_text = re.sub(r"\s+", " ", text)
            idx = compact_text.find(compact_claim)
            if idx >= 0:
                # best-effort fallback: append citation at end of draft when exact slice isn't available
                text = text.rstrip() + f"\n\n{claim_text} {cite}\n"

    references = [f"[{i}] {label}" for i, label in enumerate(ref_labels, start=1)]
    return text, references
