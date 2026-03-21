from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Optional

from cortex_engine.stakeholder_signal_matcher import normalize_lookup


def _week_key(date_text: str) -> str:
    text = str(date_text or "").strip()
    return text[:8] if len(text) >= 8 else text


def _content_similarity(left: str, right: str) -> float:
    a = " ".join(str(left or "").lower().split())
    b = " ".join(str(right or "").lower().split())
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a[:4000], b=b[:4000]).ratio()


def find_duplicate_note(store: Any, payload: Dict[str, Any], similarity_threshold: float = 0.82) -> Optional[Dict[str, Any]]:
    primary_entity = dict(payload.get("primary_entity") or {})
    note = dict(payload.get("note") or {})
    primary_name = normalize_lookup(primary_entity.get("name") or "")
    note_date = str(note.get("note_date") or "").strip()
    source_type = normalize_lookup(note.get("source_type") or "")
    content = str(note.get("content") or note.get("original_text") or "").strip()
    attachment_fingerprints = {
        str(item).strip()
        for item in note.get("attachment_fingerprints") or []
        if str(item).strip()
    }
    if not primary_name or not note_date or not content:
        if not attachment_fingerprints or not note_date:
            return None

    for message in store.list_messages():
        result_path = str(message.get("result_path") or "").strip()
        if not result_path or not Path(result_path).exists():
            continue
        try:
            result_payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        website_payload = dict(result_payload.get("website_payload") or {})
        existing_primary = normalize_lookup(((website_payload.get("primary_entity") or {}).get("name")) or "")
        existing_note = dict(website_payload.get("note") or {})
        existing_date = str(existing_note.get("note_date") or "").strip()
        existing_source_type = normalize_lookup(existing_note.get("source_type") or "")
        existing_fingerprints = {
            str(item).strip()
            for item in existing_note.get("attachment_fingerprints") or []
            if str(item).strip()
        }
        same_week = _week_key(note_date) == _week_key(existing_date)
        same_primary = primary_name and primary_name == existing_primary
        same_source_type = source_type and source_type == existing_source_type
        fingerprint_overlap = attachment_fingerprints and existing_fingerprints and bool(attachment_fingerprints.intersection(existing_fingerprints))

        if fingerprint_overlap and same_week and (same_source_type or not source_type or not existing_source_type):
            return {
                "message_key": message.get("message_key", ""),
                "trace_id": message.get("trace_id", ""),
                "result_path": result_path,
                "similarity": 1.0,
                "delivery": dict(message.get("delivery") or {}),
                "existing_intel_id": str(
                    ((message.get("delivery") or {}).get("response") or {}).get("intel_id")
                    or result_payload.get("intel_id")
                    or ""
                ).strip(),
            }

        if not same_primary or not same_week:
            continue
        similarity = _content_similarity(content, existing_note.get("content") or existing_note.get("original_text") or "")
        if similarity >= similarity_threshold:
            return {
                "message_key": message.get("message_key", ""),
                "trace_id": message.get("trace_id", ""),
                "result_path": result_path,
                "similarity": round(similarity, 3),
                "delivery": dict(message.get("delivery") or {}),
                "existing_intel_id": str(
                    ((message.get("delivery") or {}).get("response") or {}).get("intel_id")
                    or result_payload.get("intel_id")
                    or ""
                ).strip(),
            }
    return None
