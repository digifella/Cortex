from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_DEFAULT_ANTHROPIC_MODEL = os.environ.get("CORTEX_REVIEW_TABLE_ANTHROPIC_MODEL", "").strip() or "claude-sonnet-4-6"
_ROOT = Path(__file__).resolve().parent.parent


def _worker_env_value(name: str) -> str:
    env_path = _ROOT / "worker" / "config.env"
    if not env_path.exists():
        return ""
    try:
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def get_anthropic_api_key() -> str:
    direct = str(os.environ.get("ANTHROPIC_API_KEY", "") or "").strip()
    if direct:
        return direct
    fallback = _worker_env_value("ANTHROPIC_API_KEY")
    if fallback and "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = fallback
    return fallback


def anthropic_key_source() -> str:
    direct = str(os.environ.get("ANTHROPIC_API_KEY", "") or "").strip()
    if direct:
        return "environment/.env"
    if _worker_env_value("ANTHROPIC_API_KEY"):
        return "worker/config.env"
    return ""


def claude_table_rescue_available() -> bool:
    return bool(get_anthropic_api_key())


def parse_review_table_rescue_response(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    fenced = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(fenced)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _anthropic_client():
    import anthropic

    api_key = get_anthropic_api_key()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def _coerce_list_of_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_rescue_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in list(payload.get("candidates") or []):
        if not isinstance(item, dict):
            continue
        design_matches = _coerce_list_of_strings(item.get("design_matches"))
        outcome_matches = _coerce_list_of_strings(item.get("outcome_matches"))
        reference_number = str(item.get("reference_number") or "").strip()
        needs_review = bool(item.get("needs_review"))
        review_warning = str(item.get("review_warning") or "").strip()
        validation = str(item.get("reference_validation") or "").strip()
        confidence = str(item.get("confidence") or "").strip().lower()
        score = 0
        if confidence == "high":
            score += 4
        elif confidence == "medium":
            score += 2
        if design_matches:
            score += 2
        if outcome_matches:
            score += 2
        if reference_number:
            score += 1
        candidates.append(
            {
                "title": str(item.get("title") or item.get("resolved_reference_title") or "").strip(),
                "authors": str(item.get("authors") or "").strip(),
                "year": str(item.get("year") or "").strip(),
                "doi": str(item.get("doi") or "").strip(),
                "journal": str(item.get("journal") or "").strip(),
                "abstract": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "accession": "",
                "aim": "",
                "notes": "cloud table rescue",
                "source_section": "cloud_rescue",
                "meets_criteria": bool(item.get("meets_criteria")),
                "relevance_score": score,
                "design_matches": design_matches,
                "outcome_matches": outcome_matches,
                "raw_citation": str(item.get("raw_label") or item.get("title") or "").strip(),
                "raw_excerpt": str(item.get("evidence") or "").strip()[:1000],
                "reference_number": reference_number,
                "reference_match_method": "cloud_rescue",
                "reference_validation": validation,
                "needs_review": needs_review,
                "review_warning": review_warning,
                "extra_fields": {
                    "reference_number": reference_number,
                    "reference_match_method": "cloud_rescue",
                    "reference_validation": validation,
                    "needs_review": "yes" if needs_review else "",
                    "review_warning": review_warning,
                    "cloud_confidence": confidence,
                },
            }
        )
    return candidates


def run_claude_table_rescue(
    *,
    review_title: str,
    design_query: str,
    outcome_query: str,
    table_snapshots: Sequence[Dict[str, Any]],
    table_blocks: Sequence[Dict[str, Any]],
    references_text: str,
    model: str = "",
) -> Dict[str, Any]:
    if not table_snapshots and not table_blocks:
        return {
            "provider": "anthropic",
            "model": model or _DEFAULT_ANTHROPIC_MODEL,
            "candidates": [],
            "warnings": ["No table evidence available for cloud rescue."],
            "raw_response": "",
        }

    client = _anthropic_client()
    model_name = model or _DEFAULT_ANTHROPIC_MODEL

    table_markdown_parts: List[str] = []
    for block in list(table_blocks or [])[:4]:
        table_markdown_parts.append(
            f"Table {int(block.get('table_index') or 0)}\n{str(block.get('markdown') or '').strip()[:6000]}"
        )
    references_excerpt = str(references_text or "").strip()[:18000]

    system_prompt = (
        "You extract included-study citations from messy systematic review tables. "
        "Be conservative. If a row is a table label or attribute rather than a study, exclude it. "
        "If a table label and the referenced bibliography entry do not match, include the row but set needs_review=true "
        "and explain the mismatch. Return only valid JSON."
    )
    user_prompt = (
        "Analyse these systematic review table snapshots and extracted markdown tables.\n\n"
        "Task:\n"
        "1. Identify actual study rows only.\n"
        "2. Reconstruct short labels like 'Maziarz 2020 [19]' into full study citations when possible.\n"
        "3. Use the reference section to validate the study identity.\n"
        "4. Mark needs_review=true if uncertain, contradictory, or only partially reconstructed.\n"
        "5. Only mark meets_criteria=true when the table evidence supports both design and outcome criteria.\n\n"
        f"Review title: {review_title}\n"
        f"Design criteria: {design_query}\n"
        f"Outcome criteria: {outcome_query}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "candidates": [\n'
        '    {\n'
        '      "title": "",\n'
        '      "authors": "",\n'
        '      "year": "",\n'
        '      "doi": "",\n'
        '      "journal": "",\n'
        '      "reference_number": "",\n'
        '      "resolved_reference_title": "",\n'
        '      "design_matches": [],\n'
        '      "outcome_matches": [],\n'
        '      "meets_criteria": false,\n'
        '      "needs_review": false,\n'
        '      "reference_validation": "",\n'
        '      "review_warning": "",\n'
        '      "confidence": "low",\n'
        '      "raw_label": "",\n'
        '      "evidence": ""\n'
        "    }\n"
        "  ],\n"
        '  "warnings": []\n'
        "}\n\n"
        "Extracted markdown tables:\n"
        f"{chr(10).join(table_markdown_parts)[:22000]}\n\n"
        "Reference section excerpt:\n"
        f"{references_excerpt}"
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for shot in list(table_snapshots or [])[:4]:
        image_bytes = shot.get("image_bytes")
        if not image_bytes:
            continue
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                },
            }
        )

    response = client.messages.create(
        model=model_name,
        max_tokens=3000,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    parts: List[str] = []
    for block in response.content:
        text = getattr(block, "text", "")
        if text:
            parts.append(text)
    raw_response = "\n".join(parts).strip()
    parsed = parse_review_table_rescue_response(raw_response)
    return {
        "provider": "anthropic",
        "model": model_name,
        "candidates": _normalize_rescue_candidates(parsed),
        "warnings": _coerce_list_of_strings(parsed.get("warnings")),
        "raw_response": raw_response,
    }
