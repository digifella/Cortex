from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from cortex_engine.org_chart_extractor import analyse_org_chart_attachments
from cortex_engine.strategic_doc_analyser import analyse_strategic_documents, clean_strategic_role_label
from cortex_engine.stakeholder_signal_matcher import normalize_lookup

_STRATEGIC_PERSON_ROLE_MARKERS = (
    "president",
    "chair",
    "chief",
    "ceo",
    "executive",
    "director",
    "head",
    "lead",
    "dean",
    "officer",
)
_STRATEGIC_PERSON_EVIDENCE_MARKERS = (
    "foreword",
    "signed by",
    "chief executive",
    "president and chair",
    "board",
    "leadership",
)
_STRATEGIC_CREDIT_MARKERS = (
    "artwork",
    "design",
    "credit",
    "developed",
    "healing place",
    "cultural elements",
)
_ANNUAL_REPORT_PERSON_STOPWORDS = (
    "attendance",
    "committee",
    "integrated water management",
    "risk management",
    "review",
    "governance reform",
    "current directors",
    "term ceased",
    "cpd",
)


def _clean_annual_report_name(name: str) -> str:
    text = str(name or "").strip()
    text = text.lstrip("•*-· ").strip()
    text = " ".join(text.split())
    return text


def _merge_strategic_leadership_entities(output_data: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    strategic_doc = dict(analysis.get("strategic_doc") or {})
    org_name = str(strategic_doc.get("org_name") or "").strip()
    leadership_people = list(strategic_doc.get("leadership_people") or [])
    if not leadership_people:
        return

    existing_entities = list(output_data.get("entities") or [])
    existing_people = list(output_data.get("people") or [])
    seen_entity_names = {
        normalize_lookup(item.get("canonical_name") or item.get("name") or "")
        for item in existing_entities
        if str(item.get("canonical_name") or item.get("name") or "").strip()
    }
    seen_people_names = {
        normalize_lookup(item.get("canonical_name") or item.get("name") or "")
        for item in existing_people
        if str(item.get("canonical_name") or item.get("name") or "").strip()
    }

    for person in leadership_people:
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        key = normalize_lookup(name)
        if key not in seen_entity_names:
            existing_entities.append(
                {
                    "canonical_name": name,
                    "name": name,
                    "target_type": "person",
                    "current_employer": str(person.get("current_employer") or "").strip(),
                    "current_role": clean_strategic_role_label(str(person.get("current_role") or "").strip(), org_name),
                    "confidence": float(person.get("confidence") or 0.92),
                    "evidence": str(person.get("evidence") or "").strip(),
                }
            )
            seen_entity_names.add(key)
        if key not in seen_people_names:
            existing_people.append(
                {
                    "canonical_name": name,
                    "name": name,
                    "current_employer": str(person.get("current_employer") or "").strip(),
                    "current_role": clean_strategic_role_label(str(person.get("current_role") or "").strip(), org_name),
                    "confidence": float(person.get("confidence") or 0.92),
                    "evidence": str(person.get("evidence") or "").strip(),
                }
            )
            seen_people_names.add(key)

    output_data["entities"] = existing_entities
    output_data["people"] = existing_people


def _apply_strategic_summary(output_data: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    strategic_doc = dict(analysis.get("strategic_doc") or {})
    doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
    if doc_type not in {"strategic_plan", "annual_report", "industry_report"}:
        return

    org_name = str(strategic_doc.get("org_name") or "").strip()
    signals = [str(item.get("headline") or "").strip() for item in strategic_doc.get("strategic_signals") or [] if str(item.get("headline") or "").strip()]
    performance_indicators = [
        item
        for item in strategic_doc.get("performance_indicators") or []
        if str(item.get("label") or "").strip()
    ]
    leaders = []
    for item in strategic_doc.get("leadership_people") or []:
        name = str(item.get("name") or "").strip()
        role = clean_strategic_role_label(str(item.get("current_role") or "").strip(), org_name)
        if not name or not role:
            continue
        leaders.append(f"{name} ({role})")

    parts = []
    if doc_type == "strategic_plan" and org_name:
        parts.append(f"Strategic planning document from {org_name} outlining the current direction and operating priorities.")
    elif org_name:
        parts.append(f"Document from {org_name}.")
    if leaders:
        parts.append(f"Leadership identified: {', '.join(leaders[:2])}.")
    if signals:
        parts.append(f"Key strategic signals: {', '.join(signals[:5])}.")
    if performance_indicators:
        parts.append(
            "Performance indicators: "
            + ", ".join(
                f"{item['label']} ({item['value']})"
                for item in performance_indicators[:4]
                if str(item.get("value") or "").strip()
            )
            + "."
        )

    if parts:
        output_data["summary"] = " ".join(parts).strip()


def _should_keep_strategic_entity(entity: Dict[str, Any], strategic_org_name: str) -> bool:
    target_type = str(entity.get("target_type") or "").strip().lower()
    name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
    role = str(entity.get("current_role") or "").strip()
    employer = str(entity.get("current_employer") or "").strip()
    evidence = str(entity.get("evidence") or "").strip()
    lowered_evidence = normalize_lookup(evidence)
    strategic_org_key = normalize_lookup(strategic_org_name)
    name_key = normalize_lookup(name)
    employer_key = normalize_lookup(employer)

    if target_type == "person":
        if any(marker in lowered_evidence for marker in _STRATEGIC_CREDIT_MARKERS):
            return False
        if role and any(marker in normalize_lookup(role) for marker in _STRATEGIC_PERSON_ROLE_MARKERS):
            return True
        if any(marker in lowered_evidence for marker in _STRATEGIC_PERSON_EVIDENCE_MARKERS):
            return True
        if "named in document" in lowered_evidence:
            return False
        return False

    if target_type == "organisation":
        if strategic_org_key and name_key == strategic_org_key:
            return True
        if any(marker in lowered_evidence for marker in _STRATEGIC_CREDIT_MARKERS):
            return False
        if employer_key and employer_key == strategic_org_key:
            return False
        return False

    return True


def _looks_like_curated_stakeholder_name(name: str) -> bool:
    cleaned = _clean_annual_report_name(name)
    lowered = normalize_lookup(cleaned)
    if not lowered:
        return False
    if any(marker in lowered for marker in _ANNUAL_REPORT_PERSON_STOPWORDS):
        return False
    if any(char.isdigit() for char in cleaned):
        return False
    parts = [part for part in cleaned.replace(",", " ").split() if part]
    if len(parts) < 2 or len(parts) > 7:
        return False
    return True


def _filter_annual_report_entities(output_data: Dict[str, Any], strategic_doc: Dict[str, Any]) -> None:
    strategic_org_name = str(strategic_doc.get("org_name") or "").strip()
    strategic_org_key = normalize_lookup(strategic_org_name)
    curated_people = [
        item
        for item in strategic_doc.get("key_stakeholders") or strategic_doc.get("leadership_people") or []
        if _looks_like_curated_stakeholder_name(str(item.get("name") or "").strip())
    ]
    curated_people_keys = {
        normalize_lookup(_clean_annual_report_name(str(item.get("name") or "").strip()))
        for item in curated_people
        if str(item.get("name") or "").strip()
    }

    filtered_entities = []
    organisation_candidates = []
    for item in list(output_data.get("entities") or []):
        target_type = str(item.get("target_type") or "").strip().lower()
        name = _clean_annual_report_name(str(item.get("canonical_name") or item.get("name") or "").strip())
        name_key = normalize_lookup(name)
        if not name:
            continue
        if target_type == "organisation":
            clean_item = dict(item)
            clean_item["canonical_name"] = name
            clean_item["name"] = name
            organisation_candidates.append(clean_item)
            if strategic_org_key and name_key == strategic_org_key:
                filtered_entities.append(clean_item)
            continue
        if target_type == "person" and name_key in curated_people_keys:
            item = dict(item)
            item["canonical_name"] = name
            item["name"] = name
            filtered_entities.append(item)

    if organisation_candidates and not any(str(item.get("target_type") or "").strip().lower() == "organisation" for item in filtered_entities):
        best_org = max(organisation_candidates, key=lambda item: float(item.get("confidence") or 0.0))
        filtered_entities.append(best_org)

    # Ensure curated stakeholders are restored even when extractor missed them.
    seen_keys = {
        normalize_lookup(str(item.get("canonical_name") or item.get("name") or "").strip())
        for item in filtered_entities
        if str(item.get("canonical_name") or item.get("name") or "").strip()
    }
    for person in curated_people:
        name = _clean_annual_report_name(str(person.get("name") or "").strip())
        key = normalize_lookup(name)
        if not name or key in seen_keys:
            continue
        filtered_entities.append(
            {
                "canonical_name": name,
                "name": name,
                "target_type": "person",
                "current_employer": str(person.get("current_employer") or strategic_org_name).strip(),
                "current_role": clean_strategic_role_label(str(person.get("current_role") or "").strip(), strategic_org_name),
                "confidence": float(person.get("confidence") or 0.92),
                "evidence": str(person.get("evidence") or "").strip(),
            }
        )
        seen_keys.add(key)

    output_data["entities"] = filtered_entities
    output_data["contacts"] = filtered_entities
    output_data["extracted"] = filtered_entities
    output_data["entity_count"] = len(filtered_entities)

    output_data["people"] = [
        {
            **item,
            "canonical_name": _clean_annual_report_name(str(item.get("canonical_name") or item.get("name") or "").strip()),
            "name": _clean_annual_report_name(str(item.get("canonical_name") or item.get("name") or "").strip()),
        }
        for item in list(output_data.get("people") or [])
        if normalize_lookup(_clean_annual_report_name(str(item.get("canonical_name") or item.get("name") or "").strip())) in curated_people_keys
    ]
    existing_people_keys = {
        normalize_lookup(_clean_annual_report_name(str(item.get("canonical_name") or item.get("name") or "").strip()))
        for item in output_data["people"]
    }
    for person in curated_people:
        clean_name = _clean_annual_report_name(str(person.get("name") or "").strip())
        key = normalize_lookup(clean_name)
        if not key or key in existing_people_keys:
            continue
        output_data["people"].append(
            {
                "canonical_name": clean_name,
                "name": clean_name,
                "current_employer": str(person.get("current_employer") or strategic_org_name).strip(),
                "current_role": clean_strategic_role_label(str(person.get("current_role") or "").strip(), strategic_org_name),
                "confidence": float(person.get("confidence") or 0.92),
                "evidence": str(person.get("evidence") or "").strip(),
            }
        )
        existing_people_keys.add(key)

    output_data["organisations"] = [
        item
        for item in list(output_data.get("organisations") or [])
        if normalize_lookup(str(item.get("canonical_name") or item.get("name") or "").strip()) == strategic_org_key
    ]
    if not output_data["organisations"] and organisation_candidates:
        best_org = max(organisation_candidates, key=lambda item: float(item.get("confidence") or 0.0))
        output_data["organisations"] = [
            {
                "canonical_name": str(best_org.get("canonical_name") or best_org.get("name") or "").strip(),
                "name": str(best_org.get("name") or best_org.get("canonical_name") or "").strip(),
                "evidence": str(best_org.get("evidence") or "").strip(),
            }
        ]


def _filter_strategic_document_entities(output_data: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    strategic_doc = dict(analysis.get("strategic_doc") or {})
    doc_type = str(strategic_doc.get("doc_type") or "").strip().lower()
    if doc_type not in {"strategic_plan", "annual_report", "industry_report"}:
        return
    if doc_type == "annual_report":
        _filter_annual_report_entities(output_data, strategic_doc)
        return
    strategic_org_name = str(strategic_doc.get("org_name") or "").strip()
    entities = list(output_data.get("entities") or [])
    if not entities:
        return

    filtered_entities = [item for item in entities if _should_keep_strategic_entity(item, strategic_org_name)]
    output_data["entities"] = filtered_entities
    output_data["contacts"] = filtered_entities
    output_data["extracted"] = filtered_entities
    output_data["entity_count"] = len(filtered_entities)

    people = list(output_data.get("people") or [])
    if people:
        output_data["people"] = [
            item
            for item in people
            if _should_keep_strategic_entity(
                {
                    "canonical_name": item.get("canonical_name") or item.get("name"),
                    "target_type": "person",
                    "current_role": item.get("current_role") or item.get("role"),
                    "current_employer": item.get("current_employer") or item.get("employer"),
                    "evidence": item.get("evidence"),
                },
                strategic_org_name,
            )
        ]

    organisations = list(output_data.get("organisations") or [])
    if organisations:
        output_data["organisations"] = [
            item
            for item in organisations
            if _should_keep_strategic_entity(
                {
                    "canonical_name": item.get("canonical_name") or item.get("name"),
                    "target_type": "organisation",
                    "evidence": item.get("evidence") or item.get("notes"),
                },
                strategic_org_name,
            )
        ]


class IntelNoteProcessor:
    def __init__(self, extractor: Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Optional[Path]]]):
        self.extractor = extractor

    def process(self, payload: Dict[str, Any], message_kind: str) -> Tuple[Dict[str, Any], Optional[Path], Dict[str, Any]]:
        output_data, output_file = self.extractor(payload)
        attachments = list(output_data.get("attachments") or [])
        analysis: Dict[str, Any] = {"message_kind": message_kind}
        extraction_depth = str(payload.get("extraction_depth") or "default").strip().lower() or "default"
        analysis["extraction_depth"] = extraction_depth
        if isinstance(output_data.get("email_triage"), dict):
            analysis["email_triage"] = dict(output_data.get("email_triage") or {})
        if message_kind == "document_analysis":
            analysis["org_chart"] = analyse_org_chart_attachments(attachments)
            analysis["strategic_doc"] = analyse_strategic_documents(
                attachments,
                str(output_data.get("summary") or ""),
                subject=str(payload.get("subject") or ""),
                raw_text=str(payload.get("raw_text") or ""),
                extraction_depth=extraction_depth,
            )
            _merge_strategic_leadership_entities(output_data, analysis)
            _filter_strategic_document_entities(output_data, analysis)
            _apply_strategic_summary(output_data, analysis)
        return output_data, output_file, analysis
