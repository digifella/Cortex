from __future__ import annotations

import copy
import io
import json
import zipfile
from typing import Any, Dict, List

from cortex_engine.handoff_contract import normalize_handoff_metadata, validate_research_resolve_input


def included_study_group_label(group: Dict[str, Any]) -> str:
    group_label = str(group.get("group_label") or "").strip()
    trial_label = str(group.get("trial_label") or "").strip()
    if group_label and trial_label:
        return f"{group_label} / {trial_label}"
    return group_label or trial_label


def _row_keep_selected(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    return normalized in {"true", "1", "yes", "y", "on"}


def enrich_included_study_tables_with_bibliography(
    tables: List[Dict[str, Any]], bibliography_entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    numbered_entries = {
        str(entry.get("reference_number") or "").strip(): dict(entry)
        for entry in bibliography_entries or []
        if str(entry.get("reference_number") or "").strip()
    }
    if not numbered_entries:
        return [copy.deepcopy(table) for table in tables or []]

    enriched_tables: List[Dict[str, Any]] = []
    for table in tables or []:
        table_copy = copy.deepcopy(table)
        for group in list(table_copy.get("groups") or []):
            for citation in list(group.get("citations") or []):
                reference_number = str(citation.get("reference_number") or "").strip()
                if not reference_number:
                    continue
                entry = numbered_entries.get(reference_number)
                if not entry:
                    continue
                entry_title = str(entry.get("title") or "").strip()
                entry_authors = str(entry.get("authors") or "").strip()
                entry_year = str(entry.get("year") or "").strip()
                entry_journal = str(entry.get("journal") or "").strip()
                entry_doi = str(entry.get("doi") or "").strip()
                entry_text = str(entry.get("entry_text") or "").strip()

                if entry_title:
                    citation["resolved_title"] = entry_title
                if entry_authors:
                    citation["resolved_authors"] = entry_authors
                if entry_year:
                    citation["resolved_year"] = entry_year
                    if not str(citation.get("year") or "").strip():
                        citation["year"] = entry_year
                if entry_journal:
                    citation["resolved_journal"] = entry_journal
                if entry_doi:
                    citation["resolved_doi"] = entry_doi
                if entry_authors and not str(citation.get("authors") or "").strip():
                    citation["authors"] = entry_authors
                citation["bibliography_entry_text"] = entry_text
                citation["bibliography_match_method"] = "reference_number"
        enriched_tables.append(table_copy)
    return enriched_tables


def included_study_editor_rows(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    row_id = 1
    for table in tables or []:
        table_number = str(table.get("table_number") or "").strip()
        table_title = str(table.get("table_title") or "").strip()
        grouping_basis = str(table.get("grouping_basis") or "").strip()
        for group in list(table.get("groups") or []):
            combined_group = included_study_group_label(group)
            group_notes = str(group.get("notes") or "").strip()
            for citation in list(group.get("citations") or []):
                resolved_title = str(citation.get("resolved_title") or "").strip()
                display = str(citation.get("display") or "").strip()
                resolved_authors = str(citation.get("resolved_authors") or citation.get("authors") or "").strip()
                resolved_year = str(citation.get("resolved_year") or citation.get("year") or "").strip()
                resolved_journal = str(citation.get("resolved_journal") or "").strip()
                resolved_doi = str(citation.get("resolved_doi") or "").strip()
                notes = str(citation.get("notes") or group_notes).strip()
                needs_review = bool(citation.get("needs_review"))
                rows.append(
                    {
                        "keep": not needs_review,
                        "row_id": row_id,
                        "table_number": table_number,
                        "table_title": table_title,
                        "grouping_basis": grouping_basis,
                        "group_label": str(group.get("group_label") or "").strip(),
                        "trial_label": str(group.get("trial_label") or "").strip(),
                        "combined_group": combined_group,
                        "citation_display": display,
                        "title": resolved_title or display,
                        "authors": resolved_authors,
                        "year": resolved_year,
                        "doi": resolved_doi,
                        "journal": resolved_journal,
                        "reference_number": str(citation.get("reference_number") or "").strip(),
                        "study_design": str(citation.get("study_design") or "").strip(),
                        "sample_size": str(citation.get("sample_size") or "").strip(),
                        "outcome_measure": str(citation.get("outcome_measure") or "").strip(),
                        "outcome_result": str(citation.get("outcome_result") or "").strip(),
                        "bibliography_entry_text": str(citation.get("bibliography_entry_text") or "").strip(),
                        "bibliography_match_method": str(citation.get("bibliography_match_method") or "").strip(),
                        "notes": notes,
                        "needs_review": "yes" if needs_review else "",
                    }
                )
                row_id += 1
    return rows


def merge_included_study_editor_rows(editor_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for row in editor_rows or []:
        if not _row_keep_selected(row.get("keep", True)):
            continue
        merged.append(
            {
                "row_id": int(row.get("row_id") or 0) or len(merged) + 1,
                "title": str(row.get("title") or row.get("citation_display") or "").strip(),
                "authors": str(row.get("authors") or "").strip(),
                "year": str(row.get("year") or "").strip(),
                "doi": str(row.get("doi") or "").strip(),
                "journal": str(row.get("journal") or "").strip(),
                "notes": str(row.get("notes") or "").strip(),
                "extra_fields": {
                    "table_number": str(row.get("table_number") or "").strip(),
                    "table_title": str(row.get("table_title") or "").strip(),
                    "grouping_basis": str(row.get("grouping_basis") or "").strip(),
                    "group_label": str(row.get("group_label") or "").strip(),
                    "trial_label": str(row.get("trial_label") or "").strip(),
                    "combined_group": str(row.get("combined_group") or "").strip(),
                    "reference_number": str(row.get("reference_number") or "").strip(),
                    "citation_display": str(row.get("citation_display") or "").strip(),
                    "needs_review": str(row.get("needs_review") or "").strip(),
                    "study_design": str(row.get("study_design") or "").strip(),
                    "sample_size": str(row.get("sample_size") or "").strip(),
                    "outcome_measure": str(row.get("outcome_measure") or "").strip(),
                    "outcome_result": str(row.get("outcome_result") or "").strip(),
                    "bibliography_entry_text": str(row.get("bibliography_entry_text") or "").strip(),
                    "bibliography_match_method": str(row.get("bibliography_match_method") or "").strip(),
                },
            }
        )
    return merged


def build_included_study_research_payload(
    editor_rows: List[Dict[str, Any]],
    *,
    check_open_access: bool,
    enrich_sjr: bool,
    unpaywall_email: str,
    extraction_scope: str,
    output_detail: str,
    focus_label: str = "",
) -> Dict[str, Any]:
    citations = merge_included_study_editor_rows(editor_rows)
    payload = validate_research_resolve_input(
        {
            "citations": citations,
            "options": {
                "check_open_access": bool(check_open_access),
                "enrich_sjr": bool(enrich_sjr),
                "unpaywall_email": str(unpaywall_email or "").strip(),
            },
        }
    )
    payload["source_workflow"] = "included_study_extractor"
    payload["included_study_context"] = {
        "extraction_scope": str(extraction_scope or "").strip(),
        "output_detail": str(output_detail or "").strip(),
        "focused_table_label": str(focus_label or "").strip(),
    }
    return payload


def build_included_study_research_queue_job(
    editor_rows: List[Dict[str, Any]],
    *,
    check_open_access: bool,
    enrich_sjr: bool,
    unpaywall_email: str,
    extraction_scope: str,
    output_detail: str,
    focus_label: str = "",
) -> Dict[str, Any]:
    input_data = build_included_study_research_payload(
        editor_rows,
        check_open_access=check_open_access,
        enrich_sjr=enrich_sjr,
        unpaywall_email=unpaywall_email,
        extraction_scope=extraction_scope,
        output_detail=output_detail,
        focus_label=focus_label,
    )
    metadata = normalize_handoff_metadata(
        input_data={
            "source_system": "cortex_streamlit",
            "project_id": "included_study_extractor",
        }
    )
    return {
        **metadata,
        "job_type": "research_resolve",
        "job_label": f"Included Study Resolver - {str(focus_label or 'selected tables').strip()}",
        "source_workflow": "included_study_extractor",
        "input_data": validate_research_resolve_input(input_data),
    }


def build_included_study_website_payload(
    editor_rows: List[Dict[str, Any]],
    *,
    extraction_scope: str,
    output_detail: str,
    focus_label: str = "",
    resolver_payload: Dict[str, Any] | None = None,
    resolver_queue_job: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected_rows = [dict(row) for row in editor_rows or [] if bool(row.get("keep", True))]
    tables: List[Dict[str, Any]] = []
    table_lookup: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    for row in selected_rows:
        table_number = str(row.get("table_number") or "").strip()
        table_title = str(row.get("table_title") or "").strip()
        grouping_basis = str(row.get("grouping_basis") or "").strip()
        table_key = (table_number, table_title, grouping_basis)
        table_entry = table_lookup.get(table_key)
        if table_entry is None:
            table_entry = {
                "table_number": table_number,
                "table_title": table_title,
                "grouping_basis": grouping_basis,
                "groups": [],
            }
            table_entry["_group_lookup"] = {}
            table_lookup[table_key] = table_entry
            tables.append(table_entry)

        group_label = str(row.get("group_label") or "").strip()
        trial_label = str(row.get("trial_label") or "").strip()
        combined_group = str(row.get("combined_group") or "").strip()
        group_key = (group_label, trial_label, combined_group)
        group_lookup = table_entry["_group_lookup"]
        group_entry = group_lookup.get(group_key)
        if group_entry is None:
            group_entry = {
                "group_label": group_label,
                "trial_label": trial_label,
                "combined_group": combined_group,
                "citations": [],
            }
            group_lookup[group_key] = group_entry
            table_entry["groups"].append(group_entry)

        group_entry["citations"].append(
            {
                "citation_display": str(row.get("citation_display") or "").strip(),
                "title": str(row.get("title") or "").strip(),
                "authors": str(row.get("authors") or "").strip(),
                "year": str(row.get("year") or "").strip(),
                "doi": str(row.get("doi") or "").strip(),
                "journal": str(row.get("journal") or "").strip(),
                "reference_number": str(row.get("reference_number") or "").strip(),
                "notes": str(row.get("notes") or "").strip(),
                "needs_review": str(row.get("needs_review") or "").strip().lower() == "yes",
                "study_design": str(row.get("study_design") or "").strip(),
                "sample_size": str(row.get("sample_size") or "").strip(),
                "outcome_measure": str(row.get("outcome_measure") or "").strip(),
                "outcome_result": str(row.get("outcome_result") or "").strip(),
            }
        )

    for table_entry in tables:
        table_entry.pop("_group_lookup", None)

    payload = {
        "action": "included_study_extract_handoff",
        "source_workflow": "included_study_extractor",
        "included_study_context": {
            "extraction_scope": str(extraction_scope or "").strip(),
            "output_detail": str(output_detail or "").strip(),
            "focused_table_label": str(focus_label or "").strip(),
        },
        "selection_summary": {
            "selected_paper_count": len(selected_rows),
            "table_count": len(tables),
            "group_count": sum(len(list(item.get("groups") or [])) for item in tables),
        },
        "tables": tables,
    }
    if resolver_payload:
        payload["resolver_payload"] = dict(resolver_payload)
    if resolver_queue_job:
        payload["resolver_queue_job"] = dict(resolver_queue_job)
    return payload


def included_study_rows_to_xlsx_bytes(rows: List[Dict[str, Any]], sheet_name: str = "included_study") -> bytes:
    import pandas as pd

    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine="openpyxl") as writer:
        pd.DataFrame(rows or []).to_excel(writer, index=False, sheet_name=str(sheet_name or "included_study")[:31])
    return mem.getvalue()


def build_included_study_handoff_bundle_bytes(
    editor_rows: List[Dict[str, Any]],
    *,
    extraction_scope: str,
    output_detail: str,
    focus_label: str = "",
    resolver_payload: Dict[str, Any],
) -> bytes:
    import pandas as pd

    resolver_queue_job = build_included_study_research_queue_job(
        editor_rows,
        check_open_access=bool(dict(resolver_payload or {}).get("options", {}).get("check_open_access", True)),
        enrich_sjr=bool(dict(resolver_payload or {}).get("options", {}).get("enrich_sjr", True)),
        unpaywall_email=str(dict(resolver_payload or {}).get("options", {}).get("unpaywall_email", "") or "").strip(),
        extraction_scope=extraction_scope,
        output_detail=output_detail,
        focus_label=focus_label,
    )
    website_payload = build_included_study_website_payload(
        editor_rows,
        extraction_scope=extraction_scope,
        output_detail=output_detail,
        focus_label=focus_label,
        resolver_payload=resolver_payload,
        resolver_queue_job=resolver_queue_job,
    )
    selected_rows = [dict(row) for row in editor_rows or [] if bool(row.get("keep", True))]
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("included_study_selection.csv", pd.DataFrame(editor_rows or []).to_csv(index=False))
        zf.writestr("included_study_selected_rows.csv", pd.DataFrame(selected_rows or []).to_csv(index=False))
        zf.writestr("research_resolver_payload.json", json.dumps(resolver_payload, indent=2))
        zf.writestr("research_resolver_queue_job.json", json.dumps(resolver_queue_job, indent=2))
        zf.writestr("included_study_website_handoff.json", json.dumps(website_payload, indent=2))
    return mem.getvalue()
