from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

from cortex_engine.handoff_contract import validate_included_study_extract_input
from cortex_engine.included_study_extractor import (
    included_study_extractor_available,
    run_included_study_table_extractor,
)
from cortex_engine.included_study_handoff import (
    build_included_study_research_payload,
    build_included_study_research_queue_job,
    build_included_study_website_payload,
    enrich_included_study_tables_with_bibliography,
    included_study_editor_rows,
    included_study_rows_to_xlsx_bytes,
)
from cortex_engine.included_study_slicer import slice_review_pdf


def _extract_table_with_fallback(
    *,
    pdf_path: str,
    bibliography_text: str,
    provider: str,
    model: str,
    review_title: str,
    table_label: str,
    table_title: str,
    table_kind: str,
    extraction_scope: str,
    output_detail: str,
    fallback_provider: str,
    fallback_model: str,
) -> Dict[str, any]:
    providers_to_try: List[tuple[str, str]] = []
    primary = (str(provider or "").strip().lower(), str(model or "").strip())
    fallback = (str(fallback_provider or "").strip().lower(), str(fallback_model or "").strip())
    if primary[0]:
        providers_to_try.append(primary)
    if fallback[0] and fallback[0] != primary[0]:
        providers_to_try.append(fallback)

    last_error: Exception | None = None
    for provider_name, model_name in providers_to_try:
        if not included_study_extractor_available(provider_name):
            last_error = RuntimeError(f"Included-study extractor provider unavailable: {provider_name}")
            continue
        try:
            return run_included_study_table_extractor(
                pdf_path=pdf_path,
                bibliography_text=bibliography_text,
                provider=provider_name,
                model=model_name,
                review_title=review_title,
                table_label=table_label,
                table_title=table_title,
                table_kind=table_kind,
                extraction_scope=extraction_scope,
                output_detail=output_detail,
            )
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("No included-study extractor provider configured")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _zip_directory(source_dir: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(source_dir.rglob("*")):
            if not item.is_file():
                continue
            zf.write(item, item.relative_to(source_dir))
    return zip_path


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    payload = validate_included_study_extract_input(input_data or {})
    if not input_path or not input_path.exists():
        raise ValueError("included_study_extract requires a downloaded PDF input file")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError("included_study_extract requires a PDF input file")

    if progress_cb:
        progress_cb(5, "Slicing review PDF into per-table artifacts", "slice")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before slicing started")

    work_dir = input_path.parent / "included_study_extract"
    work_dir.mkdir(parents=True, exist_ok=True)
    slice_result = slice_review_pdf(str(input_path), work_dir=str(work_dir / "slices"))

    table_slices = list(slice_result.get("table_slices") or [])
    if not bool(payload.get("include_low_value_tables", False)):
        table_slices = [item for item in table_slices if str(item.get("kind") or "").strip().lower() != "hta"]
    if not table_slices:
        raise RuntimeError("No included-study table slices were detected in the review PDF")

    artifacts_root = work_dir / "artifacts"
    tables_root = artifacts_root / "tables"
    bibliography_root = artifacts_root / "bibliography"
    bibliography_root.mkdir(parents=True, exist_ok=True)
    bibliography_txt = Path(str(slice_result.get("bibliography_txt_path") or "")).read_text(encoding="utf-8", errors="ignore")
    _write_text(bibliography_root / "bibliography.txt", bibliography_txt)
    bibliography_csv_path = str(slice_result.get("bibliography_csv_path") or "").strip()
    if bibliography_csv_path and Path(bibliography_csv_path).exists():
        shutil.copy2(bibliography_csv_path, bibliography_root / "bibliography.csv")

    provider = str(payload.get("provider") or "anthropic").strip().lower()
    model = str(payload.get("model") or "").strip()
    fallback_provider = str(payload.get("fallback_provider") or "").strip().lower()
    fallback_model = str(payload.get("fallback_model") or "").strip()
    review_title = str(payload.get("review_title") or input_path.stem).strip() or input_path.stem
    extraction_scope = str(payload.get("extraction_scope") or "all_trials").strip()
    output_detail = str(payload.get("output_detail") or "reference_map").strip()
    resolver_defaults = dict(payload.get("resolver_defaults") or {})
    check_open_access = bool(resolver_defaults.get("check_open_access", True))
    enrich_sjr = bool(resolver_defaults.get("enrich_sjr", True))
    unpaywall_email = str(resolver_defaults.get("unpaywall_email") or "").strip()

    table_summaries: List[Dict[str, object]] = []
    combined_rows: List[Dict[str, object]] = []
    combined_tables: List[Dict[str, object]] = []
    warnings: List[str] = []
    successful_tables = 0
    bibliography_entries = list(slice_result.get("bibliography_entries") or [])

    total = max(1, len(table_slices))
    for idx, table_slice in enumerate(table_slices, start=1):
        if is_cancelled_cb and is_cancelled_cb():
            raise RuntimeError("Cancelled during included-study extraction")

        label = str(table_slice.get("label") or f"table {idx}").strip()
        table_number = str(table_slice.get("table_number") or "").strip() or str(idx)
        table_title = str(table_slice.get("table_title") or "").strip()
        table_kind = str(table_slice.get("kind") or "").strip()
        if progress_cb:
            pct = 10 + int((idx - 1) / total * 70)
            progress_cb(pct, f"Extracting {label} with {provider} / {model}", "extract")

        extraction = _extract_table_with_fallback(
            pdf_path=str(table_slice.get("pdf_path") or ""),
            bibliography_text=str(slice_result.get("bibliography_text") or ""),
            provider=provider,
            model=model,
            review_title=review_title,
            table_label=label,
            table_title=table_title,
            table_kind=table_kind,
            extraction_scope=extraction_scope,
            output_detail=output_detail,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )

        extraction_tables = enrich_included_study_tables_with_bibliography(
            list(extraction.get("tables") or []),
            bibliography_entries,
        )
        if extraction_tables:
            successful_tables += 1
        for warning in list(extraction.get("warnings") or []):
            warnings.append(f"{label}: {warning}")

        slice_dir = tables_root / f"table_{table_number}"
        slice_dir.mkdir(parents=True, exist_ok=True)
        source_pdf_path = Path(str(table_slice.get("pdf_path") or ""))
        if source_pdf_path.exists():
            shutil.copy2(source_pdf_path, slice_dir / source_pdf_path.name)
        _write_text(slice_dir / "extraction_raw_output.txt", str(extraction.get("raw_response") or ""))
        _write_text(slice_dir / "extraction.json", json.dumps(extraction, indent=2))

        editor_rows = included_study_editor_rows(extraction_tables)
        combined_rows.extend(editor_rows)
        combined_tables.extend(extraction_tables)

        if editor_rows:
            import pandas as pd

            (slice_dir / "included_study_selection.csv").write_text(
                pd.DataFrame(editor_rows).to_csv(index=False),
                encoding="utf-8",
            )
            (slice_dir / "included_study_selection.xlsx").write_bytes(
                included_study_rows_to_xlsx_bytes(editor_rows, sheet_name=f"table_{table_number}")
            )
        else:
            warnings.append(f"{label}: no citations were extracted; skipped resolver payload generation")

        resolver_payload = None
        resolver_queue_job = None
        website_handoff = None
        selected_rows = [row for row in editor_rows if bool(row.get("keep", True))]
        if selected_rows:
            resolver_payload = build_included_study_research_payload(
                editor_rows,
                check_open_access=check_open_access,
                enrich_sjr=enrich_sjr,
                unpaywall_email=unpaywall_email,
                extraction_scope=extraction_scope,
                output_detail=output_detail,
                focus_label=label,
            )
            resolver_queue_job = build_included_study_research_queue_job(
                editor_rows,
                check_open_access=check_open_access,
                enrich_sjr=enrich_sjr,
                unpaywall_email=unpaywall_email,
                extraction_scope=extraction_scope,
                output_detail=output_detail,
                focus_label=label,
            )
            website_handoff = build_included_study_website_payload(
                editor_rows,
                extraction_scope=extraction_scope,
                output_detail=output_detail,
                focus_label=label,
                resolver_payload=resolver_payload,
                resolver_queue_job=resolver_queue_job,
            )
            _write_text(slice_dir / "research_resolver_payload.json", json.dumps(resolver_payload, indent=2))
            _write_text(slice_dir / "research_resolver_queue_job.json", json.dumps(resolver_queue_job, indent=2))
            _write_text(slice_dir / "included_study_website_handoff.json", json.dumps(website_handoff, indent=2))
        elif editor_rows:
            warnings.append(f"{label}: extracted citations were all marked needs_review; skipped resolver payload generation")

        artifacts = {
            "table_pdf": f"tables/table_{table_number}/{source_pdf_path.name}" if source_pdf_path.exists() else "",
            "raw_output_txt": f"tables/table_{table_number}/extraction_raw_output.txt",
        }
        if editor_rows:
            artifacts.update(
                {
                    "selection_csv": f"tables/table_{table_number}/included_study_selection.csv",
                    "selection_xlsx": f"tables/table_{table_number}/included_study_selection.xlsx",
                }
            )
        if resolver_payload is not None:
            artifacts["resolver_payload_json"] = f"tables/table_{table_number}/research_resolver_payload.json"
        if resolver_queue_job is not None:
            artifacts["resolver_queue_job_json"] = f"tables/table_{table_number}/research_resolver_queue_job.json"
        if website_handoff is not None:
            artifacts["website_handoff_json"] = f"tables/table_{table_number}/included_study_website_handoff.json"
        table_summaries.append(
            {
                "table_label": label,
                "table_number": table_number,
                "table_title": table_title,
                "kind": table_kind,
                "group_count": sum(len(list(item.get("groups") or [])) for item in extraction_tables),
                "citation_count": len(editor_rows),
                "selected_citation_count": len(selected_rows),
                "needs_review_count": sum(1 for row in editor_rows if str(row.get("needs_review") or "").strip().lower() == "yes"),
                "page_numbers": list(table_slice.get("page_numbers") or []),
                "warnings": list(extraction.get("warnings") or []),
                "artifacts": artifacts,
            }
        )

    if successful_tables == 0:
        raise RuntimeError("Included-study extraction completed, but no structured tables were parsed")

    combined_root = artifacts_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)
    combined_selected_rows = [row for row in combined_rows if bool(row.get("keep", True))]
    if combined_selected_rows:
        combined_resolver_payload = build_included_study_research_payload(
            combined_rows,
            check_open_access=check_open_access,
            enrich_sjr=enrich_sjr,
            unpaywall_email=unpaywall_email,
            extraction_scope=extraction_scope,
            output_detail=output_detail,
            focus_label="combined",
        )
        combined_resolver_job = build_included_study_research_queue_job(
            combined_rows,
            check_open_access=check_open_access,
            enrich_sjr=enrich_sjr,
            unpaywall_email=unpaywall_email,
            extraction_scope=extraction_scope,
            output_detail=output_detail,
            focus_label="combined",
        )
        combined_website_handoff = build_included_study_website_payload(
            combined_rows,
            extraction_scope=extraction_scope,
            output_detail=output_detail,
            focus_label="combined",
            resolver_payload=combined_resolver_payload,
            resolver_queue_job=combined_resolver_job,
        )
        _write_text(combined_root / "research_resolver_payload.json", json.dumps(combined_resolver_payload, indent=2))
        _write_text(combined_root / "research_resolver_queue_job.json", json.dumps(combined_resolver_job, indent=2))
        _write_text(combined_root / "included_study_website_handoff.json", json.dumps(combined_website_handoff, indent=2))
    elif combined_rows:
        warnings.append("Combined extracted citations were all marked needs_review; skipped combined resolver payload generation")
    else:
        warnings.append("No combined citations were available for resolver payload generation")
    _write_text(
        artifacts_root / "manifest.json",
        json.dumps(
            {
                "review_title": review_title,
                "input_filename": input_path.name,
                "extraction_scope": extraction_scope,
                "output_detail": output_detail,
                "provider": provider,
                "model": model,
                "table_count": len(table_summaries),
                "warnings": warnings,
                "tables": table_summaries,
            },
            indent=2,
        ),
    )

    zip_path = _zip_directory(artifacts_root, work_dir / "included_study_extract_bundle.zip")
    if progress_cb:
        progress_cb(100, f"Completed {len(table_summaries)} included-study table exports", "done")

    output_data = {
        "status": "completed",
        "source_workflow": "included_study_extractor",
        "provider": provider,
        "model": model,
        "review_title": review_title,
        "input_filename": input_path.name,
        "extraction_scope": extraction_scope,
        "output_detail": output_detail,
        "bibliography_entry_count": len(list(slice_result.get("bibliography_entries") or [])),
        "table_count": len(table_summaries),
        "tables": table_summaries,
        "warnings": warnings,
        "result_bundle_name": zip_path.name,
    }
    return {"output_data": output_data, "output_file": zip_path}
