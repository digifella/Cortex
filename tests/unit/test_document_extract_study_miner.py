from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


def _load_document_extract_module():
    path = Path("/home/longboardfella/cortex_suite/pages/7_Document_Extract.py")
    spec = importlib.util.spec_from_file_location("doc_extract_page", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_structural_candidate_detection_handles_repeated_header_tokens():
    module = _load_document_extract_module()

    item = {
        "title": "Region Study Author year",
        "raw_citation": "Region Study Author year",
        "extra_fields": {"study": "", "author_year": ""},
    }

    assert module._is_structural_study_miner_candidate(item) is True


def test_export_rows_drop_structural_entries_and_merge_duplicate_reference_rows():
    module = _load_document_extract_module()

    candidates = [
        {
            "row_id": 1,
            "title": "Region Study Author year",
            "raw_citation": "Region Study Author year",
            "extra_fields": {"study": "", "author_year": ""},
        },
        {
            "row_id": 2,
            "title": "Patrick 2021 [17]",
            "authors": "Patrick",
            "year": "2021",
            "reference_number": "17",
            "reference_match_method": "table_inline_reference",
            "source_section": "local_rescue",
            "raw_citation": "Patrick 2021 [17]",
            "raw_excerpt": "first pass",
            "extra_fields": {
                "region": "US",
                "study": "TRANSCEND",
                "author_year": "Patrick 2021 [17]",
                "assessment_method": "EORTC QLQ-C30 global health status",
                "table_index": 1,
                "table_label": "table 1",
                "reference_number": "17",
            },
        },
        {
            "row_id": 3,
            "title": "Patrick 2021",
            "authors": "Patrick",
            "year": "2021",
            "reference_number": "17",
            "reference_match_method": "local_reconciliation",
            "source_section": "local_reconciliation",
            "raw_citation": "Patrick 2021",
            "raw_excerpt": "reconciliation pass",
            "extra_fields": {
                "region": "US",
                "study": "TRANSCEND",
                "author_year": "Patrick 2021",
                "treatment": "Liso-cel",
                "table_index": 1,
                "table_label": "table 1",
                "reference_number": "17",
            },
        },
    ]

    editor_rows = [
        {
            "keep": False,
            "row_id": 1,
            "source_review_title": "test",
            "table_group": "",
            "table_citation": "",
            "title": "Region Study Author year",
            "reference_number": "",
            "reference_match_method": "",
            "reference_link": "",
        },
        {
            "keep": False,
            "row_id": 2,
            "source_review_title": "test",
            "table_index": 1,
            "table_label": "table 1",
            "table_group": "US / TRANSCEND",
            "table_citation": "Patrick 2021 [17]",
            "title": "Patrick 2021 [17]",
            "authors": "Patrick",
            "year": "2021",
            "reference_number": "17",
            "reference_match_method": "table_inline_reference",
            "reference_link": "ref 17 (table_inline_reference)",
            "source_section": "local_rescue",
        },
        {
            "keep": False,
            "row_id": 3,
            "source_review_title": "test",
            "table_index": 1,
            "table_label": "table 1",
            "table_group": "US / TRANSCEND",
            "table_citation": "Patrick 2021",
            "title": "Patrick 2021",
            "authors": "Patrick",
            "year": "2021",
            "reference_number": "17",
            "reference_match_method": "local_reconciliation",
            "reference_link": "ref 17 (local_reconciliation)",
            "source_section": "local_reconciliation",
        },
    ]

    exported = module._study_miner_export_rows(editor_rows, candidates)

    assert len(exported) == 1
    assert exported[0]["table_group"] == "US / TRANSCEND"
    assert exported[0]["reference_number"] == "17"
    assert "local_rescue" in exported[0]["source_section"]
    assert exported[0]["table_index"] == 1


def test_export_rows_keep_same_reference_separate_across_table_slices():
    module = _load_document_extract_module()

    candidates = [
        {
            "row_id": 1,
            "title": "Patrick 2021 [17]",
            "authors": "Patrick",
            "year": "2021",
            "reference_number": "17",
            "reference_match_method": "table_inline_reference",
            "source_section": "local_rescue",
            "raw_citation": "Patrick 2021 [17]",
            "raw_excerpt": "table 1",
            "extra_fields": {
                "region": "US",
                "study": "TRANSCEND",
                "author_year": "Patrick 2021 [17]",
                "table_index": 1,
                "table_label": "table 1",
                "reference_number": "17",
            },
        },
        {
            "row_id": 2,
            "title": "Patrick 2021 [17]",
            "authors": "Patrick",
            "year": "2021",
            "reference_number": "17",
            "reference_match_method": "table_inline_reference",
            "source_section": "local_rescue",
            "raw_citation": "Patrick 2021 [17]",
            "raw_excerpt": "table 2",
            "extra_fields": {
                "region": "US",
                "study": "CEA",
                "author_year": "Patrick 2021 [17]",
                "table_index": 2,
                "table_label": "table 2",
                "reference_number": "17",
            },
        },
    ]

    editor_rows = [
        {
            "keep": True,
            "row_id": 1,
            "source_review_title": "test",
            "table_index": 1,
            "table_label": "table 1",
            "table_group": "US / TRANSCEND",
            "table_citation": "Patrick 2021 [17]",
            "title": "Patrick 2021 [17]",
            "authors": "Patrick",
            "year": "2021",
            "reference_number": "17",
            "reference_match_method": "table_inline_reference",
            "reference_link": "ref 17 (table_inline_reference)",
            "source_section": "local_rescue",
        },
        {
            "keep": True,
            "row_id": 2,
            "source_review_title": "test",
            "table_index": 2,
            "table_label": "table 2",
            "table_group": "US / CEA",
            "table_citation": "Patrick 2021 [17]",
            "title": "Patrick 2021 [17]",
            "authors": "Patrick",
            "year": "2021",
            "reference_number": "17",
            "reference_match_method": "table_inline_reference",
            "reference_link": "ref 17 (table_inline_reference)",
            "source_section": "local_rescue",
        },
    ]

    exported = module._study_miner_export_rows(editor_rows, candidates)

    assert len(exported) == 2
    assert sorted(row["table_index"] for row in exported) == [1, 2]


def test_study_miner_table_slices_split_blocks_by_table_index():
    module = _load_document_extract_module()

    document = {
        "table_blocks": [
            {"table_index": 1, "header": ["Study"], "context_text": "Table 3"},
            {"table_index": 2, "header": ["Country"], "context_text": "Table 4"},
        ],
        "table_snapshots": [
            {"table_index": 1, "page_number": 1},
            {"table_index": 1, "page_number": 2},
            {"table_index": 2, "page_number": 3},
        ],
    }

    slices = module._study_miner_table_slices(document)

    assert len(slices) == 2
    assert slices[0]["label"] == "table 3"
    assert [item["page_number"] for item in slices[0]["table_snapshots"]] == [1, 2]
    assert slices[1]["label"] == "table 4"
    assert [item["page_number"] for item in slices[1]["table_snapshots"]] == [3]


def test_study_miner_table_slices_merge_adjacent_blocks_into_table_families():
    module = _load_document_extract_module()

    document = {
        "table_blocks": [
            {
                "table_index": 1,
                "header": ["Study", "Author year", "Utility"],
                "context_before": "Table 3 (continued)",
                "context_text": "After table:\nTable 3 (continued)",
            },
            {
                "table_index": 2,
                "header": ["Baseline mean", "Mean change", "Utility"],
                "context_before": "Table 3 (continued)",
                "context_text": "Before table:\nTable 3 (continued)",
            },
            {
                "table_index": 3,
                "header": ["Utility values in remission", "Utility values on-treatment"],
                "context_before": "Table",
                "context_text": "",
            },
            {
                "table_index": 4,
                "header": ["Utility values in remission", "Utility values on-treatment"],
                "context_before": "Table 4 Overview of economic studies",
                "context_text": "Table 4 Overview of economic studies reporting health state utility values",
            },
        ],
        "table_snapshots": [],
    }

    slices = module._study_miner_table_slices(document)

    assert len(slices) == 2
    assert [item["table_index"] for item in slices] == [3, 4]
    assert slices[0]["source_block_indices"] == [1, 2]
    assert slices[1]["source_block_indices"] == [3, 4]


def test_estimate_slice_page_numbers_splits_pdf_across_table_families(tmp_path):
    module = _load_document_extract_module()

    import fitz

    pdf_path = tmp_path / "tables.pdf"
    doc = fitz.open()
    for _ in range(4):
        page = doc.new_page()
        page.insert_text((72, 72), "table page")
    doc.save(pdf_path)
    doc.close()

    slices = [
        {"label": "table 3", "table_index": 3, "block_count": 2, "table_blocks": [{}, {}], "table_snapshots": []},
        {"label": "table 4", "table_index": 4, "block_count": 2, "table_blocks": [{}, {}], "table_snapshots": []},
    ]

    estimated = module._estimate_slice_page_numbers(str(pdf_path), slices)

    assert estimated[0]["heuristic_page_numbers"] == [1, 2]
    assert estimated[1]["heuristic_page_numbers"] == [3, 4]


def test_slice_has_explicit_table_context_requires_matching_table_number():
    module = _load_document_extract_module()

    good_slice = {
        "table_index": 4,
        "table_blocks": [
            {
                "context_before": "Table 4 Overview of economic studies",
                "context_text": "Table 4 (continued)",
                "context_after": "",
            }
        ],
    }
    bad_slice = {
        "table_index": 5,
        "table_blocks": [
            {
                "context_before": "Overall, the evidence synthesized in this review...",
                "context_text": "Source of utility values",
                "context_after": "",
            }
        ],
    }

    assert module._slice_has_explicit_table_context(good_slice) is True
    assert module._slice_has_explicit_table_context(bad_slice) is False


def test_filter_base_table_candidates_after_local_rescue_drops_original_table_rows():
    module = _load_document_extract_module()

    candidates = [
        {"source_review": "test.pdf", "source_section": "table", "title": "base row"},
        {"source_review": "test.pdf", "source_section": "local_rescue", "title": "rescued row"},
        {"source_review": "other.pdf", "source_section": "table", "title": "other base row"},
    ]

    filtered = module._filter_base_table_candidates_after_local_rescue(candidates, review_source_name="test.pdf")

    assert [item["title"] for item in filtered] == ["rescued row", "other base row"]


def test_relabel_outlier_group_candidates_splits_economic_rows_out_of_trial_group():
    module = _load_document_extract_module()

    candidates = [
        {
            "row_id": 1,
            "reference_number": "17",
            "raw_citation": "Patrick 2021 [17]",
            "raw_excerpt": "US | TRANSCEND NHL 001 | Patrick 2021 [17] | Liso-cel",
            "extra_fields": {
                "table_index": 3,
                "region": "US",
                "study": "TRANSCEND NHL 001",
                "author_year": "Patrick 2021 [17]",
                "treatment": "Liso-cel",
                "assessment_method": "EQ-5D-5L",
            },
        },
        {
            "row_id": 2,
            "reference_number": "36",
            "raw_citation": "Patrick 2019 [36]",
            "raw_excerpt": "US | TRANSCEND NHL 001 | Patrick 2019 [36] | Liso-cel",
            "extra_fields": {
                "table_index": 3,
                "region": "US",
                "study": "TRANSCEND NHL 001",
                "author_year": "Patrick 2019 [36]",
                "treatment": "Liso-cel",
                "assessment_method": "EQ-5D-5L",
            },
        },
        {
            "row_id": 3,
            "reference_number": "44",
            "raw_citation": "Howell 2022 [44]",
            "raw_excerpt": "UK | TRANSCEND NHL 001 | Howell 2022 [44] | hypothetical CAR T pathway | TTO",
            "extra_fields": {
                "table_index": 3,
                "region": "UK",
                "study": "TRANSCEND NHL 001",
                "author_year": "Howell 2022 [44]",
                "study_design": "Vignette-based study",
                "patient_population": "General adult population in UK",
                "treatment": "Hypothetical CAR T cell therapy pathway",
                "assessment_method": "TTO (1-year time horizon)",
            },
        },
        {
            "row_id": 4,
            "reference_number": "46",
            "raw_citation": "Orlova 2022 [46]",
            "raw_excerpt": "NR | TRANSCEND NHL 001 | Orlova 2022 [46] | Narlatuzumab entansine + rituximab",
            "extra_fields": {
                "table_index": 3,
                "region": "NR",
                "study": "TRANSCEND NHL 001",
                "author_year": "Orlova 2022 [46]",
                "study_design": "Phase 2 single-arm trial",
                "patient_population": "2L and heavily pre-treated patients with R/R DLBCL",
                "treatment": "Narlatuzumab entansine + rituximab",
                "assessment_method": "EQ-5D",
            },
        },
    ]

    relabeled = module._study_miner_relabel_outlier_group_candidates(candidates)
    by_ref = {item["reference_number"]: item for item in relabeled}

    assert by_ref["17"]["extra_fields"]["study"] == "TRANSCEND NHL 001"
    assert by_ref["36"]["extra_fields"]["study"] == "TRANSCEND NHL 001"
    assert by_ref["44"]["extra_fields"]["study"] == "TTO"
    assert by_ref["44"]["needs_review"] is True
    assert by_ref["46"]["extra_fields"]["study"] == "Orlova 2022 [46]"
    assert by_ref["46"]["needs_review"] is True


def test_reassign_candidates_to_matching_table_slices_moves_economic_rows_to_economic_slice():
    module = _load_document_extract_module()

    table_slices = [
        {
            "table_index": 3,
            "table_blocks": [
                {"context_text": "Table 3 Overview of included studies on HRQOL measures", "header": ["Author year", "Utilities", "EQ-5D"]}
            ],
        },
        {
            "table_index": 4,
            "table_blocks": [
                {"context_text": "Table 4 Overview of economic studies reporting health state utility values", "header": ["Utility values", "CUA", "CEA"]}
            ],
        },
    ]
    candidates = [
        {
            "reference_number": "45",
            "raw_excerpt": "Global | CheckMate 436 | Zinzani 2019 [45] | Nivolumab",
            "extra_fields": {
                "table_index": 3,
                "table_label": "table 3",
                "region": "Global",
                "study": "CheckMate 436",
                "author_year": "Zinzani 2019 [45]",
                "treatment": "Nivolumab + brentuximab vedotin",
                "assessment_method": "EQ-5D-3L",
            },
        },
        {
            "reference_number": "54",
            "raw_excerpt": "China | CUA | Li 2022 | Disutility of chemotherapy",
            "extra_fields": {
                "table_index": 3,
                "table_label": "table 3",
                "region": "China",
                "study": "CUA",
                "author_year": "Li 2022 [54]",
                "treatment": "Axi-cel, Salvage chemotherapy",
                "assessment_method": "Disutility of chemotherapy",
            },
        },
    ]

    reassigned = module._reassign_candidates_to_matching_table_slices(candidates, table_slices)
    by_ref = {item["reference_number"]: item for item in reassigned}

    assert by_ref["45"]["extra_fields"]["table_index"] == 3
    assert by_ref["54"]["extra_fields"]["table_index"] == 4


def test_classify_table_slice_kind_distinguishes_hrqol_from_economic_tables():
    module = _load_document_extract_module()

    hrqol_slice = {
        "table_blocks": [
            {
                "context_text": "Table 2 Overview of included studies on HRQOL measures",
                "header": ["Author year", "Utilities", "EQ-5D"],
            }
        ]
    }
    economic_slice = {
        "table_blocks": [
            {
                "context_text": "Table 4 Overview of economic studies reporting health state utility values",
                "header": ["Utility values in remission", "CUA", "CEA"],
            }
        ]
    }

    assert module._classify_table_slice_kind(hrqol_slice) == "hrqol"
    assert module._classify_table_slice_kind(economic_slice) == "economic"


def test_filter_study_miner_export_groups_hides_low_value_reference_free_tables_by_default():
    module = _load_document_extract_module()

    groups = [
        {
            "source_review_title": "review",
            "table_index": 4,
            "table_label": "table 4",
            "rows": [
                {"reference_number": "54", "table_group": "China / CUA"},
                {"reference_number": "55", "table_group": "Japan / CUA"},
            ],
        },
        {
            "source_review_title": "review",
            "table_index": 5,
            "table_label": "table 5",
            "rows": [
                {"reference_number": "", "table_group": "Juliet study"},
                {"reference_number": "", "table_group": "TRANSCEND NHL 001"},
            ],
        },
    ]

    filtered = module._filter_study_miner_export_groups(groups)

    assert len(filtered) == 1
    assert filtered[0]["table_index"] == 4
    assert filtered[0]["low_value"] is False

    included = module._filter_study_miner_export_groups(groups, include_low_value=True)
    assert len(included) == 2
    by_index = {item["table_index"]: item for item in included}
    assert by_index[4]["low_value"] is False
    assert by_index[5]["low_value"] is True


def test_study_miner_harmonize_group_labels_upgrades_weak_trial_label_from_stronger_overlap():
    module = _load_document_extract_module()

    rows = [
        {
            "source_review_title": "review",
            "region": "US",
            "study": "TRANSCE",
            "table_group": "US / TRANSCE",
            "reference_number": "17",
        },
        {
            "source_review_title": "review",
            "region": "US",
            "study": "TRANSCE",
            "table_group": "US / TRANSCE",
            "reference_number": "37",
        },
        {
            "source_review_title": "review",
            "region": "US",
            "study": "TRANSCEND NHL 001",
            "table_group": "US / TRANSCEND NHL 001",
            "reference_number": "37",
        },
    ]

    harmonized = module._study_miner_harmonize_group_labels(rows)

    weak_rows = [row for row in harmonized if row["reference_number"] in {"17", "37"}]
    assert all(row["table_group"] == "US / TRANSCEND NHL 001" for row in weak_rows)


def test_normalize_study_miner_study_label_collapses_obvious_uppercase_ocr_repeats():
    module = _load_document_extract_module()

    assert module._normalize_study_miner_study_label("ORCHARRD") == "ORCHARD"
    assert module._normalize_study_miner_study_label("JULIET") == "JULIET"


def test_build_study_miner_research_payload_normalizes_selected_candidates():
    module = _load_document_extract_module()

    payload = module._build_study_miner_research_payload(
        [
            {
                "row_id": "9",
                "title": " Example paper ",
                "authors": "A. Author",
                "year": "2021",
                "doi": "10.1000/example",
                "extra_fields": {"source_review_title": "Review A", "reference_number": "17"},
            }
        ],
        check_open_access=True,
        enrich_sjr=False,
        unpaywall_email=" person@example.com ",
    )

    assert payload["citations"][0]["row_id"] == 9
    assert payload["citations"][0]["title"] == "Example paper"
    assert payload["citations"][0]["extra_fields"]["reference_number"] == "17"
    assert payload["options"]["check_open_access"] is True
    assert payload["options"]["enrich_sjr"] is False
    assert payload["options"]["unpaywall_email"] == "person@example.com"


def test_run_study_miner_paper_retrieval_chains_resolver_and_url_ingestor(temp_dir, monkeypatch):
    module = _load_document_extract_module()

    captured = {"resolver_payload": None, "urls": None}

    def _fake_run_research_resolve(*, payload, run_dir, progress_cb=None, is_cancelled_cb=None):
        captured["resolver_payload"] = payload
        if progress_cb:
            progress_cb(20, "Resolving row 1/1", "research_resolve_lookup")
        return {
            "status": "completed",
            "resolved": [
                {
                    "row_id": 1,
                    "input_title": "Example paper",
                    "resolved_url": "https://doi.org/10.1000/example",
                    "open_access": {
                        "is_oa": True,
                        "oa_status": "gold",
                        "pdf_url": "https://example.org/paper.pdf",
                    },
                }
            ],
            "unresolved": [],
            "stats": {"total": 1, "resolved_high": 1, "resolved_low": 0, "unresolved": 0},
        }

    class _FakeURLIngestor:
        def __init__(self, output_root, timeout=25):
            self.output_root = Path(output_root)
            self.timeout = timeout
            self.output_root.mkdir(parents=True, exist_ok=True)

        def process_urls(
            self,
            urls,
            convert_to_md=False,
            use_vision_for_md=False,
            textify_options=None,
            capture_web_md_on_no_pdf=False,
            progress_cb=None,
            event_cb=None,
        ):
            captured["urls"] = list(urls)
            if progress_cb:
                progress_cb(1, len(urls), "Completed 1/1")
            if event_cb:
                event_cb("retrieved pdf")
            return [
                SimpleNamespace(
                    input_url="https://example.org/paper.pdf",
                    final_url="https://example.org/paper.pdf",
                    status="downloaded",
                    reason="",
                    pdf_path=str(self.output_root / "pdfs" / "paper.pdf"),
                    md_path=str(self.output_root / "markdown" / "paper.md"),
                    converted_to_md=True,
                    web_captured=False,
                    open_access_pdf_found=True,
                    page_title="Example paper",
                    http_code="200",
                    elapsed_seconds=1.0,
                )
            ]

        def build_reports(self, results):
            csv_path = self.output_root / "report.csv"
            json_path = self.output_root / "report.json"
            csv_path.write_text("status\n", encoding="utf-8")
            json_path.write_text("[]", encoding="utf-8")
            return csv_path, json_path

        @staticmethod
        def build_zip_bytes(results, csv_path, json_path):
            return b"zip-bytes"

    monkeypatch.setattr(module, "run_research_resolve", _fake_run_research_resolve)
    monkeypatch.setattr(module, "URLIngestor", _FakeURLIngestor)

    messages = []
    output = module._run_study_miner_paper_retrieval(
        candidates=[
            {
                "row_id": 1,
                "title": "Example paper",
                "authors": "A. Author",
                "year": "2021",
                "doi": "",
                "journal": "",
                "extra_fields": {"source_review_title": "Review A", "reference_number": "17"},
            }
        ],
        db_root=temp_dir,
        resolver_options={
            "check_open_access": True,
            "enrich_sjr": True,
            "unpaywall_email": "person@example.com",
        },
        ingest_options={
            "convert_to_md": True,
            "use_vision_for_md": False,
            "capture_web_md_on_no_pdf": True,
            "timeout_seconds": 15,
            "textify_options": {"pdf_strategy": "hybrid"},
        },
        progress_cb=messages.append,
    )

    assert captured["resolver_payload"]["citations"][0]["title"] == "Example paper"
    assert captured["urls"] == ["https://example.org/paper.pdf"]
    assert output["preferred_urls"] == ["https://example.org/paper.pdf"]
    assert output["resolver_output"]["stats"]["resolved_high"] == 1
    assert output["url_csv_path"].endswith("report.csv")
    assert output["url_json_path"].endswith("report.json")
    assert output["url_zip_bytes"] == b"zip-bytes"
    assert any("Research Resolver:" in message for message in messages)
    assert any("URL Ingestor:" in message for message in messages)


def test_included_study_editor_rows_flattens_grouped_tables_for_selection():
    module = _load_document_extract_module()

    rows = module._included_study_editor_rows(
        [
            {
                "table_number": "2",
                "table_title": "Overview of Included Studies on HRQOL Measures",
                "grouping_basis": "Grouped by instrument",
                "groups": [
                    {
                        "group_label": "EORTC QLQ-C30",
                        "trial_label": "TRANSCEND NHL 001",
                        "citations": [
                            {
                                "display": "Patrick 2021 [17]",
                                "resolved_title": "Health-related quality of life with lisocabtagene maraleucel",
                                "resolved_authors": "Patrick D",
                                "resolved_year": "2021",
                                "reference_number": "17",
                                "study_design": "Phase 1 single-arm trial",
                                "sample_size": "256",
                                "outcome_measure": "EORTC QLQ-C30 global health status",
                                "outcome_result": "19.7 (25.6)",
                                "needs_review": False,
                            }
                        ],
                    }
                ],
            }
        ]
    )

    assert len(rows) == 1
    assert rows[0]["combined_group"] == "EORTC QLQ-C30 / TRANSCEND NHL 001"
    assert rows[0]["title"] == "Health-related quality of life with lisocabtagene maraleucel"
    assert rows[0]["reference_number"] == "17"
    assert rows[0]["study_design"] == "Phase 1 single-arm trial"
    assert rows[0]["sample_size"] == "256"


def test_merge_included_study_editor_rows_builds_research_payload_rows():
    module = _load_document_extract_module()

    merged = module._merge_included_study_editor_rows(
        [
            {
                "keep": True,
                "row_id": 1,
                "table_number": "2",
                "table_title": "Overview of Included Studies on HRQOL Measures",
                "grouping_basis": "Grouped by instrument",
                "group_label": "EORTC QLQ-C30",
                "trial_label": "TRANSCEND NHL 001",
                "combined_group": "EORTC QLQ-C30 / TRANSCEND NHL 001",
                "citation_display": "Patrick 2021 [17]",
                "title": "Health-related quality of life with lisocabtagene maraleucel",
                "authors": "Patrick D",
                "year": "2021",
                "doi": "10.1000/example",
                "journal": "Blood Advances",
                "reference_number": "17",
                "study_design": "Phase 1 single-arm trial",
                "sample_size": "256",
                "outcome_measure": "EORTC QLQ-C30 global health status",
                "outcome_result": "19.7 (25.6)",
                "notes": "",
                "needs_review": "",
            }
        ]
    )

    assert len(merged) == 1
    assert merged[0]["title"] == "Health-related quality of life with lisocabtagene maraleucel"
    assert merged[0]["extra_fields"]["table_number"] == "2"
    assert merged[0]["extra_fields"]["combined_group"] == "EORTC QLQ-C30 / TRANSCEND NHL 001"
    assert merged[0]["extra_fields"]["study_design"] == "Phase 1 single-arm trial"


def test_merge_included_study_editor_rows_coerces_keep_strings():
    module = _load_document_extract_module()

    merged = module._merge_included_study_editor_rows(
        [
            {
                "keep": "TRUE",
                "row_id": 1,
                "table_number": "3",
                "table_title": "Overview of included studies on health state utility values",
                "grouping_basis": "Utility assessment method and region",
                "group_label": "EQ-5D / Global",
                "trial_label": "SADAL",
                "combined_group": "EQ-5D / Global / SADAL",
                "citation_display": "Shah 2021 [38]",
                "title": "Health-related quality of life and utility outcomes with selinexor in relapsed/refractory diffuse large B-cell lymphoma",
                "authors": "Shah J",
                "year": "2021",
                "doi": "",
                "journal": "Future Oncology",
                "reference_number": "38",
                "notes": "",
                "needs_review": "",
            },
            {
                "keep": "False",
                "row_id": 2,
                "table_number": "3",
                "table_title": "Overview of included studies on health state utility values",
                "grouping_basis": "Utility assessment method and region",
                "group_label": "TTO / UK",
                "trial_label": "Hypothetical CAR T vignette study",
                "combined_group": "TTO / UK / Hypothetical CAR T vignette study",
                "citation_display": "Howell 2022 [44]",
                "title": "Howell 2022 [44]",
                "authors": "Howell D",
                "year": "2022",
                "doi": "",
                "journal": "",
                "reference_number": "44",
                "notes": "",
                "needs_review": "yes",
            },
        ]
    )

    assert len(merged) == 1
    assert merged[0]["row_id"] == 1
    assert merged[0]["extra_fields"]["reference_number"] == "38"


def test_build_included_study_website_payload_groups_selected_rows():
    module = _load_document_extract_module()

    editor_rows = [
        {
            "keep": True,
            "row_id": 1,
            "table_number": "2",
            "table_title": "Overview of Included Studies on HRQOL Measures",
            "grouping_basis": "Grouped by instrument",
            "group_label": "EORTC QLQ-C30",
            "trial_label": "TRANSCEND NHL 001",
            "combined_group": "EORTC QLQ-C30 / TRANSCEND NHL 001",
            "citation_display": "Patrick 2021 [17]",
            "title": "Health-related quality of life with lisocabtagene maraleucel",
            "authors": "Patrick D",
            "year": "2021",
            "doi": "",
            "journal": "",
            "reference_number": "17",
            "study_design": "",
            "sample_size": "",
            "outcome_measure": "",
            "outcome_result": "",
            "notes": "",
            "needs_review": "",
        },
        {
            "keep": True,
            "row_id": 2,
            "table_number": "2",
            "table_title": "Overview of Included Studies on HRQOL Measures",
            "grouping_basis": "Grouped by instrument",
            "group_label": "EORTC QLQ-C30",
            "trial_label": "TRANSCEND NHL 001",
            "combined_group": "EORTC QLQ-C30 / TRANSCEND NHL 001",
            "citation_display": "Patrick 2020 [37]",
            "title": "Impact of lisocabtagene maraleucel",
            "authors": "Patrick D",
            "year": "2020",
            "doi": "",
            "journal": "",
            "reference_number": "37",
            "study_design": "",
            "sample_size": "",
            "outcome_measure": "",
            "outcome_result": "",
            "notes": "Co-citation",
            "needs_review": "",
        },
        {
            "keep": False,
            "row_id": 3,
            "table_number": "3",
            "table_title": "Overview of Included Studies on Health State Utility Values",
            "grouping_basis": "Grouped by instrument",
            "group_label": "EQ-5D",
            "trial_label": "ZUMA-1",
            "combined_group": "EQ-5D / ZUMA-1",
            "citation_display": "Lin 2019 [42]",
            "title": "Preference-weighted health status",
            "authors": "Lin V",
            "year": "2019",
            "doi": "",
            "journal": "",
            "reference_number": "42",
            "study_design": "",
            "sample_size": "",
            "outcome_measure": "",
            "outcome_result": "",
            "notes": "",
            "needs_review": "",
        },
    ]

    resolver_payload = {"citations": [{"row_id": 1, "title": "Health-related quality of life with lisocabtagene maraleucel"}]}
    payload = module._build_included_study_website_payload(
        editor_rows,
        extraction_scope="rct_or_clinical",
        output_detail="reference_map",
        focus_label="table 2",
        resolver_payload=resolver_payload,
    )

    assert payload["action"] == "included_study_extract_handoff"
    assert payload["included_study_context"]["focused_table_label"] == "table 2"
    assert payload["selection_summary"]["selected_paper_count"] == 2
    assert payload["selection_summary"]["table_count"] == 1
    assert payload["selection_summary"]["group_count"] == 1
    assert len(payload["tables"]) == 1
    assert payload["tables"][0]["table_number"] == "2"
    assert payload["tables"][0]["groups"][0]["combined_group"] == "EORTC QLQ-C30 / TRANSCEND NHL 001"
    assert [item["reference_number"] for item in payload["tables"][0]["groups"][0]["citations"]] == ["17", "37"]
    assert payload["resolver_payload"] == resolver_payload


def test_build_included_study_research_queue_job_wraps_resolver_payload():
    module = _load_document_extract_module()

    editor_rows = [
        {
            "keep": True,
            "row_id": 1,
            "table_number": "3",
            "table_title": "Overview of included studies on health state utility values",
            "grouping_basis": "Grouped by instrument",
            "group_label": "EQ-5D / US",
            "trial_label": "TRANSCEND NHL 001",
            "combined_group": "EQ-5D / US / TRANSCEND NHL 001",
            "citation_display": "Patrick 2021 [17]",
            "title": "Effect of lisocabtagene maraleucel on HRQoL",
            "authors": "Patrick DL",
            "year": "2021",
            "doi": "",
            "journal": "",
            "reference_number": "17",
            "study_design": "",
            "sample_size": "",
            "outcome_measure": "",
            "outcome_result": "",
            "notes": "",
            "needs_review": "",
        }
    ]

    payload = module._build_included_study_research_queue_job(
        editor_rows,
        check_open_access=True,
        enrich_sjr=False,
        unpaywall_email=" person@example.com ",
        extraction_scope="rct_or_clinical",
        output_detail="reference_map",
        focus_label="table 3",
    )

    assert payload["job_type"] == "research_resolve"
    assert payload["source_workflow"] == "included_study_extractor"
    assert payload["project_id"] == "included_study_extractor"
    assert payload["source_system"] == "cortex_streamlit"
    assert payload["trace_id"].startswith("trace-")
    assert payload["input_data"]["citations"][0]["title"] == "Effect of lisocabtagene maraleucel on HRQoL"
    assert payload["input_data"]["citations"][0]["extra_fields"]["reference_number"] == "17"
    assert payload["input_data"]["options"]["unpaywall_email"] == "person@example.com"


def test_parse_included_study_handoff_bundle_loads_resolver_payload():
    module = _load_document_extract_module()
    payload = {
        "citations": [
            {
                "row_id": 1,
                "title": "Example trial",
                "authors": "Example",
                "year": "2024",
                "doi": "10.1000/example",
            }
        ],
        "options": {"check_open_access": True, "enrich_sjr": False},
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("research_resolver_payload.json", json.dumps(payload))

    parsed = module._parse_included_study_handoff_bundle(buf.getvalue(), bundle_name="handoff.zip")

    assert parsed["source_name"] == "handoff.zip"
    assert parsed["citations"][0]["title"] == "Example trial"
    assert parsed["source_payload"]["options"]["check_open_access"] is True


def test_build_included_study_handoff_bundle_includes_reference_list_exports():
    module = _load_document_extract_module()
    editor_rows = [
        {
            "keep": True,
            "row_id": 1,
            "title": "Example trial",
            "authors": "Example",
            "year": "2024",
            "doi": "10.1000/example",
            "journal": "Example Journal",
        }
    ]
    resolver_payload = module._build_included_study_research_payload(
        editor_rows,
        check_open_access=True,
        enrich_sjr=False,
        unpaywall_email="person@example.com",
        extraction_scope="all_trials",
        output_detail="reference_map",
    )

    bundle = module._build_included_study_handoff_bundle_bytes(
        editor_rows,
        extraction_scope="all_trials",
        output_detail="reference_map",
        resolver_payload=resolver_payload,
        bibliography_entries=[
            {
                "authors": "Cohen",
                "year": "2004",
                "title": "Psychological adjustment and sleep quality",
                "journal": "Cancer",
                "reference_section": "included",
                "entry_text": "Cohen 2004...",
            }
        ],
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = set(zf.namelist())
        csv_text = zf.read("bibliography.csv").decode("utf-8")

    assert "bibliography.csv" in names
    assert "bibliography.xlsx" in names
    assert "reference_section" in csv_text
    assert "Cohen" in csv_text


def test_build_study_miner_handoff_bundle_loads_in_research_resolver():
    module = _load_document_extract_module()
    candidates = [
        {
            "row_id": 1,
            "title": "Example mined trial",
            "authors": "Miner",
            "year": "2022",
            "doi": "10.1000/mined",
            "journal": "Example Journal",
            "reference_number": "12",
            "source_section": "table",
            "extra_fields": {
                "table_index": 2,
                "table_label": "table 2",
                "reference_number": "12",
            },
        }
    ]
    editor_rows = [
        {
            "keep": True,
            "row_id": 1,
            "title": "Example mined trial",
            "authors": "Miner",
            "year": "2022",
            "doi": "10.1000/mined",
            "journal": "Example Journal",
            "source_review_title": "Example review",
            "table_index": 2,
            "table_label": "table 2",
            "reference_number": "12",
        }
    ]

    bundle = module._build_study_miner_handoff_bundle_bytes(
        editor_rows,
        candidates,
        check_open_access=True,
        enrich_sjr=False,
        unpaywall_email="person@example.com",
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = set(zf.namelist())
        payload = json.loads(zf.read("research_resolver_payload.json").decode("utf-8"))
        manifest = json.loads(zf.read("study_miner_handoff.json").decode("utf-8"))

    parsed = module._parse_included_study_handoff_bundle(bundle, bundle_name="study_miner.zip")

    assert "study_miner_selected_rows.csv" in names
    assert payload["citations"][0]["title"] == "Example mined trial"
    assert payload["options"]["unpaywall_email"] == "person@example.com"
    assert manifest["source_workflow"] == "study_miner"
    assert parsed["source_name"] == "study_miner.zip"
    assert parsed["citations"][0]["doi"] == "10.1000/mined"


def test_run_included_study_table_slice_retries_multiple_quota_waits_before_success():
    module = _load_document_extract_module()
    waits = []
    calls = {"count": 0}

    def _fake_run(**kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise module.IncludedStudyExtractorQuotaError(
                "gemini",
                429,
                f"Gemini quota/rate limit exceeded. Please retry in {40 + calls['count']}s.",
            )
        return {"tables": [{"table_number": "2"}], "warnings": [], "raw_response": "{}"}

    progress = []
    module.run_included_study_table_extractor = _fake_run

    result = module._run_included_study_table_slice(
        table_slice={"label": "table 2", "pdf_path": "/tmp/table_2.pdf"},
        bibliography_text="refs",
        provider="gemini",
        model="gemini-2.5-flash",
        review_title="Review",
        extraction_scope="all_trials",
        output_detail="reference_map",
        auto_retry_quota=True,
        retry_wait_cap=75,
        max_quota_retries=3,
        progress_callback=progress.append,
        sleep_fn=waits.append,
    )

    assert result["tables"][0]["table_number"] == "2"
    assert calls["count"] == 3
    assert waits == [43.0, 44.0]
    assert any("waiting 43.0s before retry (1/3)" in item for item in progress)
    assert any("waiting 44.0s before retry (2/3)" in item for item in progress)


def test_run_included_study_table_slice_raises_after_retry_budget_exhausted():
    module = _load_document_extract_module()
    waits = []
    calls = {"count": 0}

    def _always_quota(**kwargs):
        calls["count"] += 1
        raise module.IncludedStudyExtractorQuotaError(
            "gemini",
            429,
            "Gemini quota/rate limit exceeded. Please retry in 30s.",
        )

    module.run_included_study_table_extractor = _always_quota

    with pytest.raises(module.IncludedStudyExtractorQuotaError):
        module._run_included_study_table_slice(
            table_slice={"label": "table 3", "pdf_path": "/tmp/table_3.pdf"},
            bibliography_text="refs",
            provider="gemini",
            model="gemini-2.5-flash",
            review_title="Review",
            extraction_scope="all_trials",
            output_detail="reference_map",
            auto_retry_quota=True,
            retry_wait_cap=75,
            max_quota_retries=2,
            progress_callback=None,
            sleep_fn=waits.append,
        )

    assert calls["count"] == 3
    assert waits == [32.0, 32.0]


def test_upsert_included_study_slice_run_replaces_same_label():
    module = _load_document_extract_module()

    updated = module._upsert_included_study_slice_run(
        [
            {"label": "table 2", "extraction": {"tables": [{"table_number": "2"}]}},
            {"label": "table 3", "extraction": {"tables": [{"table_number": "3"}]}},
        ],
        {"label": "table 2", "extraction": {"tables": [{"table_number": "2b"}]}},
    )

    assert len(updated) == 2
    assert updated[0]["extraction"]["tables"][0]["table_number"] == "2b"
    assert updated[1]["label"] == "table 3"


def test_upsert_included_study_slice_run_appends_new_label():
    module = _load_document_extract_module()

    updated = module._upsert_included_study_slice_run(
        [{"label": "table 2", "extraction": {"tables": [{"table_number": "2"}]}}],
        {"label": "table 4", "extraction": {"tables": [{"table_number": "4"}]}},
    )

    assert [item["label"] for item in updated] == ["table 2", "table 4"]


def test_included_study_slice_run_is_completed_for_warning_only_extraction():
    module = _load_document_extract_module()

    assert module._included_study_slice_run_is_completed(
        {
            "label": "table 2",
            "extraction": {"tables": [], "warnings": ["No included-study tables parsed"], "raw_response": "{}"},
        }
    ) is True

    assert module._included_study_slice_run_is_completed({"label": "table 3"}) is False


def test_clear_included_study_extractor_state_removes_derived_outputs(tmp_path, monkeypatch):
    module = _load_document_extract_module()
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))

    work_dir = tmp_path / "included_study_slices_case"
    work_dir.mkdir()
    (work_dir / "table_901.pdf").write_bytes(b"pdf")
    upload_dir = tmp_path / "cortex_included_study"
    upload_dir.mkdir()
    uploaded = upload_dir / "upload_123_review.pdf"
    uploaded.write_bytes(b"pdf")

    module.st.session_state.clear()
    module.st.session_state["included_study_provider"] = "anthropic"
    module.st.session_state["included_study_model"] = "claude-sonnet-4-6"
    module.st.session_state["included_study_upload_version"] = 2
    module.st.session_state["included_study_upload_v2"] = object()
    module.st.session_state["included_study_slice_result"] = {"work_dir": str(work_dir)}
    module.st.session_state["included_study_slice_runs"] = [{"label": "Cochrane included studies"}]
    module.st.session_state["included_study_result"] = {"tables": []}
    module.st.session_state["included_study_editor_rows"] = [{"row_id": 1}]
    module.st.session_state["research_parse_result"] = {"source_name": "Included Study Extractor"}
    module.st.session_state["research_editor_rows"] = [{"row_id": 1}]
    module.st.session_state["url_ingestor_zip_bytes"] = b"zip"

    module._clear_included_study_extractor_state()

    assert not work_dir.exists()
    assert not uploaded.exists()
    assert module.st.session_state["included_study_upload_version"] == 3
    assert module.st.session_state["included_study_provider"] == "anthropic"
    assert module.st.session_state["included_study_model"] == "claude-sonnet-4-6"
    assert "included_study_slice_result" not in module.st.session_state
    assert "included_study_upload_v2" not in module.st.session_state
    assert "research_parse_result" not in module.st.session_state
    assert "url_ingestor_zip_bytes" not in module.st.session_state


def test_included_study_defaults_prefer_anthropic_when_available():
    module = _load_document_extract_module()
    module.included_study_extractor_available = lambda provider: provider == "anthropic"

    assert module._included_study_default_provider() == "anthropic"
    assert module._included_study_default_model("anthropic") == "claude-sonnet-4-6"
    assert module._included_study_default_model("gemini") == "gemini-2.5-flash"


def test_included_study_defaults_fall_back_to_gemini_when_anthropic_unavailable():
    module = _load_document_extract_module()
    module.included_study_extractor_available = lambda provider: False

    assert module._included_study_default_provider() == "gemini"


def test_included_study_rows_to_xlsx_bytes_round_trips_basic_rows():
    module = _load_document_extract_module()

    payload = module._included_study_rows_to_xlsx_bytes(
        [{"table_number": "2", "trial_label": "JULIET", "reference_number": "19"}],
        sheet_name="table_2",
    )

    assert payload.startswith(b"PK")


def test_parse_research_resolve_bundle_builds_review_rows():
    module = _load_document_extract_module()

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "resolved_citations.csv",
            "\n".join(
                [
                    "row_id,table_number,table_title,combined_group,citation_display,reference_number,input_title,matched_title,resolved_doi,resolved_url,source_api,resolution_method,confidence,publisher,journal_name,open_access_pdf_url",
                    "1,3,Table 3,EQ-5D / Global / CheckMate 436,Zinzani 2019 [45],45,Input A,Matched A,10.1200/jco.19.01492,https://doi.org/10.1200/jco.19.01492,review_bibliography,bibliography_reference_number,high,ASCO,Journal of Clinical Oncology,",
                    "2,3,Table 3,EQ-5D / NR / Orfanos 2022,Orfanos 2022 [46],46,Input B,Matched B,10.1016/j.jval.2021.11.1195,https://doi.org/10.1016/j.jval.2021.11.1195,review_bibliography,bibliography_reference_number,high,Elsevier,Value in Health,",
                ]
            ),
        )
        zf.writestr("unresolved_citations.csv", "row_id,input_title,reason,best_candidate\n3,Missing paper,no_match,\n")
        zf.writestr("preferred_urls.txt", "https://doi.org/10.1200/jco.19.01492\nhttps://doi.org/10.1016/j.jval.2021.11.1195\n")
        zf.writestr("research_resolve_result.json", json.dumps({"status": "completed", "stats": {"total": 2}}))
        zf.writestr(
            "retrieval/reports/url_ingest_report_1.csv",
            "\n".join(
                [
                    "input_url,final_url,page_title,status,reason,http_code,open_access_pdf_found,pdf_url,pdf_path,md_path,converted_to_md,web_captured,elapsed_seconds",
                    "https://doi.org/10.1200/jco.19.01492,https://ascopubs.org/doi/10.1200/JCO.19.01492,Example,web_markdown,web_page_captured_http_error,403,FALSE,,,/tmp/run/retrieval/markdown/asco.md,FALSE,TRUE,0.8",
                    "https://doi.org/10.1016/j.jval.2021.11.1195,https://linkinghub.elsevier.com/retrieve/pii/S1098301521029909,Example,downloaded,,200,TRUE,https://example.org/file.pdf,/tmp/run/retrieval/pdfs/example.pdf,/tmp/run/retrieval/markdown/example.md,TRUE,FALSE,0.4",
                ]
            ),
        )
        zf.writestr("retrieval/markdown/asco.md", "# ASCO landing")
        zf.writestr("retrieval/markdown/example.md", "# Example")
        zf.writestr("retrieval/pdfs/example.pdf", b"%PDF-1.4 test")

    parsed = module._parse_research_resolve_bundle(mem.getvalue(), "bundle.zip")

    assert parsed["bundle_name"] == "bundle.zip"
    assert len(parsed["review_rows"]) == 2
    row_by_id = {row["row_id"]: row for row in parsed["review_rows"]}
    assert row_by_id["1"]["auto_markdown_present"] is True
    assert row_by_id["1"]["auto_pdf_present"] is False
    assert row_by_id["1"]["final_status"] == "retrieved_auto_web"
    assert row_by_id["2"]["auto_pdf_present"] is True
    assert row_by_id["2"]["final_status"] == "retrieved_auto_pdf"
    assert len(parsed["unresolved_rows"]) == 1


def test_build_retrieval_researcher_package_includes_manual_pdf():
    module = _load_document_extract_module()

    source_bundle = io.BytesIO()
    with zipfile.ZipFile(source_bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("retrieval/markdown/example.md", "# Example")
        zf.writestr("retrieval/pdfs/example.pdf", b"%PDF-1.4 auto")
        zf.writestr("resolved_citations.csv", "row_id\n1\n")

    review_rows = [
        {
            "row_id": "1",
            "reference_number": "45",
            "citation_display": "Zinzani 2019 [45]",
            "matched_title": "Matched A",
            "final_status": "retrieved_auto_pdf",
            "auto_pdf_present": True,
            "auto_markdown_present": True,
            "manual_pdf_filename": "",
            "review_notes": "",
        },
        {
            "row_id": "2",
            "reference_number": "46",
            "citation_display": "Orfanos 2022 [46]",
            "matched_title": "Matched B",
            "final_status": "retrieved_manual_pdf",
            "auto_pdf_present": False,
            "auto_markdown_present": False,
            "manual_pdf_filename": "manual.pdf",
            "review_notes": "uploaded manually",
        },
    ]
    manual_uploads = {"2": {"name": "manual.pdf", "data": b"%PDF-1.4 manual"}}

    package_bytes = module._build_retrieval_researcher_package(
        source_bundle.getvalue(),
        review_rows,
        manual_uploads,
        source_bundle_name="bundle.zip",
    )

    with zipfile.ZipFile(io.BytesIO(package_bytes)) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "final_reviewed_citations.csv" in names
        assert "missing_or_unavailable.csv" in names
        assert "resolved/pdfs/example.pdf" in names
        assert "resolved/markdown/example.md" in names
        assert "resolved/manual_pdfs/manual.pdf" in names
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["manual_pdf_count"] == 1
