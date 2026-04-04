from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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
