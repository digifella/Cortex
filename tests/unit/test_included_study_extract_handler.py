import json
import zipfile

from worker.handlers import included_study_extract as handler


def test_included_study_extract_handler_returns_summary_and_zip(monkeypatch, tmp_path):
    input_pdf = tmp_path / "review.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    slice_pdf = tmp_path / "table_2.pdf"
    slice_pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(
        handler,
        "slice_review_pdf",
        lambda pdf_path, work_dir="": {
            "pdf_path": pdf_path,
            "work_dir": work_dir,
            "table_slices": [
                {
                    "table_number": "2",
                    "table_title": "Overview of included studies on HRQOL measures",
                    "kind": "included_studies",
                    "page_numbers": [6, 7],
                    "pdf_path": str(slice_pdf),
                    "pdf_file_name": "table_2.pdf",
                    "label": "table 2",
                }
            ],
            "bibliography_entries": [
                {
                    "reference_number": "17",
                    "authors": "Patrick DL, Powers A, Jun MP, et al.",
                    "year": "2021",
                    "title": "Effect of lisocabtagene maraleucel on HRQoL and symptom severity in relapsed/refractory large B-cell lymphoma",
                    "journal": "Blood Advances",
                    "doi": "10.1182/bloodadvances.2021004412",
                    "entry_text": "17. Patrick DL, Powers A, Jun MP, et al. Effect of lisocabtagene maraleucel on HRQoL and symptom severity in relapsed/refractory large B-cell lymphoma. Blood Advances. 2021.",
                }
            ],
            "bibliography_text": "[17] Patrick 2021 ...",
            "bibliography_txt_path": str(tmp_path / "bibliography.txt"),
            "bibliography_csv_path": str(tmp_path / "bibliography.csv"),
        },
    )
    (tmp_path / "bibliography.txt").write_text("[17] Patrick 2021", encoding="utf-8")
    (tmp_path / "bibliography.csv").write_text("reference_number\n17\n", encoding="utf-8")

    monkeypatch.setattr(handler, "included_study_extractor_available", lambda provider: True)
    monkeypatch.setattr(
        handler,
        "run_included_study_table_extractor",
        lambda **kwargs: {
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "tables": [
                {
                    "table_number": "2",
                    "table_title": "Overview of included studies on HRQOL measures",
                    "grouping_basis": "Grouped by instrument",
                    "groups": [
                        {
                            "group_label": "EORTC QLQ-C30",
                            "trial_label": "TRANSCEND NHL 001",
                            "notes": "",
                            "citations": [
                                {
                                    "display": "Patrick 2021 [17]",
                                    "authors": "Patrick et al.",
                                    "year": "2021",
                                    "reference_number": "17",
                                    "notes": "",
                                    "needs_review": False,
                                }
                            ],
                        }
                    ],
                }
            ],
            "warnings": [],
            "raw_response": '{"tables":[...]}',
        },
    )

    result = handler.handle(
        input_pdf,
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "extraction_scope": "rct_or_clinical",
            "output_detail": "reference_map",
        },
        {"id": 123},
    )

    output_data = result["output_data"]
    output_file = result["output_file"]
    assert output_data["status"] == "completed"
    assert output_data["table_count"] == 1
    assert output_data["tables"][0]["table_number"] == "2"
    assert output_data["tables"][0]["selected_citation_count"] == 1
    assert output_file.exists()

    with zipfile.ZipFile(output_file, "r") as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "bibliography/bibliography.txt" in names
        assert "tables/table_2/included_study_selection.csv" in names
        assert "tables/table_2/included_study_selection.xlsx" in names
        assert "tables/table_2/research_resolver_payload.json" in names
        assert "tables/table_2/research_resolver_queue_job.json" in names
        assert "tables/table_2/included_study_website_handoff.json" in names
        assert "combined/research_resolver_queue_job.json" in names

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["table_count"] == 1
        website_payload = json.loads(zf.read("tables/table_2/included_study_website_handoff.json").decode("utf-8"))
        assert website_payload["tables"][0]["table_number"] == "2"
        assert website_payload["resolver_queue_job"]["job_type"] == "research_resolve"
        resolver_payload = json.loads(zf.read("tables/table_2/research_resolver_payload.json").decode("utf-8"))
        assert resolver_payload["citations"][0]["title"] == (
            "Effect of lisocabtagene maraleucel on HRQoL and symptom severity in relapsed/refractory large B-cell lymphoma"
        )
        assert resolver_payload["citations"][0]["authors"] == "Patrick DL, Powers A, Jun MP, et al."
        assert resolver_payload["citations"][0]["journal"] == "Blood Advances"
        assert resolver_payload["citations"][0]["doi"] == "10.1182/bloodadvances.2021004412"


def test_included_study_extract_handler_skips_empty_table_payload_generation(monkeypatch, tmp_path):
    input_pdf = tmp_path / "review.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    slice_pdf_2 = tmp_path / "table_2.pdf"
    slice_pdf_3 = tmp_path / "table_3.pdf"
    slice_pdf_2.write_bytes(b"%PDF-1.4\n")
    slice_pdf_3.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(
        handler,
        "slice_review_pdf",
        lambda pdf_path, work_dir="": {
            "pdf_path": pdf_path,
            "work_dir": work_dir,
            "table_slices": [
                {
                    "table_number": "2",
                    "table_title": "Overview of included studies on HRQOL measures",
                    "kind": "included_studies",
                    "page_numbers": [6, 7],
                    "pdf_path": str(slice_pdf_2),
                    "pdf_file_name": "table_2.pdf",
                    "label": "table 2",
                },
                {
                    "table_number": "3",
                    "table_title": "Overview of included studies on health state utility values",
                    "kind": "included_studies",
                    "page_numbers": [11, 12],
                    "pdf_path": str(slice_pdf_3),
                    "pdf_file_name": "table_3.pdf",
                    "label": "table 3",
                },
            ],
            "bibliography_entries": [{"reference_number": "17"}],
            "bibliography_text": "[17] Patrick 2021 ...",
            "bibliography_txt_path": str(tmp_path / "bibliography.txt"),
            "bibliography_csv_path": str(tmp_path / "bibliography.csv"),
        },
    )
    (tmp_path / "bibliography.txt").write_text("[17] Patrick 2021", encoding="utf-8")
    (tmp_path / "bibliography.csv").write_text("reference_number\n17\n", encoding="utf-8")

    monkeypatch.setattr(handler, "included_study_extractor_available", lambda provider: True)

    def fake_extract(**kwargs):
        if kwargs["table_label"] == "table 3":
            return {
                "provider": kwargs["provider"],
                "model": kwargs["model"],
                "tables": [],
                "warnings": ["No included-study rows found"],
                "raw_response": "No structured table could be extracted.",
            }
        return {
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "tables": [
                {
                    "table_number": "2",
                    "table_title": "Overview of included studies on HRQOL measures",
                    "grouping_basis": "Grouped by instrument",
                    "groups": [
                        {
                            "group_label": "EORTC QLQ-C30",
                            "trial_label": "TRANSCEND NHL 001",
                            "notes": "",
                            "citations": [
                                {
                                    "display": "Patrick 2021 [17]",
                                    "authors": "Patrick et al.",
                                    "year": "2021",
                                    "reference_number": "17",
                                    "notes": "",
                                    "needs_review": False,
                                }
                            ],
                        }
                    ],
                }
            ],
            "warnings": [],
            "raw_response": '{"tables":[...]}',
        }

    monkeypatch.setattr(handler, "run_included_study_table_extractor", fake_extract)

    result = handler.handle(
        input_pdf,
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "extraction_scope": "rct_or_clinical",
            "output_detail": "reference_map",
        },
        {"id": 456},
    )

    output_data = result["output_data"]
    assert output_data["status"] == "completed"
    assert output_data["table_count"] == 2
    assert any("table 3: no citations were extracted" in warning for warning in output_data["warnings"])

    with zipfile.ZipFile(result["output_file"], "r") as zf:
        names = set(zf.namelist())
        assert "tables/table_3/extraction.json" in names
        assert "tables/table_3/extraction_raw_output.txt" in names
        assert "tables/table_3/research_resolver_payload.json" not in names
        assert "tables/table_3/research_resolver_queue_job.json" not in names
        assert "tables/table_3/included_study_website_handoff.json" not in names
        assert "combined/research_resolver_queue_job.json" in names


def test_included_study_extract_handler_skips_resolver_payload_when_all_rows_need_review(monkeypatch, tmp_path):
    input_pdf = tmp_path / "review.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    slice_pdf = tmp_path / "table_4.pdf"
    slice_pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(
        handler,
        "slice_review_pdf",
        lambda pdf_path, work_dir="": {
            "pdf_path": pdf_path,
            "work_dir": work_dir,
            "table_slices": [
                {
                    "table_number": "4",
                    "table_title": "Overview of economic studies reporting health state utility values",
                    "kind": "economic",
                    "page_numbers": [13, 14],
                    "pdf_path": str(slice_pdf),
                    "pdf_file_name": "table_4.pdf",
                    "label": "table 4",
                }
            ],
            "bibliography_entries": [{"reference_number": "54"}],
            "bibliography_text": "[54] Li 2022 ...",
            "bibliography_txt_path": str(tmp_path / "bibliography.txt"),
            "bibliography_csv_path": str(tmp_path / "bibliography.csv"),
        },
    )
    (tmp_path / "bibliography.txt").write_text("[54] Li 2022", encoding="utf-8")
    (tmp_path / "bibliography.csv").write_text("reference_number\n54\n", encoding="utf-8")

    monkeypatch.setattr(handler, "included_study_extractor_available", lambda provider: True)
    monkeypatch.setattr(
        handler,
        "run_included_study_table_extractor",
        lambda **kwargs: {
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "tables": [
                {
                    "table_number": "4",
                    "table_title": "Overview of economic studies reporting health state utility values",
                    "grouping_basis": "Economic studies",
                    "groups": [
                        {
                            "group_label": "China / CUA",
                            "trial_label": "",
                            "notes": "",
                            "citations": [
                                {
                                    "display": "Li 2022 [54]",
                                    "authors": "Li et al.",
                                    "year": "2022",
                                    "reference_number": "54",
                                    "notes": "Model-based economic study",
                                    "needs_review": True,
                                }
                            ],
                        }
                    ],
                }
            ],
            "warnings": [],
            "raw_response": '{"tables":[...]}',
        },
    )

    result = handler.handle(
        input_pdf,
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "extraction_scope": "rct_or_clinical",
            "output_detail": "reference_map",
        },
        {"id": 789},
    )

    output_data = result["output_data"]
    assert output_data["status"] == "completed"
    assert output_data["table_count"] == 1
    assert output_data["tables"][0]["citation_count"] == 1
    assert output_data["tables"][0]["selected_citation_count"] == 0
    assert any("all marked needs_review" in warning for warning in output_data["warnings"])
    assert "resolver_payload_json" not in output_data["tables"][0]["artifacts"]
    assert "resolver_queue_job_json" not in output_data["tables"][0]["artifacts"]
    assert "website_handoff_json" not in output_data["tables"][0]["artifacts"]

    with zipfile.ZipFile(result["output_file"], "r") as zf:
        names = set(zf.namelist())
        assert "tables/table_4/included_study_selection.csv" in names
        assert "tables/table_4/included_study_selection.xlsx" in names
        assert "tables/table_4/research_resolver_payload.json" not in names
        assert "tables/table_4/research_resolver_queue_job.json" not in names
        assert "tables/table_4/included_study_website_handoff.json" not in names
        assert "combined/research_resolver_queue_job.json" not in names
