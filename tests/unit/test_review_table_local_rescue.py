from __future__ import annotations

import json
from types import SimpleNamespace

from cortex_engine.review_table_local_rescue import (
    _align_reconciled_candidates_to_provisional_groups,
    annotate_local_candidate_completeness,
    merge_local_table_candidates,
    _ollama_chat,
    _normalize_reconciled_candidates,
    _normalize_local_rescue_candidates,
    run_local_table_rescue,
)


def test_normalize_local_rescue_candidates_links_reference_entries():
    payload = {
        "rows": [
            {
                "region": "JULIET",
                "study": "JULIET",
                "author_year": "Maziarz 2020 [19]",
                "study_design": "Phase 2 single-arm trial",
                "patient_population": "Adult patients with R/R DLBCL with 2+ prior lines of therapy",
                "treatment": "Tisagenlecleucel",
                "assessment_method": "FACT-G TS",
                "scale": "0 to 108",
                "baseline_mean_sd": "77.4 (16.1)",
                "mean_change_last_followup_sd": "+10 (11.1)",
                "hrqol_measures_mappable_to_direct_utility_values": "Global",
                "source_pages": [1, 2],
            }
        ]
    }
    references_text = """
References
[19] Maziarz RT, Schuster SJ. Randomized clinical trial of health-related quality of life after CAR-T therapy in lymphoma. Journal of Clinical Oncology. 2020. 10.1000/maziarz2020
"""

    candidates = _normalize_local_rescue_candidates(
        payload,
        design_query="RCT clinical trial randomised randomized",
        outcome_query="health-related quality of life HRQoL QoL",
        references_text=references_text,
    )

    assert len(candidates) == 1
    assert candidates[0]["reference_number"] == "19"
    assert candidates[0]["doi"] == "10.1000/maziarz2020"
    assert candidates[0]["meets_criteria"] is True
    assert candidates[0]["extra_fields"]["region"] == "JULIET"


def test_run_local_table_rescue_sends_page_images(monkeypatch):
    captured = {}

    def _fake_ollama_chat(model, prompt, image_bytes_list, host=""):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["images"] = list(image_bytes_list)
        return '{"rows":[{"region":"JULIET","study":"JULIET","author_year":"Maziarz 2020 [19]","study_design":"Phase 2 single-arm trial","patient_population":"Adult patients","treatment":"Tisagenlecleucel","assessment_method":"FACT-G TS","scale":"0 to 108","baseline_mean_sd":"77.4 (16.1)","mean_change_last_followup_sd":"+10 (11.1)","hrqol_measures_mappable_to_direct_utility_values":"Global","source_pages":[1]}],"warnings":[]}'

    monkeypatch.setattr("cortex_engine.review_table_local_rescue._ollama_chat", _fake_ollama_chat)
    monkeypatch.setattr("cortex_engine.review_table_local_rescue._resolve_ollama_model_name", lambda model="", host="": model or "qwen3-vl:30b-a3b-instruct")
    monkeypatch.setattr(
        "cortex_engine.review_table_local_rescue._render_pdf_pages_for_local_rescue",
        lambda pdf_path, table_snapshots, **kwargs: [
            {"page_number": 1, "image_bytes": b"page-one", "text_sample": "Table 2 continued"}
        ],
    )

    result = run_local_table_rescue(
        review_title="Supportive care review",
        design_query="randomized trial",
        outcome_query="quality of life",
        table_snapshots=[],
        table_blocks=[{"table_index": 1, "header": ["Region", "Study", "Author/Year"], "context_text": "Table 2 continued"}],
        references_text="[19] Maziarz RT. Trial paper. Journal. 2020.",
        pdf_path="test.pdf",
        model="qwen2.5-vl-32b-instruct",
    )

    assert captured["model"] == "qwen2.5-vl-32b-instruct"
    assert "Table 2 continued" in captured["prompt"]
    assert captured["images"] == [b"page-one"]
    assert result["used_page_images"] is True
    assert result["pages_used"] == [1]


def test_normalize_local_rescue_candidates_repairs_continuation_rows():
    payload = {
        "rows": [
            {
                "region": "Global",
                "study": "TRANSFORM",
                "author_year": "Abramson 2021 [18]",
                "study_design": "Phase 3 trial",
                "patient_population": "Adults aged <=75 years with R/R LBCL <=12 months after 1L therapy",
                "treatment": "Liso-cel",
                "followup_times_assessed": "Baseline, 1 month, 2 months, 3 months, 6 months",
                "assessment_method": "FACT-Lym subscale",
                "scale": "0 to 168",
                "baseline_mean_sd": "NR",
                "mean_change_last_followup_sd": "+3.08",
                "source_pages": [3],
            },
            {
                "region": "Global",
                "study": "EORTC QLQ-C30",
                "author_year": "",
                "study_design": "",
                "patient_population": "",
                "treatment": "NR",
                "followup_times_assessed": "",
                "assessment_method": "EORTC QLQ-C30 global health status",
                "scale": "0 to 100",
                "baseline_mean_sd": "NR",
                "mean_change_last_followup_sd": "+0.04",
                "source_pages": [4],
            },
        ]
    }
    references_text = """
References
[18] Abramson JS, Ghosh N, Sehgal AR, et al. Health-related quality of life with lisocabtagene maraleucel in relapsed/refractory large B-cell lymphoma. Blood. 2021.
"""

    candidates = _normalize_local_rescue_candidates(
        payload,
        design_query="phase 3 clinical trial",
        outcome_query="quality of life FACT EORTC",
        references_text=references_text,
    )

    assert len(candidates) == 1
    assert candidates[0]["reference_number"] == "18"
    assert candidates[0]["extra_fields"]["study"] == "TRANSFORM"
    assert candidates[0]["extra_fields"]["author_year"] == "Abramson 2021 [18]"
    assert candidates[0]["extra_fields"]["treatment"] == "Liso-cel"
    assert "FACT-Lym subscale" in candidates[0]["extra_fields"]["assessment_method"]
    assert "EORTC QLQ-C30 global health status" in candidates[0]["extra_fields"]["assessment_method"]


def test_normalize_local_rescue_candidates_does_not_carry_forward_study_across_region_change_with_new_citation():
    payload = {
        "rows": [
            {
                "region": "US",
                "study": "TRANSCEND NHL 001",
                "author_year": "Patrick 2021 [17]",
                "study_design": "Phase 1 single-arm trial",
                "patient_population": "Adult patients with R/R DLBCL",
                "treatment": "Liso-cel",
                "assessment_method": "EORTC QLQ-C30",
                "source_pages": [2],
            },
            {
                "region": "UK",
                "study": "",
                "author_year": "Howell 2022 [44]",
                "study_design": "",
                "patient_population": "",
                "treatment": "",
                "assessment_method": "TTO",
                "source_pages": [3],
            },
        ]
    }

    candidates = _normalize_local_rescue_candidates(
        payload,
        design_query="trial cost utility",
        outcome_query="quality of life utility TTO",
        references_text="",
    )

    assert len(candidates) == 2
    by_ref = {item["reference_number"]: item for item in candidates}
    assert by_ref["17"]["extra_fields"]["study"] == "TRANSCEND NHL 001"
    assert by_ref["44"]["extra_fields"]["study"] == ""


def test_normalize_local_rescue_candidates_splits_multi_reference_cells():
    payload = {
        "rows": [
            {
                "region": "Global",
                "study": "JULIET",
                "author_year": "Maziarz 2020 [19], Tam 2019 [67], Maziarz 2017 [68]",
                "study_design": "Phase 2 single-arm trial",
                "patient_population": "Adult patients with R/R DLBCL with 2+ prior lines of therapy",
                "treatment": "Tisagenlecleucel",
                "assessment_method": "FACT-Lym TS",
                "scale": "0 to 168",
                "baseline_mean_sd": "76.8 (16.4)",
                "mean_change_last_followup_sd": "+11.0",
                "source_pages": [2],
            }
        ]
    }
    references_text = """
[19] Maziarz RT, Waller EK, Jaeger U, et al. Patient-reported long-term quality of life after tisagenlecleucel in relapsed/refractory diffuse large B-cell lymphoma. Blood Adv. 2020.
[67] Tam CS, et al. JULIET quality of life report. 2019.
[68] Maziarz RT, et al. JULIET interim quality of life report. 2017.
"""

    candidates = _normalize_local_rescue_candidates(
        payload,
        design_query="trial phase 2",
        outcome_query="quality of life FACT",
        references_text=references_text,
    )

    assert len(candidates) == 3
    assert sorted(item["reference_number"] for item in candidates) == ["19", "67", "68"]
    assert all(item["extra_fields"]["study"] == "JULIET" for item in candidates)


def test_normalize_local_rescue_candidates_preserves_inline_reference_number_without_bibliography_match():
    payload = {
        "rows": [
            {
                "region": "Global",
                "study": "JULIET",
                "author_year": "Tam 2019 [67]",
                "study_design": "Phase 2 single-arm trial",
                "patient_population": "Adult patients with R/R DLBCL",
                "treatment": "Tisagenlecleucel",
                "assessment_method": "FACT-Lym TOI",
                "source_pages": [2],
            }
        ]
    }

    candidates = _normalize_local_rescue_candidates(
        payload,
        design_query="trial phase 2",
        outcome_query="quality of life FACT",
        references_text="",
    )

    assert len(candidates) == 1
    assert candidates[0]["reference_number"] == "67"
    assert candidates[0]["reference_match_method"] == "table_inline_reference"


def test_normalize_local_rescue_candidates_uses_linked_reference_for_citation_display():
    payload = {
        "rows": [
            {
                "region": "Global",
                "study": "SADAL",
                "author_year": "Shan 2021 [38]",
                "study_design": "Phase 2b single-arm trial",
                "patient_population": "Adult patients with R/R DLBCL",
                "treatment": "Selinexor",
                "assessment_method": "FACT-G",
                "source_pages": [2],
            },
            {
                "region": "Global",
                "study": "SADAL",
                "author_year": "2020 [69]",
                "study_design": "Phase 2b single-arm trial",
                "patient_population": "Adult patients with R/R DLBCL",
                "treatment": "Selinexor",
                "assessment_method": "FACT-G",
                "source_pages": [2],
            },
        ]
    }
    references_text = """
[38] Shah GL, et al. Selinexor quality of life study in relapsed or refractory diffuse large B-cell lymphoma. 2021.
[69] Casasnovas RO, et al. Additional SADAL report. 2020.
"""

    candidates = _normalize_local_rescue_candidates(
        payload,
        design_query="phase 2 trial",
        outcome_query="quality of life FACT",
        references_text=references_text,
    )

    by_ref = {item["reference_number"]: item for item in candidates}
    assert by_ref["38"]["extra_fields"]["author_year"] == "Shah 2021 [38]"
    assert by_ref["69"]["extra_fields"]["author_year"] == "Casasnovas 2020 [69]"


def test_normalize_local_rescue_candidates_strips_leading_study_code_from_citation():
    payload = {
        "rows": [
            {
                "region": "US",
                "study": "TRANSCEND",
                "author_year": "NHL 001 Patrick 2021 [17]",
                "study_design": "Phase 1 single-arm trial",
                "patient_population": "Adult patients with R/R DLBCL",
                "treatment": "Liso-cel",
                "assessment_method": "EORTC QLQ-C30 global health status",
                "source_pages": [3],
            }
        ]
    }

    candidates = _normalize_local_rescue_candidates(
        payload,
        design_query="trial phase 1",
        outcome_query="quality of life EORTC",
        references_text="",
    )

    assert len(candidates) == 1
    assert candidates[0]["raw_citation"] == "Patrick 2021 [17]"
    assert candidates[0]["authors"] == "Patrick"
    assert candidates[0]["reference_number"] == "17"


def test_normalize_reconciled_candidates_expands_multi_citation_notes():
    payload = {
        "candidates": [
            {
                "region": "US",
                "study": "TRANSCEND",
                "author_year": "Patrick 2021",
                "study_design": "Phase 1 single-arm trial",
                "patient_population": "Adult patients with DLBCL with 2+ prior lines of therapy",
                "treatment": "Liso-cel",
                "followup_times_assessed": "Baseline, 1 month, 2 months",
                "assessment_methods": ["EORTC QLQ-C30 global health status"],
                "mapped_utility_measures": ["EORTC QLQ-C30"],
                "notes": "Citations Patrick 2021 [17], Patrick 2020 [37], Patrick 2019 [35], Patrick 2019 [36] grouped under TRANSCEND.",
                "source_pages": [3, 4],
            }
        ]
    }

    candidates = _normalize_reconciled_candidates(
        payload,
        design_query="trial phase 1",
        outcome_query="quality of life EORTC",
        references_text="",
    )

    assert sorted(item["reference_number"] for item in candidates) == ["17", "35", "36", "37"]
    assert all(item["extra_fields"]["study"] == "TRANSCEND" for item in candidates)


def test_align_reconciled_candidates_to_provisional_groups_reuses_provisional_group_labels():
    provisional_candidates = [
        {
            "reference_number": "45",
            "raw_citation": "Liu 2021 [45]",
            "relevance_score": 8,
            "extra_fields": {
                "region": "China",
                "study": "CUA",
                "author_year": "Liu 2021 [45]",
            },
        }
    ]
    reconciled_candidates = [
        {
            "reference_number": "45",
            "raw_citation": "Liu 2021 [45]",
            "relevance_score": 7,
            "extra_fields": {
                "region": "China",
                "study": "Axi-cel vs Salvage",
                "author_year": "Liu 2021 [45]",
            },
        },
        {
            "reference_number": "99",
            "raw_citation": "Extra 2020 [99]",
            "relevance_score": 5,
            "extra_fields": {
                "region": "US",
                "study": "Off-pattern group",
                "author_year": "Extra 2020 [99]",
            },
        },
    ]

    aligned = _align_reconciled_candidates_to_provisional_groups(
        provisional_candidates,
        reconciled_candidates,
    )

    assert len(aligned) == 1
    assert aligned[0]["reference_number"] == "45"
    assert aligned[0]["extra_fields"]["region"] == "China"
    assert aligned[0]["extra_fields"]["study"] == "CUA"


def test_ollama_chat_disables_thinking_and_reads_content(monkeypatch):
    captured = {}

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": "[]",
                        "thinking": "should not be used",
                    }
                }
            ).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("cortex_engine.review_table_local_rescue.urlopen", _fake_urlopen)

    reply = _ollama_chat("qwen3.5:35b-a3b", "Return []", [b"image-bytes"], host="http://127.0.0.1:11434")

    assert reply == "[]"
    assert captured["timeout"] == 240
    assert captured["body"]["think"] is False
    assert captured["body"]["messages"][0]["images"]


def test_merge_local_table_candidates_preserves_missing_provisional_refs():
    provisional = [
        {
            "title": "JULIET quality of life report",
            "authors": "Tam CS",
            "year": "2019",
            "doi": "",
            "reference_number": "67",
            "relevance_score": 5,
            "needs_review": False,
            "reference_validation": "",
            "design_matches": ["trial"],
            "outcome_matches": ["quality of life"],
            "extra_fields": {"study": "JULIET", "region": "Global", "author_year": "Tam 2019 [67]"},
        }
    ]
    reconciled = [
        {
            "title": "Patient-reported long-term quality of life after tisagenlecleucel in relapsed/refractory diffuse large B-cell lymphoma",
            "authors": "Maziarz RT",
            "year": "2020",
            "doi": "",
            "reference_number": "19",
            "relevance_score": 8,
            "needs_review": False,
            "reference_validation": "",
            "design_matches": ["trial"],
            "outcome_matches": ["quality of life"],
            "extra_fields": {"study": "JULIET", "region": "Global", "author_year": "Maziarz 2020 [19]"},
        }
    ]

    merged = merge_local_table_candidates(provisional, reconciled)

    assert sorted(item["reference_number"] for item in merged) == ["19", "67"]


def test_annotate_local_candidate_completeness_flags_likely_missing_group_refs():
    candidates = [
        {
            "title": "Treatment-related quality of life in diffuse large B-cell lymphoma patients receiving lisocabtagene maraleucel",
            "authors": "Patrick DL",
            "year": "2021",
            "doi": "",
            "reference_number": "17",
            "relevance_score": 8,
            "needs_review": False,
            "review_warning": "",
            "reference_validation": "",
            "raw_citation": "Patrick 2021 [17]",
            "raw_excerpt": "TRANSCEND NHL 001 | Liso-cel | EORTC QLQ-C30 global health status",
            "extra_fields": {
                "region": "US",
                "study": "TRANSCEND NHL 001",
                "author_year": "Patrick 2021 [17]",
                "treatment": "Liso-cel",
                "needs_review": "",
                "review_warning": "",
            },
        }
    ]
    references_text = """
[17] Patrick DL, Ghosh N, Reagan PM, et al. Treatment-related quality of life in diffuse large B-cell lymphoma patients receiving lisocabtagene maraleucel. Lancet Haematol. 2021.
[35] Patrick DL, et al. Health-related quality of life after lisocabtagene maraleucel in relapsed/refractory large B-cell lymphoma. 2019.
[37] Patrick DL, et al. Patient-reported outcomes with lisocabtagene maraleucel in relapsed/refractory large B-cell lymphoma. 2020.
"""

    annotated = annotate_local_candidate_completeness(candidates, references_text=references_text)

    assert annotated[0]["needs_review"] is True
    assert "[35]" in annotated[0]["review_warning"]
