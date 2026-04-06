from __future__ import annotations

from cortex_engine.review_study_miner import (
    _parse_reference_entries,
    ReviewMiningOptions,
    assess_review_documents,
    extract_review_table_blocks,
    mine_review_documents,
    mine_review_studies_from_text,
)


def test_review_study_miner_detects_systematic_review_and_table_candidates():
    markdown = """
# Systematic review of supportive care interventions

Methods
We conducted a systematic review following PRISMA guidance. Search strategy, eligibility criteria,
study selection, and characteristics of included studies are reported below.

| Study | Design | Outcomes |
| --- | --- | --- |
| Smith et al. 2020. Quality of life after intervention | Randomized controlled trial | Health-related quality of life, fatigue |
| Jones et al. 2019. Observational cohort | Cohort study | Survival only |
"""

    result = mine_review_studies_from_text(
        markdown,
        source_name="review.md",
        review_title="Supportive care review",
        options=ReviewMiningOptions(
            design_query="RCT clinical trial randomised randomized",
            outcome_query="health-related quality of life HRQoL QoL",
            require_all_criteria=True,
            include_reference_list_scan=False,
        ),
    )

    assert result["review_assessment"]["is_systematic_review"] is True
    assert result["stats"]["tables_detected"] == 1
    assert result["stats"]["candidates_total"] == 2
    assert result["stats"]["candidates_matching"] == 1
    assert result["candidates"][0]["meets_criteria"] is True
    assert "randomized" in " ".join(result["candidates"][0]["design_matches"]).lower()


def test_extract_review_table_blocks_returns_markdown_tables():
    markdown = """
| Study | Design |
| --- | --- |
| Smith 2020 | Randomized trial |
"""

    blocks = extract_review_table_blocks(markdown)

    assert len(blocks) == 1
    assert blocks[0]["row_count"] == 1
    assert "| Study | Design |" in blocks[0]["markdown"]


def test_extract_review_table_blocks_captures_nearby_context():
    markdown = """
# Systematic review of lymphoma interventions

Table 2. Characteristics of included studies
Outcomes were grouped by quality of life domains.
| Study | Design |
| --- | --- |
| Smith 2020 | Randomized trial |
Follow-up was typically 12 months.
"""

    blocks = extract_review_table_blocks(markdown)

    assert len(blocks) == 1
    assert "Table 2. Characteristics of included studies" in blocks[0]["context_text"]
    assert "Outcomes were grouped by quality of life domains." in blocks[0]["context_text"]
    assert "Follow-up was typically 12 months." in blocks[0]["context_text"]


def test_parse_reference_entries_accepts_bare_reference_excerpt():
    entries = _parse_reference_entries(
        """
        [19] Maziarz RT, Schuster SJ. Quality of life outcomes after CAR-T therapy in lymphoma. Leukemia & Lymphoma. 2020.
        [20] Patrick DL, Ghosh N. Treatment-related quality of life in diffuse large B-cell lymphoma. Lancet Haematology. 2021.
        """
    )

    assert len(entries) == 2
    assert entries[0]["reference_number"] == "19"
    assert entries[1]["year"] == "2021"


def test_parse_reference_entries_normalizes_wrapped_words_and_urls():
    entries = _parse_reference_entries(
        """
        70. ClinicalTrials.gov. Study evaluating the safety and efficacy of KTE-C19 in adult participants with refractory aggressive non-
        Hodgkin lymphoma (ZUMA-1 (NCT02348216). 2015. https://
        clini​caltr​ials.​gov/​ct2/​show/​NCT02​348216. Accessed 13 Dec 2022.
        """
    )

    assert len(entries) == 1
    assert entries[0]["reference_number"] == "70"
    assert "non-Hodgkin lymphoma" in entries[0]["title"]
    assert "https://clinicaltrials.gov/ct2/show/NCT02348216" in entries[0]["entry_text"]


def test_parse_reference_entries_repairs_mojibake_punctuation():
    entries = _parse_reference_entries(
        """
        69. Casasnovas RO, Daniele P, Tremblay G, et al. PCN325 Health utility in relapsed/refractory diffuse large B-cell lymphoma (RR-DLBCL) patientsâ€”results of a phase II trial with ORAL selinexor. Value Health. 2020;23(suppl 2):S479-80.
        """
    )

    assert len(entries) == 1
    assert "â€”" not in entries[0]["title"]
    assert entries[0]["title"] == "PCN325 Health utility in relapsed/refractory diffuse large B-cell lymphoma (RR-DLBCL) patients-results of a phase II trial with ORAL selinexor"
    assert entries[0]["journal"] == "Value Health"


def test_parse_reference_entries_keeps_vs_titles_intact():
    entries = _parse_reference_entries(
        """
        54. Li Y, et al. Cost-effectiveness analysis of axicabtagene ciloleucel vs. salvage chemotherapy for relapsed or refractory adult diffuse large B-cell lymphoma in China. Front Public Health. 2022;10:123.
        """
    )

    assert len(entries) == 1
    assert entries[0]["title"] == "Cost-effectiveness analysis of axicabtagene ciloleucel vs salvage chemotherapy for relapsed or refractory adult diffuse large B-cell lymphoma in China"
    assert entries[0]["journal"] == "Front Public Health"


def test_review_study_miner_scans_reference_list_for_keyword_matches():
    markdown = """
# Meta-analysis of survivorship interventions

This meta-analysis and systematic review followed PRISMA.

References
1. Smith J, Brown A. Randomized clinical trial of quality of life support in oncology. Journal of Supportive Care. 2021.
2. Lee P, Wong R. Biomarker assay development for oncology workflows. Lab Medicine. 2020.
"""

    result = mine_review_studies_from_text(
        markdown,
        source_name="review.md",
        review_title="Survivorship review",
        options=ReviewMiningOptions(
            design_query="RCT clinical trial",
            outcome_query="quality of life",
            require_all_criteria=True,
            include_reference_list_scan=True,
        ),
    )

    assert result["review_assessment"]["is_systematic_review"] is True
    assert result["stats"]["reference_entries_detected"] >= 1
    assert result["stats"]["candidates_matching"] >= 1
    assert any(item["source_section"] == "references" and item["meets_criteria"] for item in result["candidates"])


def test_review_study_miner_links_table_row_to_numbered_reference_entry():
    markdown = """
# Systematic review of lymphoma interventions

This systematic review followed PRISMA and reports included studies.

| Study | Design | Outcomes |
| --- | --- | --- |
| Maziarz 2020 [19] | Randomized clinical trial | Health-related quality of life |

References
[19] Maziarz RT, Schuster SJ. Randomized clinical trial of health-related quality of life after CAR-T therapy in lymphoma. Journal of Clinical Oncology. 2020. 10.1000/maziarz2020
"""

    result = mine_review_studies_from_text(
        markdown,
        source_name="review.md",
        review_title="Lymphoma review",
        options=ReviewMiningOptions(
            design_query="RCT clinical trial randomised randomized",
            outcome_query="health-related quality of life HRQoL QoL",
            require_all_criteria=True,
            include_reference_list_scan=False,
        ),
    )

    assert result["stats"]["table_reference_links"] == 1
    assert result["candidates"][0]["reference_number"] == "19"
    assert result["candidates"][0]["reference_match_method"] == "reference_number"
    assert result["candidates"][0]["doi"] == "10.1000/maziarz2020"
    assert "Randomized clinical trial of health-related quality of life" in result["candidates"][0]["title"]


def test_review_study_miner_falls_back_to_author_year_reference_match():
    markdown = """
# Systematic review of lymphoma survivorship

This systematic review followed PRISMA and included studies are summarized below.

| Study | Design | Outcomes |
| --- | --- | --- |
| Maziarz 2020 | Randomized clinical trial | Quality of life |

References
19. Maziarz RT, Schuster SJ. Quality of life outcomes after CAR-T therapy in lymphoma. Leukemia & Lymphoma. 2020.
"""

    result = mine_review_studies_from_text(
        markdown,
        source_name="review.md",
        review_title="Lymphoma survivorship review",
        options=ReviewMiningOptions(
            design_query="RCT clinical trial",
            outcome_query="quality of life",
            require_all_criteria=True,
            include_reference_list_scan=False,
        ),
    )

    assert result["stats"]["table_reference_links"] == 1
    assert result["candidates"][0]["reference_number"] == "19"
    assert result["candidates"][0]["reference_match_method"] == "author_year_fuzzy"
    assert "Quality of life outcomes after CAR-T therapy in lymphoma" in result["candidates"][0]["title"]


def test_review_study_miner_filters_structural_table_labels():
    markdown = """
# Systematic review of utility values

This systematic review followed PRISMA and included studies are summarized below.

| Study | Design | Outcomes |
| --- | --- | --- |
| HRQOL outcomes | Region | Study design |
| Source of utility values | Region | Utility values in remission |
| Maziarz 2020 [19] | Randomized clinical trial | Health-related quality of life |

References
[19] Maziarz RT, Schuster SJ. Randomized clinical trial of health-related quality of life after CAR-T therapy in lymphoma. Journal of Clinical Oncology. 2020.
"""

    result = mine_review_studies_from_text(
        markdown,
        source_name="review.md",
        review_title="Utility review",
        options=ReviewMiningOptions(
            design_query="RCT clinical trial randomised randomized",
            outcome_query="health-related quality of life HRQoL QoL",
            require_all_criteria=True,
            include_reference_list_scan=False,
        ),
    )

    assert result["stats"]["candidates_total"] == 1
    assert result["candidates"][0]["reference_number"] == "19"
    assert result["candidates"][0]["title"].startswith("Randomized clinical trial")


def test_review_study_miner_flags_reference_mismatch_for_human_review():
    markdown = """
# Systematic review of lymphoma interventions

This systematic review followed PRISMA and reports included studies.

| Study | Design | Outcomes |
| --- | --- | --- |
| Maziarz 2020 [19] | Randomized clinical trial | Health-related quality of life |

References
[19] Smith J, Brown A. Completely different lymphoma supportive care paper. Journal of Supportive Care. 2018.
"""

    result = mine_review_studies_from_text(
        markdown,
        source_name="review.md",
        review_title="Lymphoma review",
        options=ReviewMiningOptions(
            design_query="RCT clinical trial randomised randomized",
            outcome_query="health-related quality of life HRQoL QoL",
            require_all_criteria=True,
            include_reference_list_scan=False,
        ),
    )

    assert result["stats"]["reference_mismatches"] == 1
    assert result["stats"]["needs_review"] == 1
    assert result["candidates"][0]["reference_validation"] == "mismatch"
    assert result["candidates"][0]["needs_review"] is True
    assert "does not match" in result["candidates"][0]["review_warning"]


def test_assess_review_documents_marks_systematic_review_candidates():
    result = assess_review_documents(
        [
            {"source_name": "review_a.md", "review_title": "Review A", "text": "Systematic review with PRISMA and included studies."},
            {"source_name": "note_b.md", "review_title": "Note B", "text": "Short commentary without review signals."},
        ]
    )

    assert result["stats"]["documents_total"] == 2
    assert result["stats"]["systematic_review_likely"] == 1
    assert result["documents"][0]["confirm_review"] is True
    assert result["documents"][1]["confirm_review"] is False


def test_mine_review_documents_groups_candidates_by_confirmed_review():
    documents = [
        {
            "source_name": "review_a.md",
            "review_title": "Review A",
            "text": """
            Systematic review with PRISMA and characteristics of included studies.
            | Study | Design | Outcomes |
            | --- | --- | --- |
            | Smith et al. 2020. Trial of quality of life support | Randomized controlled trial | Health-related quality of life |
            """,
        },
        {
            "source_name": "review_b.md",
            "review_title": "Review B",
            "text": """
            Systematic review with PRISMA and included studies.
            | Study | Design | Outcomes |
            | --- | --- | --- |
            | Jones et al. 2019. Trial without QoL outcome | Randomized controlled trial | Survival |
            """,
        },
    ]

    result = mine_review_documents(
        documents,
        options=ReviewMiningOptions(
            design_query="RCT clinical trial randomised randomized",
            outcome_query="health-related quality of life quality of life",
            require_all_criteria=True,
            include_reference_list_scan=False,
        ),
        confirmed_doc_ids=[1],
    )

    assert result["stats"]["documents_total"] == 2
    assert result["stats"]["documents_confirmed"] == 1
    assert result["stats"]["reviews_mined"] == 1
    assert result["stats"]["candidates_total"] == 1
    assert result["candidates"][0]["source_review"] == "review_a.md"
    assert result["candidates"][0]["extra_fields"]["source_review_title"] == "Review A"
