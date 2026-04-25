from __future__ import annotations

import fitz

from cortex_engine.included_study_slicer import slice_review_pdf


def test_slice_review_pdf_creates_table_pdfs_and_bibliography(tmp_path):
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Table 2 Overview of included studies on HRQOL measures\nFACT-G JULIET Maziarz 2020 [19]")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Table 3 Overview of economic studies reporting health state utility values\nUS CUA Betts 2020 [61]")
    page3 = doc.new_page()
    page3.insert_text((72, 72), "References\n[19] Maziarz RT. Patient-reported long-term quality of life. 2020.\n[61] Betts KA. Cost utility analysis. 2020.")
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out"))

    assert [item["table_number"] for item in result["table_slices"]] == ["2", "3"]
    assert result["bibliography_pages"] == [3]
    assert len(result["bibliography_entries"]) >= 2
    for item in result["table_slices"]:
        assert item["pdf_path"]


def test_slice_review_pdf_ignores_body_mentions_and_detects_reference_pages(tmp_path):
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Narrative text says utilities were reported in Table 3 and Table 4.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Table 2 Overview of included studies on HRQOL measures\nFACT-G JULIET Maziarz 2020 [19]")
    page3 = doc.new_page()
    page3.insert_text((72, 72), "Table 2 (continued)\nMore rows")
    page4 = doc.new_page()
    page4.insert_text((72, 72), "18.\t First reference entry.\n19.\t Second reference entry.\n20.\t Third reference entry.\n21.\t Fourth reference entry.")
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out2"))

    assert [item["table_number"] for item in result["table_slices"]] == ["2"]
    assert result["table_slices"][0]["page_numbers"] == [2, 3]
    assert result["bibliography_pages"] == [4]


def test_slice_review_pdf_recognises_cont_abbreviation_and_study_design_title(tmp_path):
    """MDPI-style reviews use 'Table 1. Cont.' for continuations and titles like
    'Study design and participants' characteristics' for the included-studies table."""
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    for _ in range(5):
        doc.new_page().insert_text((72, 72), "Body text only; no table headings here.")
    doc.new_page().insert_text((72, 72), "Table 1. Study design and participants characteristics.\nRow A")
    doc.new_page().insert_text((72, 72), "Table 1. Cont.\nRow B")
    doc.new_page().insert_text((72, 72), "Table 2. Exercise intervention characteristics.\nRow C")
    doc.new_page().insert_text((72, 72), "Table 2. Cont.\nRow D")
    doc.new_page().insert_text((72, 72), "Table 2. Cont.\nRow E")
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out3"))

    assert [item["table_number"] for item in result["table_slices"]] == ["1", "2"]
    assert result["table_slices"][0]["page_numbers"] == [6, 7]
    assert result["table_slices"][1]["page_numbers"] == [8, 9, 10]
    assert all(item["kind"] == "included_studies" for item in result["table_slices"])


def test_slice_review_pdf_detects_bare_numbered_references(tmp_path):
    """Many journals (e.g. MDPI Healthcare) format references with the number on
    its own line: '4.\\nAuthors...'. The bibliography detector must still fire."""
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    for idx in range(6):
        doc.new_page().insert_text((72, 72), f"Body page {idx + 1}. No tables.")
    doc.new_page().insert_text(
        (72, 72),
        "4.\nGoswami, P. HM-PRO reliability. Front. Pharmacol. 2020, 11, 571066.\n"
        "5.\nArpinelli, F. FDA Guidance. Health Qual Life Outcomes 2006, 4, 85.\n"
        "6.\nLin, X.-J. Methodological Issues in HRQoL. Tzu Chi Med. J. 2013, 25, 8.",
    )
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out4"))

    assert result["diagnostics"]["bibliography_start_page"] == 7
    assert result["bibliography_pages"] == [7]


def test_slice_review_pdf_detects_references_heading_low_on_page(tmp_path):
    pdf_path = tmp_path / "late_references.pdf"
    doc = fitz.open()
    for idx in range(5):
        doc.new_page().insert_text((72, 72), f"Body page {idx + 1}.")
    doc.new_page().insert_text(
        (72, 72),
        "\n".join(
            [
                "Conclusions",
                "Exercise should be considered supportive care.",
                "Author Contributions: A.B. and C.D.",
                "Funding: none.",
                "Conflicts of Interest: none.",
                "References",
                "1.",
                "Karagianni, P.; Giannouli, S. Hematologic malignancy. Journal 2021, 22, 6321.",
                "2.",
                "Zhang, N.; Wu, J. Global burden of hematologic malignancies. Blood Cancer J. 2023, 13, 82.",
                "3.",
                "Hemminki, K.; Hemminki, J. Survival in hematological malignancies. Leukemia 2023, 37, 854-863.",
            ]
        ),
    )
    doc.new_page().insert_text(
        (72, 72),
        "4.\nGoswami, P. Reliability of a hematological malignancy PRO measure. Front. Pharmacol. 2020, 11, 571066.",
    )
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out_late_refs"))

    assert result["bibliography_pages"] == [6, 7]
    assert [entry["reference_number"] for entry in result["bibliography_entries"][:4]] == ["1", "2", "3", "4"]
    assert [entry["authors"] for entry in result["bibliography_entries"][:3]] == ["Karagianni", "Zhang", "Hemminki"]


def test_slice_review_pdf_supports_roman_numeral_tables(tmp_path):
    """Older journal formats (e.g. Wiley Transfusion, some BMC/Arden papers)
    number tables with Roman numerals. The slicer must detect 'Table I' /
    'Table II' and track continuations across pages."""
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table I. Description of measures used in the review.\nRow A")
    doc.new_page().insert_text((72, 72), "Table II. Study characteristics of included RCTs.\nStudy Alpha")
    doc.new_page().insert_text((72, 72), "Table II. Continued.\nStudy Beta")
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out_roman"))

    # Table I is a "measures used" table -> classified 'other' and dropped.
    # Table II is included-studies -> kept, with its continuation merged.
    assert [item["table_number"] for item in result["table_slices"]] == ["2"]
    assert result["table_slices"][0]["page_numbers"] == [2, 3]
    kinds = {c["table_number"]: c["kind"] for c in result["diagnostics"]["classified_slices"]}
    assert kinds == {"1": "other", "2": "included_studies"}


def test_slice_review_pdf_handles_mid_page_multiple_tables(tmp_path):
    """Some reviews place multiple table captions on a single page (Estcourt
    2013 has Tables 3 and 4 stacked on the same layout page). The slicer must
    scan every line rather than the top N and treat each as its own start."""
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Introduction page with no headings.")
    doc.new_page().insert_text(
        (72, 72),
        "Methods used three phases of screening.\n"
        "Table 3. Study characteristics of included RCTs.\nStudy A\n"
        "Table 4. Risk of bias of included studies.\nStudy B",
    )
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out_midpage"))

    numbers = sorted(item["table_number"] for item in result["table_slices"])
    assert numbers == ["3", "4"]


def test_slice_review_pdf_detects_tables_after_bibliography(tmp_path):
    """Some reviews place supplementary tables after References. The slicer
    must still emit those slices and cap bibliography coverage at the first
    post-references table page instead of swallowing the whole tail."""
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    for idx in range(4):
        doc.new_page().insert_text((72, 72), f"Body page {idx + 1}. No tables.")
    doc.new_page().insert_text(
        (72, 72),
        "References\n[1] Foo et al. 2020.\n[2] Bar et al. 2021.\n[3] Baz et al. 2022.",
    )
    doc.new_page().insert_text(
        (72, 72),
        "[4] Qux et al. 2023. Journal of Reviews.\n[5] Quux et al. 2024. Journal of Reviews.",
    )
    doc.new_page().insert_text(
        (72, 72),
        "Table 1. Characteristics of included studies.\nStudy Alpha\nStudy Beta",
    )
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out_postbib"))

    assert [item["table_number"] for item in result["table_slices"]] == ["1"]
    assert result["table_slices"][0]["page_numbers"] == [7]
    assert result["bibliography_pages"] == [5, 6]
    assert result["diagnostics"]["bibliography_start_page"] == 5


def test_slice_review_pdf_rejects_pronoun_and_artifact_captions(tmp_path):
    """Line-start 'Table N' matching is aggressive; the caption filter must
    reject body text like 'Table I. We identified...' (pronoun opening) and
    column-break artefacts like 'Table 3). The prospective...' while keeping
    real captions on the same page."""
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (72, 72),
        "Table I. We identified 482 records in PubMed.\n"
        "Table 3). The prospective studies were small.\n"
        "Table 2. Study characteristics of included RCTs.\nStudy A",
    )
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out_pronoun"))

    assert [item["table_number"] for item in result["table_slices"]] == ["2"]
    numbers_detected = {h["table_number"] for h in result["diagnostics"]["headings_detected"]}
    assert 1 not in numbers_detected
    assert 3 not in numbers_detected
    assert 2 in numbers_detected


def test_slice_review_pdf_extends_table_across_unmarked_continuation_pages(tmp_path):
    """Reviews like Arden 2010 flow table rows onto subsequent pages without a
    'Table N (continued)' marker. The slicer must still merge those pages and
    stop at boundaries (next table start, bibliography, or Figure caption)."""
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    for idx in range(4):
        doc.new_page().insert_text((72, 72), f"Body page {idx + 1}. No tables.")
    doc.new_page().insert_text(
        (72, 72),
        "References\n1. Smith et al. 2020.\n2. Jones et al. 2021.\n3. Brown et al. 2022.",
    )
    doc.new_page().insert_text(
        (72, 72),
        "Table II: Correlates of quality of life: studies with comparison group\n"
        "1. Alpha et al. (2001) [10]\nRow A",
    )
    doc.new_page().insert_text(
        (72, 72),
        "4. Beta et al. (2003) [11]\nRow B continuing Table II",
    )
    doc.new_page().insert_text(
        (72, 72),
        "7. Gamma et al. (2005) [12]\nRow C continuing Table II",
    )
    doc.new_page().insert_text(
        (72, 72),
        "Table III: Correlates of quality of life: studies without comparison group\n"
        "1. Delta et al. (2007) [13]\nRow D",
    )
    doc.new_page().insert_text(
        (72, 72),
        "6. Epsilon et al. (2009) [14]\nRow E continuing Table III",
    )
    doc.new_page().insert_text(
        (72, 72),
        "Figure 1: Search process\nKeyword search generated hits",
    )
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out_unmarked"))

    numbers = [item["table_number"] for item in result["table_slices"]]
    assert numbers == ["2", "3"]
    table_pages = {item["table_number"]: item["page_numbers"] for item in result["table_slices"]}
    assert table_pages["2"] == [6, 7, 8]
    assert table_pages["3"] == [9, 10]


def test_slice_review_pdf_diagnostics_record_drops(tmp_path):
    """When all candidate slices are classified as 'other', the diagnostics block
    must expose the headings and classifications so operators can see why."""
    pdf_path = tmp_path / "review.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 1. Search strategy.\nPubMed: ...")
    doc.new_page().insert_text((72, 72), "Table 2. PRISMA flow counts.\nRecords identified 4820")
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out5"))

    assert result["table_slices"] == []
    diag = result["diagnostics"]
    assert diag["total_pages"] == 2
    assert len(diag["headings_detected"]) == 2


def test_slice_review_pdf_detects_cochrane_characteristics_and_reference_sections(tmp_path):
    pdf_path = tmp_path / "cochrane.pdf"
    doc = fitz.open()
    for idx in range(3):
        doc.new_page().insert_text((72, 72), f"Body page {idx + 1}. No tables.")
    doc.new_page().insert_text(
        (72, 72),
        "R E F E R E N C E S\n"
        "References to studies included in this review\n"
        "Cohen 2004 {published data only}\n"
        "* Cohen L, Warneke C. Randomized trial of Tibetan yoga in lymphoma. Cancer 2004;100(10):2253-60.\n"
        "References to studies excluded from this review\n"
        "Adamsen 2006 {published data only}\n"
        "Adamsen L, Quist M. Exercise intervention in cancer patients. Supportive Care in Cancer 2006;14(2):116-27.",
    )
    doc.new_page().insert_text(
        (72, 72),
        "C H A R A C T E R I S T I C S   O F   S T U D I E S\n"
        "Characteristics of included studies [ordered by study ID]\n"
        "Methods\nStudy design: RCT.\nParticipants\nPatients (N = 39)\nCohen 2004",
    )
    doc.new_page().insert_text(
        (72, 72),
        "Cohen 2004  (Continued)\nOutcomes\nQuality of sleep assessed by PSQI.",
    )
    doc.new_page().insert_text(
        (72, 72),
        "Characteristics of excluded studies [ordered by study ID]\n"
        "Study\nReason for exclusion\n"
        "Adamsen 2006\nThis study was excluded as it was not an RCT.",
    )
    doc.save(str(pdf_path))
    doc.close()

    result = slice_review_pdf(str(pdf_path), work_dir=str(tmp_path / "out_cochrane"))

    assert result["bibliography_pages"] == [4]
    assert {entry["authors"] for entry in result["bibliography_entries"]} >= {"Cohen", "Adamsen"}
    assert [(item["label"], item["kind"], item["page_numbers"]) for item in result["table_slices"]] == [
        ("Cochrane included: Cohen 2004", "included_studies", [5, 6]),
        ("Cochrane excluded studies", "excluded_studies", [7]),
    ]
    assert result["table_slices"][0]["cochrane_study_id"] == "Cohen 2004"
    assert result["diagnostics"]["cochrane_sections_detected"] == [
        {
            "page": 5,
            "section": "included",
            "table_number": 901,
            "title": "Characteristics of included studies",
            "kind": "included_studies",
        },
        {
            "page": 7,
            "section": "excluded",
            "table_number": 902,
            "title": "Characteristics of excluded studies",
            "kind": "excluded_studies",
        },
    ]
