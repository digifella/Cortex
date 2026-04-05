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
