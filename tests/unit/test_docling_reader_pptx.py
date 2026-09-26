import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

from cortex_engine.docling_reader import DoclingDocumentReader

HEATMAP_FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_findings_heatmap.pptx"
PICTURE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_cop_findings_with_picture.pptx"


def test_docling_converts_pptx_text():
    # This fixture is built entirely from native PPTX shapes (text boxes,
    # connectors) -- no embedded picture -- so it exercises PPTX text
    # extraction specifically, without exercising figure extraction.
    reader = DoclingDocumentReader(ocr_enabled=False, table_structure_recognition=False,
                                    skip_vlm_processing=True)
    assert reader.is_available, "Docling must be available for this test"
    docs = reader.load_data(str(HEATMAP_FIXTURE))
    assert len(docs) == 1
    assert "Ease of Implementation" in docs[0].text


def test_docling_reports_figure_area_for_a_real_embedded_picture():
    reader = DoclingDocumentReader(ocr_enabled=False, table_structure_recognition=False,
                                    skip_vlm_processing=True)
    docs = reader.load_data(str(PICTURE_FIXTURE))
    assert len(docs) == 1
    figures = docs[0].metadata.get("docling_figures") or []
    assert figures, "expected at least one figure entry from a file with real embedded pictures"
    # every figure_entry must carry an area_frac key (value may be None, but key must exist)
    for entry in figures:
        assert "area_frac" in entry
    # at least one figure on a genuine slide/page must have a numeric area_frac
    assert any(isinstance(entry["area_frac"], float) for entry in figures)
