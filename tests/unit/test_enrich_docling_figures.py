import base64
import io
from unittest.mock import patch

from llama_index.core import Document
from PIL import Image

from cortex_engine.enhanced_ingest_cortex import EnhancedDocumentProcessor


def _png_b64(size=(400, 400)) -> str:
    img = Image.new("RGB", size, color=(10, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_doc_with_figures(area_fracs):
    doc = Document(text="Some body text")
    figures = []
    payloads = []
    for i, frac in enumerate(area_fracs):
        figures.append({"index": i, "caption": "", "area_frac": frac})
        payloads.append({"index": i, "image_base64": _png_b64(), "area_frac": frac})
    doc.metadata["docling_figures"] = figures
    doc.metadata["docling_figures_payload"] = payloads
    return doc


def test_small_figure_below_threshold_is_never_captioned(tmp_path, monkeypatch):
    monkeypatch.setattr("cortex_engine.enhanced_ingest_cortex.DEFAULT_CACHE_PATH",
                         str(tmp_path / "cache.db"))
    processor = EnhancedDocumentProcessor(enable_docling=False)
    doc = _make_doc_with_figures([0.02])  # 2% -- below the 10% threshold
    with patch("cortex_engine.enhanced_ingest_cortex.caption_figure_with_haiku") as mock_caption:
        processor._enrich_docling_figures(doc, skip_image_processing=False)
        mock_caption.assert_not_called()
    assert "Figure Intelligence" not in doc.text


def test_large_figure_above_threshold_is_captioned(tmp_path, monkeypatch):
    monkeypatch.setattr("cortex_engine.enhanced_ingest_cortex.DEFAULT_CACHE_PATH",
                         str(tmp_path / "cache.db"))
    processor = EnhancedDocumentProcessor(enable_docling=False)
    doc = _make_doc_with_figures([0.35])  # 35% -- above the 10% threshold
    with patch("cortex_engine.enhanced_ingest_cortex.caption_figure_with_haiku",
               return_value="A blue rectangle.") as mock_caption:
        processor._enrich_docling_figures(doc, skip_image_processing=False)
        mock_caption.assert_called_once()
    assert "Figure Intelligence" in doc.text
    assert "A blue rectangle." in doc.text


def test_dedup_cache_prevents_second_haiku_call(tmp_path, monkeypatch):
    from cortex_engine.image_caption_cache import ImageCaptionCache
    cache_path = str(tmp_path / "dedup.db")
    monkeypatch.setattr("cortex_engine.enhanced_ingest_cortex.DEFAULT_CACHE_PATH", cache_path)

    processor = EnhancedDocumentProcessor(enable_docling=False)
    same_png = _png_b64()
    doc1 = Document(text="Doc one")
    doc1.metadata["docling_figures"] = [{"index": 0, "caption": "", "area_frac": 0.5}]
    doc1.metadata["docling_figures_payload"] = [{"index": 0, "image_base64": same_png, "area_frac": 0.5}]
    doc2 = Document(text="Doc two")
    doc2.metadata["docling_figures"] = [{"index": 0, "caption": "", "area_frac": 0.5}]
    doc2.metadata["docling_figures_payload"] = [{"index": 0, "image_base64": same_png, "area_frac": 0.5}]

    with patch("cortex_engine.enhanced_ingest_cortex.caption_figure_with_haiku",
               return_value="A shared logo.") as mock_caption:
        processor._enrich_docling_figures(doc1, skip_image_processing=False)
        processor._enrich_docling_figures(doc2, skip_image_processing=False)
        assert mock_caption.call_count == 1, "second identical image must hit the dedup cache, not call Haiku again"
    assert "A shared logo." in doc1.text
    assert "A shared logo." in doc2.text
