from __future__ import annotations

import io

import fitz
from PIL import Image

from cortex_engine.textifier import DocumentTextifier


def _write_blank_pdf(path, page_count: int = 1) -> None:
    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page(width=612, height=792)
    doc.save(path)
    doc.close()


def test_historical_scan_strategy_emits_page_ocr(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    _write_blank_pdf(pdf_path, page_count=2)

    textifier = DocumentTextifier(use_vision=False, pdf_strategy="historical_scan")
    ocr_results = iter(
        [
            {"text": "CONTENTS\nPART I. MAGNETISM.", "avg_conf": 91.0, "word_count": 4},
            {"text": "Fig. 67a - Details of a motor generator.", "avg_conf": 88.0, "word_count": 8},
        ]
    )

    monkeypatch.setattr(textifier, "_render_pdf_page_png", lambda page, zoom=2.0: b"png")
    monkeypatch.setattr(textifier, "_historical_scan_ocr_image", lambda image_bytes: next(ocr_results))
    monkeypatch.setattr(textifier, "_historical_scan_page_needs_vision", lambda *args, **kwargs: False)

    output = textifier.textify_pdf(str(pdf_path))

    assert "## Page 1" in output
    assert "PART I. MAGNETISM." in output
    assert "## Page 2" in output
    assert "Fig. 67a - Details of a motor generator." in output


def test_historical_scan_adds_vision_summary_for_weak_page(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    _write_blank_pdf(pdf_path)

    textifier = DocumentTextifier(use_vision=True, pdf_strategy="historical_scan")
    monkeypatch.setattr(textifier, "_render_pdf_page_png", lambda page, zoom=2.0: b"png")
    monkeypatch.setattr(
        textifier,
        "_historical_scan_ocr_image",
        lambda image_bytes: {"text": "Fig. 230", "avg_conf": 31.5, "word_count": 2},
    )
    monkeypatch.setattr(textifier, "_historical_scan_page_needs_vision", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        textifier,
        "_describe_image_with_timeout",
        lambda image_bytes: "A rotated graph with curves and a caption about constant K.",
    )

    output = textifier.textify_pdf(str(pdf_path))

    assert "> **[Page 1 image summary]**: A rotated graph with curves" in output
    assert "Low average word confidence (31.5)" in output


def test_parse_tesseract_tsv_reconstructs_lines():
    tsv = "\n".join(
        [
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t96.0\tMotor",
            "5\t1\t1\t1\t1\t2\t12\t0\t10\t10\t95.0\tGenerator",
            "5\t1\t1\t1\t2\t1\t0\t20\t10\t10\t90.0\tAssembled",
        ]
    )

    parsed = DocumentTextifier._parse_tesseract_tsv(tsv)

    assert parsed["text"] == "Motor Generator\nAssembled"
    assert parsed["word_count"] == 3
    assert parsed["avg_conf"] == 93.66666666666667


def test_screenshot_article_strategy_crops_and_ocr_pages(monkeypatch, tmp_path):
    pdf_path = tmp_path / "article.pdf"
    _write_blank_pdf(pdf_path, page_count=2)

    textifier = DocumentTextifier(use_vision=False, pdf_strategy="screenshot_article")
    ocr_results = iter(
        [
            {
                "text": "THE END OF READING IS HERE\nBy Rose Horowitch\nArticle paragraph one.",
                "avg_conf": 91.0,
                "word_count": 12,
            },
            {"text": "Sale ad", "avg_conf": 27.0, "word_count": 2},
        ]
    )

    monkeypatch.setattr(textifier, "_render_screenshot_article_page_png", lambda page: b"cropped")
    monkeypatch.setattr(textifier, "_screenshot_article_ocr_image", lambda image_bytes: next(ocr_results))

    output = textifier.textify_pdf(str(pdf_path))

    assert "THE END OF READING IS HERE" in output
    assert "Article paragraph one." in output
    assert "> **[Skipped]**: Low-confidence or low-text screenshot page" in output


def test_screenshot_article_crop_box_uses_configured_ratios(monkeypatch):
    monkeypatch.setenv("CORTEX_SCREENSHOT_ARTICLE_CROP_LEFT", "0.25")
    monkeypatch.setenv("CORTEX_SCREENSHOT_ARTICLE_CROP_TOP", "0.10")
    monkeypatch.setenv("CORTEX_SCREENSHOT_ARTICLE_CROP_RIGHT", "0.90")
    monkeypatch.setenv("CORTEX_SCREENSHOT_ARTICLE_CROP_BOTTOM", "0.95")

    assert DocumentTextifier._screenshot_article_crop_box(1000, 800) == (250, 80, 900, 760)


def test_crop_screenshot_article_image_removes_left_sidebar(monkeypatch):
    img = Image.new("RGB", (1000, 800), "white")
    path_bytes = io.BytesIO()
    img.save(path_bytes, format="PNG")

    textifier = DocumentTextifier(use_vision=False, pdf_strategy="screenshot_article")
    cropped = Image.open(io.BytesIO(textifier._crop_screenshot_article_image(path_bytes.getvalue())))

    assert cropped.size == (745, 760)
