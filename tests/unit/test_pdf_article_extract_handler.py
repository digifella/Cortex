from __future__ import annotations

import json
import zipfile
from pathlib import Path

import fitz

from worker.handlers import pdf_article_extract as handler


def _build_bookmarked_pdf(path: Path) -> None:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "FEATURES", fontsize=12)
    page1.insert_text((72, 110), "The First Story", fontsize=22)
    page1.insert_text((72, 145), "BY TEST WRITER", fontsize=10)
    page1.insert_text((72, 190), "The first article starts here and keeps going.", fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Continuation of the first article.", fontsize=11)
    page2.insert_text((72, 300), "SCIENCE", fontsize=12)
    page2.insert_text((72, 340), "The Second Story", fontsize=22)
    page2.insert_text((72, 380), "BY NEXT WRITER", fontsize=10)
    page2.insert_text((72, 420), "The second article starts mid-page.", fontsize=11)

    page3 = doc.new_page()
    page3.insert_text((72, 72), "The second article continues on page three.", fontsize=11)

    doc.set_toc(
        [
            [1, "The First Story", 1],
            [1, "The Second Story", 2],
        ]
    )
    doc.save(str(path))
    doc.close()


def test_pdf_article_extract_handler_returns_zip_bundle(tmp_path):
    pdf_path = tmp_path / "stories.pdf"
    _build_bookmarked_pdf(pdf_path)

    result = handler.handle(
        input_path=pdf_path,
        input_data={"article_extract_options": {"segmentation_strategy": "pdf_bookmarks"}},
        job={"id": 10, "type": "pdf_article_extract"},
    )

    output_file = Path(result["output_file"])
    assert output_file.exists()
    assert output_file.suffix == ".zip"
    assert result["output_data"]["article_count"] == 2

    with zipfile.ZipFile(output_file, "r") as zf:
        assert "manifest.json" in zf.namelist()
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["article_count"] == 2
        first_md = zf.read(manifest["articles"][0]["output_file"]).decode("utf-8")
        second_md = zf.read(manifest["articles"][1]["output_file"]).decode("utf-8")

    assert "# The First Story" in first_md
    assert "Continuation of the first article" in first_md
    assert "The second article starts mid-page." in second_md
    assert "The second article continues on page three." in second_md
