from worker.handlers import intel_extract as handler
from cortex_engine.intel_extractor import _build_entity_record, _merge_structured_results, _ocr_image_text


def test_intel_extract_handler_returns_structured_output(monkeypatch, tmp_path):
    output_path = tmp_path / "extract.md"
    output_path.write_text("# Extract", encoding="utf-8")

    monkeypatch.setattr(
        handler,
        "extract_intel",
        lambda payload: (
            {
                "status": "extracted",
                "org_name": payload["org_name"],
                "entity_count": 1,
                "entities": [{"canonical_name": "Carolyn Bell", "target_type": "person"}],
                "emails": [{"email": "cbell@example.com"}],
                "target_update_suggestions": [],
                "warnings": [],
            },
            output_path,
        ),
    )

    result = handler.handle(
        None,
        {
            "org_name": "Longboardfella",
            "subject": "Carolyn Bell screenshot",
        },
        {},
    )

    assert result["output_data"]["status"] == "extracted"
    assert result["output_data"]["entity_count"] == 1
    assert result["output_file"] == output_path


def test_build_entity_record_accepts_list_evidence():
    entity = _build_entity_record(
        {
            "name": "Carolyn Bell",
            "target_type": "person",
            "current_employer": "Silverchain",
            "confidence": 0.9,
            "evidence": ["Screenshot OCR", "Email body"],
        }
    )

    assert entity["canonical_name"] == "Carolyn Bell"
    assert entity["evidence"] == ["Screenshot OCR", "Email body"]


def test_merge_structured_results_combines_text_and_image_extractions():
    merged = _merge_structured_results(
        {
            "people": [{"name": "Carolyn Bell"}],
            "organisations": [],
            "emails": [],
            "career_events": [],
            "summary": "Body extraction.",
        },
        {
            "people": [],
            "organisations": [{"name": "Silverchain Group"}],
            "emails": [{"email": "cbell@example.com"}],
            "career_events": [],
            "summary": "Image extraction.",
        },
    )

    assert merged["people"][0]["name"] == "Carolyn Bell"
    assert merged["organisations"][0]["name"] == "Silverchain Group"
    assert merged["emails"][0]["email"] == "cbell@example.com"
    assert "Body extraction." in merged["summary"]
    assert "Image extraction." in merged["summary"]


def test_ocr_image_text_prefers_most_readable_output(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")

    monkeypatch.setattr("cortex_engine.intel_extractor.shutil.which", lambda name: "/usr/bin/tesseract")

    class _Result:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    outputs = iter([
        _Result("Paul Cooper\npaul@project-143.com\nProject 143"),
        _Result("A list"),
    ])

    monkeypatch.setattr(
        "cortex_engine.intel_extractor.subprocess.run",
        lambda *args, **kwargs: next(outputs),
    )

    text = _ocr_image_text(image_path)

    assert "Paul Cooper" in text
    assert "project-143.com" in text
