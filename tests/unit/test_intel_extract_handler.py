import sys
import types
import io
from base64 import b64decode

from PIL import Image

from worker.handlers import intel_extract as handler
from cortex_engine.intel_extractor import (
    _build_entity_record,
    _extract_linkedin_feed_structured,
    _merge_structured_results,
    _ocr_image_text,
    _prepare_anthropic_image_bytes,
    _triage_email_payload,
    _call_haiku_image_extract,
    extract_intel,
)
from cortex_engine.org_chart_extractor import analyse_org_chart_attachments, extract_org_chart_structured


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


def test_extract_linkedin_feed_structured_recovers_people_and_company():
    text = """
    Feed post
    [image: View Rachel Choong's profile]
    <https://www.linkedin.com/in/rachel-choong-80160014b/>
    Rachel Choong
    • 1st
    CEO and CIO Awards Winner | Cyber Security Program Lead | AI Cybersecurity Specialist
    1h •

    [image: View company: BoardEffect, a Diligent Brand]
    <https://www.linkedin.com/company/boardeffect/posts/>
    BoardEffect, a Diligent Brand
    AI introduces campus-wide risks and opportunities. Use proven governance frameworks to guide responsible adoption.
    """

    structured = _extract_linkedin_feed_structured(text)

    assert structured["people"][0]["name"] == "Rachel Choong"
    assert "Cyber Security Program Lead" in structured["people"][0]["current_role"]
    assert structured["people"][0]["linkedin_url"] == "https://www.linkedin.com/in/rachel-choong-80160014b/"
    assert structured["organisations"][0]["name"] == "BoardEffect, a Diligent Brand"


def test_extract_linkedin_feed_structured_cleans_company_names_with_embedded_urls():
    text = """
    [image: View company: Canva] <https://www.linkedin.com/company/canva/posts/>
    <https://www.linkedin.com/company/canva/posts/>
    Canva
    Teams are using Canva to speed up campaign design and collaborative brand work across the organisation.
    """

    structured = _extract_linkedin_feed_structured(text)

    assert structured["organisations"][0]["name"] == "Canva"


def test_extract_linkedin_feed_structured_skips_passive_like_or_promoted_blocks():
    text = """
    [image: View Russell Yardley's profile]
    <https://www.linkedin.com/in/russellyardley/>
    Russell Yardley
    Professional Non-Exec Director
    Sue O'Connor and 5 others reacted

    [image: View company: BoardEffect, a Diligent Brand]
    <https://www.linkedin.com/company/boardeffect/posts/>
    BoardEffect, a Diligent Brand
    Promoted
    """

    structured = _extract_linkedin_feed_structured(text)

    assert structured["people"] == []
    assert structured["organisations"] == []


def test_extract_linkedin_feed_structured_keeps_commented_case():
    text = """
    [image: View Russell Yardley's profile]
    <https://www.linkedin.com/in/russellyardley/>
    Russell Yardley
    Professional Non-Exec Director
    commented on this
    Clear substantive comment text about governance and public policy in healthcare systems.
    """

    structured = _extract_linkedin_feed_structured(text)

    assert structured["people"][0]["name"] == "Russell Yardley"


def test_extract_intel_uses_linkedin_feed_fallback_when_model_returns_empty(monkeypatch):
    monkeypatch.setattr("cortex_engine.intel_extractor._call_haiku_extract", lambda payload, combined_text: {})
    monkeypatch.setattr("cortex_engine.intel_extractor._maybe_textifier", lambda: None)
    monkeypatch.setattr("cortex_engine.intel_extractor.StakeholderSignalStore", lambda: type("S", (), {"list_profiles": lambda self, org_name="": []})())

    result, output_path = extract_intel(
        {
            "org_name": "Longboardfella",
            "subject": "",
            "raw_text": """
            [image: View Rachel Choong's profile]
            <https://www.linkedin.com/in/rachel-choong-80160014b/>
            Rachel Choong
            CEO and CIO Awards Winner | Cyber Security Program Lead | AI Cybersecurity Specialist
            """,
            "html_text": "",
            "attachments": [],
            "message_id": "<sample>",
        }
    )

    assert result["entity_count"] >= 1
    assert any(item["canonical_name"] == "Rachel Choong" for item in result["entities"])
    assert output_path is not None


def test_extract_intel_does_not_emit_profile_updates_from_heuristic_feed_noise(monkeypatch):
    class _Store:
        def list_profiles(self, org_name=""):
            return [
                {
                    "profile_key": "pk_russell",
                    "target_type": "person",
                    "canonical_name": "Russell Yardley",
                    "current_role": "Professional Non-Exec Director",
                    "linkedin_url": "https://www.linkedin.com/in/russellyardley/",
                    "affiliations": [],
                    "aliases": [],
                    "known_employers": [],
                }
            ]

    monkeypatch.setattr("cortex_engine.intel_extractor._call_haiku_extract", lambda payload, combined_text: {})
    monkeypatch.setattr("cortex_engine.intel_extractor._maybe_textifier", lambda: None)
    monkeypatch.setattr("cortex_engine.intel_extractor.StakeholderSignalStore", lambda: _Store())

    result, _ = extract_intel(
        {
            "org_name": "Longboardfella",
            "subject": "Fwd:",
            "raw_text": """
            [image: View Rachel Choong's profile]
            <https://www.linkedin.com/in/rachel-choong-80160014b/>
            Rachel Choong
            CEO and CIO Awards Winner | Cyber Security Program Lead | AI Cybersecurity Specialist

            [image: View Russell Yardley's profile]
            <https://www.linkedin.com/in/russellyardley/>
            Russell Yardley
            Professional Non-Exec Director
            """,
            "html_text": "",
            "attachments": [],
            "message_id": "<sample2>",
        }
    )

    assert any(item["canonical_name"] == "Rachel Choong" for item in result["entities"])
    assert result["target_update_suggestions"] == []


def test_prepare_anthropic_image_bytes_downscales_oversized_png(tmp_path):
    image_path = tmp_path / "org-chart.png"
    Image.new("RGB", (9001, 240), color="white").save(image_path, format="PNG")

    prepared_bytes, prepared_mime = _prepare_anthropic_image_bytes(image_path, "image/png")

    assert prepared_mime in {"image/png", "image/jpeg"}
    with Image.open(image_path) as original_img:
        original_size = original_img.size
    with Image.open(io.BytesIO(prepared_bytes)) as prepared_img:
        assert max(prepared_img.size) <= 4096
        assert prepared_img.size[0] < original_size[0]


def test_call_haiku_image_extract_sends_resized_image_to_anthropic(monkeypatch, tmp_path):
    image_path = tmp_path / "org-chart.png"
    Image.new("RGB", (9001, 240), color="white").save(image_path, format="PNG")
    captured = {}

    class _FakeResponse:
        content = [types.SimpleNamespace(text='{"people":[],"organisations":[],"emails":[],"career_events":[],"summary":"ok"}')]

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr("cortex_engine.intel_extractor._anthropic_client", lambda: _FakeClient())

    result = _call_haiku_image_extract(
        {"org_name": "Escient", "subject": "Barwon Water"},
        {
            "filename": "org-chart.png",
            "stored_path": str(image_path),
            "mime_type": "image/png",
        },
    )

    assert result["summary"] == "ok"
    source = captured["messages"][0]["content"][1]["source"]
    prepared_bytes = b64decode(source["data"])
    assert source["media_type"] in {"image/png", "image/jpeg"}
    with Image.open(io.BytesIO(prepared_bytes)) as prepared_img:
        assert max(prepared_img.size) <= 4096


def test_extract_org_chart_structured_recovers_name_and_role_pairs():
    structured = extract_org_chart_structured(
        attachment_texts=[
            """
            Attachment: racp_org_chart.pdf
            Chief Executive Officer
            Mic Cavazzini
            Head of Strategy & Transformation
            Gillian Dunn
            Director, Digital Health
            Karen Smith
            """
        ],
        attachment_summaries=[
            {
                "filename": "racp_org_chart.pdf",
                "status": "processed",
            }
        ],
        employer_hint="RACP",
    )

    assert any(item["name"] == "Mic Cavazzini" and item["current_role"] == "Chief Executive Officer" for item in structured["people"])
    assert any(item["name"] == "Gillian Dunn" and "Strategy" in item["current_role"] for item in structured["people"])


def test_analyse_org_chart_attachments_recognizes_management_structure_filename():
    analysis = analyse_org_chart_attachments(
        [
            {
                "filename": "senior-management-structure-department-of-health-jan-2026.pdf",
                "status": "processed",
                "excerpt": "SECRETARY JENNY ATTA CHIEF EXECUTIVE OFFICER",
            }
        ]
    )

    assert analysis["attachment_count"] == 1


def test_extract_intel_uses_org_chart_heuristic_for_attachment_text(monkeypatch, tmp_path):
    attachment_path = tmp_path / "racp_org_chart.pdf"
    attachment_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr("cortex_engine.intel_extractor._call_haiku_extract", lambda payload, combined_text: {})
    monkeypatch.setattr("cortex_engine.intel_extractor._maybe_textifier", lambda: type("T", (), {"textify_file": lambda self, path: """
        Chief Executive Officer
        Mic Cavazzini
        Head of Strategy & Transformation
        Gillian Dunn
    """})())
    monkeypatch.setattr("cortex_engine.intel_extractor.StakeholderSignalStore", lambda: type("S", (), {"list_profiles": lambda self, org_name='': []})())

    result, output_path = extract_intel(
        {
            "org_name": "Longboardfella",
            "subject": "RACP org chart",
            "raw_text": "",
            "html_text": "",
            "attachments": [
                {
                    "filename": "racp_org_chart.pdf",
                    "stored_path": str(attachment_path),
                    "mime_type": "application/pdf",
                    "kind": "document",
                }
            ],
            "message_id": "<orgchart1>",
            "parsed_candidate_employer": "RACP",
        }
    )

    assert any(item["canonical_name"] == "Mic Cavazzini" and item.get("current_role") == "Chief Executive Officer" for item in result["entities"])
    assert any(item["canonical_name"] == "Gillian Dunn" and "Strategy" in item.get("current_role", "") for item in result["entities"])
    assert output_path is not None


def test_extract_org_chart_structured_ignores_non_chart_documents():
    structured = extract_org_chart_structured(
        attachment_texts=[
            """
            Attachment: 2026-2030-strategic-direction-document_compressed.pdf
            Royal Australasian College of Physicians
            Strategic Direction 2026 to 2030
            Professor Jennifer Martin
            President and Chair of the Board
            Steffen Faurby
            Chief Executive Officer
            """
        ],
        attachment_summaries=[
            {
                "filename": "2026-2030-strategic-direction-document_compressed.pdf",
                "status": "processed",
            }
        ],
        employer_hint="Royal Australasian College of Physicians",
    )

    assert structured["people"] == []


def test_extract_intel_forces_pdf_fallback_when_org_chart_text_is_only_visual_summary(monkeypatch, tmp_path):
    attachment_path = tmp_path / "senior-management-structure-department-of-health-jan-2026.pdf"
    attachment_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr("cortex_engine.intel_extractor._call_haiku_extract", lambda payload, combined_text: {})
    monkeypatch.setattr(
        "cortex_engine.intel_extractor._maybe_textifier",
        lambda: type(
            "T",
            (),
            {
                "textify_file": lambda self, path: "> **[Image 1]**: The image is an organizational chart for Victoria's Department of Health. It shows the structure with various departments and roles."
            },
        )(),
    )
    monkeypatch.setattr(
        "cortex_engine.intel_extractor._extract_pdf_text_local",
        lambda path: """
        Chief Executive Officer
        Mic Cavazzini
        Head of Strategy & Transformation
        Gillian Dunn
        """,
    )
    monkeypatch.setattr("cortex_engine.intel_extractor.StakeholderSignalStore", lambda: type("S", (), {"list_profiles": lambda self, org_name='': []})())

    result, _ = extract_intel(
        {
            "org_name": "Longboardfella",
            "subject": "Fw: Fw:",
            "raw_text": "",
            "html_text": "",
            "attachments": [
                {
                    "filename": attachment_path.name,
                    "stored_path": str(attachment_path),
                    "mime_type": "application/pdf",
                    "kind": "document",
                }
            ],
            "message_id": "<orgchart-fallback>",
            "parsed_candidate_employer": "Department of Health",
        }
    )

    assert any(item["canonical_name"] == "Mic Cavazzini" and item.get("current_role") == "Chief Executive Officer" for item in result["entities"])
    assert any(item["canonical_name"] == "Gillian Dunn" and "Strategy" in item.get("current_role", "") for item in result["entities"])


def test_extract_intel_suppresses_sender_signature_when_large_document_attachment_present(monkeypatch, tmp_path):
    attachment_path = tmp_path / "2026-2030-strategic-direction-document_compressed.pdf"
    attachment_path.write_text("placeholder", encoding="utf-8")
    captured = {}

    def _fake_extract(_payload, combined_text):
        captured["combined_text"] = combined_text
        return {
            "people": [
                {
                    "name": "Professor Jennifer Martin",
                    "current_employer": "Royal Australasian College of Physicians",
                    "current_role": "President and Chair of the Board",
                }
            ],
            "organisations": [
                {
                    "name": "Royal Australasian College of Physicians",
                    "industry": "Medical Education & Professional Services",
                }
            ],
            "emails": [],
            "career_events": [],
            "summary": "RACP strategic direction document.",
        }

    monkeypatch.setattr("cortex_engine.intel_extractor._call_haiku_extract", _fake_extract)
    monkeypatch.setattr(
        "cortex_engine.intel_extractor._maybe_textifier",
        lambda: type("T", (), {"textify_file": lambda self, path: ("Royal Australasian College of Physicians\nStrategic Direction 2026 to 2030\n" * 40)})(),
    )
    monkeypatch.setattr("cortex_engine.intel_extractor.StakeholderSignalStore", lambda: type("S", (), {"list_profiles": lambda self, org_name='': []})())

    result, _ = extract_intel(
        {
            "org_name": "Longboardfella",
            "subject": "",
            "raw_text": "Dr. Paul Cooper\nDirector, Longboardfella Consulting Pty Ltd\nwww.longboardfella.com\npaul@longboardfella.com.au\nLinkedIn: https://www.linkedin.com/in/digitalfella/\nZoom: 939-985-8184",
            "html_text": "",
            "attachments": [
                {
                    "filename": attachment_path.name,
                    "stored_path": str(attachment_path),
                    "mime_type": "application/pdf",
                    "kind": "document",
                }
            ],
            "message_id": "<strategic-doc>",
        }
    )

    assert "paul@longboardfella.com.au" not in captured["combined_text"].lower()
    assert not any(item["canonical_name"] == "Dr. Paul Cooper" for item in result["entities"])
    assert any(item["canonical_name"] == "Royal Australasian College of Physicians" for item in result["entities"])


def test_extract_intel_strips_forwarding_headers_and_keeps_multiple_attachments(monkeypatch, tmp_path):
    first_attachment = tmp_path / "aidh_strategy.pdf"
    second_attachment = tmp_path / "aidh_appendix.pdf"
    first_attachment.write_text("placeholder", encoding="utf-8")
    second_attachment.write_text("placeholder", encoding="utf-8")
    captured = {}

    def _fake_extract(_payload, combined_text):
        captured["combined_text"] = combined_text
        return {
            "people": [],
            "organisations": [{"name": "Australian Institute of Digital Health"}],
            "emails": [],
            "career_events": [],
            "summary": "AIDH strategy and appendix.",
        }

    monkeypatch.setattr("cortex_engine.intel_extractor._call_haiku_extract", _fake_extract)
    monkeypatch.setattr(
        "cortex_engine.intel_extractor._maybe_textifier",
        lambda: type(
            "T",
            (),
            {
                "textify_file": lambda self, path: (
                    "Australian Institute of Digital Health\nStrategic Plan 2026-2030\nDigital capability and workforce uplift"
                    if str(path).endswith("aidh_strategy.pdf")
                    else "Appendix\nPriority actions\nHealth interoperability\nDigital maturity"
                )
            },
        )(),
    )
    monkeypatch.setattr("cortex_engine.intel_extractor.StakeholderSignalStore", lambda: type("S", (), {"list_profiles": lambda self, org_name='': []})())

    extract_intel(
        {
            "org_name": "Longboardfella",
            "subject": "Fwd: Re: AIDH strategy",
            "raw_text": "-----Original Message-----\nFrom: Paul Cooper <paul@longboardfella.com.au>\nSent: Friday, 20 March 2026 9:00 AM\nTo: intel@example.com\nSubject: Re: AIDH strategy\n\nPlease see attached.",
            "html_text": "",
            "attachments": [
                {
                    "filename": "aidh_strategy.pdf",
                    "stored_path": str(first_attachment),
                    "mime_type": "application/pdf",
                    "kind": "document",
                },
                {
                    "filename": "aidh_appendix.pdf",
                    "stored_path": str(second_attachment),
                    "mime_type": "application/pdf",
                    "kind": "document",
                },
            ],
            "message_id": "<aidh-multi>",
        }
    )

    assert "original message" not in captured["combined_text"].lower()
    assert "from: paul cooper" not in captured["combined_text"].lower()
    assert "attachment: aidh_strategy.pdf" in captured["combined_text"].lower()
    assert "attachment: aidh_appendix.pdf" in captured["combined_text"].lower()


def test_email_triage_uses_qwen_json_when_available(monkeypatch):
    class _Client:
        def __init__(self, timeout=30):
            self.timeout = timeout

        def list(self):
            return {"models": [{"name": "qwen3.5:9b"}]}

        def chat(self, model, messages, options=None):
            assert model == "qwen3.5:9b"
            assert "Attachments JSON" in messages[0]["content"]
            return {
                "message": {
                    "content": (
                        '{"processing_mode":"attachments_only","clean_subject":"AIDH strategy",'
                        '"actionable_body_text":"","wrapper_text":"forwarded wrapper","signature_text":"Paul Cooper signature",'
                        '"include_attachment_filenames":["aidh_strategy.pdf","aidh_appendix.pdf"],'
                        '"ignore_attachment_filenames":["Outlook-logo.png"],"confidence":0.91}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=_Client))

    triage = _triage_email_payload(
        {
            "subject": "Fwd: Re: AIDH strategy",
            "raw_text": "Paul Cooper signature block\npaul@longboardfella.com.au",
            "html_text": "",
            "attachments": [
                {"filename": "aidh_strategy.pdf", "kind": "document", "mime_type": "application/pdf"},
                {"filename": "aidh_appendix.pdf", "kind": "document", "mime_type": "application/pdf"},
                {"filename": "Outlook-logo.png", "kind": "image", "mime_type": "image/png"},
            ],
        }
    )

    assert triage["processing_mode"] == "attachments_only"
    assert triage["clean_subject"] == "AIDH strategy"
    assert triage["include_attachment_filenames"] == ["aidh_strategy.pdf", "aidh_appendix.pdf"]
    assert triage["ignore_attachment_filenames"] == ["Outlook-logo.png"]
    assert triage["used_model"] == "qwen3.5:9b"
