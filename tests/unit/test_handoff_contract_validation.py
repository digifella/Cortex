from __future__ import annotations

import pytest

from cortex_engine.handoff_contract import (
    validate_cortex_sync_input,
    validate_intel_extract_input,
    validate_pdf_textify_input,
    validate_signal_ingest_input,
    validate_signal_digest_input,
    validate_stakeholder_profile_sync_input,
    validate_url_ingest_input,
)


def test_pdf_textify_normalizes_valid_options():
    payload = validate_pdf_textify_input(
        {
            "textify_options": {
                "use_vision": "true",
                "pdf_strategy": "docling",
                "cleanup_provider": "lmstudio",
                "cleanup_model": "qwen2.5:32b",
                "docling_timeout_seconds": "300",
                "image_description_timeout_seconds": 8,
                "image_enrich_max_seconds": "45",
            }
        }
    )
    opts = payload["textify_options"]
    assert opts["use_vision"] is True
    assert opts["pdf_strategy"] == "docling"
    assert opts["cleanup_provider"] == "lmstudio"
    assert opts["cleanup_model"] == "qwen2.5:32b"
    assert opts["docling_timeout_seconds"] == 300.0
    assert opts["image_description_timeout_seconds"] == 8.0
    assert opts["image_enrich_max_seconds"] == 45.0


def test_pdf_textify_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid pdf_strategy"):
        validate_pdf_textify_input({"textify_options": {"pdf_strategy": "bad_mode"}})


def test_pdf_textify_rejects_invalid_provider():
    with pytest.raises(ValueError, match="Invalid cleanup_provider"):
        validate_pdf_textify_input({"textify_options": {"cleanup_provider": "openai"}})


def test_pdf_textify_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="docling_timeout_seconds must be > 0"):
        validate_pdf_textify_input(
            {"textify_options": {"docling_timeout_seconds": 0}}
        )


def test_url_ingest_requires_urls_or_text():
    with pytest.raises(ValueError, match="requires input_data.urls"):
        validate_url_ingest_input({})


def test_url_ingest_rejects_non_object_ingest_options():
    with pytest.raises(ValueError, match="ingest_options must be an object"):
        validate_url_ingest_input(
            {
                "urls": ["https://example.com"],
                "ingest_options": "not-an-object",
            }
        )


def test_url_ingest_coerces_booleans_timeout_and_textify():
    payload = validate_url_ingest_input(
        {
            "urls": ["https://example.com/a", " ", "https://example.com/b"],
            "ingest_options": {
                "convert_to_md": "1",
                "use_vision": "yes",
                "capture_web_md_on_no_pdf": "0",
            },
            "timeout_seconds": "35",
            "pdf_strategy": "hybrid",
            "docling_timeout_seconds": "240",
            "image_description_timeout_seconds": "10",
            "image_enrich_max_seconds": "60",
        }
    )
    assert payload["urls"] == ["https://example.com/a", "https://example.com/b"]
    assert payload["ingest_options"]["convert_to_md"] is True
    assert payload["ingest_options"]["use_vision"] is True
    assert payload["ingest_options"]["capture_web_md_on_no_pdf"] is False
    assert payload["timeout_seconds"] == 35
    assert payload["textify_options"]["pdf_strategy"] == "hybrid"
    assert payload["textify_options"]["docling_timeout_seconds"] == 240.0


def test_cortex_sync_validates_required_fields():
    payload = validate_cortex_sync_input(
        {
            "file_paths": [" /tmp/a.md ", "/tmp/b.md"],
            "collection_name": "Website - OECD",
            "fresh": "1",
        }
    )
    assert payload["file_paths"] == ["/tmp/a.md", "/tmp/b.md"]
    assert payload["collection_name"] == "Website - OECD"
    assert payload["fresh"] is True


def test_cortex_sync_rejects_missing_fields():
    payload = validate_cortex_sync_input({"collection_name": "x"})
    assert payload["file_paths"] == []
    payload2 = validate_cortex_sync_input({"file_paths": ["/tmp/a.md"]})
    assert payload2["collection_name"] == "default"


def test_cortex_sync_accepts_manifest_mode():
    payload = validate_cortex_sync_input(
        {
            "collection_name": "Website - Digital Health",
            "manifest": [{"zip_path": "digital-health/doc1.md", "doc_id": 1}],
        }
    )
    assert payload["file_paths"] == []
    assert payload["manifest"][0]["zip_path"] == "digital-health/doc1.md"


def test_cortex_sync_defaults_blank_collection_to_default():
    payload = validate_cortex_sync_input(
        {
            "collection_name": "",
            "manifest": [{"zip_path": "digital-health/doc1.md"}],
        }
    )
    assert payload["collection_name"] == "default"


def test_cortex_sync_rejects_bad_manifest():
    with pytest.raises(ValueError, match="manifest\\[0\\]\\.zip_path"):
        validate_cortex_sync_input(
            {
                "collection_name": "Website - Digital Health",
                "manifest": [{}],
            }
        )


def test_intel_extract_normalizes_attachments():
    payload = validate_intel_extract_input(
        {
            "org_name": "Longboardfella",
            "subject": "Forwarded screenshot",
            "attachments": [
                {
                    "filename": "Screenshot.jpg",
                    "mime_type": "image/jpeg",
                    "stored_path": "C:\\temp\\Screenshot.jpg",
                    "kind": "image",
                }
            ],
        }
    )
    assert payload["org_name"] == "Longboardfella"
    assert payload["attachments"][0]["filename"] == "Screenshot.jpg"
    assert payload["attachments"][0]["kind"] == "image"


def test_stakeholder_profile_sync_normalizes_affiliations():
    payload = validate_stakeholder_profile_sync_input(
        {
            "org_name": "Longboardfella",
            "org_alumni": ["SMS", "Escient", "sms"],
            "profiles": [
                {
                    "canonical_name": "Jane Smith",
                    "current_employer": "Acme Corp",
                    "current_role": "Chief Strategy Officer",
                    "affiliations": [
                        {"org_name_text": "Acme Corp", "role": "Chief Strategy Officer", "is_primary": "1"},
                        {"org_name_text": "Board Co", "role": "Director", "affiliation_type": "board", "confidence": "probable"},
                    ],
                }
            ],
        }
    )
    profile = payload["profiles"][0]
    assert profile["current_employer"] == "Acme Corp"
    assert profile["current_role"] == "Chief Strategy Officer"
    assert len(profile["affiliations"]) == 2
    assert profile["affiliations"][0]["is_primary"] == 1
    assert profile["affiliations"][1]["affiliation_type"] == "board"
    assert profile["affiliations"][1]["confidence"] == "probable"
    assert payload["org_alumni"] == ["SMS", "Escient"]


def test_signal_digest_validation_accepts_depth_and_tier():
    payload = validate_signal_digest_input(
        {
            "org_name": "Longboardfella",
            "profile_keys": ["abc123"],
            "priority_profile_keys": ["abc123"],
            "digest_tier": "priority",
            "report_depth": "strategic",
            "deep_analysis": "true",
            "member_alumni": ["SMS"],
            "org_alumni": ["SMS", "Deloitte"],
        }
    )
    assert payload["digest_tier"] == "priority"
    assert payload["report_depth"] == "strategic"
    assert payload["deep_analysis"] is True
    assert payload["priority_profile_keys"] == ["abc123"]
    assert payload["member_alumni"] == ["SMS"]
    assert payload["org_alumni"] == ["SMS", "Deloitte"]


def test_signal_ingest_validation_accepts_watch_signal_batches():
    payload = validate_signal_ingest_input(
        {
            "org_name": "Longboardfella",
            "source": "market_radar_watch",
            "source_job": "321",
            "signals": [
                {
                    "target": "Jane Smith",
                    "type": "person",
                    "current_employer": "BigBank",
                    "headline": "Jane Smith joins BigBank digital team",
                    "url": "https://example.com/jane",
                    "date": "2026-03-16T00:00:00Z",
                    "snippet": "Appointment confirmed in trade media.",
                    "source_type": "news",
                }
            ],
        }
    )
    assert payload["source_system"] == "market_radar_watch"
    assert payload["source_job"] == "321"
    assert payload["signals"][0]["target_name"] == "Jane Smith"
    assert payload["signals"][0]["headline"] == "Jane Smith joins BigBank digital team"
