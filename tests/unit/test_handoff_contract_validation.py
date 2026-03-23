from __future__ import annotations

import pytest

from cortex_engine.handoff_contract import (
    validate_csv_profile_import_input,
    validate_cortex_sync_input,
    validate_intel_extract_input,
    validate_org_profile_refresh_input,
    validate_pdf_textify_input,
    validate_stakeholder_graph_view_input,
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


def test_org_profile_refresh_requires_org_and_target():
    with pytest.raises(ValueError, match="requires org_name"):
        validate_org_profile_refresh_input({"target_org_name": "Barwon Water"})

    with pytest.raises(ValueError, match="requires target_org_name"):
        validate_org_profile_refresh_input({"org_name": "Escient"})


def test_org_profile_refresh_uses_snapshot_website_and_defaults():
    payload = validate_org_profile_refresh_input(
        {
            "profile_id": 123,
            "org_name": "Escient",
            "target_org_name": "Barwon Water",
            "current_profile_snapshot": {"website_url": "https://barwonwater.vic.gov.au"},
        }
    )

    assert payload["profile_id"] == "123"
    assert payload["website_url"] == "https://barwonwater.vic.gov.au"
    assert payload["requested_docs"] == ["annual_report", "strategic_plan", "org_chart"]
    assert payload["discovery_mode"] == "official_sources_first"


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


def test_csv_profile_import_normalizes_rows():
    payload = validate_csv_profile_import_input(
        {
            "org_name": "Longboardfella",
            "on_behalf_of": "PAUL@example.com",
            "dry_run": "yes",
            "rows": [
                {
                    "canonical_name": "Jane Smith",
                    "target_type": "person",
                    "watch": "1",
                    "status": "ACTIVE",
                }
            ],
        }
    )
    assert payload["on_behalf_of"] == "paul@example.com"
    assert payload["dry_run"] is True
    assert payload["rows"][0]["watch"] == "yes"
    assert payload["rows"][0]["status"] == "active"


def test_csv_profile_import_rejects_missing_rows():
    with pytest.raises(ValueError, match="requires rows"):
        validate_csv_profile_import_input(
            {
                "org_name": "Longboardfella",
                "on_behalf_of": "paul@example.com",
                "rows": [],
            }
        )


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


def test_stakeholder_profile_sync_accepts_industry_metadata_and_affiliations():
    payload = validate_stakeholder_profile_sync_input(
        {
            "org_name": "Longboardfella",
            "profiles": [
                {
                    "canonical_name": "Digital Health",
                    "target_type": "industry",
                    "description": "Sector profile",
                    "key_themes": ["telehealth", "regulation", "telehealth"],
                    "regulatory_context": "TGA and privacy reform",
                    "market_size": "$10B",
                },
                {
                    "canonical_name": "Healthdirect Australia",
                    "target_type": "organisation",
                    "industry_affiliations": [
                        {"industry_profile_key": "ind-1", "industry_name": "Digital Health", "role": "key player"},
                    ],
                },
            ],
        }
    )
    industry = payload["profiles"][0]
    org = payload["profiles"][1]
    assert industry["target_type"] == "industry"
    assert industry["key_themes"] == ["telehealth", "regulation"]
    assert industry["regulatory_context"] == "TGA and privacy reform"
    assert org["industry_affiliations"][0]["industry_name"] == "Digital Health"


def test_stakeholder_profile_sync_accepts_org_strategic_profile():
    payload = validate_stakeholder_profile_sync_input(
        {
            "org_name": "Longboardfella",
            "org_strategic_profile": {
                "description": "Subscriber strategic focus",
                "industries": ["Healthcare", {"industry_name": "Digital Health"}, "healthcare"],
                "priority_industries": ["Healthcare"],
                "key_themes": ["transformation", "digital"],
                "strategic_objectives": ["board access"],
            },
            "profiles": [
                {
                    "canonical_name": "Mic Cavazzini",
                    "target_type": "person",
                }
            ],
        }
    )
    strategic = payload["org_strategic_profile"]
    assert strategic["description"] == "Subscriber strategic focus"
    assert strategic["industries"] == ["Healthcare", "Digital Health"]
    assert strategic["priority_industries"] == ["Healthcare"]
    assert strategic["key_themes"] == ["transformation", "digital"]


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


def test_signal_digest_validation_accepts_industry_scope():
    payload = validate_signal_digest_input(
        {
            "org_name": "Longboardfella",
            "scope_type": "industry",
            "scope_profile_key": "ind-1",
            "child_profile_keys": ["org-1", "person-1"],
            "child_org_names": ["Healthdirect Australia"],
            "key_themes": ["telehealth"],
            "regulatory_context": "Privacy reform",
            "market_size": "$10B",
            "shared_with_orgs": ["Escient"],
        }
    )
    assert payload["scope_type"] == "industry"
    assert payload["scope_profile_key"] == "ind-1"
    assert payload["child_profile_keys"] == ["org-1", "person-1"]
    assert payload["child_org_names"] == ["Healthdirect Australia"]
    assert payload["shared_with_orgs"] == ["Escient"]


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


def test_signal_ingest_validation_accepts_visibility_and_hierarchy_fields():
    payload = validate_signal_ingest_input(
        {
            "org_name": "Longboardfella",
            "subject": "Digital health regulation changes",
            "raw_text": "Sector update",
            "source_org_name": "Escient",
            "visible_to_orgs": ["Longboardfella", "Escient"],
            "shared_with_orgs": ["Longboardfella"],
            "scope_profile_key": "ind-1",
            "child_profile_keys": ["org-1"],
            "child_org_names": ["Healthdirect Australia"],
            "key_themes": ["telehealth"],
        }
    )
    assert payload["source_org_name"] == "Escient"
    assert payload["visible_to_orgs"] == ["Longboardfella", "Escient"]
    assert payload["scope_profile_key"] == "ind-1"
    assert payload["child_org_names"] == ["Healthdirect Australia"]


def test_stakeholder_graph_view_validation_accepts_defaults_and_filters():
    payload = validate_stakeholder_graph_view_input(
        {
            "org_name": "Longboardfella",
            "view_mode": "warm_intro",
            "profile_keys": ["abc123"],
            "edge_types": ["works_at", "linkedin_connection", "alumni_of"],
            "include_sources": "true",
            "max_nodes": "140",
            "top_k_paths": "7",
        }
    )
    assert payload["org_name"] == "Longboardfella"
    assert payload["view_mode"] == "warm_intro"
    assert payload["profile_keys"] == ["abc123"]
    assert payload["include_sources"] is True
    assert payload["max_nodes"] == 140
    assert payload["top_k_paths"] == 7
    assert payload["edge_types"] == ["works_at", "linkedin_connection", "alumni_of"]


def test_stakeholder_graph_view_validation_accepts_industry_network():
    payload = validate_stakeholder_graph_view_input(
        {
            "org_name": "Longboardfella",
            "view_mode": "industry_network",
            "focus_profile_key": "ind-1",
            "child_profile_keys": ["org-1", "person-1"],
            "edge_types": ["belongs_to_industry", "works_at"],
        }
    )
    assert payload["view_mode"] == "industry_network"
    assert payload["focus_profile_key"] == "ind-1"
    assert payload["edge_types"] == ["belongs_to_industry", "works_at"]


def test_stakeholder_graph_view_validation_rejects_missing_focus_inputs():
    with pytest.raises(ValueError, match="focus_profile_key"):
        validate_stakeholder_graph_view_input({"org_name": "Longboardfella", "view_mode": "ego"})

    with pytest.raises(ValueError, match="focus_org_name"):
        validate_stakeholder_graph_view_input({"org_name": "Longboardfella", "view_mode": "org_focus"})


def test_stakeholder_graph_view_validation_rejects_invalid_edge_type():
    with pytest.raises(ValueError, match="Invalid edge_types"):
        validate_stakeholder_graph_view_input(
            {
                "org_name": "Longboardfella",
                "edge_types": ["works_at", "unknown_edge"],
            }
        )
