from __future__ import annotations

import pytest

from cortex_engine.handoff_contract import (
    validate_cortex_sync_input,
    validate_pdf_textify_input,
    validate_portal_ingest_input,
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


def test_portal_ingest_defaults_scope_fields():
    payload = validate_portal_ingest_input({})
    assert payload["portal_document_id"] == ""
    assert payload["project_id"] == "default"
    assert payload["tenant_id"] == "default"
    assert payload["chunk_target_chars"] == 2000
    assert payload["chunk_min_chars"] == 200
    assert payload["max_chunks"] == 250


def test_portal_ingest_normalizes_supplied_scope_fields():
    payload = validate_portal_ingest_input(
        {
            "portal_document_id": 12345,
            "project_id": "project-alpha",
            "tenant_id": "tenant-a",
            "chunk_target_chars": "3000",
            "chunk_min_chars": "300",
            "max_chunks": "50",
        }
    )
    assert payload["portal_document_id"] == "12345"
    assert payload["project_id"] == "project-alpha"
    assert payload["tenant_id"] == "tenant-a"
    assert payload["chunk_target_chars"] == 3000
    assert payload["chunk_min_chars"] == 300
    assert payload["max_chunks"] == 50


def test_portal_ingest_rejects_invalid_chunk_bounds():
    with pytest.raises(ValueError, match="chunk_min_chars must be <= chunk_target_chars"):
        validate_portal_ingest_input(
            {
                "chunk_target_chars": 100,
                "chunk_min_chars": 200,
            }
        )


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
    with pytest.raises(ValueError, match="collection_name"):
        validate_cortex_sync_input({"file_paths": ["/tmp/a.md"]})
