from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex_engine.email_handlers.csv_profile_import import (
    CsvProfileImportError,
    CsvProfileImportProcessor,
    detect_csv_profile_import,
    parse_csv_profile_rows,
    subject_looks_like_csv_profile_import,
)


def test_subject_detection_and_attachment_detection():
    attachments = [{"filename": "profiles.csv", "mime_type": "text/csv"}]
    assert subject_looks_like_csv_profile_import("PROFILES DRY RUN")
    assert detect_csv_profile_import("PROFILES DRY RUN", attachments)
    assert not detect_csv_profile_import("General update", attachments)


def test_parse_csv_profile_rows_supports_comments_bom_and_org_detection(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "\ufeffcanonical name,current employer,current role,watch,website_url\n"
        "# comment row should be ignored\n"
        "Jane Smith,Acme Corp,CEO,yes,\n"
        "Acme Corp,,,no,https://acme.com\n",
        encoding="utf-8",
    )
    parsed = parse_csv_profile_rows(str(csv_path))
    assert parsed["row_count"] == 2
    assert parsed["rows"][0]["canonical_name"] == "Jane Smith"
    assert parsed["rows"][0]["watch"] == "yes"
    assert parsed["rows"][1]["target_type"] == "organisation"


def test_parse_csv_profile_rows_skips_missing_canonical_name(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "canonical_name,current_employer\n"
        ",Acme Corp\n"
        "Jane Smith,Acme Corp\n",
        encoding="utf-8",
    )
    parsed = parse_csv_profile_rows(str(csv_path))
    assert len(parsed["rows"]) == 1
    assert parsed["skipped_errors"] == ["Row 2: canonical_name is required"]


def test_parse_csv_profile_rows_rejects_header_only(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text("canonical_name,current_employer\n", encoding="utf-8")
    with pytest.raises(CsvProfileImportError, match="no data rows found"):
        parse_csv_profile_rows(str(csv_path))


def test_csv_profile_import_processor_posts_expected_payload(monkeypatch, tmp_path):
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "canonical_name,current_employer,current_role,watch\n"
        "Jane Smith,Acme Corp,CEO,yes\n",
        encoding="utf-8",
    )

    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "created": 1, "updated": 0, "skipped": 0, "errors": []}

    def _fake_post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("cortex_engine.email_handlers.csv_profile_import.requests.post", _fake_post)

    processor = CsvProfileImportProcessor(
        import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        queue_secret="secret-key",
        timeout=33,
    )
    result = processor.process_message(
        message={"subject": "PROFILES", "from_email": "paul@example.com"},
        persisted={"attachments": [{"filename": "profiles.csv", "mime_type": "text/csv", "stored_path": str(csv_path)}]},
        org_name="Longboardfella",
    )

    assert result["created"] == 1
    assert seen["timeout"] == 33
    assert seen["json"]["queue_secret"] == "secret-key"
    assert seen["json"]["on_behalf_of"] == "paul@example.com"
    assert seen["json"]["rows"][0]["canonical_name"] == "Jane Smith"
