import base64

import pytest

from worker.hermes_delivery_service import DeliveryRuntime, _safe_filename


class FakeConfig:
    recipient = "owner@example.com"
    api_token = "test-token"
    max_bytes = 1024
    rate_per_hour = 2
    audit_log = None


class FakeMailer:
    def __init__(self):
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)


def test_filename_policy():
    assert _safe_filename("report.pdf") == "report.pdf"
    with pytest.raises(ValueError):
        _safe_filename("../report.pdf")
    with pytest.raises(ValueError):
        _safe_filename("payload.exe")


def test_fixed_recipient_fields_are_rejected(tmp_path):
    config = FakeConfig()
    config.audit_log = tmp_path / "audit.jsonl"
    runtime = DeliveryRuntime(config, FakeMailer())
    with pytest.raises(ValueError, match="recipient fields are forbidden"):
        runtime.deliver({"filename": "report.pdf", "content_b64": "YQ==", "to": "x@y.z"})


def test_delivery_uses_configured_recipient_and_audits(tmp_path):
    config = FakeConfig()
    config.audit_log = tmp_path / "audit.jsonl"
    mailer = FakeMailer()
    runtime = DeliveryRuntime(config, mailer)
    result = runtime.deliver({
        "filename": "report.pdf",
        "content_b64": base64.b64encode(b"safe report").decode(),
        "subject": "Requested report",
        "body": "Attached.",
    })
    assert result == {"ok": True, "filename": "report.pdf", "bytes": 11}
    assert mailer.calls[0]["filename"] == "report.pdf"
    assert "owner@example.com" in config.audit_log.read_text()


def test_rate_limit(tmp_path):
    config = FakeConfig()
    config.audit_log = tmp_path / "audit.jsonl"
    runtime = DeliveryRuntime(config, FakeMailer())
    payload = {"filename": "a.txt", "content_b64": "YQ=="}
    runtime.deliver(payload)
    runtime.deliver(payload)
    with pytest.raises(PermissionError, match="hourly delivery limit"):
        runtime.deliver(payload)
