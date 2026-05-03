from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worker.lab_mailbox_worker import (
    LabGraphClient,
    LabMailboxConfig,
    normalize_graph_message,
)


def _config(fetch_attachments: bool = True) -> LabMailboxConfig:
    return LabMailboxConfig(
        transport_mode="graph",
        poll_interval=60,
        graph_tenant_id="t",
        graph_client_id="c",
        graph_client_secret="s",
        graph_mailbox="lab@example.com",
        graph_page_size=10,
        graph_timeout=30,
        webhook_url="https://example.com/webhook",
        webhook_secret="x",
        suppress_replies=False,
        state_path=Path("/tmp/_unused.json"),
        fetch_attachments=fetch_attachments,
    )


def test_normalize_message_returns_empty_attachments_when_flag_off():
    """With the feature flag off, the worker preserves the legacy behaviour."""
    item = {
        "id": "graph-id-1",
        "internetMessageId": "<rfc-1@example>",
        "subject": "JOB: URL INGEST",
        "from": {"emailAddress": {"address": "p@example.com"}},
        "toRecipients": [{"emailAddress": {"address": "lab@example.com"}}],
        "body": {"contentType": "Text", "content": "https://x.test"},
        "hasAttachments": True,
        "receivedDateTime": "2026-05-03T10:00:00Z",
    }
    msg = normalize_graph_message(item, graph_client=None, fetch=False)
    assert msg["attachments"] == []


def test_normalize_message_fetches_and_filters_attachments_when_flag_on():
    """With fetch=True we call Graph and convert each attachment to the webhook shape."""
    item = {
        "id": "graph-id-2",
        "internetMessageId": "<rfc-2@example>",
        "subject": "TEXTIFY",
        "from": {"emailAddress": {"address": "p@example.com"}},
        "toRecipients": [{"emailAddress": {"address": "lab@example.com"}}],
        "body": {"contentType": "Text", "content": ""},
        "hasAttachments": True,
        "receivedDateTime": "2026-05-03T10:00:00Z",
    }
    pdf_b64 = base64.b64encode(b"%PDF-1.4 fake").decode()
    png_b64 = base64.b64encode(b"\x89PNGfake").decode()
    fake_client = MagicMock()
    fake_client.list_attachments.return_value = [
        {
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": "report.pdf",
            "contentType": "application/pdf",
            "size": 12345,
            "isInline": False,
            "contentBytes": pdf_b64,
        },
        {
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": "logo.png",
            "contentType": "image/png",
            "size": 1024,
            "isInline": True,
            "contentBytes": png_b64,
        },
    ]
    msg = normalize_graph_message(item, graph_client=fake_client, fetch=True)
    fake_client.list_attachments.assert_called_once_with("graph-id-2")
    assert len(msg["attachments"]) == 2
    pdf, png = msg["attachments"]
    assert pdf == {
        "filename": "report.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 12345,
        "is_inline": False,
        "content_base64": pdf_b64,
    }
    assert png["is_inline"] is True
    assert png["filename"] == "logo.png"


def test_normalize_message_skips_item_attachments():
    """Reference / item attachments without bytes are dropped."""
    item = {
        "id": "graph-id-3",
        "internetMessageId": "<rfc-3@example>",
        "subject": "TEXTIFY",
        "from": {"emailAddress": {"address": "p@example.com"}},
        "toRecipients": [{"emailAddress": {"address": "lab@example.com"}}],
        "body": {"contentType": "Text", "content": ""},
        "hasAttachments": True,
        "receivedDateTime": "2026-05-03T10:00:00Z",
    }
    fake_client = MagicMock()
    fake_client.list_attachments.return_value = [
        {
            "@odata.type": "#microsoft.graph.itemAttachment",
            "name": "Some forwarded mail",
            "size": 4096,
            "isInline": False,
        },
        {
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": "kept.pdf",
            "contentType": "application/pdf",
            "size": 8192,
            "isInline": False,
            "contentBytes": base64.b64encode(b"hi").decode(),
        },
    ]
    msg = normalize_graph_message(item, graph_client=fake_client, fetch=True)
    assert len(msg["attachments"]) == 1
    assert msg["attachments"][0]["filename"] == "kept.pdf"


def test_list_attachments_calls_graph_with_message_id():
    """LabGraphClient.list_attachments hits the right endpoint."""
    cfg = _config()
    client = LabGraphClient(cfg)
    client._access_token = "fake-token"
    client._token_expires_at = 1e12
    fake_response = MagicMock()
    fake_response.json.return_value = {"value": [{"name": "a.pdf"}]}
    fake_response.raise_for_status.return_value = None
    with patch("worker.lab_mailbox_worker.requests.get", return_value=fake_response) as gm:
        result = client.list_attachments("MSG123")
    assert result == [{"name": "a.pdf"}]
    called_url = gm.call_args.args[0]
    assert "/messages/MSG123/attachments" in called_url


def test_config_from_env_reads_fetch_attachments():
    """LAB_FETCH_ATTACHMENTS=1 in env flips the dataclass field on."""
    from worker.lab_mailbox_worker import LabMailboxConfig as _Cfg
    env = {
        "LAB_TRANSPORT_MODE": "graph",
        "LAB_GRAPH_TENANT_ID": "t",
        "LAB_GRAPH_CLIENT_ID": "c",
        "LAB_GRAPH_CLIENT_SECRET": "s",
        "LAB_WEBHOOK_URL": "https://example.com/webhook",
        "LAB_WEBHOOK_SECRET": "x",
        "LAB_FETCH_ATTACHMENTS": "1",
    }
    cfg = _Cfg.from_env(env)
    assert cfg.fetch_attachments is True

    env["LAB_FETCH_ATTACHMENTS"] = "0"
    cfg2 = _Cfg.from_env(env)
    assert cfg2.fetch_attachments is False
