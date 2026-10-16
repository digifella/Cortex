from pathlib import Path
import base64
import hashlib

import pytest

from worker.notes_mailbox_worker import NotesGraphClient, NotesMailboxConfig, NotesMailboxProcessor, _normalize_graph_message


def _notes_config(tmp_path) -> NotesMailboxConfig:
    return NotesMailboxConfig(
        source_system="notes_mailbox",
        poll_interval=60,
        transport_mode="manual",
        mailbox_identity="notes@longboardfella.com.au",
        public_outbox_dir=str(tmp_path / "public"),
        private_outbox_dir=str(tmp_path / "private"),
        public_vault_dir=str(tmp_path / "vault_public" / "Inbox"),
        private_vault_dir=str(tmp_path / "vault_private" / "notes"),
        write_vault_markdown=True,
        graph_tenant_id="",
        graph_client_id="",
        graph_client_secret="",
        graph_mailbox="notes@longboardfella.com.au",
        graph_page_size=10,
        graph_timeout=30,
        public_review_vault_dir=str(tmp_path / "vault_private" / "Public_Review" / "notes"),
    )


def test_notes_mailbox_processor_defaults_to_private_note(tmp_path):
    cfg = _notes_config(tmp_path)
    processor = NotesMailboxProcessor(cfg)

    result = processor.process_message(
        {
            "subject": "Meeting notes from today",
            "text_body": "Capture this privately by default.",
            "from_email": "paul@example.com",
        }
    )

    assert result["status"] == "outbox_written"
    assert result["route"] == "private_vault"
    assert Path(result["path"]).exists()
    assert Path(result["vault_path"]).exists()
    assert result["vault_path"].endswith(".md")
    assert "vault_private" in result["vault_path"]
    assert "wiki-ready: false" in Path(result["vault_path"]).read_text(encoding="utf-8")
    assert not list((tmp_path / "vault_public").rglob("*.md"))


def test_notes_mailbox_preserves_source_date_and_receipt_fallback(tmp_path):
    processor = NotesMailboxProcessor(_notes_config(tmp_path))
    dated = processor.process_message({
        "subject": "Research article", "text_body": "Published: 12 September 2026\nFindings follow.",
        "received_at": "2026-10-15T04:00:00Z", "from_email": "paul@example.com",
    })
    markdown = Path(dated["vault_path"]).read_text(encoding="utf-8")
    assert 'source_date: "2026-09-12"' in markdown
    assert 'source_date_source: "header_publication"' in markdown

    undated = processor.process_message({
        "subject": "Undated personal note", "text_body": "A thought with no date.",
        "received_at": "2026-10-15T05:00:00Z", "from_email": "paul@example.com",
    })
    markdown = Path(undated["vault_path"]).read_text(encoding="utf-8")
    assert 'source_date: "2026-10-15"' in markdown
    assert 'source_date_source: "email_received_fallback"' in markdown


def test_notes_mailbox_explicit_public_goes_to_private_review(tmp_path):
    processor = NotesMailboxProcessor(_notes_config(tmp_path))
    result = processor.process_message({
        "subject": "PUBLIC: Museum trip notes",
        "text_body": "Candidate for the public wiki.",
        "from_email": "paul@example.com",
    })
    assert result["route"] == "public_review"
    assert "Public_Review" in result["vault_path"]
    assert "wiki-ready: false" in Path(result["vault_path"]).read_text(encoding="utf-8")
    assert not list((tmp_path / "vault_public").rglob("*.md"))


def test_notes_mailbox_private_marker_overrides_public_candidate(tmp_path):
    processor = NotesMailboxProcessor(_notes_config(tmp_path))
    result = processor.process_message({
        "subject": "PUBLIC: PRIVATE client meeting",
        "text_body": "Client details must stay private.",
        "from_email": "paul@example.com",
    })
    assert result["route"] == "private_vault"
    assert "Public_Review" not in result["vault_path"]
    assert not list((tmp_path / "vault_public").rglob("*.md"))


def test_notes_mailbox_preserves_file_attachments_privately(tmp_path):
    processor = NotesMailboxProcessor(_notes_config(tmp_path))
    body = b"Private report contents"
    result = processor.process_message({
        "subject": "Notes with report",
        "text_body": "See attached report.",
        "from_email": "paul@example.com",
        "attachments": [{"@odata.type": "#microsoft.graph.fileAttachment",
                         "name": "../report.pdf", "contentBytes": base64.b64encode(body).decode()}],
    })
    attachment = tmp_path / "attachments" / f"{hashlib.sha256(body).hexdigest()[:16]}-report.pdf"
    assert attachment.read_bytes() == body
    assert attachment.stat().st_mode & 0o077 == 0
    markdown = Path(result["vault_path"]).read_text(encoding="utf-8")
    assert str(attachment) in markdown
    assert not list((tmp_path / "vault_public").rglob("*.pdf"))


def test_notes_mailbox_does_not_file_unsupported_attachment(tmp_path):
    processor = NotesMailboxProcessor(_notes_config(tmp_path))
    with pytest.raises(ValueError, match="Unsupported Graph attachment"):
        processor.process_message({
            "subject": "Calendar invite", "text_body": "See attached",
            "attachments": [{"@odata.type": "#microsoft.graph.itemAttachment", "name": "invite"}],
        })
    assert not list(tmp_path.rglob("*.md"))


def test_notes_mailbox_processor_writes_private_note(tmp_path):
    cfg = _notes_config(tmp_path)
    processor = NotesMailboxProcessor(cfg)

    result = processor.process_message(
        {
            "subject": "PRIVATE: Sensitive meeting notes",
            "text_body": "Only private vault should receive this.",
            "from_email": "paul@example.com",
        }
    )

    assert result["status"] == "outbox_written"
    assert result["route"] == "private_vault"
    assert Path(result["path"]).exists()
    assert Path(result["vault_path"]).exists()
    assert "wiki-ready: false" in Path(result["vault_path"]).read_text(encoding="utf-8")


def test_notes_mailbox_processor_rejects_market_intel_shape(tmp_path):
    cfg = _notes_config(tmp_path)
    processor = NotesMailboxProcessor(cfg)

    result = processor.process_message(
        {
            "subject": "entity: Escient | Barwon Water org chart",
            "text_body": "This should stay out of the notes mailbox path.",
            "from_email": "paul@example.com",
        }
    )

    assert result["status"] == "rejected"
    assert result["route"] == "unsupported_market_intel"


def test_notes_mailbox_processor_rejects_lab_result_generation_error(tmp_path):
    cfg = _notes_config(tmp_path)
    processor = NotesMailboxProcessor(cfg)

    result = processor.process_message(
        {
            "subject": "Fwd: YouTube Summariser job #1468 is complete",
            "text_body": (
                "YouTube Summariser job #1468 is complete.\n\n"
                "---\nsource_type: youtube_summary\n---\n\n"
                "### Summary\n"
                "[Error generating summary: 400 Request contains an invalid argument.]\n\n"
                "The Lab · Longboardfella"
            ),
            "from_email": "paul@example.com",
        }
    )

    assert result["status"] == "rejected"
    assert result["route"] == "rejected_lab_result_error"
    assert not list((tmp_path / "vault_public").rglob("*.md"))
    assert not list((tmp_path / "vault_private").rglob("*.md"))


def test_notes_mailbox_processor_tracks_processed_graph_ids(tmp_path):
    cfg = _notes_config(tmp_path)
    processor = NotesMailboxProcessor(cfg)

    assert not processor.has_processed_graph_id("graph-1")

    processor.record_processed_graph_id("graph-1", "2026-10-15T07:00:00Z")

    reloaded = NotesMailboxProcessor(cfg)
    assert reloaded.has_processed_graph_id("graph-1")
    assert reloaded.state["last_received_at"] == "2026-10-15T07:00:00Z"


class _FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_normalize_graph_message_strips_html_body():
    message = _normalize_graph_message(
        {
            "id": "graph-1",
            "internetMessageId": "<msg@example.com>",
            "subject": "PRIVATE: Test",
            "receivedDateTime": "2026-04-20T00:00:00Z",
            "from": {"emailAddress": {"name": "Paul", "address": "paul@example.com"}},
            "toRecipients": [{"emailAddress": {"name": "Notes", "address": "notes@longboardfella.com.au"}}],
            "body": {"contentType": "html", "content": "<p>Hello<br>World</p>"},
        }
    )

    assert message["message_id"] == "<msg@example.com>"
    assert message["graph_message_id"] == "graph-1"
    assert message["text_body"] == "Hello\nWorld"
    assert message["html_body"] == "<p>Hello<br>World</p>"


def test_graph_client_lists_unread_and_marks_read(monkeypatch):
    cfg = NotesMailboxConfig(
        source_system="notes_mailbox",
        poll_interval=60,
        transport_mode="graph",
        mailbox_identity="notes@longboardfella.com.au",
        public_outbox_dir="/tmp/public",
        private_outbox_dir="/tmp/private",
        public_vault_dir="/tmp/vault_public",
        private_vault_dir="/tmp/vault_private",
        write_vault_markdown=False,
        graph_tenant_id="tenant",
        graph_client_id="client",
        graph_client_secret="secret",
        graph_mailbox="notes@longboardfella.com.au",
        graph_page_size=10,
        graph_timeout=30,
    )
    calls = []

    def _fake_post(url, data=None, timeout=None, **_kwargs):
        calls.append(("POST", url, data, None))
        return _FakeResponse({"access_token": "token", "expires_in": 3600})

    def _fake_get(url, headers=None, params=None, timeout=None, **_kwargs):
        calls.append(("GET", url, headers, params))
        return _FakeResponse({"value": [{"id": "graph-1", "subject": "Note", "body": {"contentType": "text", "content": "Body"}}]})

    def _fake_patch(url, headers=None, json=None, timeout=None, **_kwargs):
        calls.append(("PATCH", url, headers, json))
        return _FakeResponse({})

    monkeypatch.setattr("worker.notes_mailbox_worker.requests.post", _fake_post)
    monkeypatch.setattr("worker.notes_mailbox_worker.requests.get", _fake_get)
    monkeypatch.setattr("worker.notes_mailbox_worker.requests.patch", _fake_patch)

    client = NotesGraphClient(cfg)
    messages = client.list_unread_messages()
    client.mark_read("graph-1")

    assert messages[0]["id"] == "graph-1"
    assert calls[0][0] == "POST"
    assert calls[1][0] == "GET"
    assert calls[1][3]["$filter"] == "isRead eq false"
    assert calls[2][0] == "PATCH"
    assert calls[2][3] == {"isRead": True}


def test_graph_client_pages_unread_notes(monkeypatch):
    cfg = _notes_config(Path("/tmp/notes-graph-test"))
    cfg.transport_mode = "graph"
    cfg.graph_client_id = "client"
    cfg.graph_client_secret = "secret"
    cfg.graph_tenant_id = "tenant"
    calls = []
    monkeypatch.setattr(NotesGraphClient, "_headers", lambda self: {"Authorization": "Bearer token"})

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("params")))
        if len(calls) == 1:
            return _FakeResponse({"value": [{"id": "one"}],
                                  "@odata.nextLink": "https://graph.microsoft.com/v1.0/next?page=2"})
        return _FakeResponse({"value": [{"id": "two"}]})

    monkeypatch.setattr("worker.notes_mailbox_worker.requests.get", fake_get)
    assert [item["id"] for item in NotesGraphClient(cfg).list_unread_messages(
        since="2026-10-15T06:15:00Z")] == ["one", "two"]
    assert "receivedDateTime ge 2026-10-15T06:15:00Z" in calls[0][1]["$filter"]
    assert calls[1][1] is None


def test_graph_client_reads_attachment_pages_without_leaving_graph(monkeypatch):
    cfg = _notes_config(Path("/tmp/notes-attachments-test"))
    monkeypatch.setattr(NotesGraphClient, "_headers", lambda self: {"Authorization": "Bearer token"})
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResponse({"value": [{"name": "one.pdf", "contentBytes": "YQ=="}],
                                  "@odata.nextLink": "https://graph.microsoft.com/v1.0/next?page=2"})
        return _FakeResponse({"value": [{"name": "two.pdf", "contentBytes": "Yg=="}]})

    monkeypatch.setattr("worker.notes_mailbox_worker.requests.get", fake_get)
    attachments = NotesGraphClient(cfg).list_attachments("message/1")
    assert [item["name"] for item in attachments] == ["one.pdf", "two.pdf"]
    assert "message%2F1" in calls[0]
