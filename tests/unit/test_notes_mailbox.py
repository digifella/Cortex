from pathlib import Path

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
    )


def test_notes_mailbox_processor_writes_public_note(tmp_path):
    cfg = _notes_config(tmp_path)
    processor = NotesMailboxProcessor(cfg)

    result = processor.process_message(
        {
            "subject": "Meeting notes from today",
            "text_body": "Capture this in the public stash.",
            "from_email": "paul@example.com",
        }
    )

    assert result["status"] == "outbox_written"
    assert result["route"] == "public_stash"
    assert Path(result["path"]).exists()
    assert Path(result["vault_path"]).exists()
    assert result["vault_path"].endswith(".md")


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

    processor.record_processed_graph_id("graph-1")

    reloaded = NotesMailboxProcessor(cfg)
    assert reloaded.has_processed_graph_id("graph-1")


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
    assert message["text_body"] == "Hello World"
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
