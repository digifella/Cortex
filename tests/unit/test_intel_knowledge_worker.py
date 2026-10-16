from worker.intel_knowledge_worker import classify_intel_route, poll_once
from worker.notes_mailbox_worker import NotesMailboxConfig, NotesMailboxProcessor


def test_intel_private_marker_overrides_public_review():
    assert classify_intel_route("PRIVATE VAULT: itinerary", "Public website link")["route"] == "private_vault"
    assert classify_intel_route("Article", "Client confidential details")["route"] == "private_vault"
    assert classify_intel_route("Museum label", "A public gallery essay") ["route"] == "public_review"


def test_intel_reader_stages_and_dedupes_without_publication(tmp_path):
    cfg = NotesMailboxConfig(
        source_system="intel_mailbox", poll_interval=600, transport_mode="graph",
        mailbox_identity="intel@longboardfella.com.au",
        public_outbox_dir=str(tmp_path / "audit" / "public"),
        private_outbox_dir=str(tmp_path / "audit" / "private"),
        public_vault_dir=str(tmp_path / "public-vault"),
        private_vault_dir=str(tmp_path / "private-vault" / "Intel_Inbox"),
        public_review_vault_dir=str(tmp_path / "private-vault" / "Public_Review" / "intel"),
        write_vault_markdown=True, graph_tenant_id="", graph_client_id="",
        graph_client_secret="", graph_mailbox="intel@longboardfella.com.au",
        graph_page_size=25, graph_timeout=30,
    )
    processor = NotesMailboxProcessor(cfg, classifier=classify_intel_route)

    class FakeClient:
        def list_unread_messages(self, since, unread_only):
            assert not unread_only
            return [{"id": "graph-1", "subject": "Museum label", "receivedDateTime": "2026-10-15T08:00:00Z",
                     "body": {"contentType": "text", "content": "A public gallery essay about sculpture."},
                     "from": {"emailAddress": {"address": "paul@example.com"}}}]

    assert poll_once(FakeClient(), processor) == (1, 0)
    assert poll_once(FakeClient(), processor) == (0, 1)
    notes = list((tmp_path / "private-vault").rglob("*.md"))
    assert len(notes) == 1
    assert "wiki-ready: false" in notes[0].read_text(encoding="utf-8")
    assert not list((tmp_path / "public-vault").rglob("*.md"))
