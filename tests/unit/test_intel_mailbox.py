import json
from email.message import EmailMessage
from pathlib import Path

from cortex_engine.intel_mailbox import (
    IntelMailboxConfig,
    IntelMailboxPoller,
    IntelMailboxStore,
    parse_email_bytes,
)


def _sample_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Carolyn Bell update"
    msg["From"] = "Intel Feed <intel.longboardfella@gmail.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
    msg["Message-ID"] = "<msg-1@example.com>"
    msg.set_content("Carolyn Bell has started a new program role at Silverchain.\nContact: cbell@example.com")
    msg.add_alternative("<html><body><p>Carolyn Bell update</p></body></html>", subtype="html")
    msg.add_attachment(
        b"fake-image-bytes",
        maintype="image",
        subtype="jpeg",
        filename="Screenshot 2026-03-12 112809.jpg",
    )
    return msg.as_bytes()


def _duplicate_filename_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Stakeholders"
    msg["From"] = "Intel Feed <intel.longboardfella@gmail.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
    msg["Message-ID"] = "<msg-dup@example.com>"
    msg.set_content("[image: image.png]\n[image: image.png]")
    msg.add_attachment(b"one", maintype="image", subtype="png", filename="image.png")
    msg.add_attachment(b"two", maintype="image", subtype="png", filename="image.png")
    return msg.as_bytes()


def test_parse_email_bytes_extracts_bodies_and_attachments():
    parsed = parse_email_bytes(_sample_message_bytes())

    assert parsed["message_id"] == "<msg-1@example.com>"
    assert parsed["from_email"] == "intel.longboardfella@gmail.com"
    assert "Carolyn Bell" in parsed["raw_text"]
    assert "html" in parsed["html_text"].lower()
    assert parsed["attachments"][0]["kind"] == "image"
    assert "cbell@example.com" in parsed["extracted_emails"]


def test_store_preserves_duplicate_attachment_filenames(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    parsed = parse_email_bytes(_duplicate_filename_message_bytes())

    record = store.persist_message(parsed, _duplicate_filename_message_bytes(), parsed["attachments"])

    filenames = [item["filename"] for item in record["attachments"]]
    assert filenames == ["image.png", "image_2.png"]
    assert all(Path(item["stored_path"]).exists() for item in record["attachments"])


class _FakeSignalStore:
    def ingest_signal(self, payload):
        return {
            "signal_id": "sig_test_1",
            "matched_profile_keys": ["profile_1"],
            "needs_review": False,
            "payload": payload,
        }


class _FakeIMAP:
    def __init__(self, _host, _port):
        self.stored = []

    def login(self, _username, _password):
        return "OK", [b"logged in"]

    def select(self, _folder):
        return "OK", [b"1"]

    def search(self, _charset, _criteria):
        return "OK", [b"1"]

    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _sample_message_bytes())]

    def store(self, imap_id, mode, flag):
        self.stored.append((imap_id, mode, flag))
        return "OK", []

    def logout(self):
        return "BYE", []


def test_mailbox_poller_processes_message_and_writes_outbox(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")

    def _extractor(_payload):
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        return (
            {
                "status": "extracted",
                "entity_count": 1,
                "people": [{"canonical_name": "Carolyn Bell", "current_employer": "Silverchain"}],
                "organisations": [{"canonical_name": "Silverchain Group"}],
                "emails": [{"email": "cbell@example.com"}],
                "target_update_suggestions": [],
                "warnings": [],
            },
            out,
        )

    cfg = IntelMailboxConfig(
        host="imap.gmail.com",
        port=993,
        username="intel.longboardfella@gmail.com",
        password="secret",
        folder="INBOX",
        org_name="Longboardfella",
        poll_limit=5,
        search_criteria="UNSEEN",
        allowed_senders=(),
        source_system="cortex_mailbox",
        callback_url="",
        callback_secret="",
        callback_timeout=30,
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=_extractor,
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    assert summary["failures"] == 0
    messages = store.list_messages()
    assert len(messages) == 1
    assert messages[0]["status"] == "processed"
    assert Path(messages[0]["raw_path"]).exists()
    assert messages[0]["attachments"][0]["kind"] == "image"

    outbox_files = list((tmp_path / "intel_mailbox" / "outbox").glob("*.json"))
    assert len(outbox_files) == 1
    payload = json.loads(outbox_files[0].read_text(encoding="utf-8"))
    assert payload["result_type"] == "intel_extract_result"
    assert payload["signal"]["signal_id"] == "sig_test_1"
    assert payload["output_data"]["entity_count"] == 1
    assert payload["entities"][0]["canonical_name"] == "Carolyn Bell"
    assert payload["target_update_suggestions"] == []
    assert payload["intel_id"].startswith("mail_")
