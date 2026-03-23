import json
from email.message import EmailMessage
from pathlib import Path

from cortex_engine.intel_deduplicator import find_duplicate_note
from cortex_engine.intel_mailbox import (
    IntelMailboxConfig,
    IntelMailboxPoller,
    IntelMailboxStore,
    parse_email_bytes,
)
from cortex_engine.intel_note_processor import IntelNoteProcessor
from cortex_engine.strategic_doc_analyser import analyse_strategic_documents, clean_indicator_evidence_text, clean_strategic_role_label


def _sample_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Carolyn Bell update"
    msg["From"] = "Intel Feed <intel.longboardfella@gmail.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
    msg["Message-ID"] = "<msg-1@example.com>"
    msg.set_content("Carolyn Bell has started a new program role at Silverchain.\nContact: cbell@example.com\nSee https://example.com/carolyn")
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


def _csv_import_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "PROFILES DRY RUN"
    msg["From"] = "Paul <paul@example.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
    msg["Message-ID"] = "<msg-csv@example.com>"
    msg.set_content("Please import attached profiles CSV.")
    msg.add_attachment(
        b"canonical_name,current_employer,current_role,watch\nJane Smith,Acme Corp,CEO,yes\n",
        maintype="text",
        subtype="csv",
        filename="profiles.csv",
    )
    return msg.as_bytes()


def _intel_subject_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "INTEL: Carolyn Bell update"
    msg["From"] = "Intel Feed <intel.longboardfella@gmail.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
    msg["Message-ID"] = "<msg-intel@example.com>"
    msg.set_content("Carolyn Bell has started a new program role at Silverchain.")
    return msg.as_bytes()


def _routed_document_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "entity: Escient | Barwon Water"
    msg["From"] = "Intel Feed <intel.longboardfella@gmail.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
    msg["Message-ID"] = "<msg-routed-doc@example.com>"
    msg.set_content("Uploading Barwon Water organisation material.")
    msg.add_attachment(
        b"%PDF-1.4 fake pdf bytes",
        maintype="application",
        subtype="pdf",
        filename="barwon-water-org-chart.pdf",
    )
    return msg.as_bytes()


def _prefixed_routed_document_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Subject: entity: Escient | Barwon Water"
    msg["From"] = "Intel Feed <intel.longboardfella@gmail.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
    msg["Message-ID"] = "<msg-prefixed-routed-doc@example.com>"
    msg.set_content("Uploading Barwon Water organisation material.")
    msg.add_attachment(
        b"%PDF-1.4 fake pdf bytes",
        maintype="application",
        subtype="pdf",
        filename="barwon-water-strategy.pdf",
    )
    return msg.as_bytes()


def _prefixed_routed_org_chart_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Subject: entity: Escient | Barwon Water"
    msg["From"] = "Intel Feed <intel.longboardfella@gmail.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
    msg["Message-ID"] = "<msg-prefixed-routed-org-chart@example.com>"
    msg.set_content("Uploading Barwon Water org chart.")
    msg.add_attachment(
        b"fake-image-bytes",
        maintype="image",
        subtype="png",
        filename="barwon-water-org-chart.png",
    )
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

    def list_profiles(self, org_name=""):
        del org_name
        return []

    def get_org_context(self, org_name):
        return {"org_name": org_name, "org_alumni": [], "org_strategic_profile": {}}

    def find_relationship_paths(self, org_name, target_names, max_hops=4, limit=5):
        del org_name, target_names, max_hops, limit
        return []

    def reconcile_intel_note_delivery(self, org_name, trace_id, payload, response):
        del org_name, trace_id, payload, response
        return {"intel_id": "intel_note_1", "linked_entities": 1}

    def get_state(self):
        return {}


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


class _FakeCSVIMAP(_FakeIMAP):
    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _csv_import_message_bytes())]


class _FakeIntelIMAP(_FakeIMAP):
    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _intel_subject_message_bytes())]


class _FakeRoutedDocumentIMAP(_FakeIMAP):
    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _routed_document_message_bytes())]


class _FakePrefixedRoutedDocumentIMAP(_FakeIMAP):
    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _prefixed_routed_document_message_bytes())]


class _FakePrefixedRoutedOrgChartIMAP(_FakeIMAP):
    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _prefixed_routed_org_chart_message_bytes())]


class _FakeReplyClient:
    def send(self, to_email, subject, body, in_reply_to=""):
        return {
            "status": "sent",
            "to_email": to_email,
            "subject": subject,
            "body": body,
            "in_reply_to": in_reply_to,
        }


class _RecordingResultClient:
    def __init__(self):
        self.calls = []

    def deliver(self, message_key, payload, delivery_payload=None, callback_url_override=""):
        self.calls.append(
            {
                "message_key": message_key,
                "payload": payload,
                "delivery_payload": delivery_payload,
                "callback_url_override": callback_url_override,
            }
        )
        return {"status": "posted", "http_status": 200, "response": {"ok": True, "intel_id": "intel_note_1"}}


class _RoutingSignalStore(_FakeSignalStore):
    def list_profiles(self, org_name=""):
        profiles = [
            {
                "org_name": "Escient",
                "target_type": "organisation",
                "canonical_name": "Escient",
                "aliases": [],
            }
        ]
        if org_name:
            return [item for item in profiles if item["org_name"] == org_name]
        return profiles


class _OrgContextOnlySignalStore(_FakeSignalStore):
    def get_state(self):
        return {
            "org_contexts": {
                "escient": {
                    "org_name": "Escient",
                    "org_alumni": [],
                    "org_strategic_profile": {},
                }
            }
        }


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
        note_callback_url="",
        callback_secret="",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
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
    assert payload["website_payload"]["action"] == "ingest_intel_note"
    assert payload["website_payload"]["secret"] == "[redacted]"
    assert payload["website_payload"]["primary_entity"]["name"] == "Carolyn Bell"
    assert payload["website_payload"]["note"]["source_type"] in {"meeting_note", "general"}
    assert payload["website_payload"]["urls"][0]["url"] == "https://example.com/carolyn"


def test_mailbox_poller_uses_note_callback_url_for_structured_note_posts(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()

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
                "attachments": [],
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
        callback_url="https://example.com/admin/queue_worker_api.php?action=import_cortex_extract",
        note_callback_url="https://example.com/lab/market_radar_api.php?action=ingest_intel_note",
        callback_secret="secret",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=_extractor,
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
        result_client=result_client,
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    assert result_client.calls
    assert result_client.calls[0]["callback_url_override"] == "https://example.com/lab/market_radar_api.php?action=ingest_intel_note"
    assert result_client.calls[0]["delivery_payload"]["action"] == "ingest_intel_note"


def test_mailbox_routing_override_and_subject_org_hint_flow_into_note_payload(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()

    def _extractor(payload):
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        assert payload["org_name"] == "Escient"
        assert payload["subject"] == "Barwon Water"
        assert payload["parsed_candidate_employer"] == "Barwon Water"
        return (
            {
                "status": "extracted",
                "entity_count": 1,
                "people": [{"canonical_name": "Jane Smith", "current_employer": "", "current_role": "Managing Director"}],
                "organisations": [],
                "entities": [
                    {
                        "canonical_name": "Jane Smith",
                        "name": "Jane Smith",
                        "target_type": "person",
                        "current_employer": "",
                        "current_role": "Managing Director",
                        "confidence": 0.9,
                        "evidence": "Org chart OCR pair",
                    }
                ],
                "emails": [],
                "target_update_suggestions": [],
                "warnings": [],
                "attachments": [{"filename": "barwon-water-org-chart.pdf", "status": "processed", "excerpt": "Barwon Water organisation chart"}],
                "processing_meta": {},
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
        callback_url="https://example.com/admin/queue_worker_api.php?action=import_cortex_extract",
        note_callback_url="https://example.com/lab/market_radar_api.php?action=ingest_intel_note",
        callback_secret="secret",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=_extractor,
        imap_factory=_FakeRoutedDocumentIMAP,
        signal_store=_RoutingSignalStore(),
        result_client=result_client,
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    assert result_client.calls
    delivery_payload = result_client.calls[0]["delivery_payload"]
    assert delivery_payload["org_name"] == "Escient"
    assert delivery_payload["primary_entity"]["name"] == "Barwon Water"
    assert delivery_payload["primary_entity"]["target_type"] == "organisation"
    assert delivery_payload["mailbox_routing"]["status"] == "matched_override"
    assert delivery_payload["mailbox_routing"]["requested_org_name"] == "Escient"
    assert delivery_payload["mailbox_routing"]["subject_org_hint"] == "Barwon Water"


def test_mailbox_routing_strips_literal_subject_prefix_and_matches_org_context(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()

    def _extractor(payload):
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        assert payload["org_name"] == "Escient"
        assert payload["subject"] == "Barwon Water"
        assert payload["parsed_candidate_employer"] == "Barwon Water"
        return (
            {
                "status": "extracted",
                "entity_count": 1,
                "people": [],
                "organisations": [{"canonical_name": "Barwon Water"}],
                "entities": [
                    {
                        "canonical_name": "Barwon Water",
                        "name": "Barwon Water",
                        "target_type": "organisation",
                        "confidence": 0.95,
                        "evidence": "Document title and subject line",
                    }
                ],
                "emails": [],
                "target_update_suggestions": [],
                "warnings": [],
                "attachments": [{"filename": "barwon-water-strategy.pdf", "status": "processed", "excerpt": "Barwon Water Strategy 2030"}],
                "processing_meta": {},
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
        callback_url="https://example.com/admin/queue_worker_api.php?action=import_cortex_extract",
        note_callback_url="https://example.com/lab/market_radar_api.php?action=ingest_intel_note",
        callback_secret="secret",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=_extractor,
        imap_factory=_FakePrefixedRoutedDocumentIMAP,
        signal_store=_OrgContextOnlySignalStore(),
        result_client=result_client,
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    delivery_payload = result_client.calls[0]["delivery_payload"]
    assert delivery_payload["org_name"] == "Escient"
    assert delivery_payload["mailbox_routing"]["status"] == "matched_override"
    assert delivery_payload["mailbox_routing"]["requested_org_name"] == "Escient"
    assert delivery_payload["mailbox_routing"]["subject_org_hint"] == "Barwon Water"
    assert delivery_payload["primary_entity"]["name"] == "Barwon Water"


def test_mailbox_org_chart_image_uses_subject_org_hint_as_primary_entity(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()

    def _extractor(payload):
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        return (
            {
                "status": "extracted",
                "entity_count": 2,
                "people": [{"canonical_name": "Shaun Cumming", "current_employer": "Barwon Water", "current_role": "Managing Director"}],
                "organisations": [
                    {"canonical_name": "Longboardfella Consulting Pty Ltd"},
                    {"canonical_name": "Barwon Water"},
                ],
                "entities": [
                    {
                        "canonical_name": "Shaun Cumming",
                        "name": "Shaun Cumming",
                        "target_type": "person",
                        "current_employer": "Barwon Water",
                        "current_role": "Managing Director",
                        "confidence": 0.95,
                        "evidence": "Listed as Managing Director in org chart",
                    },
                    {
                        "canonical_name": "Longboardfella Consulting Pty Ltd",
                        "name": "Longboardfella Consulting Pty Ltd",
                        "target_type": "organisation",
                        "confidence": 0.8,
                        "evidence": "Sender signature",
                    },
                    {
                        "canonical_name": "Barwon Water",
                        "name": "Barwon Water",
                        "target_type": "organisation",
                        "confidence": 0.95,
                        "evidence": "Organizational chart and email subject line reference Barwon Water",
                    },
                ],
                "emails": [],
                "target_update_suggestions": [],
                "warnings": [],
                "attachments": [{"filename": "barwon-water-org-chart.png", "status": "processed", "excerpt": "Barwon Water org chart"}],
                "processing_meta": {},
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
        callback_url="https://example.com/admin/queue_worker_api.php?action=import_cortex_extract",
        note_callback_url="https://example.com/lab/market_radar_api.php?action=ingest_intel_note",
        callback_secret="secret",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=_extractor,
        imap_factory=_FakePrefixedRoutedOrgChartIMAP,
        signal_store=_OrgContextOnlySignalStore(),
        result_client=result_client,
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    delivery_payload = result_client.calls[0]["delivery_payload"]
    assert delivery_payload["org_name"] == "Escient"
    assert delivery_payload["note"]["title"] == "Barwon Water"
    assert delivery_payload["primary_entity"]["name"] == "Barwon Water"
    assert delivery_payload["primary_entity"]["target_type"] == "organisation"
    assert delivery_payload["mailbox_routing"]["status"] == "matched_override"


def test_find_duplicate_note_uses_attachment_fingerprint_for_document_resends(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    first_message = parse_email_bytes(_sample_message_bytes())
    persisted = store.persist_message(first_message, _sample_message_bytes(), first_message["attachments"])
    result_payload = {
        "website_payload": {
            "primary_entity": {"name": ""},
            "note": {
                "source_type": "org_chart",
                "note_date": "2026-03-19",
                "content": "Initial org chart note",
                "attachment_fingerprints": ["chart.pdf:abc123"],
            },
        }
    }
    store.record_processed(
        persisted["message_key"],
        "trace-first",
        result_payload,
        {"response": {"intel_id": "intel_123"}},
    )

    duplicate = find_duplicate_note(
        store,
        {
            "primary_entity": {"name": ""},
            "note": {
                "source_type": "org_chart",
                "note_date": "2026-03-19",
                "content": "Resent org chart with slightly different mail wrapper",
                "attachment_fingerprints": ["chart.pdf:abc123"],
            },
        },
    )

    assert duplicate is not None
    assert duplicate["existing_intel_id"] == "intel_123"
    assert duplicate["trace_id"] == "trace-first"


def test_analyse_strategic_documents_extracts_org_name_and_themes():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "department-of-health-strategic-plan-2026.pdf",
                "status": "processed",
                "excerpt": "Department of Health Strategic Plan 2026 Vision Better care for all Mission Build a stronger health system Strategic priorities Digital transformation Workforce sustainability",
            }
        ],
        extracted_summary="Department of Health strategic plan with priorities in digital transformation and workforce sustainability.",
        subject="Department of Health Strategic Plan 2026",
        raw_text="Vision Better care for all Mission Build a stronger health system Strategic priorities Digital transformation Workforce sustainability",
    )

    assert analysis["doc_type"] == "strategic_plan"
    assert analysis["org_name"]
    assert analysis["has_strategy_markers"] is True
    assert analysis["themes"]
    assert analysis["strategic_signals"]


def test_analyse_strategic_documents_extracts_richer_pressures_and_priorities():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "2026-2030-strategic-direction-document_compressed.pdf",
                "status": "processed",
                "excerpt": (
                    "Royal Australasian College of Physicians Strategic Direction 2026 to 2030 "
                    "Our members are changing Women now outnumber men in age groups under 45 and nearly 60 per cent "
                    "of current trainees are women. Member satisfaction has been declining since 2016. "
                    "Digital-first is now the baseline. Membership fees cannot increase without limit. "
                    "Workforce shortages and unsafe workplaces are placing pressure on physicians."
                ),
            }
        ],
        extracted_summary="RACP strategic direction document focused on simplification, digital-first services, and diversified revenue.",
        subject="Fwd: RACP Strategic Direction 2026-2030",
        raw_text=(
            "Members and employees have told us delivery has become too complex. "
            "Digital capability is foundational to trust, efficiency, credibility and robustness. "
            "Long-term sustainability depends on clearer priorities and diversified revenue. "
            "The College is committed to advancing Aboriginal, Torres Strait Islander and Maori health and education as core business of the RACP."
        ),
    )

    headlines = [item["headline"] for item in analysis["strategic_signals"]]

    assert analysis["org_name"] == "Royal Australasian College of Physicians"
    assert "Membership base is changing" in headlines
    assert "Member trust and value-for-money pressure" in headlines
    assert "Digital-first service transformation" in headlines
    assert "Revenue diversification and financial sustainability" in headlines
    assert "Workforce sustainability and workplace pressure" in headlines
    assert "Indigenous health and cultural safety are core business" in headlines
    assert "Key strategic signals:" in analysis["strategic_summary"]


def test_analyse_strategic_documents_extracts_leadership_signatories():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "2026-2030-strategic-direction-document_compressed.pdf",
                "status": "processed",
                "excerpt": (
                    "Royal Australasian College of Physicians Strategic Direction 2026 to 2030\n"
                    "Professor Jennifer Martin\n"
                    "President and Chair of the Board\n"
                    "The Royal Australasian College of Physicians\n"
                    "Steffen Faurby\n"
                    "Chief Executive Officer\n"
                    "The Royal Australasian College of Physicians\n"
                ),
            }
        ],
        extracted_summary="RACP strategic direction document.",
        subject="RACP Strategic Direction 2026-2030",
        raw_text="",
    )

    leaders = [(item["name"], item["current_role"]) for item in analysis["leadership_people"]]

    assert ("Professor Jennifer Martin", "President and Chair of the Board") in leaders
    assert ("Steffen Faurby", "Chief Executive Officer") in leaders


def test_analyse_strategic_documents_ignores_screenshot_style_filenames_for_org_name():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Screenshot 2026-03-19 192854.jpg",
                "status": "processed",
                "excerpt": "Image attachment showing a mailbox screenshot.",
            }
        ],
        extracted_summary="Mailbox screenshot with no organisation metadata.",
        subject="Fw:",
        raw_text="",
    )

    assert analysis["org_name"] == ""


def test_analyse_strategic_documents_extracts_org_name_from_strategic_direction_content():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "2026-2030-strategic-direction-document_compressed.pdf",
                "status": "processed",
                "excerpt": "Royal Australasian College of Physicians Strategic Direction 2026 to 2030 Our vision Better healthcare through physician leadership",
            }
        ],
        extracted_summary="RACP strategic direction document for 2026 to 2030.",
        subject="",
        raw_text="",
    )

    assert analysis["doc_type"] == "strategic_plan"
    assert analysis["org_name"] == "Royal Australasian College of Physicians"


def test_analyse_strategic_documents_extracts_annual_report_performance_and_stakeholders():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "2024-racp-annual-report.pdf",
                "status": "processed",
                "excerpt": (
                    "The Royal Australasian College of Physicians ANNUAL REPORT 2024 "
                    "32,347 All Members Annual Growth ▲ 1,184 (3.8%) "
                    "15 meetings with MPs and other key stakeholders "
                    "50% members say in survey that workforce pressures affect personal lives "
                    "Acknowledging that we still have much work to do in cultural safety, taken together, our achievements in 2024 set a strong foundation for the College's future. "
                    "In 2024, we achieved a four-year accreditation as a CPD Home from the AMC "
                    "99 per cent of our members completing their CPD requirements "
                    "Overall, the total revenue and other income for the year 2024 increased from $74.2m to $87.3m. "
                    "The deficit of $0.1m ($3.7m 2023) reflects our investment in education renewal and Information Technology.\n"
                    "Professor Jennifer Martin\n"
                    "RACP President\n"
                    "Royal Australasian College of Physicians\n"
                    "Steffen Faurby\n"
                    "Chief Executive Officer\n"
                    "Royal Australasian College of Physicians"
                ),
            }
        ],
        extracted_summary="RACP annual report covering governance reform, accreditation, technology uplift, and member service improvements.",
        subject="RACP Annual Report 2024",
        raw_text="",
    )

    labels = [item["label"] for item in analysis["performance_indicators"]]
    stakeholder_names = [item["name"] for item in analysis["key_stakeholders"]]
    stakeholder_roles = {item["name"]: item["current_role"] for item in analysis["key_stakeholders"]}
    indicators = {item["label"]: item for item in analysis["performance_indicators"]}
    strategic_signals = {item["headline"]: item for item in analysis["strategic_signals"]}

    assert analysis["doc_type"] == "annual_report"
    assert analysis["org_name"] == "Royal Australasian College of Physicians"
    assert "Membership scale" in labels
    assert "Workforce pressure on members" in labels
    assert "Total revenue growth" in labels
    assert "Operating result" in labels
    assert "CPD Home accreditation" in labels
    assert "Professor Jennifer Martin" in stakeholder_names
    assert "Steffen Faurby" in stakeholder_names
    assert stakeholder_roles["Professor Jennifer Martin"] == "President"
    assert stakeholder_roles["Steffen Faurby"] == "Chief Executive Officer"
    assert "Chief Executive Officer, RACP" not in stakeholder_names
    assert not indicators["Membership scale"]["evidence"].startswith(",")
    assert indicators["Workforce pressure on members"]["evidence"].startswith("50% members say in survey")
    assert indicators["Stakeholder engagement"]["evidence"].startswith("15 meetings with MPs and other key stakeholders")
    assert indicators["Operating result"]["evidence"].startswith("The deficit of $0.1m")
    assert "strong foundation for the College's future" in strategic_signals["Indigenous health and cultural safety are core business"]["snippet"].replace("\u2019", "'")


def test_strategic_doc_cleanup_helpers_normalise_roles_and_trim_ocr_tails():
    assert clean_strategic_role_label("Chief Executive Officer (CEO)", "Royal Australasian College of Physicians") == "Chief Executive Officer"
    assert clean_strategic_role_label("Dean of the College", "Royal Australasian College of Physicians") == "Dean of the College"
    assert clean_indicator_evidence_text(
        "75 RACP and member policy statements, submissions, endorsements to Australian and Aotearoa New Zealand Governments THE ROYAL AUSTRALASIAN COLLEGE OF PHYSICIANS I ANNUAL REPORT 2024 # % 55.0%"
    ) == "75 RACP and member policy statements, submissions, endorsements to Australian and Aotearoa New Zealand Governments"


def test_analyse_strategic_documents_filters_title_only_and_table_artifacts_from_stakeholders():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "annual-report.pdf",
                "status": "processed",
                "excerpt": (
                    "Royal Australasian College of Physicians Annual Report 2024\n"
                    "Professor Jennifer Martin\n"
                    "RACP President\n"
                    "Royal Australasian College of Physicians\n"
                    "Steffen Faurby\n"
                    "Chief Executive Officer\n"
                    "Royal Australasian College of Physicians\n"
                    "Chief Executive Officer, RACP\n"
                    "CEO's Message\n"
                    "Royal Australasian College of Physicians\n"
                    "Board Attendance\n"
                    "The following table shows attendance of Directors at Board meetings during 2024:\n"
                    "Royal Australasian College of Physicians\n"
                    "Governance Committee\n"
                    "Leadership Capabilities\n"
                    "Royal Australasian College of Physicians\n"
                    "Professor Inam Haq, FRACP, FRCP,\n"
                    "Executive General Manager,\n"
                    "Royal Australasian College of Physicians\n"
                ),
            }
        ],
        extracted_summary="Annual report with leadership and governance content.",
        subject="Annual Report 2024",
        raw_text="",
    )

    stakeholder_names = [item["name"] for item in analysis["key_stakeholders"]]

    assert stakeholder_names == [
        "Professor Jennifer Martin",
        "Steffen Faurby",
        "Professor Inam Haq, FRACP, FRCP",
    ]


def test_mailbox_handoff_prefers_real_document_org_over_generic_filename_primary(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
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
        callback_url="https://example.com/admin/queue_worker_api.php?action=import_cortex_extract",
        note_callback_url="https://example.com/lab/market_radar_api.php?action=ingest_intel_note",
        callback_secret="secret",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )
    payload = poller._build_ingest_note_payload(
        message={
            "subject": "",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-19T09:38:49+00:00",
            "raw_text": "Dr. Paul Cooper, Director, Longboardfella Consulting Pty Ltd",
        },
        persisted={
            "attachments": [
                {
                    "filename": "2026-2030-strategic-direction-document_compressed.pdf",
                    "stored_path": "",
                }
            ]
        },
        output_data={
            "status": "extracted",
            "entity_count": 5,
            "people": [
                {
                    "canonical_name": "Dr. Paul Cooper",
                    "current_employer": "Longboardfella Consulting Pty Ltd",
                    "current_role": "Director",
                },
                {
                    "canonical_name": "Professor Jennifer Martin",
                    "current_employer": "Royal Australasian College of Physicians",
                    "current_role": "President and Chair of the Board",
                },
            ],
            "organisations": [
                {
                    "canonical_name": "Longboardfella Consulting Pty Ltd",
                    "industry": "Consulting",
                    "evidence": "Business contact details with ACN 650 470 474",
                },
                {
                    "canonical_name": "Royal Australasian College of Physicians",
                    "industry": "Medical Education & Professional Services",
                    "evidence": "Strategic direction document 2026-2030 authored by RACP",
                },
            ],
            "entities": [
                {
                    "canonical_name": "Dr. Paul Cooper",
                    "name": "Dr. Paul Cooper",
                    "target_type": "person",
                    "current_employer": "Longboardfella Consulting Pty Ltd",
                    "current_role": "Director",
                    "confidence": 0.95,
                    "evidence": "Business card details",
                },
                {
                    "canonical_name": "Professor Jennifer Martin",
                    "name": "Professor Jennifer Martin",
                    "target_type": "person",
                    "current_employer": "Royal Australasian College of Physicians",
                    "current_role": "President and Chair of the Board",
                    "confidence": 0.9,
                    "evidence": "Signed foreword in RACP strategic document",
                },
                {
                    "canonical_name": "Longboardfella Consulting Pty Ltd",
                    "name": "Longboardfella Consulting Pty Ltd",
                    "target_type": "organisation",
                    "confidence": 0.95,
                    "industry": "Consulting",
                    "evidence": "Business contact details with ACN 650 470 474",
                },
                {
                    "canonical_name": "Royal Australasian College of Physicians",
                    "name": "Royal Australasian College of Physicians",
                    "target_type": "organisation",
                    "confidence": 0.95,
                    "industry": "Medical Education & Professional Services",
                    "evidence": "Strategic direction document 2026-2030 authored by RACP",
                },
            ],
            "emails": [{"email": "paul@longboardfella.com.au"}],
            "target_update_suggestions": [],
            "warnings": [],
            "summary": "Primary contact is Dr. Paul Cooper. Secondary document contains RACP 2026-2030 strategic direction.",
            "attachments": [
                {
                    "filename": "2026-2030-strategic-direction-document_compressed.pdf",
                    "status": "processed",
                    "excerpt": "Royal Australasian College of Physicians Strategic Direction 2026 to 2030",
                }
            ],
            "processing_meta": {
                "message_kind": "document_analysis",
                "strategic_doc": {
                    "doc_type": "strategic_plan",
                    "org_name": "Royal Australasian College of Physicians",
                    "strategic_summary": "Document appears to relate to Royal Australasian College of Physicians.",
                    "themes": ["Physician leadership"],
                    "initiatives": ["Training reform"],
                },
            },
        },
        signal={},
        markdown_text=(
            "# Intel Extraction Result\n\n"
            "## Summary\n\n"
            "Original extractor report.\n\n"
            "## Entities\n\n"
            "- Board Attendance (person)\n"
            "- Governance Committee (person)\n"
            "- Royal Australasian College of Physicians (organisation)\n\n"
            "## Emails\n\n"
            "- info@pkf.com.au\n"
        ),
        message_kind="document_analysis",
    )

    assert payload["primary_entity"]["name"] == "Royal Australasian College of Physicians"
    assert payload["primary_entity"]["target_type"] == "organisation"
    assert payload["note"]["source_type"] == "strategic_plan"
    assert payload["note"]["title"] == "Royal Australasian College of Physicians Strategic Direction 2026-2030"


def test_mailbox_handoff_emits_specific_strategic_signals(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
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
        note_callback_url="",
        callback_secret="",
        callback_timeout=30,
        profile_import_url="",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "Fwd: RACP strategy",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-20T09:00:00+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "Royal Australasian College of Physicians", "target_type": "organisation", "confidence": 0.95}],
            "attachments": [],
            "summary": "Strategic direction document for RACP.",
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "strategic_plan",
                    "org_name": "Royal Australasian College of Physicians",
                    "strategic_summary": (
                        "Document appears to relate to Royal Australasian College of Physicians. "
                        "Key strategic signals: Membership base is changing, Digital-first service transformation."
                    ),
                    "themes": [],
                    "initiatives": [],
                    "strategic_signals": [
                        {
                            "headline": "Membership base is changing",
                            "category": "member_base",
                            "snippet": "Women now outnumber men in age groups under 45 and nearly 60 per cent of current trainees are women.",
                        },
                        {
                            "headline": "Digital-first service transformation",
                            "category": "digital_transformation",
                            "snippet": "Digital-first is now the baseline and digital capability is foundational to trust and efficiency.",
                        },
                    ],
                }
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
    )

    headlines = [item["headline"] for item in payload["signals"]]

    assert "Membership base is changing" in headlines
    assert "Digital-first service transformation" in headlines
    assert not any(headline.startswith("Strategic theme:") for headline in headlines)


def test_mailbox_handoff_enriches_annual_report_note_content_and_performance_signals(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
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
        note_callback_url="",
        callback_secret="",
        callback_timeout=30,
        profile_import_url="",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-20T09:00:00+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [
                {"name": "Royal Australasian College of Physicians", "target_type": "organisation", "confidence": 0.95},
                {
                    "name": "Professor Jennifer Martin",
                    "canonical_name": "Professor Jennifer Martin",
                    "target_type": "person",
                    "current_employer": "Royal Australasian College of Physicians",
                    "current_role": "RACP President",
                    "confidence": 0.95,
                    "evidence": "Signed President's message in annual report",
                },
            ],
            "attachments": [],
            "summary": "RACP annual report with organisation-level highlights.",
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "annual_report",
                    "org_name": "Royal Australasian College of Physicians",
                    "strategic_summary": "Annual report highlights governance reform, member service uplift, and accreditation outcomes.",
                    "themes": ["Better value for members"],
                    "initiatives": ["Training Management Platform"],
                    "strategic_signals": [
                        {
                            "headline": "Digital-first service transformation",
                            "category": "digital_transformation",
                            "snippet": "The new Training Management Platform went live in December 2024.",
                        }
                    ],
                    "key_stakeholders": [
                        {
                            "name": "Professor Jennifer Martin",
                            "current_role": "RACP President",
                            "current_employer": "Royal Australasian College of Physicians",
                        },
                        {
                            "name": "Steffen Faurby",
                            "current_role": "Chief Executive Officer",
                            "current_employer": "Royal Australasian College of Physicians",
                        },
                    ],
                    "performance_indicators": [
                        {
                            "label": "Membership scale",
                            "category": "member_scale",
                            "value": "32,347 members",
                            "evidence": "32,347 All Members Annual Growth ▲ 1,184 (3.8%).",
                        },
                        {
                            "label": "Operating result",
                            "category": "financial_performance",
                            "value": "Deficit $0.1m",
                            "evidence": "The deficit of $0.1m ($3.7m 2023) reflects investment in education renewal and technology.",
                        },
                    ],
                }
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
    )

    headlines = [item["headline"] for item in payload["signals"]]
    note_content = payload["note"]["content"]

    assert payload["note"]["source_type"] == "annual_report"
    assert payload["note"]["title"] == "Royal Australasian College of Physicians Annual Report"
    assert note_content.startswith("## Summary")
    assert "# Intel Extraction Result" not in note_content
    assert "## Entities" not in note_content
    assert "Board Attendance" not in note_content
    assert "Governance Committee" not in note_content
    assert "Annual report highlights governance reform" not in note_content
    assert "## Key Stakeholders" in note_content
    assert "Professor Jennifer Martin | President | Royal Australasian College of Physicians" in note_content
    assert "Steffen Faurby | Chief Executive Officer | Royal Australasian College of Physicians" in note_content
    assert "## Performance Snapshot" in note_content
    assert "Membership scale: 32,347 members" in note_content
    assert "Operating result" in headlines


def test_mailbox_handoff_prefers_subject_org_hint_over_compatible_formal_org_name_for_annual_report(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
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
        note_callback_url="",
        callback_secret="",
        callback_timeout=30,
        profile_import_url="",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "entity: Escient | Barwon Water annual report",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-23T09:00:00+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [
                {"name": "Barwon Region Water Corporation", "canonical_name": "Barwon Region Water Corporation", "target_type": "organisation", "confidence": 0.95},
                {
                    "name": "Shaun Cumming",
                    "canonical_name": "Shaun Cumming",
                    "target_type": "person",
                    "current_employer": "Barwon Region Water Corporation",
                    "current_role": "Managing Director",
                    "confidence": 0.95,
                    "evidence": "Leadership listing in annual report",
                },
            ],
            "organisations": [{"canonical_name": "Barwon Region Water Corporation"}],
            "attachments": [{"filename": "barwon-water-annual-report.pdf", "status": "processed", "excerpt": "Barwon Water annual report"}],
            "summary": "Barwon Water annual report.",
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "annual_report",
                    "org_name": "Barwon Region Water Corporation",
                    "strategic_summary": "Annual report for Barwon Water.",
                    "themes": [],
                    "initiatives": [],
                    "strategic_signals": [],
                    "key_stakeholders": [],
                    "performance_indicators": [],
                }
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
        routing={"subject_org_hint": "Barwon Water", "clean_subject": "Barwon Water annual report"},
    )

    assert payload["primary_entity"]["name"] == "Barwon Water"


def test_mailbox_handoff_omits_low_signal_kpi_focus_lines_from_annual_report_notes(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
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
        note_callback_url="",
        callback_secret="",
        callback_timeout=30,
        profile_import_url="",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "Annual report",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-20T09:00:00+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "Royal Australasian College of Physicians", "target_type": "organisation", "confidence": 0.95}],
            "attachments": [],
            "summary": "RACP annual report with organisation-level highlights.",
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "annual_report",
                    "org_name": "Royal Australasian College of Physicians",
                    "strategic_summary": "Annual report highlights governance reform, member service uplift, and accreditation outcomes.",
                    "strategic_signals": [],
                    "key_stakeholders": [],
                    "performance_indicators": [],
                    "kpi_focuses": ["3. STREAMLINE", "governance", "Implement Education", "Board and College Council"],
                }
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
    )

    assert "## KPI Focus Areas" not in payload["note"]["content"]


def test_intel_note_processor_filters_weak_and_credit_entities_for_strategic_docs(tmp_path):
    def _extractor(_payload):
        return (
            {
                "status": "extracted",
                "summary": "Strategic direction document from Royal Australasian College of Physicians.",
                "entity_count": 8,
                "entities": [
                    {
                        "canonical_name": "Professor Jennifer Martin",
                        "name": "Professor Jennifer Martin",
                        "target_type": "person",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "President and Chair of the Board",
                        "confidence": 0.95,
                        "evidence": "President and Chair foreword signed by Professor Jennifer Martin, President and Chair of the Board, The Royal Australasian College of Physicians",
                    },
                    {
                        "canonical_name": "Steffen Faurby",
                        "name": "Steffen Faurby",
                        "target_type": "person",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "Chief Executive Officer",
                        "confidence": 0.95,
                        "evidence": "CEO comment signed by Steffen Faurby, Chief Executive Officer, The Royal Australasian College of Physicians",
                    },
                    {
                        "canonical_name": "Dr Danny deLore",
                        "name": "Dr Danny deLore",
                        "target_type": "person",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "",
                        "confidence": 0.75,
                        "evidence": "Named in document alongside vision statement, context suggests involvement with RACP",
                    },
                    {
                        "canonical_name": "Riki Salam",
                        "name": "Riki Salam",
                        "target_type": "person",
                        "current_employer": "We are 27",
                        "current_role": "",
                        "confidence": 0.8,
                        "evidence": "Developed 'Healing Place' artwork for RACP as part of We are 27",
                    },
                    {
                        "canonical_name": "Royal Australasian College of Physicians",
                        "name": "Royal Australasian College of Physicians",
                        "target_type": "organisation",
                        "confidence": 0.95,
                        "evidence": "Primary subject of 2026-2030 strategic direction document",
                    },
                    {
                        "canonical_name": "We are 27",
                        "name": "We are 27",
                        "target_type": "organisation",
                        "confidence": 0.8,
                        "evidence": "Created artwork 'Healing Place' for RACP Indigenous cultural elements",
                    },
                ],
                "people": [
                    {
                        "canonical_name": "Professor Jennifer Martin",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "President and Chair of the Board",
                        "evidence": "President and Chair foreword signed by Professor Jennifer Martin, President and Chair of the Board, The Royal Australasian College of Physicians",
                    },
                    {
                        "canonical_name": "Dr Danny deLore",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "",
                        "evidence": "Named in document alongside vision statement, context suggests involvement with RACP",
                    },
                    {
                        "canonical_name": "Riki Salam",
                        "current_employer": "We are 27",
                        "current_role": "",
                        "evidence": "Developed 'Healing Place' artwork for RACP as part of We are 27",
                    },
                ],
                "organisations": [
                    {
                        "canonical_name": "Royal Australasian College of Physicians",
                        "evidence": "Primary subject of 2026-2030 strategic direction document",
                    },
                    {
                        "canonical_name": "We are 27",
                        "evidence": "Created artwork 'Healing Place' for RACP Indigenous cultural elements",
                    },
                ],
                "attachments": [
                    {
                        "filename": "2026-2030-strategic-direction-document_compressed.pdf",
                        "status": "processed",
                        "excerpt": "Royal Australasian College of Physicians Strategic Direction 2026 to 2030",
                    }
                ],
            },
            tmp_path / "extract.md",
        )

    processor = IntelNoteProcessor(_extractor)
    output_data, _output_file, analysis = processor.process(
        {
            "subject": "Fwd: RACP strategy",
            "raw_text": "Please see attached strategic direction document.",
        },
        "document_analysis",
    )

    names = [item["canonical_name"] for item in output_data["entities"]]

    assert analysis["strategic_doc"]["org_name"] == "Royal Australasian College of Physicians"
    assert names == [
        "Professor Jennifer Martin",
        "Steffen Faurby",
        "Royal Australasian College of Physicians",
    ]
    assert output_data["entity_count"] == 3
    assert "Strategic planning document from Royal Australasian College of Physicians" in output_data["summary"]


def test_intel_note_processor_restores_leadership_signatories_from_strategic_doc(tmp_path):
    def _extractor(_payload):
        return (
            {
                "status": "extracted",
                "summary": "Strategic planning document from the Royal Australasian College of Physicians.",
                "entity_count": 1,
                "entities": [
                    {
                        "canonical_name": "Royal Australasian College of Physicians",
                        "name": "Royal Australasian College of Physicians",
                        "target_type": "organisation",
                        "confidence": 0.95,
                        "evidence": "Primary subject of 2026-2030 strategic direction document",
                    }
                ],
                "people": [],
                "organisations": [
                    {
                        "canonical_name": "Royal Australasian College of Physicians",
                        "evidence": "Primary subject of 2026-2030 strategic direction document",
                    }
                ],
                "attachments": [
                    {
                        "filename": "2026-2030-strategic-direction-document_compressed.pdf",
                        "status": "processed",
                        "stored_path": "/home/longboardfella/cortex_suite/2026-2030-strategic-direction-document_compressed.pdf",
                    }
                ],
            },
            tmp_path / "extract.md",
        )

    processor = IntelNoteProcessor(_extractor)
    output_data, _output_file, _analysis = processor.process(
        {
            "subject": "Fwd: RACP strategy",
            "raw_text": "Please see attached strategic direction document.",
        },
        "document_analysis",
    )

    names = [item["canonical_name"] for item in output_data["entities"]]

    assert names == [
        "Royal Australasian College of Physicians",
        "Professor Jennifer Martin",
        "Steffen Faurby",
    ]


def test_intel_note_processor_filters_noisy_annual_report_entities_to_curated_stakeholders(tmp_path):
    def _extractor(_payload):
        return (
            {
                "status": "extracted",
                "summary": "RACP annual report.",
                "entity_count": 16,
                "entities": [
                    {
                        "canonical_name": "Professor Jennifer Martin",
                        "name": "Professor Jennifer Martin",
                        "target_type": "person",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "President",
                        "confidence": 0.95,
                        "evidence": "Signed President's message",
                    },
                    {
                        "canonical_name": "Steffen Faurby",
                        "name": "Steffen Faurby",
                        "target_type": "person",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "Chief Executive Officer",
                        "confidence": 0.95,
                        "evidence": "Signed CEO's message",
                    },
                    {
                        "canonical_name": "Board Attendance",
                        "name": "Board Attendance",
                        "target_type": "person",
                        "current_employer": "",
                        "current_role": "Current Directors",
                        "confidence": 0.72,
                        "evidence": "Org chart OCR pair",
                    },
                    {
                        "canonical_name": "7% Education CPD",
                        "name": "7% Education CPD",
                        "target_type": "person",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "*Governance, OPCEO, Strategy &",
                        "confidence": 0.92,
                        "evidence": "Strategic document signatory block",
                    },
                    {
                        "canonical_name": "Governance Committee",
                        "name": "Governance Committee",
                        "target_type": "person",
                        "current_employer": "Royal Australasian College of Physicians",
                        "current_role": "Leadership Capabilities",
                        "confidence": 0.92,
                        "evidence": "Strategic document signatory block",
                    },
                    {
                        "canonical_name": "Royal Australasian College of Physicians",
                        "name": "Royal Australasian College of Physicians",
                        "target_type": "organisation",
                        "confidence": 0.95,
                        "evidence": "Source document is 2024 Annual Report",
                    },
                ],
                "people": [],
                "organisations": [
                    {
                        "canonical_name": "Royal Australasian College of Physicians",
                        "evidence": "Source document is 2024 Annual Report",
                    }
                ],
                "attachments": [
                    {
                        "filename": "2024-racp-annual-report.pdf",
                        "status": "processed",
                        "excerpt": (
                            "Royal Australasian College of Physicians Annual Report 2024\n"
                            "Professor Jennifer Martin\n"
                            "RACP President\n"
                            "Royal Australasian College of Physicians\n"
                            "Steffen Faurby\n"
                            "Chief Executive Officer\n"
                            "Royal Australasian College of Physicians"
                        ),
                    }
                ],
            },
            tmp_path / "extract.md",
        )

    processor = IntelNoteProcessor(_extractor)
    output_data, _output_file, analysis = processor.process(
        {
            "subject": "RACP annual report",
            "raw_text": "",
        },
        "document_analysis",
    )

    names = [item["canonical_name"] for item in output_data["entities"]]

    assert analysis["strategic_doc"]["doc_type"] == "annual_report"
    assert names == [
        "Professor Jennifer Martin",
        "Steffen Faurby",
        "Royal Australasian College of Physicians",
    ]


def test_intel_note_processor_annual_report_strips_bullets_and_drops_management_headings(tmp_path):
    def _extractor(_payload):
        return (
            {
                "status": "extracted",
                "summary": "Barwon Water annual report.",
                "entity_count": 8,
                "entities": [
                    {
                        "canonical_name": "• Des Powell",
                        "name": "• Des Powell",
                        "target_type": "person",
                        "current_employer": "Barwon Region Water Corporation",
                        "current_role": "Chair",
                        "confidence": 0.88,
                        "evidence": "Annual report leadership listing",
                    },
                    {
                        "canonical_name": "• Risk Management",
                        "name": "• Risk Management",
                        "target_type": "person",
                        "current_employer": "",
                        "current_role": "",
                        "confidence": 0.55,
                        "evidence": "Bullet heading from annual report",
                    },
                    {
                        "canonical_name": "Integrated Water Management",
                        "name": "Integrated Water Management",
                        "target_type": "person",
                        "current_employer": "",
                        "current_role": "",
                        "confidence": 0.55,
                        "evidence": "Heading from annual report",
                    },
                    {
                        "canonical_name": "Barwon Region Water Corporation",
                        "name": "Barwon Region Water Corporation",
                        "target_type": "organisation",
                        "confidence": 0.94,
                        "evidence": "Source document is 2025 Annual Report",
                    },
                ],
                "people": [],
                "organisations": [
                    {
                        "canonical_name": "Barwon Region Water Corporation",
                        "evidence": "Source document is 2025 Annual Report",
                    }
                ],
                "attachments": [
                    {
                        "filename": "barwon-water-annual-report-2025.pdf",
                        "status": "processed",
                        "excerpt": (
                            "Barwon Water Group Annual Report 2025\n"
                            "Des Powell\n"
                            "Chair\n"
                            "Shaun Cumming\n"
                            "Managing Director\n"
                            "Barwon Region Water Corporation"
                        ),
                    }
                ],
            },
            tmp_path / "extract.md",
        )

    processor = IntelNoteProcessor(_extractor)
    output_data, _output_file, analysis = processor.process(
        {
            "subject": "Barwon Water annual report",
            "raw_text": "",
        },
        "document_analysis",
    )

    names = [item["canonical_name"] for item in output_data["entities"]]

    assert analysis["strategic_doc"]["doc_type"] == "annual_report"
    assert "Des Powell" in names
    assert "• Des Powell" not in names
    assert "• Risk Management" not in names
    assert "Integrated Water Management" not in names
    assert "Barwon Region Water Corporation" in names


def test_mailbox_handoff_strips_re_fwd_subject_prefixes(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
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
        note_callback_url="",
        callback_secret="",
        callback_timeout=30,
        profile_import_url="",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "Fwd: Re: AIDH strategy",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-20T09:00:00+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "Australian Institute of Digital Health", "target_type": "organisation", "confidence": 0.95}],
            "attachments": [],
            "summary": "AIDH strategy note.",
            "target_update_suggestions": [],
            "processing_meta": {"strategic_doc": {"doc_type": "strategic_plan", "org_name": "Australian Institute of Digital Health", "themes": [], "initiatives": []}},
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
    )

    assert payload["note"]["title"] == "AIDH strategy"
    assert payload["signals"][0]["headline"] == "AIDH strategy"


def test_mailbox_poller_routes_csv_profile_imports(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")

    class _FailSignalStore:
        def ingest_signal(self, payload):
            raise AssertionError(f"signal ingest should not run for CSV imports: {payload}")

    class _FakeCsvImporter:
        def process_message(self, message, persisted, org_name):
            assert message["from_email"] == "paul@example.com"
            assert org_name == "Longboardfella"
            assert persisted["attachments"][0]["filename"] == "profiles.csv"
            return {
                "filename": "profiles.csv",
                "row_count": 1,
                "dry_run": True,
                "created": 1,
                "updated": 0,
                "skipped": 0,
                "errors": [],
                "api_result": {"ok": True, "created": 1, "updated": 0, "skipped": 0, "errors": []},
                "reply_subject": "Re: PROFILES DRY RUN - Preview (not saved)",
                "reply_body": "Dry run complete",
            }

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
        note_callback_url="",
        callback_secret="",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: (_ for _ in ()).throw(AssertionError(f"extractor should not run: {payload}")),
        csv_importer=_FakeCsvImporter(),
        reply_client=_FakeReplyClient(),
        imap_factory=_FakeCSVIMAP,
        signal_store=_FailSignalStore(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    messages = store.list_messages()
    assert len(messages) == 1
    assert messages[0]["status"] == "processed"
    payload = json.loads(Path(messages[0]["result_path"]).read_text(encoding="utf-8"))
    assert payload["result_type"] == "csv_profile_import_result"
    assert payload["csv_import"]["dry_run"] is True
    assert payload["reply"]["delivery"]["status"] == "sent"


def test_mailbox_poller_keeps_intel_subject_on_legacy_extract_flow(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")

    def _extractor(_payload):
        return (
            {
                "status": "extracted",
                "entity_count": 1,
                "people": [{"canonical_name": "Carolyn Bell", "current_employer": "Silverchain"}],
                "organisations": [{"canonical_name": "Silverchain"}],
                "emails": [],
                "target_update_suggestions": [],
                "warnings": [],
            },
            None,
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
        note_callback_url="",
        callback_secret="",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=_extractor,
        imap_factory=_FakeIntelIMAP,
        signal_store=_FakeSignalStore(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    outbox_files = list((tmp_path / "intel_mailbox" / "outbox").glob("*.json"))
    assert len(outbox_files) == 1
    payload = json.loads(outbox_files[0].read_text(encoding="utf-8"))
    assert payload["result_type"] == "intel_extract_result"
    assert "website_payload" not in payload


def test_mailbox_poller_reposts_note_payloads_even_when_local_match_exists(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()
    existing_bytes = _sample_message_bytes().replace(b"<msg-1@example.com>", b"<msg-existing@example.com>")
    parsed_existing = parse_email_bytes(existing_bytes)
    persisted = store.persist_message(parsed_existing, existing_bytes, parsed_existing["attachments"])
    existing_payload = {
        "result_type": "intel_extract_result",
        "org_name": "Longboardfella",
        "trace_id": "trace-existing",
        "website_payload": {
            "action": "ingest_intel_note",
            "org_name": "Longboardfella",
            "note": {
                "note_date": "2026-03-12",
                "content": "# Extract",
                "original_text": "Carolyn Bell has started a new program role at Silverchain.\nContact: cbell@example.com\nSee https://example.com/carolyn",
            },
            "primary_entity": {
                "name": "Carolyn Bell",
                "target_type": "person",
            },
        },
    }
    store.record_processed(
        persisted["message_key"],
        "trace-existing",
        existing_payload,
        {"status": "posted", "response": {"intel_id": "note_existing"}},
    )

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
                "attachments": [],
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
        callback_url="https://example.com/admin/queue_worker_api.php?action=import_cortex_extract",
        note_callback_url="https://example.com/lab/market_radar_api.php?action=ingest_intel_note",
        callback_secret="secret",
        callback_timeout=30,
        profile_import_url="https://example.com/lab/market_radar_api.php?action=bulk_import_profiles",
        profile_import_timeout=30,
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_username="intel.longboardfella@gmail.com",
        smtp_password="secret",
        smtp_use_ssl=True,
        reply_from="intel.longboardfella@gmail.com",
        mark_seen_on_success=False,
    )

    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=_extractor,
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
        result_client=result_client,
    )

    summary = poller.poll_once()
    assert summary["processed"] == 1
    assert summary["results"][0]["delivery"]["status"] == "posted"
    assert result_client.calls
    posted_record = next(item for item in store.list_messages() if item.get("trace_id") == summary["results"][0]["trace_id"])
    assert posted_record["delivery"]["status"] == "posted"
