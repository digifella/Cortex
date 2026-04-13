import json
from email.message import EmailMessage
from pathlib import Path

from cortex_engine.document_registry import build_content_fingerprint, build_document_meta, derive_period_label
from cortex_engine.intel_deduplicator import find_duplicate_note
from cortex_engine.intel_mailbox import (
    IntelMailboxConfig,
    IntelMailboxPoller,
    IntelMailboxStore,
    _compact_strategy_snippet,
    _clean_note_signal_snippet,
    _subject_org_hint,
    _should_keep_rendered_strategic_signal,
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


def _cortex_processed_reply_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Re: Re: NemoClaw Work on my ethics papers"
    msg["From"] = "intel.longboardfella@gmail.com"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Wed, 08 Apr 2026 19:56:56 +1000"
    msg["Message-ID"] = "<msg-cortex-reply@example.com>"
    msg.set_content(
        "Cortex processed your submission for Longboardfella.\n\n"
        "Depth: default\n"
        "Primary entity: Longboardfella\n\n"
        "Title: NemoClaw Work on my ethics papers\n\n"
        "# Intel Extraction Result\n\n"
        "Calendar entry created by NemoClaw for work on ethics papers.\n"
    )
    return msg.as_bytes()


def _forwarding_confirmation_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "(Project 143 Forwarding confirmation - Receive mail from paul@project-143.com"
    msg["From"] = "Paul Cooper <paul@project-143.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Wed, 08 Apr 2026 19:56:56 +1000"
    msg["Message-ID"] = "<msg-forwarding-confirmation@example.com>"
    msg.set_content(
        "paul@project-143.com has requested to automatically forward mail to your email\n"
        "address intel.longboardfella@gmail.com.\n\n"
        "To allow paul@project-143.com to automatically forward mail to your address,\n"
        "please click the link below to confirm the request.\n\n"
        "paul@project-143.com cannot automatically forward messages to your email address\n"
        "unless you confirm the request by clicking the link above.\n"
    )
    return msg.as_bytes()


def _youtube_summariser_message_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Fwd: YouTube Summariser | Weekly queue digest"
    msg["From"] = "Paul Cooper <paul@example.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Thu, 09 Apr 2026 09:15:00 +1000"
    msg["Message-ID"] = "<msg-youtube-summariser@example.com>"
    msg.set_content("Nemoclaw should handle this, not Market Radar.")
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
    assert parsed["to_email"] == "intel.longboardfella@gmail.com"
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
    def __init__(self):
        self.document_records = {}

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

    def get_document_record(self, org_name, canonical_doc_key):
        return dict(self.document_records.get((org_name.lower(), str(canonical_doc_key or "").strip()), {})) or None

    def classify_document_meta(self, org_name, document_meta):
        meta = dict(document_meta or {})
        existing = self.get_document_record(org_name, meta.get("canonical_doc_key"))
        if existing:
            meta["status"] = "known_same" if str(existing.get("content_fingerprint") or "") == str(meta.get("content_fingerprint") or "") else "changed_document"
            meta["ingest_recommendation"] = "skip" if meta["status"] == "known_same" and str(meta.get("ingest_policy") or "") == "strict" else "ingest"
            meta["existing_record"] = existing
        else:
            meta["status"] = meta.get("status") or "new_document"
            meta["ingest_recommendation"] = "ingest"
            meta["existing_record"] = {}
        return meta

    def register_document_meta(self, org_name, document_meta, latest_trace_id="", latest_intel_id=""):
        meta = dict(document_meta or {})
        meta["org_name"] = org_name
        meta["latest_trace_id"] = latest_trace_id
        meta["latest_intel_id"] = latest_intel_id
        self.document_records[(org_name.lower(), str(meta.get("canonical_doc_key") or "").strip())] = meta
        return meta


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


class _FakeCortexProcessedReplyIMAP(_FakeIMAP):
    instances = []

    def __init__(self, *args):
        super().__init__(*args)
        self.__class__.instances.append(self)

    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _cortex_processed_reply_message_bytes())]


class _FakeForwardingConfirmationIMAP(_FakeIMAP):
    instances = []

    def __init__(self, *args):
        super().__init__(*args)
        self.__class__.instances.append(self)

    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _forwarding_confirmation_message_bytes())]


class _FakeYouTubeSummariserIMAP(_FakeIMAP):
    instances = []

    def __init__(self, *args):
        super().__init__(*args)
        self.__class__.instances.append(self)

    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _youtube_summariser_message_bytes())]


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


def test_mailbox_poller_allows_trusted_self_relay_and_normalizes_submitter(tmp_path):
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
        allowed_senders=("someone.else@example.com",),
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
    payload = result_client.calls[0]["delivery_payload"]
    assert payload["note"]["submitted_by"] == "paul@longboardfella.com.au"


def test_mailbox_poller_suppresses_cortex_processed_replies_and_marks_seen(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    _FakeCortexProcessedReplyIMAP.instances = []

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
        mark_seen_on_success=True,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: (_ for _ in ()).throw(AssertionError(f"extractor should not run: {payload}")),
        imap_factory=_FakeCortexProcessedReplyIMAP,
        signal_store=_FakeSignalStore(),
        reply_client=_FakeReplyClient(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 0
    assert summary["skipped"] == 1
    assert summary["failures"] == 0
    assert store.list_messages() == []
    assert _FakeCortexProcessedReplyIMAP.instances[0].stored == [(b"1", "+FLAGS", "\\Seen")]


def test_mailbox_poller_suppresses_forwarding_confirmation_and_marks_seen(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    _FakeForwardingConfirmationIMAP.instances = []

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
        mark_seen_on_success=True,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: (_ for _ in ()).throw(AssertionError(f"extractor should not run: {payload}")),
        imap_factory=_FakeForwardingConfirmationIMAP,
        signal_store=_FakeSignalStore(),
        reply_client=_FakeReplyClient(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 0
    assert summary["skipped"] == 1
    assert summary["failures"] == 0
    assert store.list_messages() == []
    assert _FakeForwardingConfirmationIMAP.instances[0].stored == [(b"1", "+FLAGS", "\\Seen")]


def test_mailbox_poller_suppresses_youtube_summariser_subject_and_marks_seen(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    _FakeYouTubeSummariserIMAP.instances = []

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
        mark_seen_on_success=True,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: (_ for _ in ()).throw(AssertionError(f"extractor should not run: {payload}")),
        imap_factory=_FakeYouTubeSummariserIMAP,
        signal_store=_FakeSignalStore(),
        reply_client=_FakeReplyClient(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 0
    assert summary["skipped"] == 1
    assert summary["failures"] == 0
    assert store.list_messages() == []
    assert _FakeYouTubeSummariserIMAP.instances[0].stored == [(b"1", "+FLAGS", "\\Seen")]


def _nemoclaw_vault_email_bytes() -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Vault: Patterns in pi in Contact"
    msg["From"] = "Paul Cooper <paul@example.com>"
    msg["To"] = "intel.longboardfella@gmail.com"
    msg["Date"] = "Sun, 13 Apr 2026 14:00:00 +1000"
    msg.set_content("Some vault content about patterns in pi.")
    return msg.as_bytes()


class _FakeNemoClawVaultIMAP(_FakeIMAP):
    instances = []

    def __init__(self, *args):
        super().__init__(*args)
        self.__class__.instances.append(self)

    def fetch(self, _imap_id, _query):
        return "OK", [(b"1 (RFC822 {123})", _nemoclaw_vault_email_bytes())]


def test_mailbox_poller_suppresses_nemoclaw_vault_prefix_and_marks_seen(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    _FakeNemoClawVaultIMAP.instances = []

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
        mark_seen_on_success=True,
    )
    poller = IntelMailboxPoller(
        cfg,
        store=store,
        extractor=lambda payload: (_ for _ in ()).throw(AssertionError(f"extractor should not run: {payload}")),
        imap_factory=_FakeNemoClawVaultIMAP,
        signal_store=_FakeSignalStore(),
        reply_client=_FakeReplyClient(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 0
    assert summary["skipped"] == 1
    assert summary["failures"] == 0
    assert store.list_messages() == []
    assert _FakeNemoClawVaultIMAP.instances[0].stored == [(b"1", "+FLAGS", "\\Seen")]


def test_mailbox_reply_to_trusted_self_relay_uses_effective_submitter(tmp_path):
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
        reply_client=_FakeReplyClient(),
    )

    delivery = poller._send_reply(
        {
            "from_email": "intel.longboardfella@gmail.com",
            "to_email": "intel.longboardfella@gmail.com",
            "message_id": "<msg-self@example.com>",
        },
        "Re: self relay",
        "done",
    )

    assert delivery["status"] == "sent"
    assert delivery["to_email"] == "paul@longboardfella.com.au"


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


def test_mailbox_routing_parses_depth_override_and_strips_it_from_subject(tmp_path):
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
        signal_store=_RoutingSignalStore(),
    )

    routing = poller._resolve_message_routing(
        {
            "subject": "Fw: entity: Escient | depth:detailed | Barwon Water annual report",
            "from_email": "paul@longboardfella.com.au",
        },
        {
            "attachments": [
                {
                    "filename": "barwon-water-annual-report.pdf",
                    "kind": "document",
                }
            ]
        },
    )

    assert routing["effective_org_name"] == "Escient"
    assert routing["status"] == "matched_override"
    assert routing["clean_subject"] == "Barwon Water annual report"
    assert routing["subject_org_hint"] == "Barwon Water"
    assert routing["extraction_depth"] == "detailed"


def test_mailbox_routing_parses_force_override_and_strips_it_from_subject(tmp_path):
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
        signal_store=_RoutingSignalStore(),
    )

    routing = poller._resolve_message_routing(
        {
            "subject": "entity: Escient | depth:detailed | force:yes | Yarra Valley Water strategic plan",
            "from_email": "paul@longboardfella.com.au",
        },
        {"attachments": [{"filename": "strategy.pdf", "kind": "document"}]},
    )

    assert routing["effective_org_name"] == "Escient"
    assert routing["clean_subject"] == "Yarra Valley Water strategic plan"
    assert routing["force_reingest"] is True


def test_mailbox_routing_strips_privacy_markers_from_clean_subject_and_hint(tmp_path):
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
        signal_store=_RoutingSignalStore(),
    )

    routing = poller._resolve_message_routing(
        {
            "subject": "Private > entity: Escient > South East Water strategic plan",
            "from_email": "paul@longboardfella.com.au",
        },
        {"attachments": [{"filename": "strategy.pdf", "kind": "document"}]},
    )

    assert routing["effective_org_name"] == "Escient"
    assert routing["clean_subject"] == "South East Water strategic plan"
    assert routing["subject_org_hint"] == "South East Water"


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
    assert delivery_payload["note"]["source_type"] == "org_chart"
    assert delivery_payload["note"]["content"].startswith("## Summary")
    assert "# Intel Extraction Result" not in delivery_payload["note"]["content"]
    assert "Longboardfella Consulting" not in delivery_payload["note"]["content"]
    assert "## Leadership Team" in delivery_payload["note"]["content"]
    assert "Shaun Cumming | Managing Director | Barwon Water" in delivery_payload["note"]["content"]


def test_mailbox_org_chart_subject_hint_beats_sender_org_when_extractor_is_noisy(tmp_path):
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
            "subject": "org chart for Barwon Water",
            "from_name": "Paul Cooper",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-24T00:41:08+00:00",
            "raw_text": "Dr. Paul Cooper\nDirector\nLongboardfella Consulting Pty Ltd",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [
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
                    "evidence": "Org chart subject and image content",
                },
            ],
            "people": [
                {
                    "canonical_name": "Shaun Cumming",
                    "name": "Shaun Cumming",
                    "current_employer": "Barwon Water",
                    "current_role": "Managing Director",
                }
            ],
            "organisations": [
                {"canonical_name": "Longboardfella Consulting Pty Ltd"},
                {"canonical_name": "Barwon Water"},
            ],
            "attachments": [{"filename": "barwon-water-org-chart.png", "status": "processed", "excerpt": "Barwon Water org chart"}],
            "summary": "Email contains org chart for Barwon Water. Sender Dr. Paul Cooper is Director of Longboardfella Consulting.",
            "target_update_suggestions": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\nNoisy org chart extractor summary",
        message_kind="org_chart",
        routing={"subject_org_hint": "Barwon Water", "clean_subject": "org chart for Barwon Water", "has_org_chart_image_attachment": True},
    )

    assert payload["primary_entity"]["name"] == "Barwon Water"
    assert payload["note"]["title"] == "Barwon Water org chart"
    assert "Longboardfella Consulting Pty Ltd" not in payload["note"]["content"]


def test_mailbox_org_chart_note_uses_entities_when_people_bucket_missing(tmp_path):
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
            "subject": "org chart for Barwon Water",
            "from_name": "Paul Cooper",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-24T00:51:10+00:00",
            "raw_text": "Dr. Paul Cooper\nDirector\nLongboardfella Consulting Pty Ltd",
        },
        persisted={"attachments": []},
        output_data={
            "people": None,
            "entities": [
                {
                    "canonical_name": "Dr. Paul Cooper",
                    "name": "Dr. Paul Cooper",
                    "target_type": "person",
                    "current_employer": "Longboardfella Consulting Pty Ltd",
                    "current_role": "Director",
                    "evidence": "Contact details and credentials provided in source document",
                },
                {
                    "canonical_name": "Shaun Cumming",
                    "name": "Shaun Cumming",
                    "target_type": "person",
                    "current_employer": "Barwon Water",
                    "current_role": "Managing Director",
                    "evidence": "Listed as Managing Director in org chart dated July 2025",
                },
                {
                    "canonical_name": "Anna Murray",
                    "name": "Anna Murray",
                    "target_type": "person",
                    "current_employer": "Barwon Water",
                    "current_role": "General Manager, Operations",
                    "evidence": "Listed as General Manager in org chart",
                },
                {"canonical_name": "Barwon Water", "name": "Barwon Water", "target_type": "organisation"},
                {"canonical_name": "Longboardfella Consulting Pty Ltd", "name": "Longboardfella Consulting Pty Ltd", "target_type": "organisation"},
            ],
            "organisations": [{"canonical_name": "Barwon Water"}],
            "attachments": [{"filename": "barwon-water-org-chart.png", "status": "processed", "excerpt": "Barwon Water org chart"}],
            "summary": "Noisy org chart extraction summary",
            "target_update_suggestions": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\nNoisy org chart extractor summary",
        message_kind="org_chart",
        routing={"subject_org_hint": "Barwon Water", "clean_subject": "org chart for Barwon Water", "has_org_chart_image_attachment": True},
    )

    assert "## Leadership Team" in payload["note"]["content"]
    assert "Shaun Cumming | Managing Director | Barwon Water" in payload["note"]["content"]
    assert "Anna Murray | General Manager, Operations | Barwon Water" in payload["note"]["content"]
    assert "Dr. Paul Cooper" not in payload["note"]["content"]


def test_mailbox_org_chart_note_falls_back_to_functional_structure_when_no_people_found(tmp_path):
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
            "subject": "org chart for Greater Western Water",
            "from_name": "Paul Cooper",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-24T06:12:00+00:00",
            "raw_text": "Forwarding functional org chart.",
        },
        persisted={"attachments": []},
        output_data={
            "people": [],
            "entities": [
                {"canonical_name": "Greater Western Water", "name": "Greater Western Water", "target_type": "organisation"},
            ],
            "organisations": [{"canonical_name": "Greater Western Water"}],
            "attachments": [{"filename": "greater-western-water-org-chart.png", "status": "processed", "excerpt": "Functional org chart"}],
            "summary": "Functional organisational chart.",
            "target_update_suggestions": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="\n".join(
            [
                "Greater Western Water organisational chart",
                "Office of the CEO",
                "General Manager, Operations",
                "People and Culture",
                "Customer and Community",
                "Digital and Technology",
                "Corporate Services",
            ]
        ),
        message_kind="org_chart",
        routing={"subject_org_hint": "Greater Western Water", "clean_subject": "org chart for Greater Western Water", "has_org_chart_image_attachment": True},
    )

    assert "## Functional Structure" in payload["note"]["content"]
    assert "## Leadership Team" not in payload["note"]["content"]
    assert "- Office of the CEO" in payload["note"]["content"]
    assert "- General Manager, Operations" in payload["note"]["content"]
    assert "- Digital and Technology" in payload["note"]["content"]


def test_mailbox_org_chart_functional_structure_strips_typed_labels_and_org_rows(tmp_path):
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
            "subject": "entity:escient org chart for GVW",
            "from_name": "Paul Cooper",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-24T06:45:00+00:00",
            "raw_text": "Forwarding functional org chart.",
        },
        persisted={"attachments": []},
        output_data={
            "people": [],
            "entities": [
                {"canonical_name": "Asset Monitoring", "name": "Asset Monitoring", "target_type": "person"},
                {"canonical_name": "Business Intelligence", "name": "Business Intelligence", "target_type": "person"},
                {"canonical_name": "Customer Accounts", "name": "Customer Accounts", "target_type": "person"},
                {"canonical_name": "Goulburn Valley Water", "name": "Goulburn Valley Water", "target_type": "organisation"},
            ],
            "organisations": [{"canonical_name": "Goulburn Valley Water"}],
            "attachments": [{"filename": "gvw-functional-org-chart.png", "status": "processed", "excerpt": "Functional org chart"}],
            "summary": "Functional organisational chart.",
            "target_update_suggestions": [],
            "warnings": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="\n".join(
            [
                "Asset Monitoring (person)",
                "Business Intelligence (person)",
                "Customer Accounts (person)",
                "Goulburn Valley Water (organisation)",
                "Technology Programs",
                "Corporate Services",
            ]
        ),
        message_kind="org_chart",
        routing={"subject_org_hint": "GVW", "clean_subject": "org chart for GVW", "has_org_chart_image_attachment": True},
    )

    assert "Asset Monitoring (person)" not in payload["note"]["content"]
    assert "Goulburn Valley Water (organisation)" not in payload["note"]["content"]
    assert "- Goulburn Valley Water" not in payload["note"]["content"]
    assert "- Asset Monitoring" in payload["note"]["content"]
    assert "- Business Intelligence" in payload["note"]["content"]
    assert "- Technology Programs" in payload["note"]["content"]


def test_subject_org_hint_strips_leading_for_from_document_subjects():
    assert _subject_org_hint("org chart for Barwon Water") == "Barwon Water"
    assert _subject_org_hint("strategy for Barwon Water") == "Barwon Water"


def test_mailbox_routing_strips_scope_org_prefix_from_document_subject_hint(tmp_path):
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
        signal_store=_RoutingSignalStore(),
    )

    routing = poller._resolve_message_routing(
        {
            "subject": "Escient Yarra Valley Water annual report",
            "from_email": "rebecca.campbell-burns@escient.com.au",
        },
        {
            "attachments": [
                {
                    "filename": "Yarra_Valley_Water_Annual_Report_2025.pdf",
                    "kind": "document",
                }
            ]
        },
    )

    assert routing["status"] == "matched_sender_domain"
    assert routing["effective_org_name"] == "Escient"
    assert routing["subject_org_hint"] == "Yarra Valley Water"


def test_mailbox_routing_entity_override_can_appear_inline_before_document_phrase(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()

    def _extractor(payload):
        assert payload["org_name"] == "Escient"
        assert payload["subject"] == "org chart for GVW"
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        return (
            {
                "status": "extracted",
                "entity_count": 1,
                "people": [],
                "organisations": [{"canonical_name": "Goulburn Valley Water"}],
                "entities": [
                    {"canonical_name": "Goulburn Valley Water", "name": "Goulburn Valley Water", "target_type": "organisation", "confidence": 0.95},
                ],
                "attachments": [{"filename": "gvw-functional-org-chart.png", "status": "processed", "excerpt": "Functional org chart"}],
                "summary": "Functional chart.",
                "target_update_suggestions": [],
                "warnings": [],
                "processing_meta": {},
            },
            out,
        )

    class _InlineEntityOverrideIMAP(_FakeIMAP):
        def fetch(self, _imap_id, _query):
            msg = EmailMessage()
            msg["Subject"] = "entity:escient org chart for GVW"
            msg["From"] = "Paul <paul@longboardfella.com.au>"
            msg["To"] = "intel.longboardfella@gmail.com"
            msg["Date"] = "Thu, 24 Mar 2026 17:28:09 +1100"
            msg["Message-ID"] = "<msg-inline-entity@example.com>"
            msg.set_content("Please see attached org chart.")
            msg.add_attachment(b"fake-image", maintype="image", subtype="png", filename="gvw-functional-org-chart.png")
            return "OK", [(b"1 (RFC822 {123})", msg.as_bytes())]

    poller = IntelMailboxPoller(
        IntelMailboxConfig(
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
            profile_import_url="",
            profile_import_timeout=30,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            smtp_username="intel.longboardfella@gmail.com",
            smtp_password="secret",
            smtp_use_ssl=True,
            reply_from="intel.longboardfella@gmail.com",
            mark_seen_on_success=False,
        ),
        store=store,
        extractor=_extractor,
        imap_factory=_InlineEntityOverrideIMAP,
        signal_store=_RoutingSignalStore(),
        result_client=result_client,
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    delivery_payload = result_client.calls[0]["delivery_payload"]
    assert delivery_payload["org_name"] == "Escient"
    assert delivery_payload["mailbox_routing"]["requested_org_name"] == "Escient"
    assert delivery_payload["mailbox_routing"]["subject_org_hint"] == "GVW"


def test_mailbox_routing_can_match_sender_domain_to_known_org_scope(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()

    def _extractor(payload):
        assert payload["org_name"] == "Escient"
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        return (
            {
                "status": "extracted",
                "entity_count": 2,
                "people": [],
                "organisations": [{"canonical_name": "Goulburn Valley Water"}],
                "entities": [
                    {"canonical_name": "Rebecca Campbell-Burns", "name": "Rebecca Campbell-Burns", "target_type": "person", "confidence": 0.95},
                    {"canonical_name": "Goulburn Valley Water", "name": "Goulburn Valley Water", "target_type": "organisation", "confidence": 0.95},
                ],
                "attachments": [],
                "summary": "GVW RFP intelligence.",
                "target_update_suggestions": [],
                "warnings": [],
                "processing_meta": {},
            },
            out,
        )

    class _EscientDomainIMAP(_FakeIMAP):
        def fetch(self, _imap_id, _query):
            msg = EmailMessage()
            msg["Subject"] = "Intel report"
            msg["From"] = "Rebecca Campbell-Burns <rebecca.campbell-burns@escient.com.au>"
            msg["To"] = "intel.longboardfella@gmail.com"
            msg["Date"] = "Thu, 24 Mar 2026 21:13:00 +1100"
            msg["Message-ID"] = "<msg-escient-domain@example.com>"
            msg.set_content("GVW is preparing an RFP for IT strategy services.")
            return "OK", [(b"1 (RFC822 {123})", msg.as_bytes())]

    poller = IntelMailboxPoller(
        IntelMailboxConfig(
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
            profile_import_url="",
            profile_import_timeout=30,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            smtp_username="intel.longboardfella@gmail.com",
            smtp_password="secret",
            smtp_use_ssl=True,
            reply_from="intel.longboardfella@gmail.com",
            mark_seen_on_success=False,
        ),
        store=store,
        extractor=_extractor,
        imap_factory=_EscientDomainIMAP,
        signal_store=_RoutingSignalStore(),
        result_client=result_client,
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    delivery_payload = result_client.calls[0]["delivery_payload"]
    assert delivery_payload["org_name"] == "Escient"
    assert delivery_payload["mailbox_routing"]["status"] == "matched_sender_domain"
    assert delivery_payload["mailbox_routing"]["sender_domain_org_name"] == "Escient"


def test_mailbox_org_chart_subject_acronym_prefers_full_organisation_name(tmp_path):
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
            "subject": "entity:escient org chart for GVW",
            "from_name": "Paul Cooper",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-24T07:05:00+00:00",
            "raw_text": "Forwarding functional org chart.",
        },
        persisted={"attachments": []},
        output_data={
            "people": [],
            "entities": [
                {"canonical_name": "Goulburn Valley Water", "name": "Goulburn Valley Water", "target_type": "organisation", "confidence": 0.95},
                {"canonical_name": "Asset Monitoring", "name": "Asset Monitoring", "target_type": "person"},
            ],
            "organisations": [{"canonical_name": "Goulburn Valley Water"}],
            "attachments": [{"filename": "gvw-functional-org-chart.png", "status": "processed", "excerpt": "Functional org chart"}],
            "summary": "Functional organisational chart.",
            "target_update_suggestions": [],
            "warnings": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="Asset Monitoring\nBusiness Intelligence",
        message_kind="org_chart",
        routing={"subject_org_hint": "GVW", "clean_subject": "org chart for GVW", "has_org_chart_image_attachment": True},
    )

    assert payload["primary_entity"]["name"] == "Goulburn Valley Water"
    assert payload["note"]["title"] == "Goulburn Valley Water org chart"


def test_mailbox_strategy_acronym_hint_uses_known_org_label_for_title_and_primary(tmp_path):
    class _KnownGVWSignalStore(_FakeSignalStore):
        def list_profiles(self, org_name=""):
            del org_name
            return [
                {
                    "org_name": "Escient",
                    "target_type": "organisation",
                    "canonical_name": "Goulburn Valley Water",
                    "aliases": ["GVW"],
                }
            ]

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
        signal_store=_KnownGVWSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "GVW",
            "from_name": "Paul Cooper",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-24T18:26:53+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [],
            "organisations": [],
            "attachments": [{"filename": "GVW_Strategy2035.pdf", "status": "processed", "excerpt": "GVW strategy"}],
            "summary": "",
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "strategic_plan",
                    "org_name": "GVW Strategy2035",
                    "strategic_summary": "Document appears to relate to GVW Strategy2035.",
                    "strategic_signals": [],
                    "key_stakeholders": [],
                    "performance_indicators": [],
                }
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
        routing={"effective_org_name": "Escient", "subject_org_hint": "GVW", "clean_subject": "GVW"},
    )

    assert payload["primary_entity"]["name"] == "Goulburn Valley Water"
    assert payload["note"]["title"] == "Goulburn Valley Water Strategic Plan 2035"
    assert payload["document_meta"]["target_org_name"] == "Goulburn Valley Water"


def test_mailbox_strategy_subject_full_name_overrides_documentish_org_label(tmp_path):
    class _KnownGVWSignalStore(_FakeSignalStore):
        def list_profiles(self, org_name=""):
            del org_name
            return [
                {
                    "org_name": "Escient",
                    "target_type": "organisation",
                    "canonical_name": "Goulburn Valley Water",
                    "aliases": ["GVW"],
                }
            ]

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
        signal_store=_KnownGVWSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "entity: Escient | Goulburn Valley Water strategic plan",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-25T06:10:55+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [],
            "organisations": [],
            "attachments": [{"filename": "GVW_Strategy2035.pdf", "status": "processed", "excerpt": "Strategy 2035"}],
            "summary": "",
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "strategic_plan",
                    "org_name": "GVW Strategy2035",
                    "strategic_summary": "Strategic planning document from GVW Strategy2035 outlining the current direction and operating priorities.",
                    "strategic_signals": [],
                    "key_stakeholders": [],
                    "performance_indicators": [],
                }
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
        routing={"effective_org_name": "Escient", "subject_org_hint": "Goulburn Valley Water", "clean_subject": "Goulburn Valley Water strategic plan"},
    )

    assert payload["primary_entity"]["name"] == "Goulburn Valley Water"
    assert payload["note"]["title"] == "Goulburn Valley Water strategic plan"
    assert "GVW Strategy2035" not in payload["note"]["content"]
    assert payload["document_meta"]["target_org_name"] == "Goulburn Valley Water"
    assert payload["document_meta"]["period_label"] == "2035"


def test_mailbox_plain_intel_prefers_non_scope_org_entity_as_primary(tmp_path):
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
            "subject": "Opportunity intel",
            "from_email": "rebecca.campbell-burns@escient.com.au",
            "received_at": "2026-03-25T07:12:29+00:00",
            "raw_text": "GVW is preparing an RFP for IT strategy and roadmap development services.",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [
                {"name": "Escient", "target_type": "organisation", "confidence": 0.7},
                {"name": "Goulburn Valley Water", "target_type": "organisation", "confidence": 0.9},
            ],
            "summary": "Intelligence note indicating Goulburn Valley Water is planning to issue an RFP for IT strategy and roadmap development services.",
            "attachments": [],
            "target_update_suggestions": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\n## Summary\n\nGVW is preparing an RFP for IT strategy and roadmap development services.",
        message_kind="general_intelligence",
        routing={"effective_org_name": "Escient", "status": "matched_sender_domain"},
    )

    assert payload["primary_entity"]["name"] == "Goulburn Valley Water"
    assert payload["primary_entity"]["target_type"] == "organisation"


def test_mailbox_fit_assessment_uses_subscriber_strategic_profile(tmp_path):
    class _EscientSignalStore(_FakeSignalStore):
        def get_org_context(self, org_name):
            return {
                "org_name": org_name,
                "org_alumni": [],
                "org_strategic_profile": {
                    "description": "Consulting focused on digital transformation, operating model and customer strategy.",
                    "industries": ["Consulting"],
                    "priority_industries": ["Water Utilities"],
                    "key_themes": ["IT strategy", "roadmap", "digital transformation"],
                    "strategic_objectives": ["win technology strategy work"],
                },
            }

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
        signal_store=_EscientSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "GVW opportunity",
            "from_email": "rebecca.campbell-burns@escient.com.au",
            "received_at": "2026-03-25T07:12:29+00:00",
            "raw_text": "Goulburn Valley Water is planning an RFP for IT strategy and roadmap work to build on a billing system replacement.",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "Goulburn Valley Water", "target_type": "organisation", "confidence": 0.9}],
            "summary": "Goulburn Valley Water is planning an RFP for IT strategy and roadmap development services.",
            "attachments": [],
            "target_update_suggestions": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\n## Summary\n\nGVW is preparing an RFP for IT strategy and roadmap development services.",
        message_kind="general_intelligence",
        routing={"effective_org_name": "Escient", "status": "matched_sender_domain"},
    )

    assert payload["fit_assessment"]["fit_label"] == "high_fit"
    assert "IT strategy" in payload["fit_assessment"]["matched_themes"]
    assert "roadmap" in payload["fit_assessment"]["matched_themes"]
    assert "## Subscriber Fit" in payload["note"]["content"]
    assert "High fit for Escient" in payload["note"]["content"]


def test_mailbox_fit_assessment_matches_related_strategy_language(tmp_path):
    class _EscientSignalStore(_FakeSignalStore):
        def get_org_context(self, org_name):
            return {
                "org_name": org_name,
                "org_alumni": [],
                "org_strategic_profile": {
                    "description": "Consulting focused on digital transformation and operating model change.",
                    "industries": ["Consulting"],
                    "priority_industries": ["Water"],
                    "key_themes": ["Technology Strategy & Enablement", "Strategy & Planning"],
                    "strategic_objectives": ["Transform customer experiences and operational efficiency"],
                },
            }

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
        signal_store=_EscientSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "GVW opportunity",
            "from_email": "rebecca.campbell-burns@escient.com.au",
            "received_at": "2026-03-25T07:12:29+00:00",
            "raw_text": "Goulburn Valley Water is planning an RFP for IT strategy and roadmap work to build on a billing system replacement.",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "Goulburn Valley Water", "target_type": "organisation", "confidence": 0.9}],
            "summary": "Goulburn Valley Water is planning an RFP for IT strategy and roadmap development services.",
            "attachments": [],
            "target_update_suggestions": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\n## Summary\n\nGVW is preparing an RFP for IT strategy and roadmap development services.",
        message_kind="general_intelligence",
        routing={"effective_org_name": "Escient", "status": "matched_sender_domain"},
    )

    assert payload["fit_assessment"]["fit_label"] in {"medium_fit", "high_fit"}
    assert "Water" in payload["fit_assessment"]["matched_priority_industries"]
    assert "Technology Strategy & Enablement" in payload["fit_assessment"]["matched_themes"]
    assert payload["fit_assessment"]["matched_objectives"] == []


def test_result_payload_copies_fit_assessment_to_top_level(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    poller = IntelMailboxPoller(
        IntelMailboxConfig(
            host="imap.gmail.com",
            port=993,
            username="intel@example.com",
            password="pw",
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
            smtp_username="intel@example.com",
            smtp_password="pw",
            smtp_use_ssl=True,
            reply_from="intel@example.com",
            mark_seen_on_success=False,
        ),
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    result = poller._build_result_payload(
        message={"subject": "GVW opportunity", "from_email": "paul@example.com", "received_at": "2026-03-26T13:08:05+11:00"},
        persisted={"message_key": "msg1", "raw_path": "/tmp/message.eml", "attachments": []},
        trace_id="trace-test",
        output_data={"summary": "Summary", "entities": [], "target_update_suggestions": [], "warnings": []},
        signal={"signal_id": "sig1", "matched_profile_keys": [], "needs_review": False},
        scope_org_name="Escient",
        routing={"effective_org_name": "Escient"},
        fit_assessment={
            "fit_label": "high_fit",
            "matched_priority_industries": ["Water"],
            "matched_themes": ["Technology Strategy & Enablement"],
        },
    )

    assert result["fit_assessment"]["fit_label"] == "high_fit"
    assert result["fit_assessment"]["matched_priority_industries"] == ["Water"]
    assert result["fit_assessment"]["matched_themes"] == ["Technology Strategy & Enablement"]


def test_result_payload_copies_processing_meta_to_top_level_and_website_payload(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    poller = IntelMailboxPoller(
        IntelMailboxConfig(
            host="imap.gmail.com",
            port=993,
            username="intel@example.com",
            password="pw",
            folder="INBOX",
            org_name="Longboardfella",
            poll_limit=5,
            search_criteria="UNSEEN",
            allowed_senders=(),
            source_system="cortex_mailbox",
            callback_url="",
            note_callback_url="",
            callback_secret="super-secret",
            callback_timeout=30,
            profile_import_url="",
            profile_import_timeout=30,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            smtp_username="intel@example.com",
            smtp_password="pw",
            smtp_use_ssl=True,
            reply_from="intel@example.com",
            mark_seen_on_success=False,
        ),
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    processing_meta = {
        "strategic_doc": {
            "doc_type": "annual_report",
            "org_name": "Gippsland Water",
            "themes": ["Future Solutions", "Healthy Country"],
            "strategic_signals": [
                {"headline": "Future Solutions", "snippet": "Infrastructure planning and growth."},
                {"headline": "Healthy Country", "snippet": "Sustainability and stewardship."},
            ],
        }
    }
    output_data = {
        "summary": "Summary",
        "entities": [],
        "target_update_suggestions": [],
        "warnings": [],
        "processing_meta": processing_meta,
    }

    result = poller._build_result_payload(
        message={"subject": "Gippsland Water annual report", "from_email": "paul@example.com", "received_at": "2026-03-26T13:56:27+11:00"},
        persisted={"message_key": "msg1", "raw_path": "/tmp/message.eml", "attachments": []},
        trace_id="trace-test",
        output_data=output_data,
        signal={"signal_id": "sig1", "matched_profile_keys": [], "needs_review": False},
        scope_org_name="Escient",
        routing={"effective_org_name": "Escient"},
        fit_assessment={},
    )
    website_payload = poller._build_website_payload(
        {
            "action": "ingest_intel_note",
            "secret": "super-secret",
            "org_name": "Escient",
            "note": {"title": "Gippsland Water Annual Report", "content": "## Summary\n\nDocument from Gippsland Water."},
            "fit_assessment": {},
        },
        output_data,
    )

    assert result["processing_meta"]["strategic_doc"]["doc_type"] == "annual_report"
    assert result["processing_meta"]["strategic_doc"]["org_name"] == "Gippsland Water"
    assert website_payload["processing_meta"]["strategic_doc"]["doc_type"] == "annual_report"
    assert website_payload["processing_meta"]["strategic_doc"]["themes"] == ["Future Solutions", "Healthy Country"]
    assert website_payload["secret"] == "[redacted]"


def test_mailbox_prioritizes_high_fit_strategy_themes_before_background(tmp_path):
    class _EscientSignalStore(_FakeSignalStore):
        def get_org_context(self, org_name):
            return {
                "org_name": org_name,
                "org_alumni": [],
                "org_strategic_profile": {
                    "description": "Consulting focused on digital transformation, customer strategy and operating model change.",
                    "industries": ["Consulting"],
                    "priority_industries": ["Water"],
                    "key_themes": ["digital transformation", "customer transformation", "strategy and planning"],
                    "strategic_objectives": ["win technology strategy work"],
                    "low_relevance_themes": ["Helping communities thrive"],
                },
            }

    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    poller = IntelMailboxPoller(
        IntelMailboxConfig(
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
        ),
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_EscientSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "entity: Escient | Yarra Valley Water strategic plan",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-25T07:12:29+00:00",
            "raw_text": "Yarra Valley Water 2030 strategy includes digital transformation and customer planning themes.",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "Yarra Valley Water", "target_type": "organisation", "confidence": 0.9}],
            "summary": "Yarra Valley Water strategy focused on customer transformation and environmental leadership.",
            "attachments": [],
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "strategic_plan",
                    "org_name": "Yarra Valley Water",
                    "themes": [
                        "Transforming around the customer",
                        "Enabling high performance",
                        "Helping communities thrive",
                    ],
                        "strategic_signals": [
                            {"headline": "Helping communities thrive", "snippet": "Community wellbeing and local partnerships."},
                            {"headline": "Transforming around the customer", "snippet": "Digital transformation and service redesign."},
                            {"headline": "Enabling high performance", "snippet": "Capability uplift, planning and execution discipline."},
                            {"headline": "Victorians every day and our customers", "snippet": "consistently rank us highly for trust, satisfaction, value for money and reputation."},
                        ],
                    "strategic_summary": "Yarra Valley Water strategy with customer transformation, capability uplift and community themes.",
                    "key_stakeholders": [],
                    "performance_indicators": [],
                    "major_projects": [],
                    "kpi_focuses": [],
                }
            },
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\n## Summary\n\nStrategy document.",
        message_kind="document_analysis",
        routing={"effective_org_name": "Escient", "status": "matched_override"},
    )

    note_content = payload["note"]["content"]
    assert "## Priority Strategic Themes" in note_content
    assert "## Background Themes" in note_content
    assert note_content.index("Transforming around the customer") < note_content.index("Helping communities thrive")
    assert note_content.index("Enabling high performance") < note_content.index("Helping communities thrive")
    priority_section = note_content.split("## Priority Strategic Themes", 1)[1].split("## Background Themes", 1)[0]
    background_section = note_content.split("## Background Themes", 1)[1]
    assert "Helping communities thrive" not in priority_section
    assert "- Helping communities thrive" in background_section
    assert "Victorians every day and our customers" not in note_content
    assert payload["fit_assessment"]["matched_low_relevance_themes"] == ["Helping communities thrive"]


def test_mailbox_handoff_runs_final_sanity_pass_on_note_content(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    poller = IntelMailboxPoller(
        config=IntelMailboxConfig(
            host="imap.gmail.com",
            port=993,
            username="intel@example.com",
            password="pw",
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
            smtp_username="intel@example.com",
            smtp_password="pw",
            smtp_use_ssl=True,
            reply_from="intel@example.com",
            mark_seen_on_success=False,
        ),
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )
    poller._run_markdown_sanity_llm = lambda markdown_text, llm_policy: markdown_text.replace(
        "innovation stretc...", "innovation stretches through partnerships and commercialisation."
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "entity: Escient | South East Water strategic plan",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-26T09:07:02+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "South East Water", "target_type": "organisation", "confidence": 0.9}],
            "summary": "South East Water strategy focused on customers, operations and innovation.",
            "attachments": [],
            "target_update_suggestions": [],
            "llm_policy": {"local_only": True, "model": "qwen3:30b"},
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "strategic_plan",
                    "org_name": "South East Water",
                    "themes": ["Deliver For Our Customers"],
                    "strategic_signals": [
                        {
                            "headline": "Deliver For Our Customers",
                            "snippet": "We provide safe and reliable water services while our innovation stretc...",
                        }
                    ],
                    "strategic_summary": "South East Water strategy emphasising service reliability and customer experience.",
                    "key_stakeholders": [],
                    "performance_indicators": [],
                    "major_projects": [],
                    "kpi_focuses": [],
                }
            },
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\n## Summary\n\nStrategy document.",
        message_kind="document_analysis",
        routing={"effective_org_name": "Escient", "status": "matched_override", "extraction_depth": "detailed"},
    )

    note_content = payload["note"]["content"]
    assert "innovation stretches through partnerships and commercialisation." in note_content
    assert "innovation stretc..." not in note_content


def test_mailbox_detailed_mode_includes_theme_detail_notes(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    poller = IntelMailboxPoller(
        config=IntelMailboxConfig(
            host="imap.gmail.com",
            port=993,
            username="intel@example.com",
            password="pw",
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
            smtp_username="intel@example.com",
            smtp_password="pw",
            smtp_use_ssl=True,
            reply_from="intel@example.com",
            mark_seen_on_success=False,
        ),
        store=store,
        extractor=lambda payload: ({}, None),
        imap_factory=_FakeIMAP,
        signal_store=_FakeSignalStore(),
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "entity: Escient | South East Water strategic plan",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-26T09:07:02+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "South East Water", "target_type": "organisation", "confidence": 0.9}],
            "summary": "South East Water strategy focused on customers, operations and innovation.",
            "attachments": [],
            "target_update_suggestions": [],
            "llm_policy": {"local_only": True, "model": "qwen3:30b"},
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "strategic_plan",
                    "org_name": "South East Water",
                    "themes": ["Deliver For Our Customers", "Protect Our Environment"],
                    "strategic_signals": [
                        {
                            "headline": "Deliver For Our Customers",
                            "snippet": "We provide safe and reliable water and waste services while improving customer experience.",
                            "detail_points": [
                                {
                                    "heading": "Get The Basics Right, Always",
                                    "snippet": "We provide safe and reliable water and waste services, minimising interruptions and continually delighting our customers.",
                                },
                                {
                                    "heading": "Make Our Customers Experience Better",
                                    "snippet": "We’ve increased self-service options to meet the needs of our customers by streamlining our systems and processes.",
                                },
                            ],
                        },
                        {
                            "headline": "Protect Our Environment",
                            "snippet": "We work with Traditional Owners and continue reducing emissions.",
                            "detail_points": [
                                {
                                    "heading": "Care For Country",
                                    "snippet": "We walk with Traditional Owners to support self-determination and deliver water justice.",
                                }
                            ],
                        },
                    ],
                    "strategic_summary": "South East Water strategy emphasising service reliability, customer experience and sustainability.",
                    "key_stakeholders": [],
                    "performance_indicators": [],
                    "major_projects": [],
                    "kpi_focuses": [],
                }
            },
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\n## Summary\n\nStrategy document.",
        message_kind="document_analysis",
        routing={"effective_org_name": "Escient", "status": "matched_override", "extraction_depth": "detailed"},
    )

    note_content = payload["note"]["content"]
    assert "## Detailed Theme Notes" in note_content
    assert "### Deliver For Our Customers" in note_content
    assert "We provide safe and reliable water and waste services, minimising interruptions and continually delighting our customers" in note_content
    assert "We’ve increased self-service options to meet the needs of our customers by streamlining our systems and processes" in note_content
    assert "### Protect Our Environment" in note_content
    assert "We walk with Traditional Owners to support self-determination and deliver water justice" in note_content


def test_mailbox_poller_sends_reply_for_note_ingest_and_includes_depth(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()

    def _extractor(payload):
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        assert payload["extraction_depth"] == "detailed"
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
                    }
                ],
                "summary": "Detailed annual report extraction for Barwon Water.",
                "attachments": [{"filename": "barwon-water-org-chart.pdf", "status": "processed"}],
                "target_update_suggestions": [],
                "warnings": [],
                "processing_meta": {},
            },
            out,
        )

    class _FakeDetailedRoutedDocumentIMAP(_FakeIMAP):
        def fetch(self, _imap_id, _query):
            msg = EmailMessage()
            msg["Subject"] = "entity: Escient | depth:detailed | Barwon Water annual report"
            msg["From"] = "Paul <paul@example.com>"
            msg["To"] = "intel.longboardfella@gmail.com"
            msg["Date"] = "Thu, 12 Mar 2026 11:28:09 +1100"
            msg["Message-ID"] = "<msg-detailed-routed-doc@example.com>"
            msg.set_content("Uploading Barwon Water annual report.")
            msg.add_attachment(
                b"%PDF-1.4 fake pdf bytes",
                maintype="application",
                subtype="pdf",
                filename="barwon-water-annual-report.pdf",
            )
            return "OK", [(b"1 (RFC822 {123})", msg.as_bytes())]

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
        imap_factory=_FakeDetailedRoutedDocumentIMAP,
        signal_store=_RoutingSignalStore(),
        result_client=result_client,
        reply_client=_FakeReplyClient(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    messages = store.list_messages()
    payload = json.loads(Path(messages[0]["result_path"]).read_text(encoding="utf-8"))
    assert payload["reply"]["delivery"]["status"] == "sent"
    assert payload["reply"]["subject"] == "Re: entity: Escient | depth:detailed | Barwon Water annual report"
    assert "Depth: detailed" in payload["reply"]["body"]
    assert "Title: Barwon Water annual report" in payload["reply"]["body"]
    assert "Primary entity: Barwon Water" in payload["reply"]["body"]
    assert payload["mailbox_routing"]["extraction_depth"] == "detailed"


def test_mailbox_poller_sends_reply_for_legacy_intel_extract(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")

    def _extractor(_payload):
        return (
            {
                "status": "extracted",
                "entity_count": 2,
                "people": [],
                "organisations": [{"canonical_name": "Goulburn Valley Water"}],
                "entities": [
                    {
                        "canonical_name": "Goulburn Valley Water",
                        "name": "Goulburn Valley Water",
                        "target_type": "organisation",
                        "confidence": 0.95,
                    },
                    {
                        "canonical_name": "Escient",
                        "name": "Escient",
                        "target_type": "organisation",
                        "confidence": 0.8,
                    },
                ],
                "summary": "Goulburn Valley Water is preparing an RFP for IT strategy and roadmap work.",
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
        reply_client=_FakeReplyClient(),
    )

    summary = poller.poll_once()

    assert summary["processed"] == 1
    messages = store.list_messages()
    payload = json.loads(Path(messages[0]["result_path"]).read_text(encoding="utf-8"))
    assert payload["reply"]["delivery"]["status"] == "sent"
    assert payload["reply"]["subject"] == "Re: INTEL: Carolyn Bell update"
    assert "Cortex processed your submission for Longboardfella." in payload["reply"]["body"]
    assert "Primary entity: Goulburn Valley Water" in payload["reply"]["body"]
    assert "Goulburn Valley Water is preparing an RFP for IT strategy and roadmap work." in payload["reply"]["body"]


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


def test_mailbox_poller_skips_reingesting_same_annual_report_document(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()
    signal_store = _RoutingSignalStore()
    existing_fingerprint = build_content_fingerprint(
        attachment_fingerprints=["barwon-water-annual-report-2025.pdf:report123"]
    )
    existing_meta = build_document_meta(
        doc_type="annual_report",
        target_org_name="Barwon Water",
        title="Barwon Water annual report",
        period_label="2025",
        content_fingerprint=existing_fingerprint,
        source_label="Mailbox attachment",
    )
    signal_store.register_document_meta(
        "Escient",
        existing_meta,
        latest_trace_id="trace-old",
        latest_intel_id="intel_existing_1",
    )

    def _extractor(_payload):
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        return (
            {
                "status": "extracted",
                "entity_count": 1,
                "entities": [
                    {
                        "canonical_name": "Barwon Water",
                        "name": "Barwon Water",
                        "target_type": "organisation",
                        "confidence": 0.95,
                    }
                ],
                "organisations": [{"canonical_name": "Barwon Water"}],
                "attachments": [
                        {
                            "filename": "barwon-water-annual-report-2025.pdf",
                            "status": "processed",
                            "excerpt": "Barwon Water annual report 2025",
                        }
                ],
                "summary": "Barwon Water annual report.",
                "target_update_suggestions": [],
                "warnings": [],
                "processing_meta": {
                    "strategic_doc": {
                        "doc_type": "annual_report",
                        "org_name": "Barwon Water",
                        "strategic_summary": "Annual report for Barwon Water.",
                        "strategic_signals": [],
                        "key_stakeholders": [],
                        "performance_indicators": [],
                    }
                },
            },
            out,
        )

    class _AnnualReportIMAP(_FakeIMAP):
        def fetch(self, _imap_id, _query):
            msg = EmailMessage()
            msg["Subject"] = "entity: Escient | Barwon Water annual report 2025"
            msg["From"] = "Paul <paul@longboardfella.com.au>"
            msg["To"] = "intel.longboardfella@gmail.com"
            msg["Date"] = "Thu, 24 Mar 2026 11:28:09 +1100"
            msg["Message-ID"] = "<msg-annual-dedupe@example.com>"
            msg.set_content("Please see attached annual report.")
            msg.add_attachment(
                b"%PDF-1.4 fake annual report bytes",
                maintype="application",
                subtype="pdf",
                filename="barwon-water-annual-report-2025.pdf",
            )
            return "OK", [(b"1 (RFC822 {123})", msg.as_bytes())]

    poller = IntelMailboxPoller(
        IntelMailboxConfig(
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
            profile_import_url="",
            profile_import_timeout=30,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            smtp_username="intel.longboardfella@gmail.com",
            smtp_password="secret",
            smtp_use_ssl=True,
            reply_from="intel.longboardfella@gmail.com",
            mark_seen_on_success=False,
        ),
        store=store,
        extractor=_extractor,
        imap_factory=_AnnualReportIMAP,
        signal_store=signal_store,
        result_client=result_client,
    )

    original_build_attachment_fingerprints = poller._build_attachment_fingerprints
    poller._build_attachment_fingerprints = lambda persisted, output_data: ["barwon-water-annual-report-2025.pdf:report123"]  # type: ignore[method-assign]
    try:
        summary = poller.poll_once()
    finally:
        poller._build_attachment_fingerprints = original_build_attachment_fingerprints  # type: ignore[method-assign]

    assert summary["processed"] == 1
    assert result_client.calls == []
    messages = store.list_messages()
    payload = json.loads(Path(messages[0]["result_path"]).read_text(encoding="utf-8"))
    assert messages[0]["delivery"]["status"] == "duplicate_document"
    assert payload["document_meta"]["status"] == "known_same"


def test_mailbox_poller_force_override_reingests_same_annual_report_document(tmp_path):
    store = IntelMailboxStore(base_path=tmp_path / "intel_mailbox")
    result_client = _RecordingResultClient()
    signal_store = _RoutingSignalStore()
    existing_fingerprint = build_content_fingerprint(
        attachment_fingerprints=["barwon-water-annual-report-2025.pdf:report123"]
    )
    existing_meta = build_document_meta(
        doc_type="annual_report",
        target_org_name="Barwon Water",
        title="Barwon Water annual report",
        period_label="2025",
        content_fingerprint=existing_fingerprint,
        source_label="Mailbox attachment",
    )
    signal_store.register_document_meta(
        "Escient",
        existing_meta,
        latest_trace_id="trace-old",
        latest_intel_id="intel_existing_1",
    )

    def _extractor(_payload):
        out = tmp_path / "extract.md"
        out.write_text("# Extract", encoding="utf-8")
        return (
            {
                "status": "extracted",
                "entity_count": 1,
                "entities": [
                    {
                        "canonical_name": "Barwon Water",
                        "name": "Barwon Water",
                        "target_type": "organisation",
                        "confidence": 0.95,
                    }
                ],
                "organisations": [{"canonical_name": "Barwon Water"}],
                "attachments": [
                    {
                        "filename": "barwon-water-annual-report-2025.pdf",
                        "status": "processed",
                        "excerpt": "Barwon Water annual report 2025",
                    }
                ],
                "summary": "Barwon Water annual report.",
                "target_update_suggestions": [],
                "warnings": [],
                "processing_meta": {
                    "strategic_doc": {
                        "doc_type": "annual_report",
                        "org_name": "Barwon Water",
                        "strategic_summary": "Annual report for Barwon Water.",
                        "strategic_signals": [],
                        "key_stakeholders": [],
                        "performance_indicators": [],
                    }
                },
            },
            out,
        )

    class _ForceAnnualReportIMAP(_FakeIMAP):
        def fetch(self, _imap_id, _query):
            msg = EmailMessage()
            msg["Subject"] = "entity: Escient | force:yes | Barwon Water annual report 2025"
            msg["From"] = "Paul <paul@longboardfella.com.au>"
            msg["To"] = "intel.longboardfella@gmail.com"
            msg["Date"] = "Thu, 24 Mar 2026 11:28:09 +1100"
            msg["Message-ID"] = "<msg-force-annual-dedupe@example.com>"
            msg.set_content("Please see attached annual report.")
            msg.add_attachment(
                b"%PDF-1.4 fake annual report bytes",
                maintype="application",
                subtype="pdf",
                filename="barwon-water-annual-report-2025.pdf",
            )
            return "OK", [(b"1 (RFC822 {123})", msg.as_bytes())]

    poller = IntelMailboxPoller(
        IntelMailboxConfig(
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
            profile_import_url="",
            profile_import_timeout=30,
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            smtp_username="intel.longboardfella@gmail.com",
            smtp_password="secret",
            smtp_use_ssl=True,
            reply_from="intel.longboardfella@gmail.com",
            mark_seen_on_success=False,
        ),
        store=store,
        extractor=_extractor,
        imap_factory=_ForceAnnualReportIMAP,
        signal_store=signal_store,
        result_client=result_client,
        reply_client=_FakeReplyClient(),
    )

    original_build_attachment_fingerprints = poller._build_attachment_fingerprints
    poller._build_attachment_fingerprints = lambda persisted, output_data: ["barwon-water-annual-report-2025.pdf:report123"]  # type: ignore[method-assign]
    try:
        summary = poller.poll_once()
    finally:
        poller._build_attachment_fingerprints = original_build_attachment_fingerprints  # type: ignore[method-assign]

    assert summary["processed"] == 1
    assert result_client.calls
    payload = json.loads(Path(store.list_messages()[0]["result_path"]).read_text(encoding="utf-8"))
    assert payload["mailbox_routing"]["force_reingest"] is True
    assert payload["document_meta"]["status"] == "changed_document"


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
    assert "First Nations and cultural commitments" in headlines
    assert "Key strategic signals:" in analysis["strategic_summary"]


def test_analyse_strategic_documents_prefers_explicit_strategy_pillars_for_yarra_plan():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "2030_Strategy-website_0.pdf",
                "status": "processed",
                "excerpt": (
                    "Yarra Valley Water 2030 Strategy "
                    "Our purpose is to support the health and wellbeing of customers and create a brighter future for communities and the natural environment. "
                    "Honouring and healing Country "
                    "Transforming around the customer "
                    "Helping communities thrive "
                    "Leading for our environmental future "
                    "Enabling high performance "
                    "Ongoing digital transformation will allow us to better respond to our customers."
                ),
            }
        ],
        extracted_summary="Yarra Valley Water strategy focused on customer transformation and environmental leadership.",
        subject="Yarra Valley Water strategic plan",
        raw_text="",
    )

    assert analysis["org_name"] == "Yarra Valley Water"
    assert "Honouring and healing Country" in analysis["themes"]
    assert "Transforming around the customer" in analysis["themes"]
    assert "Helping communities thrive" in analysis["themes"]
    assert "Leading for our environmental future" in analysis["themes"]
    assert "Enabling high performance" in analysis["themes"]
    assert analysis["strategic_signals"][0]["headline"] == "Honouring and healing Country"


def test_analyse_strategic_documents_prefers_seqwater_focus_areas_and_objectives():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Seqwater-Strategic-Plan-2024-28.pdf",
                "status": "processed",
                "excerpt": (
                    "Seqwater Strategic Plan 2024-28\n"
                    "Strategic objectives\n"
                    "Improve safety and\n"
                    "organisational culture\n"
                    "Improve processes,\n"
                    "systems and planning\n"
                    "Strengthen financial\n"
                    "sustainability\n"
                    "Increase water supply\n"
                    "certainty\n"
                    "Increase stakeholder, customer and\n"
                    "community satisfaction and support\n"
                    "Strategic outcomes\n"
                    "Focus areas\n"
                    "CAPITAL DELIVERY TRANSFORMATION\n"
                    "A continuation of the program of work to uplift capital delivery capability to meet the portfolio requirements to 2035.\n"
                    "OPERATIONAL TRANSFORMATION\n"
                    "Develop a multi-year program improving systems, processes and assets to reduce risk, improve productivity amidst long-term operational challenges.\n"
                    "SHAREHOLDER EXPECTATIONS AND\n"
                    "SOCIAL LICENCE\n"
                    "Building Seqwater’s social licence to operate as a key enabler for operational activities and delivery of capital projects.\n"
                    "TALENT, CULTURE AND PERFORMANCE\n"
                    "Harness and leverage technology, empowering our people and creating a culture of high-performance.\n"
                ),
            }
        ],
        extracted_summary="Seqwater strategic plan covering focus areas and strategic objectives.",
        subject="Seqwater strategic plan 2024-28",
        raw_text="",
        extraction_depth="default",
    )

    headlines = [item["headline"] for item in analysis["strategic_signals"]]

    assert analysis["doc_type"] == "strategic_plan"
    assert analysis["org_name"] == "Seqwater"
    assert "Capital Delivery Transformation" in headlines
    assert "Operational Transformation" in headlines
    assert "Shareholder Expectations And Social Licence" in headlines
    assert "Talent, Culture And Performance" in headlines
    assert "Improve safety and organisational culture" in headlines
    assert "social licence. Performance" not in headlines


def test_analyse_strategic_documents_prefers_later_focus_area_matrix_over_preamble():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Our_Corporate_Strategy_2028.pdf",
                "status": "processed",
                "excerpt": (
                    "Our Corporate Strategy 2028\n"
                    "Acknowledgement\n"
                    "We acknowledge their songlines, cultural lore and continuing connection to the land and water.\n"
                    "The risks and opportunities we're facing\n"
                    "Digital transformation\n"
                    "Water security\n"
                    "Cyber security\n"
                    "Growing population\n"
                    "Our strategic focus areas\n"
                    "Empower our people\n"
                    "Deliver for our customers\n"
                    "Protect our environment\n"
                    "Optimise our operations\n"
                    "Drive innovation at scale\n"
                    "We’re one team that reflects the diversity of our customers and builds a safe space where people find inspiring opportunities in water.\n"
                    "We know how important it is to get the basics right and make our customers’ experience better every time.\n"
                    "We’re driving long-term water security, net zero emissions and repurposing waste to protect our environment.\n"
                    "Committed to refining our processes, products, assets and service, we strive for continuous improvement.\n"
                    "Our innovation stretches beyond prototypes and through partnerships and commercialisation we share data, expertise and technology.\n"
                ),
            }
        ],
        extracted_summary="South East Water corporate strategy focused on customer, operations and innovation.",
        subject="South East Water strategic plan",
        raw_text="",
        extraction_depth="default",
    )

    headlines = [item["headline"] for item in analysis["strategic_signals"]]

    assert analysis["doc_type"] == "strategic_plan"
    assert str(analysis["org_name"]).startswith("South East Water")
    assert "Empower Our People" in headlines
    assert "Deliver For Our Customers" in headlines
    assert "Protect Our Environment" in headlines
    assert "Optimise Our Operations" in headlines
    assert "Drive Innovation At Scale" in headlines
    assert "Elders past, present and future" not in headlines
    assert "we're facing Digital" not in headlines


def test_analyse_strategic_documents_prefers_ambition_sections_for_south_east_water_strategy():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Our_Corporate_Strategy_2028.pdf",
                "status": "processed",
                "excerpt": (
                    "Our Corporate Strategy 2028\n"
                    "Our strategic focus areas\n"
                    "Empower our people\n"
                    "Deliver for our customers\n"
                    "Protect our environment\n"
                    "Optimise our operations\n"
                    "Drive innovation at scale\n"
                    "We’re one team that reflects the diversity of our customers.\n"
                    "Our ambitions for 2028\n"
                    "Deliver\n"
                    "for our\n"
                    "customers\n"
                    "Intent statements and measures\n"
                    "Get the basics\n"
                    "right, always\n"
                    "We provide safe and reliable water and waste services, minimising interruptions and continually delighting our customers.\n"
                    "We’ve increased self-service options to meet the needs of our customers by streamlining our systems and processes.\n"
                    "Support\n"
                    "our community\n"
                    "We have deeper understanding of and relationships with our community, so we can better meet their needs.\n"
                    "Our ambitions for 2028\n"
                    "Protect our\n"
                    "environment\n"
                    "Intent statements and measures\n"
                    "Care for country\n"
                    "We walk with Traditional Owners to support self-determination and deliver water justice.\n"
                    "We’ve achieved net zero scope 1 and 2 emissions and continued to reduce scope 3 emissions.\n"
                    "Our ambitions for 2028\n"
                    "Optimise our\n"
                    "operations\n"
                    "Intent statements and measures\n"
                    "Digital customer &\n"
                    "employee experience\n"
                    "With most of our customers having digital meters installed, more customers have a better experience and can use the tools and information to save money and water.\n"
                    "We’re reducing water losses by using insights gained from our digital meters and sensors to identify leaks early.\n"
                    "Our ambitions for 2028\n"
                    "Drive\n"
                    "innovation\n"
                    "at scale\n"
                    "Intent statements and measures\n"
                    "Commercialisation\n"
                    "& partnership impact\n"
                    "We regularly draw on established partnerships to broaden the reach and scale of our innovation, providing better solutions to customers and the sector.\n"
                ),
            }
        ],
        extracted_summary="South East Water corporate strategy focused on customer, operations and innovation.",
        subject="South East Water strategic plan",
        raw_text="",
        extraction_depth="detailed",
    )

    strategic_signals = {item["headline"]: item for item in analysis["strategic_signals"]}

    assert analysis["doc_type"] == "strategic_plan"
    assert analysis["org_name"] == "South East Water"
    assert "Deliver For Our Customers" in strategic_signals
    assert strategic_signals["Deliver For Our Customers"]["snippet"].startswith(
        "We provide safe and reliable water and waste services"
    )
    assert "Protect Our Environment" in strategic_signals
    assert "water justice" in strategic_signals["Protect Our Environment"]["snippet"]
    assert "Optimise Our Operations" in strategic_signals
    assert "digital meters" in strategic_signals["Optimise Our Operations"]["snippet"]
    assert "Drive Innovation At Scale" in strategic_signals
    assert "broaden the reach and scale of our innovation" in strategic_signals["Drive Innovation At Scale"]["snippet"]
    assert "We’Re One Team That" not in strategic_signals


def test_analyse_strategic_documents_prefers_explicit_priority_plan_section_for_gippsland_water():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Web_version_2024_Corporate_Plan.pdf",
                "status": "processed",
                "excerpt": (
                    "Corporate Plan 2024 - 29\n"
                    "Table of contents\n"
                    "Aboriginal acknowledgement\n"
                    "Our values\n"
                    "‘Go home safe’\n"
                    "The safety and wellbeing of our employees and community is our priority.\n"
                    "‘Customer first’\n"
                    "Customers are at the heart of everything we do.\n"
                    "Our strategic priorities\n"
                    "Our strategic priorities represent the highest order initiatives we will focus on in the coming five-year period.\n"
                    "Our 2024-29 Strategic Priorities Plan outlines priorities including:\n"
                    "Healthy country\n"
                    "• Embedding of the requirements of the new EPA General Environmental Duty throughout our business.\n"
                    "• Delivery of education and awareness campaigns that focus on water conservation and sustainability.\n"
                    "Climate preparedness\n"
                    "• Delivery of our Energy Management Strategy which will outline how we reduce our carbon footprint and achieve 100% renewable energy by 2025.\n"
                    "• Delivery of our Climate Change Strategy which outlines how we will minimise our emissions and offset our residual Scope 1 to achieve net-zero emissions by 2030.\n"
                    "Affordable bills\n"
                    "• Delivery of an App for Gippsland Water customers to have greater control over their bill and water usage.\n"
                    "• Replacement of our finance system to support efficiencies in the business and affordable bills for customers.\n"
                    "Future solutions\n"
                    "• Undertaking phases one and two of the Gippsland Regional Organics expansion plan to deliver increased processing capacity.\n"
                    "• Digitise our works management practice to transform how we schedule and engage with infrastructure work orders.\n"
                    "Our 2050 Vision\n"
                ),
            }
        ],
        extracted_summary="Gippsland Water corporate plan outlining strategic priorities.",
        subject="Gippsland Water strategic plan",
        raw_text="",
        extraction_depth="detailed",
    )

    strategic_signals = {item["headline"]: item for item in analysis["strategic_signals"]}

    assert analysis["doc_type"] == "strategic_plan"
    assert analysis["org_name"] == "Gippsland Water"
    assert "Healthy Country" in strategic_signals
    assert "water conservation and sustainability" in strategic_signals["Healthy Country"]["snippet"]
    assert "Climate Preparedness" in strategic_signals
    assert "100% renewable energy by 2025" in strategic_signals["Climate Preparedness"]["snippet"]
    assert "Affordable Bills" in strategic_signals
    assert "greater control over their bill and water usage" in strategic_signals["Affordable Bills"]["snippet"]
    assert "Future Solutions" in strategic_signals
    assert "Regional Organics expansion plan" in strategic_signals["Future Solutions"]["snippet"]
    assert "OFFICIAL" not in strategic_signals
    assert "Aboriginal Acknowledgement" not in strategic_signals
    assert "‘Customer First’" not in strategic_signals


def test_analyse_annual_report_filters_official_markers_and_credential_only_stakeholders():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Annual_Report_2024-25.pdf",
                "status": "processed",
                "excerpt": (
                    "OFFICIAL\n"
                    "F\n"
                    "Annual report 2024-25\n"
                    "Board Chair Tom Mollenkopf AO, Minister for Water Gayle Tierney and Managing Director Sarah Cumming\n"
                    "It is a pleasure to present the 2024-25 Gippsland Water Annual Report.\n"
                    "This year we made strong progress against the four pillars of our strategic framework, future solutions, Healthy Country, affordable bills and climate preparedness.\n"
                    "Tom Mollenkopf AO\n"
                    "Board Chair\n"
                    "Sarah Cumming\n"
                    "Managing Director\n"
                    "Our Board\n"
                    "Tom Mollenkopf, AO (Chair)\n"
                    "Sarah Cumming (Managing Director)\n"
                    "BA(Hons), AMusA\n"
                    "Penny has extensive board and committee experience, including\n"
                    "Cert. Lab. Tech.\n"
                    "Simon is responsible for leading the commercial and operational\n"
                    "Key Management Personnel\n"
                    "Remuneration of Executives\n"
                    "Central Gippsland Region Water Corporation, or Gippsland Water, is a statutory body.\n"
                ),
            }
        ],
        extracted_summary="Gippsland Water annual report covering strategic framework and annual performance.",
        subject="Gippsland Water annual report",
        raw_text="",
        extraction_depth="detailed",
    )

    stakeholders = [item["name"] for item in analysis["key_stakeholders"]]

    assert analysis["doc_type"] == "annual_report"
    assert analysis["org_name"] == "Gippsland Water"
    assert "Tom Mollenkopf AO" in stakeholders
    assert "Sarah Cumming" in stakeholders
    assert "BA(Hons), AMusA" not in stakeholders
    assert "Cert. Lab. Tech" not in stakeholders
    assert "Key Management Personnel" not in stakeholders


def test_analyse_annual_report_prefers_framework_pillars_over_generic_cross_doc_themes():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Annual_Report_2024-25.pdf",
                "status": "processed",
                "excerpt": (
                    "Annual report 2024-25\n"
                    "It is a pleasure to present the 2024-25 Gippsland Water Annual Report.\n"
                    "This year we made strong progress against the four pillars of our strategic framework,\n"
                    "future solutions, Healthy Country, affordable bills and climate preparedness.\n"
                    "We delivered major projects that enabled regional growth and strengthened resilience.\n"
                    "We invested $65.9 million in capital expenditure to proactively plan and build for the future.\n"
                    "Demonstrating our steadfast commitment to environmental stewardship, we’ve worked to be successfully positioned to operating on 100% renewable energy from 1 July 2025.\n"
                    "To support customers impacted by the cost of living we’ve absorbed some of the inflationary pressures to maintain price increases below CPI.\n"
                    "Our educational campaigns promoted Healthy Country and sustainability across the community.\n"
                    "Tom Mollenkopf AO\n"
                    "Board Chair\n"
                    "Sarah Cumming\n"
                    "Managing Director\n"
                    "Tom Mollenkopf\n"
                    "Board Chair\n"
                ),
            }
        ],
        extracted_summary="Gippsland Water annual report covering strategic framework and annual performance.",
        subject="Gippsland Water annual report",
        raw_text="",
        extraction_depth="detailed",
    )

    headlines = [item["headline"] for item in analysis["strategic_signals"]]
    stakeholders = [item["name"] for item in analysis["key_stakeholders"]]

    assert analysis["doc_type"] == "annual_report"
    assert "Future Solutions" in headlines
    assert "Healthy Country" in headlines
    assert "Affordable Bills" in headlines
    assert "Climate Preparedness" in headlines
    assert "Member trust and value-for-money pressure" not in headlines
    assert stakeholders.count("Tom Mollenkopf AO") == 1
    assert "Tom Mollenkopf" not in stakeholders


def test_compact_strategy_snippet_repairs_mid_sentence_ocr_periods():
    snippet = _compact_strategy_snippet(
        "Engagement with customers to shape our Urban Water Strategy for the delivery of. sustainable and affordable water supplies to meet current and future demand.",
        local_only=True,
        limit=520,
        allow_two_sentences=True,
    )
    assert snippet == (
        "Engagement with customers to shape our Urban Water Strategy for the delivery of sustainable and affordable water supplies to meet current and future demand"
    )


def test_clean_note_signal_snippet_drops_noisy_strategy_plan_fragments():
    snippet = _clean_note_signal_snippet(
        "Our ambitions for 2028 Intent statements and measures Safety Maturity Model >95% safety training completion by all employees",
        doc_type="strategic_plan",
    )
    assert snippet == ""


def test_should_keep_rendered_strategic_signal_filters_pronoun_led_fragments():
    assert not _should_keep_rendered_strategic_signal("We’Re One Team That", doc_type="strategic_plan")
    assert _should_keep_rendered_strategic_signal("Drive Innovation At Scale", doc_type="strategic_plan")


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
    assert "strong foundation for the College's future" in strategic_signals["First Nations and cultural commitments"]["snippet"].replace("\u2019", "'")


def test_analyse_strategic_documents_extracts_barwon_financial_highlights_and_major_projects():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Barwon-Water-Group-Annual-Report-2025-accessible.pdf",
                "status": "processed",
                "excerpt": (
                    "Barwon Water Annual Report 2024-25 "
                    "Revenue for the year increased to $344.2 million, up from $292.8 million. "
                    "During the financial year, we received revenue of $344.2 million, contributing towards a $29.4 million net surplus. "
                    "Further, we delivered a $158.1 million capital works program, as total assets reached $3.9 billion. "
                    "Total debt increased by $114.0 million to $721.9 million, with cash and cash equivalents increasing by $2.4 million to $15.0 million. "
                    "We delivered $158.1 million of capital and related infrastructure works during 2024-25. "
                    "The largest water supply investments included the Melbourne to Geelong Pipeline extension to Pettavel ($16 million), "
                    "the Water Reticulation Main Renewal ($5.8 million), the Gellibrand Water Treatment Plant Upgrade ($5.4 million) "
                    "and the Marengo Basin Upgrade ($4.9 million). "
                    "The largest wastewater system investments included the Mains Rehabilitation and Replacement ($14 million), "
                    "the Northern Growth Area Advanced Works ($15.2 million) and the Ocean Grove Rising Main No. 2 ($6.1 million). "
                    "We achieved 15.2% water recycling in 2024-25. "
                    "Jo Plummer\nChair\nBarwon Water\nShaun Cumming\nManaging Director\nBarwon Water"
                ),
            }
        ],
        extracted_summary="Barwon Water annual report covering drought response, capital delivery and financial resilience.",
        subject="Barwon Water annual report 2024-25",
        raw_text="",
    )

    indicators = {item["label"]: item for item in analysis["performance_indicators"]}
    project_names = [item["name"] for item in analysis["major_projects"]]

    assert analysis["doc_type"] == "annual_report"
    assert analysis["org_name"] == "Barwon Water"
    assert indicators["Annual revenue"]["value"] == "$344.2 million (from $292.8 million)"
    assert indicators["Net result"]["value"] == "$29.4 million net surplus"
    assert indicators["Capital works program"]["value"] == "$158.1 million"
    assert indicators["Total assets"]["value"] == "$3.9 billion"
    assert indicators["Total debt"]["value"] == "$721.9 million total debt"
    assert indicators["Cash position"]["value"] == "$15.0 million cash"
    assert indicators["Water recycling rate"]["value"] == "15.2%"
    assert "Melbourne to Geelong Pipeline extension to Pettavel" in project_names
    assert "Northern Growth Area Advanced Works" in project_names
    assert "Ocean Grove Rising Main No. 2" in project_names


def test_analyse_strategic_documents_extracts_seqwater_operational_annual_report_signals():
    analysis = analyse_strategic_documents(
        [
            {
                "filename": "Seqwater-Annual-Report-2024-25.pdf",
                "status": "processed",
                "excerpt": (
                    "Seqwater Annual Report 2024-25 "
                    "Operational transformation Operational resilience Asset information and management "
                    "Capital delivery transformation Engaging with stakeholders Working with customers "
                    "Process improvement and use of technology as an enabler Information systems, record keeping and cyber security "
                    "Seqwater is proud to deliver safe affordable, reliable and sustainable water to more than 3.7 million SEQ residents. "
                    "In 2024–25, Seqwater supplied approximately 326,060 megalitres (ML) of safe, reliable drinking water. "
                    "Additionally, we welcomed 2.5 million recreational visitors to our lakes and parks. "
                    "The SEQ Water Grid recorded a combined storage level of 85.9% on 30 June 2025. "
                    "Our Flood Operations Centre was activated nine times in response to weather events. "
                    "In 2024–25, Seqwater invested over $389 million in infrastructure including its dam improvement and water security programs. "
                    "John McEvoy\nChairperson, Seqwater Board\nSeqwater\n"
                    "Emma Thomas\nChief Executive Officer\nSeqwater\n"
                    "Responsible Ministers\nBoard Committees\nSeqwater\n"
                ),
            }
        ],
        extracted_summary="Seqwater annual report covering asset programs, resilience and customer delivery.",
        subject="Seqwater Annual Report 2024-25",
        raw_text="",
        extraction_depth="detailed",
    )

    indicators = {item["label"]: item for item in analysis["performance_indicators"]}
    signal_headlines = [item["headline"] for item in analysis["strategic_signals"]]
    stakeholder_names = [item["name"] for item in analysis["key_stakeholders"]]

    assert analysis["doc_type"] == "annual_report"
    assert analysis["org_name"] == "Seqwater"
    assert indicators["Population served"]["value"] == "3.7m"
    assert indicators["Bulk water supplied"]["value"] == "326,060 ML"
    assert indicators["Recreational visitors"]["value"] == "2.5 million"
    assert indicators["Water storage level"]["value"] == "85.9%"
    assert indicators["Infrastructure investment"]["value"] == "$389 million"
    assert "Operational transformation and resilience" in signal_headlines
    assert "Capital delivery and asset transformation" in signal_headlines
    assert "Stakeholder, customer and community engagement" in signal_headlines
    assert "Technology and cyber enablement" in signal_headlines
    assert "Emma Thomas" in stakeholder_names
    assert "Responsible Ministers" not in stakeholder_names


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
                    "major_projects": [
                        {
                            "name": "Training Management Platform",
                            "value": "$2.4 million",
                            "evidence": "The new Training Management Platform went live in December 2024.",
                        }
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
    assert "## Key Projects" in note_content
    assert "Training Management Platform: $2.4 million" in note_content
    assert "Operating result" in headlines
    assert "Major project: Training Management Platform" in headlines


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
    assert payload["note"]["title"].lower() == "barwon water annual report"
    assert "Barwon Region Water Corporation" not in payload["primary_entity"]["name"]


def test_mailbox_handoff_normalizes_annual_report_summary_org_label_and_filters_pseudo_stakeholders(tmp_path):
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
            "entities": [{"name": "Barwon Water", "target_type": "organisation", "confidence": 0.95}],
            "attachments": [{"filename": "barwon-water-annual-report.pdf", "status": "processed"}],
            "summary": (
                "Document from Barwon Water Group accessible. Leadership identified: Shaun Cumming (Chair), "
                "Jo Plummer (Chair, Barwon Water)."
            ),
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "annual_report",
                    "org_name": "Barwon Region Water Corporation",
                    "strategic_summary": "Annual report for Barwon Water.",
                    "strategic_signals": [],
                    "key_stakeholders": [
                        {"name": "Shaun Cumming", "current_role": "Chair", "current_employer": "Barwon Water"},
                        {"name": "• Des Powell", "current_role": "Barwon Asset Solutions Board", "current_employer": "Barwon Water"},
                        {
                            "name": "Barwon Region Water Corporation",
                            "current_role": "Directors’ and Chief Finance and Accounting Officer’s declaration",
                            "current_employer": "Barwon Water",
                        },
                        {
                            "name": "LINDA NIEUWENHUIZEN",
                            "current_role": "| DEPUTY CHAIR",
                            "current_employer": "I am pleased to present the Barwon Region Water Corporation",
                        },
                    ],
                    "performance_indicators": [],
                }
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
        routing={"subject_org_hint": "Barwon Water", "clean_subject": "Barwon Water annual report"},
    )

    note_content = payload["note"]["content"]
    assert "Document from Barwon Water." in note_content
    assert "Barwon Water Group accessible" not in note_content
    assert "- Des Powell | Barwon Asset Solutions Board | Barwon Water" in note_content
    assert "Barwon Region Water Corporation | Directors’ and Chief Finance and Accounting Officer’s declaration" not in note_content
    assert "- LINDA NIEUWENHUIZEN | DEPUTY CHAIR | Barwon Water" in note_content
    assert "I am pleased to present" not in note_content
    assert "LEAD INDICATORS" not in note_content


def test_mailbox_handoff_filters_seqwater_governance_artifacts_from_annual_report_note(tmp_path):
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
            "subject": "entity: Escient | Seqwater annual report",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-25T09:00:00+00:00",
            "raw_text": "",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "Seqwater", "target_type": "organisation", "confidence": 0.95}],
            "attachments": [{"filename": "seqwater-annual-report.pdf", "status": "processed"}],
            "summary": "Document from Seqwater. Leadership identified: John McEvoy (Chairperson), Emma Thomas (Chief Executive Officer).",
            "target_update_suggestions": [],
            "processing_meta": {
                "strategic_doc": {
                    "doc_type": "annual_report",
                    "org_name": "Seqwater",
                    "strategic_summary": "Annual report for Seqwater.",
                    "strategic_signals": [],
                    "key_stakeholders": [
                        {"name": "John McEvoy", "current_role": "Chairperson, Seqwater Board", "current_employer": "Seqwater"},
                        {"name": "Emma Thomas", "current_role": "Chief Executive Officer", "current_employer": "Seqwater"},
                        {"name": "Responsible Ministers", "current_role": "Board Committees", "current_employer": "Seqwater"},
                        {"name": "AND INFORMATION", "current_role": "Executive General Manager", "current_employer": "Seqwater"},
                        {"name": "Independent Auditor’s Report", "current_role": "External scrutiny", "current_employer": "Seqwater"},
                    ],
                    "performance_indicators": [],
                }
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
        routing={"subject_org_hint": "Seqwater", "clean_subject": "Seqwater annual report"},
    )

    note_content = payload["note"]["content"]
    assert "- John McEvoy | Chairperson, Seqwater Board | Seqwater" in note_content
    assert "- Emma Thomas | Chief Executive Officer | Seqwater" in note_content
    assert "Responsible Ministers" not in note_content
    assert "AND INFORMATION" not in note_content
    assert "Independent Auditor’s Report" not in note_content


def test_derive_period_label_falls_back_to_embedded_year_in_filename():
    assert derive_period_label("strategic_plan", "GVW_Strategy2035.pdf") == "2035"


def test_derive_period_label_preserves_annual_report_year_range():
    assert derive_period_label("annual_report", "Seqwater Annual Report 2024-25") == "2024-25"
    assert derive_period_label("annual_report", "Annual Report 2024-2025") == "2024-25"


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


def test_mailbox_handoff_uses_compact_provenance_for_attachment_led_signature_only_email(tmp_path):
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
            "subject": "Barwon Water annual report",
            "from_name": "Paul Cooper",
            "from_email": "paul@longboardfella.com.au",
            "received_at": "2026-03-24T08:47:21+00:00",
            "raw_text": "Dr. Paul Cooper, Ph.D\nDirector\nwww.longboardfella.com\npaul@longboardfella.com.au\nLinkedIn: https://linkedin.com/in/digitalfella",
        },
        persisted={"attachments": []},
        output_data={
            "entities": [{"name": "Barwon Water", "target_type": "organisation", "confidence": 0.95}],
            "attachments": [{"filename": "barwon-water-annual-report.pdf", "status": "processed"}],
            "summary": "Barwon Water annual report.",
            "target_update_suggestions": [],
            "processing_meta": {
                "email_triage": {
                    "processing_mode": "attachments_only",
                    "actionable_body_text": "",
                    "signature_text": "Dr. Paul Cooper, Ph.D\nDirector\nwww.longboardfella.com",
                },
                "strategic_doc": {
                    "doc_type": "annual_report",
                    "org_name": "Barwon Water",
                    "strategic_summary": "Annual report for Barwon Water.",
                    "strategic_signals": [],
                    "key_stakeholders": [],
                    "performance_indicators": [],
                },
            },
        },
        signal={},
        markdown_text="# Extract",
        message_kind="document_analysis",
    )

    assert payload["note"]["original_text"] == "Paul Cooper <paul@longboardfella.com.au> | 2026-03-24T08:47:21+00:00"
    assert "LinkedIn" not in payload["note"]["original_text"]


def test_mailbox_handoff_trims_signature_and_disclaimer_from_plain_intelligence_email(tmp_path):
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

    raw_email = "\n".join(
        [
            "Heard that Goulburn Valley water are about to come out to market (via the panel we are on) to help develop an IT strategy and roadmap.",
            "",
            "Should be out via RFP in the next fortnight.",
            "",
            "Will need to build on the plans for the billing system replacement.",
            "",
            "Rebecca Campbell-Burns (she/her)",
            "Chief Executive Officer",
            "0404 108 481",
            "escient.com.au",
            "LinkedIn",
            "—",
            "",
            "I acknowledge the Aboriginal and Torres Strait Islander peoples as the first inhabitants of Australia and the Traditional Owners and Custodians of Country where I live, learn and work.",
            "",
            "Escient encourages flexible working and as I access and support these arrangements, I am sending this message now because it suits me.",
        ]
    )

    payload = poller._build_ingest_note_payload(
        message={
            "subject": "Intel report",
            "from_name": "Rebecca Campbell-Burns",
            "from_email": "rebecca.campbell-burns@escient.com.au",
            "received_at": "2026-03-24T21:13:00+00:00",
            "raw_text": raw_email,
        },
        persisted={"attachments": []},
        output_data={
            "entities": [
                {"canonical_name": "Rebecca Campbell-Burns", "name": "Rebecca Campbell-Burns", "target_type": "person", "confidence": 0.95},
                {"canonical_name": "Escient", "name": "Escient", "target_type": "organisation", "confidence": 0.95},
                {"canonical_name": "Goulburn Valley Water", "name": "Goulburn Valley Water", "target_type": "organisation", "confidence": 0.95},
            ],
            "organisations": [{"canonical_name": "Escient"}, {"canonical_name": "Goulburn Valley Water"}],
            "attachments": [],
            "summary": "GVW is preparing an RFP for IT strategy services.",
            "target_update_suggestions": [],
            "processing_meta": {},
        },
        signal={},
        markdown_text="# Intel Extraction Result\n\n## Summary\n\nGVW is preparing an RFP for IT strategy services.",
        message_kind="document_analysis",
        routing={"effective_org_name": "Escient", "status": "matched_sender_domain", "sender_domain_org_name": "Escient"},
    )

    assert "Rebecca Campbell-Burns (she/her)" not in payload["note"]["original_text"]
    assert "I acknowledge the Aboriginal and Torres Strait Islander peoples" not in payload["note"]["original_text"]
    assert "Escient encourages flexible working" not in payload["note"]["original_text"]
    assert "Heard that Goulburn Valley water are about to come out to market" in payload["note"]["original_text"]
    assert "Will need to build on the plans for the billing system replacement." in payload["note"]["original_text"]


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
