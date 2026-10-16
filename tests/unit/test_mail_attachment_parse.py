import hashlib
import json
from pathlib import Path

from worker import mail_attachment_parse as parser


def test_attachment_is_parsed_privately_with_publication_date(tmp_path, monkeypatch):
    root = tmp_path / "intel-mailbox"
    audit_dir = root / "private"
    attach_dir = root / "attachments"
    audit_dir.mkdir(parents=True)
    attach_dir.mkdir()
    vault = tmp_path / "vault"
    original = vault / "Public_Review" / "intel" / "test.md"
    original.parent.mkdir(parents=True)
    original.write_text("---\nwiki-ready: false\n---\n\n# Test\n")
    attachment = attach_dir / "report.pdf"
    attachment.write_bytes(b"%PDF-synthetic")
    digest = hashlib.sha256(attachment.read_bytes()).hexdigest()
    audit = audit_dir / "message.json"
    audit.write_text(json.dumps({
        "vault_path": str(original), "mailbox_identity": "intel@longboardfella.com.au",
        "message": {"received_at": "2026-10-15T07:00:00Z", "saved_attachments": [
            {"name": "report.pdf", "path": str(attachment), "sha256": digest}]},
    }))
    monkeypatch.setattr(parser, "ROOTS", (root,))
    monkeypatch.setattr(parser, "PRIVATE_VAULT", vault)

    class FakeTextifier:
        def textify_file(self, _path):
            return "Publication Date: 12 September 2026\nThis is the document's substantive content."

    result = parser.parse_one(FakeTextifier())
    sidecar = Path(result["sidecar"])
    assert result["source_date"]["source_date"] == "2026-09-12"
    assert 'source_date_source: "header_publication"' in sidecar.read_text()
    assert "wiki-ready: false" in sidecar.read_text()
    assert "[[Mail_Attachments/" in original.read_text()
    assert parser.parse_one(FakeTextifier())["status"] == "idle"


def test_pdf_page_footer_date_survives_docling_footer_omission(tmp_path):
    import fitz

    path = tmp_path / "journal-2026-10-15.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((40, 60), "Article title and substantive body")
    page.insert_text((40, 770), "Journal of Examples 109, 1190-1198, July 7, 2022")
    pdf.save(path)
    pdf.close()

    date = parser._source_date("Article title and substantive body", path,
                               "2026-10-15T08:00:00Z")
    assert date == {"source_date": "2022-07-07",
                    "source_date_source": "pdf_first_page_footer",
                    "source_date_precision": "day"}
