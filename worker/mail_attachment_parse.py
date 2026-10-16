"""Parse one saved notes@/intel@ attachment into the private search vault."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cortex_engine.document_date import infer_document_date
from cortex_engine.textifier import DocumentTextifier

HOME = Path.home()
ROOTS = (HOME / "vault-rag-db/notes-mailbox", HOME / "vault-rag-db/intel-mailbox")
PRIVATE_VAULT = Path("/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive").resolve()
SUPPORTED = {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".txt", ".md"}


def _source_date(text: str, path: Path, received_at: str) -> dict[str, str]:
    result = infer_document_date(text, str(path))
    # Layout-aware PDF extraction can omit a journal's running page footer.
    # Check the original first/last page footer before using file or mail time.
    if path.suffix.lower() == ".pdf" and result.get("source_date_source") in {
        "filename", "file_created", "file_modified_fallback"
    }:
        try:
            import fitz

            with fitz.open(path) as pdf:
                for page_number, label in ((0, "pdf_first_page_footer"),
                                           (len(pdf) - 1, "pdf_last_page_footer")):
                    page = pdf[page_number]
                    footer = "\n".join(block[4] for block in page.get_text("blocks")
                                       if block[1] >= page.rect.height * 0.85)
                    lines = [line.strip() for line in footer.splitlines() if line.strip()]
                    if not lines:
                        continue
                    footer_date = infer_document_date("\n".join(lines), str(path))
                    if footer_date.get("source_date_source") in {"header", "footer",
                                                               "header_publication", "footer_publication"}:
                        return {**footer_date, "source_date_source": label}
        except (OSError, ValueError, RuntimeError):
            pass
    if result.get("source_date_source") in {"file_created", "file_modified_fallback"} and received_at[:10]:
        return {"source_date": received_at[:10], "source_date_source": "email_received_fallback",
                "source_date_precision": "day"}
    return result


def _render(parsed: str, *, name: str, digest: str, mailbox: str, original_note: Path,
            source_date: dict[str, str]) -> str:
    front = {
        "title": name, "source_type": "email-attachment", "source": mailbox,
        "original_note": str(original_note), "attachment_sha256": digest,
        **source_date,
    }
    lines = ["---", *[f"{key}: {json.dumps(str(value))}" for key, value in front.items()],
             "wiki-ready: false", "---", "", f"# {name}", "", parsed.strip(), ""]
    return "\n".join(lines)


def _parse_file(path: Path, textifier: DocumentTextifier) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    return textifier.textify_file(str(path))


def parse_one(textifier: DocumentTextifier) -> dict:
    for root in ROOTS:
        for audit in sorted((root / "private").glob("*.json")):
            record = json.loads(audit.read_text(encoding="utf-8"))
            message = record.get("message") or {}
            original_note_text = record.get("vault_path")
            if not original_note_text:
                continue
            original_note = Path(original_note_text).resolve()
            if not original_note.is_relative_to(PRIVATE_VAULT) or not original_note.is_file():
                raise RuntimeError(f"Unsafe or missing original note in {audit.name}")
            for attachment in message.get("saved_attachments") or []:
                path = Path(attachment["path"]).resolve()
                if not path.is_relative_to(root / "attachments") or not path.is_file():
                    raise RuntimeError(f"Unsafe or missing saved attachment in {audit.name}")
                if path.suffix.lower() not in SUPPORTED:
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != attachment.get("sha256"):
                    raise RuntimeError(f"Saved attachment hash changed in {audit.name}")
                stem = f"{digest[:16]}-{audit.stem[:40]}"
                sidecar = original_note.parent / "Mail_Attachments" / f"{stem}.md"
                sidecar_audit = root / "private" / f"attach-{stem}.json"
                if sidecar.is_file() and sidecar_audit.is_file():
                    continue
                parsed = _parse_file(path, textifier).strip()
                if len(parsed) < 20:
                    raise RuntimeError(f"Attachment extraction too short in {audit.name}")
                date = _source_date(parsed, path, str(message.get("received_at") or ""))
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(_render(parsed, name=attachment["name"], digest=digest,
                                           mailbox=record.get("mailbox_identity", ""),
                                           original_note=original_note, source_date=date), encoding="utf-8")
                sidecar_audit.write_text(json.dumps({"vault_path": str(sidecar),
                                                      "source_audit": audit.name,
                                                      "attachment_sha256": digest}, indent=2) + "\n",
                                         encoding="utf-8")
                sidecar_audit.chmod(0o600)
                marker = f"[[Mail_Attachments/{sidecar.stem}]]"
                parent_text = original_note.read_text(encoding="utf-8")
                if marker not in parent_text:
                    original_note.write_text(parent_text.rstrip() + "\n\n## Parsed attachment\n\n" + marker + "\n",
                                             encoding="utf-8")
                return {"status": "parsed", "source": audit.name, "sidecar": str(sidecar),
                        "source_date": date}
    return {"status": "idle"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Parse at most one attachment")
    parser.parse_args()
    textifier = DocumentTextifier(use_vision=True, pdf_strategy="docling", auto_select_vision=False)
    result = parse_one(textifier)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
