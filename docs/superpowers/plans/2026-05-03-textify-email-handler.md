# TEXTIFY / MARKDOWN email handler — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let approved Lab members email `lab@longboardfella.com.au` with `TEXTIFY` or `MARKDOWN` in the subject and one or more PDF / DOCX / PPTX / image attachments, and receive the converted text back (inline if a single short doc, otherwise as MIME attachments).

**Architecture:** Three-stage pipeline already exists end-to-end. We close three gaps: (1) the lab mailbox worker fetches attachments via Microsoft Graph and includes them in the webhook payload; (2) the PHP webhook detects the new keywords, filters attachments, and dispatches one `pdf_textify` queue job per file with a shared correlation id; (3) the existing `pdf_textify` handler also writes a plain-text sibling, and `queue_notify.php` learns a new group-aware completion branch that sends one MIME email when all siblings finish.

**Tech Stack:**
- Python 3.11 (cortex_suite worker, `requests`, `pytest`, existing `DocumentTextifier`)
- PHP 8 (longboardfella_website webhook + cron, SQLite via `SQLite3` extension, native `mail()` with multipart MIME)
- Microsoft Graph (mail polling + attachment fetch)

**Spec:** `docs/superpowers/specs/2026-05-03-textify-email-handler-design.md`

**Two repos are touched.** Each task explicitly states the repo (`cortex_suite/` or `longboardfella_website/`) and commits in the right working tree. Do **not** mix changes across repos in a single commit.

---

## File map

| Path | Repo | Change |
|---|---|---|
| `worker/handlers/pdf_textify.py` | cortex_suite | Modify — also write `.txt` (via `DocumentTextifier.markdown_to_plaintext`) when `email_textify_mode` is set; pick which file is `output_file` |
| `tests/unit/test_pdf_textify_email_modes.py` | cortex_suite | New — covers the .txt sibling and output_file selection |
| `worker/lab_mailbox_worker.py` | cortex_suite | Modify — fetch attachments via Graph, base64-encode, include in webhook payload (gated by `LAB_FETCH_ATTACHMENTS` env flag) |
| `tests/unit/test_lab_mailbox_attachments.py` | cortex_suite | New — covers Graph attachment fetch + normalisation |
| `worker/config.env.example` | cortex_suite | Modify — document `LAB_FETCH_ATTACHMENTS` |
| `site/admin/queue_api_shared.php` | longboardfella_website | Modify — add `email_correlation_id TEXT DEFAULT ''` column + index in `getQueueDB()` |
| `site/admin/email_job_ingest_lib.php` | longboardfella_website | Modify — detect `TEXTIFY` / `MARKDOWN`, run `_jilib_handleTextify()`, filter attachments, dispatch jobs, build confirmation reply |
| `site/admin/queue_notify.php` | longboardfella_website | Modify — new branch for grouped `pdf_textify` email completion (inline or MIME multipart) |
| `tests/fixtures/email_ingest/job_textify.json` | longboardfella_website | New — single PDF, plain-text mode |
| `tests/fixtures/email_ingest/job_markdown_multi.json` | longboardfella_website | New — three attachments, markdown mode |
| `tests/fixtures/email_ingest/job_textify_filtered.json` | longboardfella_website | New — XLSX + small icon + PDF, only PDF processed |
| `tests/fixtures/email_ingest/job_textify_none.json` | longboardfella_website | New — XLSX only, rejected |

---

## Task 0: Spec recap (5 min, no code)

- [ ] **Step 1: Re-read the spec end-to-end**

Open `docs/superpowers/specs/2026-05-03-textify-email-handler-design.md` and confirm:
- Subject keywords: `TEXTIFY` (plain text) or `MARKDOWN` (markdown). `MARKDOWN` wins if both present.
- Attachment filter: keep `.pdf`/`.docx`/`.pptx` (any size), images ≥ 50 KB, drop `.xlsx`/`.xls`/`.csv`/inline images/small images.
- Inline cap: 20 KB **and** single document → inline body; otherwise MIME multipart.
- Reply transport: PHP `mail()` from `noreply@longboardfella.com.au` (consistent with FULL/PLAIN/TEXT for RIP/STRIP).
- Access: open to all approved Lab members; no new capability flag.
- Rollout flag: `LAB_FETCH_ATTACHMENTS=1` in `worker/config.env`.

- [ ] **Step 2: Resolve the open questions**

Open Q1: store `email_correlation_id` as a dedicated column on `jobs`. Reason: the queue notifier needs to filter by it (`WHERE email_correlation_id != ''`) without parsing JSON for every pending textify job.

Open Q2: do **not** list filtered attachments in the confirmation reply. Reason: keeps the reply terse; users can inspect their sent folder if they care.

---

## Task 1: Database migration for `email_correlation_id`

**Repo:** longboardfella_website

**Files:**
- Modify: `site/admin/queue_api_shared.php` (around the migrations block at line ~276–287)

- [ ] **Step 1: Add the migration**

Locate the `// ── Migrations: add columns if missing ──` block in `getQueueDB()` and append, immediately after the `progress_note` migration:

```php
if (!in_array('email_correlation_id', $cols)) $db->exec("ALTER TABLE jobs ADD COLUMN email_correlation_id TEXT DEFAULT ''");
```

Then add an index (just after the existing `idx_jobs_type` line at ~260):

```php
$db->exec('CREATE INDEX IF NOT EXISTS idx_jobs_email_correlation ON jobs(email_correlation_id) WHERE email_correlation_id != \'\'');
```

- [ ] **Step 2: Verify the migration runs cleanly on the dev DB**

```bash
cd /home/longboardfella/longboardfella_website
php -r 'require "site/admin/queue_api_shared.php"; $db = getQueueDB(); $r = $db->query("PRAGMA table_info(jobs)"); while ($row = $r->fetchArray(SQLITE3_ASSOC)) echo $row["name"]."\n";' | grep email_correlation_id
```

Expected output: `email_correlation_id`

- [ ] **Step 3: Commit**

```bash
cd /home/longboardfella/longboardfella_website
git add site/admin/queue_api_shared.php
git commit -m "feat: add email_correlation_id column to jobs for grouped textify replies"
```

---

## Task 2: `pdf_textify` handler — also emit `.txt` for email mode

**Repo:** cortex_suite

**Files:**
- Modify: `worker/handlers/pdf_textify.py`
- Create: `tests/unit/test_pdf_textify_email_modes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_pdf_textify_email_modes.py` with:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from worker.handlers import pdf_textify


@pytest.fixture
def fake_input(tmp_path: Path) -> Path:
    p = tmp_path / "report.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def _patch_textifier(markdown: str = "# Title\n\nBody.\n"):
    """Patch DocumentTextifier so we don't actually run Docling."""
    class _FakeTextifier:
        @classmethod
        def from_options(cls, options, on_progress=None):
            return cls()

        def textify_file(self, _path):
            return markdown

    return patch.object(pdf_textify, "DocumentTextifier", _FakeTextifier)


def test_legacy_invocation_writes_only_md(fake_input):
    """Without email_textify_mode, behaviour matches the original handler."""
    with _patch_textifier():
        result = pdf_textify.handle(
            input_path=fake_input,
            input_data={"textify_options": {}},
            job={"id": 1},
        )
    assert result["output_file"].suffix == ".md"
    assert result["output_file"].exists()
    txt_path = fake_input.with_suffix(".txt")
    assert not txt_path.exists()


def test_email_text_mode_writes_both_and_returns_txt(fake_input):
    with _patch_textifier(markdown="# Heading\n\n- bullet\n"):
        result = pdf_textify.handle(
            input_path=fake_input,
            input_data={
                "textify_options": {},
                "email_textify_mode": "text",
            },
            job={"id": 2},
        )
    assert result["output_file"].suffix == ".txt"
    md_path = fake_input.with_suffix(".md")
    txt_path = fake_input.with_suffix(".txt")
    assert md_path.exists() and txt_path.exists()
    txt_content = txt_path.read_text(encoding="utf-8")
    # markdown_to_plaintext strips the leading '# '
    assert "Heading" in txt_content
    assert "#" not in txt_content.splitlines()[0]
    assert result["output_data"]["alt_output_file"].suffix == ".md"


def test_email_markdown_mode_writes_both_and_returns_md(fake_input):
    with _patch_textifier(markdown="# Heading\n"):
        result = pdf_textify.handle(
            input_path=fake_input,
            input_data={
                "textify_options": {},
                "email_textify_mode": "markdown",
            },
            job={"id": 3},
        )
    assert result["output_file"].suffix == ".md"
    md_path = fake_input.with_suffix(".md")
    txt_path = fake_input.with_suffix(".txt")
    assert md_path.exists() and txt_path.exists()
    assert result["output_data"]["alt_output_file"].suffix == ".txt"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd /home/longboardfella/cortex_suite
source venv/bin/activate
pytest tests/unit/test_pdf_textify_email_modes.py -v
```

Expected: `test_email_text_mode_writes_both_and_returns_txt` and `test_email_markdown_mode_writes_both_and_returns_md` fail (they expect `.txt` and `alt_output_file` in `output_data` that the current handler doesn't produce). `test_legacy_invocation_writes_only_md` should pass since it asserts current behaviour.

- [ ] **Step 3: Modify the handler to emit both formats when in email mode**

Replace the section of `worker/handlers/pdf_textify.py` from the existing `# Write output .md file alongside the input` comment through the end of `output_data` construction (current lines 68–88) with:

```python
    # Determine which output(s) to write
    email_mode = str((input_data or {}).get("email_textify_mode") or "").strip().lower()

    md_filename = input_path.stem + ".md"
    md_path = input_path.parent / md_filename
    md_path.write_text(markdown_text, encoding="utf-8")

    txt_path = None
    txt_filename = None
    if email_mode in {"text", "markdown"}:
        txt_filename = input_path.stem + ".txt"
        txt_path = input_path.parent / txt_filename
        plaintext = DocumentTextifier.markdown_to_plaintext(markdown_text, width=80)
        txt_path.write_text(plaintext, encoding="utf-8")

    if email_mode == "text" and txt_path is not None:
        primary_path = txt_path
        primary_filename = txt_filename
        alt_path = md_path
    else:
        primary_path = md_path
        primary_filename = md_filename
        alt_path = txt_path  # may be None when email_mode is empty

    # Stats are computed against the markdown (more meaningful than against plain text)
    line_count = len(markdown_text.splitlines())
    table_count = markdown_text.count("|---")
    heading_count = sum(1 for line in markdown_text.splitlines() if line.startswith("#"))

    output_data = {
        "summary": "Converted via cortex_engine.textifier.DocumentTextifier",
        "source_filename": input_path.name,
        "output_filename": primary_filename,
        "markdown_length": len(markdown_text),
        "line_count": line_count,
        "headings_found": heading_count,
        "tables_found": table_count,
        "use_vision": use_vision,
        "pdf_strategy": mode,
        "email_textify_mode": email_mode or "",
    }
    if alt_path is not None:
        output_data["alt_output_file"] = alt_path

    logger.info(
        "Textified %s -> %s (%d lines, %d headings, %d tables, mode=%s)",
        input_path.name,
        primary_filename,
        line_count,
        heading_count,
        table_count,
        email_mode or "default",
    )
    if progress_cb:
        progress_cb(100, "Textification complete", "done")

    return {"output_data": output_data, "output_file": primary_path}
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
cd /home/longboardfella/cortex_suite
source venv/bin/activate
pytest tests/unit/test_pdf_textify_email_modes.py -v
```

Expected: all three tests pass.

- [ ] **Step 5: Run the full handler-related test suite as a regression check**

```bash
pytest tests/unit/test_handoff_contract_validation.py -v
```

Expected: no failures (we didn't touch the handoff contract, but `pdf_textify`'s validator is the gateway and we want to be sure the new keys don't break anything).

- [ ] **Step 6: Commit**

```bash
cd /home/longboardfella/cortex_suite
git add worker/handlers/pdf_textify.py tests/unit/test_pdf_textify_email_modes.py
git commit -m "feat(textify): emit .txt sibling and select primary output by email_textify_mode"
```

---

## Task 3: Lab mailbox worker — fetch attachments via Graph

**Repo:** cortex_suite

**Files:**
- Modify: `worker/lab_mailbox_worker.py`
- Create: `tests/unit/test_lab_mailbox_attachments.py`
- Modify: `worker/config.env.example`

- [ ] **Step 1: Document the new env var**

Append to `worker/config.env.example`:

```ini

# Lab mailbox: fetch attachments via Graph and include in webhook payload.
# Set to 1 to enable TEXTIFY/MARKDOWN inbound conversion.
LAB_FETCH_ATTACHMENTS=0
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_lab_mailbox_attachments.py`:

```python
from __future__ import annotations

import base64
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
        state_path=__import__("pathlib").Path("/tmp/_unused.json"),
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
    fake_client = MagicMock()
    fake_client.list_attachments.return_value = [
        {
            "name": "report.pdf",
            "contentType": "application/pdf",
            "size": 12345,
            "isInline": False,
            "contentBytes": base64.b64encode(b"%PDF-1.4 fake").decode(),
        },
        {
            "name": "logo.png",
            "contentType": "image/png",
            "size": 1024,
            "isInline": True,
            "contentBytes": base64.b64encode(b"\x89PNGfake").decode(),
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
        "content_base64": base64.b64encode(b"%PDF-1.4 fake").decode(),
    }
    assert png["is_inline"] is True
    assert png["filename"] == "logo.png"


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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/longboardfella/cortex_suite
source venv/bin/activate
pytest tests/unit/test_lab_mailbox_attachments.py -v
```

Expected: `ImportError`/`TypeError` failures because `LabMailboxConfig` does not yet have `fetch_attachments`, `LabGraphClient.list_attachments` does not exist, and `normalize_graph_message` does not accept `graph_client`/`fetch`.

- [ ] **Step 4: Add `fetch_attachments` to the config dataclass**

In `worker/lab_mailbox_worker.py`, modify the `LabMailboxConfig` dataclass — add a new field after `suppress_replies` and a corresponding getter in `from_env`:

```python
@dataclass
class LabMailboxConfig:
    transport_mode: str
    poll_interval: int
    graph_tenant_id: str
    graph_client_id: str
    graph_client_secret: str
    graph_mailbox: str
    graph_page_size: int
    graph_timeout: int
    webhook_url: str
    webhook_secret: str
    suppress_replies: bool
    state_path: Path
    fetch_attachments: bool = False
```

In `from_env`, add to the constructor call (after `suppress_replies=...,`):

```python
            fetch_attachments=get_bool("LAB_FETCH_ATTACHMENTS", "0"),
```

- [ ] **Step 5: Add `LabGraphClient.list_attachments`**

In `worker/lab_mailbox_worker.py`, inside the `LabGraphClient` class, add this method after `list_unread_messages`:

```python
    def list_attachments(self, message_id: str) -> list[dict]:
        """Fetch all attachments for a Graph message.

        Returns the raw Graph attachment objects (each with name, contentType,
        size, isInline, contentBytes). Caller filters/normalises.
        """
        if not message_id:
            return []
        response = requests.get(
            f"https://graph.microsoft.com/v1.0/users/{self.config.graph_mailbox}/messages/{message_id}/attachments",
            headers=self._headers(),
            params={"$select": "id,name,contentType,size,isInline,contentBytes"},
            timeout=self.config.graph_timeout,
        )
        response.raise_for_status()
        return list(response.json().get("value") or [])
```

- [ ] **Step 6: Update `normalize_graph_message` signature and body**

Replace the entire `normalize_graph_message` function in `worker/lab_mailbox_worker.py` with:

```python
def normalize_graph_message(item: dict, graph_client: "LabGraphClient | None" = None, fetch: bool = False) -> dict:
    from_name, from_email = email_address(dict(item.get("from") or {}))
    recipients = list(item.get("toRecipients") or [])
    to_name, to_email = email_address(dict(recipients[0] if recipients else {}))
    body = dict(item.get("body") or {})
    content = str(body.get("content") or "")
    content_type = str(body.get("contentType") or "").strip().lower()
    text_body = strip_html(content) if content_type == "html" else content.strip()
    html_body = content if content_type == "html" else ""

    attachments: list[dict] = []
    if fetch and graph_client is not None and item.get("hasAttachments"):
        graph_id = str(item.get("id") or "").strip()
        try:
            raw = graph_client.list_attachments(graph_id)
        except Exception:
            logging.exception("Failed to fetch attachments for message %s", graph_id)
            raw = []
        for att in raw:
            # Skip itemAttachment and referenceAttachment — only handle fileAttachment
            if str(att.get("@odata.type", "")).endswith("fileAttachment"):
                pass  # explicit pass for readability
            elif att.get("@odata.type") not in (None, "") and "fileAttachment" not in str(att.get("@odata.type", "")):
                continue
            attachments.append({
                "filename": str(att.get("name") or "").strip(),
                "mime_type": str(att.get("contentType") or "").strip(),
                "size_bytes": int(att.get("size") or 0),
                "is_inline": bool(att.get("isInline")),
                "content_base64": str(att.get("contentBytes") or ""),
            })

    return {
        "message_id": str(item.get("internetMessageId") or item.get("id") or "").strip(),
        "graph_message_id": str(item.get("id") or "").strip(),
        "from_name": from_name,
        "from_email": from_email,
        "to_name": to_name,
        "to_email": to_email,
        "subject": str(item.get("subject") or "").strip(),
        "received_at": str(item.get("receivedDateTime") or datetime.now(timezone.utc).isoformat()).strip(),
        "text_body": text_body,
        "html_body": html_body,
        "attachments": attachments,
    }
```

- [ ] **Step 7: Update `poll_once` to thread the graph client + flag through**

In `worker/lab_mailbox_worker.py`, replace the `for item in graph.list_unread_messages():` loop with:

```python
    for item in graph.list_unread_messages():
        message = normalize_graph_message(
            item,
            graph_client=graph,
            fetch=config.fetch_attachments,
        )
        graph_id = str(message.get("graph_message_id") or "").strip()
```

(Leave the rest of the loop body untouched.)

- [ ] **Step 8: Run the tests to confirm they pass**

```bash
cd /home/longboardfella/cortex_suite
source venv/bin/activate
pytest tests/unit/test_lab_mailbox_attachments.py -v
```

Expected: all three tests pass.

- [ ] **Step 9: Smoke-check the worker still imports**

```bash
python -c "from worker import lab_mailbox_worker; print(lab_mailbox_worker.LabMailboxConfig)"
```

Expected: prints `<class 'worker.lab_mailbox_worker.LabMailboxConfig'>` with no errors.

- [ ] **Step 10: Commit**

```bash
cd /home/longboardfella/cortex_suite
git add worker/lab_mailbox_worker.py worker/config.env.example tests/unit/test_lab_mailbox_attachments.py
git commit -m "feat(lab-worker): fetch Graph attachments behind LAB_FETCH_ATTACHMENTS flag"
```

---

## Task 4: Webhook — `_jilib_handleTextify` keyword router and dispatcher

**Repo:** longboardfella_website

**Files:**
- Modify: `site/admin/email_job_ingest_lib.php`
- Create: 4 fixture JSON files in `tests/fixtures/email_ingest/`

- [ ] **Step 1: Create the fixtures**

Each is a minimal normalized-payload JSON. **Replace `<base64-of-N-bytes>` placeholders** with `python -c "import base64; print(base64.b64encode(b'A'*N).decode())"` (any valid base64 of approximately the right size — actual content is irrelevant for the routing/dispatch test under `--dry-run`).

Create `tests/fixtures/email_ingest/job_textify.json`:

```json
{
  "message_id": "<fixture-textify-001@example.com>",
  "from_email": "paul@longboardfella.com.au",
  "from_name": "Paul Cooper",
  "to_email": "lab@longboardfella.com.au",
  "subject": "TEXTIFY",
  "received_at": "2026-05-03T10:00:00Z",
  "text_body": "",
  "html_body": "",
  "attachments": [
    {
      "filename": "report.pdf",
      "mime_type": "application/pdf",
      "size_bytes": 102400,
      "is_inline": false,
      "content_base64": "JVBERi0xLjQgZmFrZQ=="
    }
  ],
  "transport_meta": {"transport": "fixture"}
}
```

Create `tests/fixtures/email_ingest/job_markdown_multi.json`:

```json
{
  "message_id": "<fixture-textify-multi-001@example.com>",
  "from_email": "paul@longboardfella.com.au",
  "from_name": "Paul Cooper",
  "to_email": "lab@longboardfella.com.au",
  "subject": "JOB: MARKDOWN",
  "received_at": "2026-05-03T10:05:00Z",
  "text_body": "",
  "html_body": "",
  "attachments": [
    {"filename": "a.pdf",  "mime_type": "application/pdf", "size_bytes": 51200,  "is_inline": false, "content_base64": "JVBERi0xLjQgZmFrZQ=="},
    {"filename": "b.docx", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size_bytes": 51200, "is_inline": false, "content_base64": "UEsDBBQAAAA="},
    {"filename": "c.pptx", "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "size_bytes": 81920, "is_inline": false, "content_base64": "UEsDBBQAAAA="}
  ],
  "transport_meta": {"transport": "fixture"}
}
```

Create `tests/fixtures/email_ingest/job_textify_filtered.json`:

```json
{
  "message_id": "<fixture-textify-filter-001@example.com>",
  "from_email": "paul@longboardfella.com.au",
  "from_name": "Paul Cooper",
  "to_email": "lab@longboardfella.com.au",
  "subject": "Textify please",
  "received_at": "2026-05-03T10:10:00Z",
  "text_body": "",
  "html_body": "",
  "attachments": [
    {"filename": "data.xlsx",  "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "size_bytes": 250000, "is_inline": false, "content_base64": "UEsDBBQAAAA="},
    {"filename": "logo.png",   "mime_type": "image/png", "size_bytes": 4096,    "is_inline": true,  "content_base64": "iVBORw0KGgo="},
    {"filename": "icon.png",   "mime_type": "image/png", "size_bytes": 8192,    "is_inline": false, "content_base64": "iVBORw0KGgo="},
    {"filename": "report.pdf", "mime_type": "application/pdf", "size_bytes": 200000, "is_inline": false, "content_base64": "JVBERi0xLjQgZmFrZQ=="}
  ],
  "transport_meta": {"transport": "fixture"}
}
```

Create `tests/fixtures/email_ingest/job_textify_none.json`:

```json
{
  "message_id": "<fixture-textify-none-001@example.com>",
  "from_email": "paul@longboardfella.com.au",
  "from_name": "Paul Cooper",
  "to_email": "lab@longboardfella.com.au",
  "subject": "MARKDOWN",
  "received_at": "2026-05-03T10:15:00Z",
  "text_body": "",
  "html_body": "",
  "attachments": [
    {"filename": "data.xlsx", "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "size_bytes": 250000, "is_inline": false, "content_base64": "UEsDBBQAAAA="}
  ],
  "transport_meta": {"transport": "fixture"}
}
```

- [ ] **Step 2: Run replay against all four fixtures and confirm none route to TEXTIFY today**

```bash
cd /home/longboardfella/longboardfella_website
for f in tests/fixtures/email_ingest/job_textify.json tests/fixtures/email_ingest/job_markdown_multi.json tests/fixtures/email_ingest/job_textify_filtered.json tests/fixtures/email_ingest/job_textify_none.json; do
  echo "── $f ──"
  php scripts/replay_email_ingest.php job "$f" --dry-run --suppress-replies | head -30
done
```

Expected: each shows `outcome: skipped` or `outcome: parse_failed` (none recognise the new keywords yet). The first fixture (`TEXTIFY` alone) parses as `skipped` because the subject doesn't start with `JOB:` and the flex parser doesn't recognise `TEXTIFY`.

- [ ] **Step 3: Add the keyword detection + dispatch handler**

Open `site/admin/email_job_ingest_lib.php`. Locate the section starting `// ── DISPATCH / APPEND handling (before normal job parsing) ──` (currently around line 159). **Immediately before** that block (still after dedupe + sender verification), add:

```php
    // ── TEXTIFY / MARKDOWN handling (before DISPATCH/APPEND and JOB parsing) ──
    $textifyMode = _jilib_detectTextifyMode($subject);
    if ($textifyMode !== null) {
        $r = _jilib_handleTextify(
            $fromEmail,
            $textifyMode,
            (array)($message['attachments'] ?? []),
            $textBody,
            $db,
            $messageId,
            $dryRun,
            $suppressReplies
        );
        $db->close();
        foreach ($r['log'] as $l) { $log($l); }
        $result['outcome'] = $r['outcome'];
        $result['error']   = $r['error'];
        $result['reply']   = $r['reply'];
        $result['job_ids'] = $r['job_ids'];
        return $result;
    }
```

Note: `$message['attachments']` is fetched from the **original** $message argument, not from any local var. If the local body parsing has shadowed it, use `$message` directly — confirm by reading the surrounding code.

- [ ] **Step 4: Add the helper functions at the bottom of the file (before the closing `?>` if any)**

Append to `site/admin/email_job_ingest_lib.php`:

```php
// ─────────────────────────────────────────────
// TEXTIFY / MARKDOWN handler
// ─────────────────────────────────────────────

const TEXTIFY_PROCESSABLE_DOC_EXTS   = ['pdf', 'docx', 'pptx'];
const TEXTIFY_PROCESSABLE_IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'webp', 'tif', 'tiff'];
const TEXTIFY_SKIPPED_EXTS           = ['xlsx', 'xls', 'csv'];
const TEXTIFY_IMAGE_MIN_BYTES        = 50000;

/**
 * Returns 'text', 'markdown', or null. MARKDOWN wins if both are present.
 */
function _jilib_detectTextifyMode(string $subject): ?string {
    $hasMd = (bool)preg_match('/\bMARKDOWN\b/i', $subject);
    $hasTx = (bool)preg_match('/\bTEXTIFY\b/i', $subject);
    if ($hasMd) return 'markdown';
    if ($hasTx) return 'text';
    return null;
}

/**
 * Decide which attachments to keep for textification.
 * Returns ['kept' => [...], 'ignored' => [...]] where each item is the original attachment array.
 */
function _jilib_filterTextifyAttachments(array $attachments): array {
    $kept = [];
    $ignored = [];
    foreach ($attachments as $a) {
        if (!is_array($a)) { continue; }
        $name = strtolower((string)($a['filename'] ?? ''));
        $isInline = !empty($a['is_inline']);
        $size = (int)($a['size_bytes'] ?? 0);
        $ext = '';
        if (preg_match('/\.([a-z0-9]+)$/i', $name, $m)) { $ext = strtolower($m[1]); }

        if ($isInline) { $ignored[] = $a; continue; }
        if (in_array($ext, TEXTIFY_SKIPPED_EXTS, true)) { $ignored[] = $a; continue; }
        if (in_array($ext, TEXTIFY_PROCESSABLE_DOC_EXTS, true)) { $kept[] = $a; continue; }
        if (in_array($ext, TEXTIFY_PROCESSABLE_IMAGE_EXTS, true)) {
            if ($size >= TEXTIFY_IMAGE_MIN_BYTES) { $kept[] = $a; } else { $ignored[] = $a; }
            continue;
        }
        $ignored[] = $a;
    }
    return ['kept' => $kept, 'ignored' => $ignored];
}

/**
 * Decode the attachment payload and write it to queue_files/inputs/<jobId>_<safe_name>.
 * Returns the input_filename (basename) on success or null on failure.
 */
function _jilib_writeTextifyInput(int $jobId, array $attachment): ?string {
    global $QUEUE_FILES_DIR;
    $rawB64 = (string)($attachment['content_base64'] ?? '');
    if ($rawB64 === '') return null;
    $decoded = base64_decode($rawB64, true);
    if ($decoded === false) return null;

    ensureQueueDirs();
    $safe = preg_replace('/[^a-zA-Z0-9._-]/', '_', (string)($attachment['filename'] ?? 'file.bin'));
    if ($safe === '' || $safe === '.' || $safe === '..') $safe = 'file.bin';
    $inputFilename = $jobId . '_' . $safe;
    $dest = $QUEUE_FILES_DIR . '/inputs/' . $inputFilename;
    $bytes = file_put_contents($dest, $decoded);
    if ($bytes === false) return null;
    return $inputFilename;
}

function _jilib_handleTextify(
    string $fromEmail,
    string $textifyMode,
    array $attachments,
    string $textBody,
    SQLite3 $logDb,
    string $messageId,
    bool $dryRun,
    bool $suppressReplies
): array {
    $out = ['outcome' => 'error', 'error' => '', 'reply' => null, 'job_ids' => [], 'log' => []];
    $modeLabel = $textifyMode === 'text' ? 'plain text' : 'markdown';
    $subjectForReply = $textifyMode === 'text' ? 'TEXTIFY' : 'MARKDOWN';

    $filtered = _jilib_filterTextifyAttachments($attachments);
    $kept = $filtered['kept'];
    $ignored = $filtered['ignored'];

    $out['log'][] = "  → Textify keyword detected (mode={$textifyMode}); " . count($kept) . " kept, " . count($ignored) . " ignored";

    if (empty($kept)) {
        _jilib_logIngest($fromEmail, $subjectForReply, 'no_attachments', [], 'No convertible attachments', $messageId, $logDb);
        $out['outcome'] = 'no_attachments';
        $out['error']   = 'No convertible attachments';
        if (!$suppressReplies) {
            $out['reply'] = _jilib_buildReply($fromEmail, $subjectForReply, false,
                "No convertible documents were found.\n\n"
                . "Supported: PDF, DOCX, PPTX, and images >= 50 KB.\n"
                . "Excel files and small inline icons are ignored.\n\n"
                . "No job submitted.");
        }
        return $out;
    }

    if ($dryRun) {
        $names = array_map(fn($a) => (string)($a['filename'] ?? '?'), $kept);
        $out['log'][] = "  → [DRY RUN] would queue " . count($kept) . " textify job(s) for: " . implode(', ', $names);
        $out['outcome'] = 'dry_run';
        return $out;
    }

    // Optional SENDTO: parameter (parse from text body)
    $sendTo = '';
    foreach (explode("\n", $textBody) as $line) {
        if (preg_match('/^\s*(SENDTO|SEND\s*TO|FORWARD)\s*:\s*(\S+)/i', $line, $m)) {
            $candidate = trim($m[2]);
            if (filter_var($candidate, FILTER_VALIDATE_EMAIL)) { $sendTo = $candidate; }
            break;
        }
    }

    // Generate correlation id (UUID v4)
    $correlationId = sprintf('%04x%04x-%04x-%04x-%04x-%04x%04x%04x',
        random_int(0, 0xffff), random_int(0, 0xffff),
        random_int(0, 0xffff),
        random_int(0, 0x0fff) | 0x4000,
        random_int(0, 0x3fff) | 0x8000,
        random_int(0, 0xffff), random_int(0, 0xffff), random_int(0, 0xffff)
    );

    $qdb = getQueueDB();
    $now = gmdate('Y-m-d H:i:s');
    $jobIds = [];
    $writtenNames = [];
    $writeFailures = [];

    try {
        foreach ($kept as $att) {
            $inputData = json_encode([
                'original_filename'        => (string)($att['filename'] ?? 'file.bin'),
                'textify_options'          => ['use_vision' => true, 'add_metadata_preface' => true],
                'textify_advanced'         => ['pdf_strategy' => 'hybrid'],
                'email_textify_mode'       => $textifyMode,
                'email_correlation_id'     => $correlationId,
                'email_correlation_count'  => count($kept),
                'source_system'            => 'email',
                'send_to'                  => $sendTo,
            ], JSON_INVALID_UTF8_SUBSTITUTE);

            $traceId = generateTraceId();
            $stmt = $qdb->prepare(
                "INSERT INTO jobs (type, status, input_data, created_at, created_by, submitter_email, trace_id, schedule_type, notify_status, email_correlation_id) "
                . "VALUES ('pdf_textify', 'pending', :data, :now, :by, :email, :trace, 'anytime', 'pending', :corr)"
            );
            $stmt->bindValue(':data',  $inputData,    SQLITE3_TEXT);
            $stmt->bindValue(':now',   $now,          SQLITE3_TEXT);
            $stmt->bindValue(':by',    $fromEmail,    SQLITE3_TEXT);
            $stmt->bindValue(':email', $fromEmail,    SQLITE3_TEXT);
            $stmt->bindValue(':trace', $traceId,      SQLITE3_TEXT);
            $stmt->bindValue(':corr',  $correlationId, SQLITE3_TEXT);
            $stmt->execute();
            $jobId = (int)$qdb->lastInsertRowID();

            $inputFilename = _jilib_writeTextifyInput($jobId, $att);
            if ($inputFilename === null) {
                $writeFailures[] = (string)($att['filename'] ?? '?');
                $del = $qdb->prepare("DELETE FROM jobs WHERE id = :id");
                $del->bindValue(':id', $jobId, SQLITE3_INTEGER);
                $del->execute();
                continue;
            }

            $upd = $qdb->prepare("UPDATE jobs SET input_filename = :fn WHERE id = :id");
            $upd->bindValue(':fn', $inputFilename, SQLITE3_TEXT);
            $upd->bindValue(':id', $jobId, SQLITE3_INTEGER);
            $upd->execute();

            $jobIds[] = $jobId;
            $writtenNames[] = (string)($att['filename'] ?? '?');
        }
    } catch (Throwable $e) {
        $qdb->close();
        _jilib_logIngest($fromEmail, $subjectForReply, 'error', $jobIds, 'Queue insert failed: ' . $e->getMessage(), $messageId, $logDb);
        $out['outcome'] = 'error';
        $out['error']   = 'Queue insert failed: ' . $e->getMessage();
        $out['job_ids'] = $jobIds;
        $out['log'][]   = "  → QUEUE ERROR: " . $e->getMessage();
        return $out;
    }
    $qdb->close();

    if (empty($jobIds)) {
        _jilib_logIngest($fromEmail, $subjectForReply, 'error', [], 'All attachment writes failed', $messageId, $logDb);
        $out['outcome'] = 'error';
        $out['error']   = 'All attachment writes failed';
        if (!$suppressReplies) {
            $out['reply'] = _jilib_buildReply($fromEmail, $subjectForReply, false,
                "Could not save any of your attachments. Please retry, or contact Paul if it keeps happening.");
        }
        return $out;
    }

    _jilib_logIngest($fromEmail, $subjectForReply, 'accepted', $jobIds, '', $messageId, $logDb);
    $out['log'][] = "  → Inserted " . count($jobIds) . " textify job(s): " . implode(', ', $jobIds);
    $out['outcome'] = 'accepted';
    $out['job_ids'] = $jobIds;

    if (!$suppressReplies) {
        $idsStr = implode(', ', array_map(fn($id) => "#{$id}", $jobIds));
        $msg  = "✅ Textify job(s) submitted.\n\n";
        $msg .= "Mode:    {$modeLabel}\n";
        $msg .= "Files:   " . implode(', ', $writtenNames) . "\n";
        $msg .= "Job IDs: {$idsStr}\n";
        if ($sendTo) $msg .= "Send to: {$sendTo} (result will also be forwarded)\n";
        if (!empty($writeFailures)) {
            $msg .= "\n⚠️  Could not save: " . implode(', ', $writeFailures) . "\n";
        }
        $msg .= "\nYou'll receive a single reply with the converted result(s) when all jobs are complete.\n";
        $msg .= "To cancel: reply with subject CANCEL #{$jobIds[0]}";
        if (count($jobIds) > 1) {
            $lastId = end($jobIds);
            $msg .= " (or any sibling id, e.g. #{$lastId})";
        }
        $msg .= "\nView queue: https://longboardfella.com.au/lab/queue.php";
        $out['reply'] = _jilib_buildReply($fromEmail, $subjectForReply, true, $msg);
    }
    return $out;
}
```

- [ ] **Step 5: Re-run the four fixtures with `--dry-run`**

```bash
cd /home/longboardfella/longboardfella_website
for f in tests/fixtures/email_ingest/job_textify.json tests/fixtures/email_ingest/job_markdown_multi.json tests/fixtures/email_ingest/job_textify_filtered.json tests/fixtures/email_ingest/job_textify_none.json; do
  echo "── $f ──"
  php scripts/replay_email_ingest.php job "$f" --dry-run --suppress-replies
  echo
done
```

Expected:
- `job_textify.json` → `outcome: dry_run`, log shows `1 kept, 0 ignored`.
- `job_markdown_multi.json` → `outcome: dry_run`, log shows `3 kept, 0 ignored`.
- `job_textify_filtered.json` → `outcome: dry_run`, log shows `1 kept, 3 ignored`.
- `job_textify_none.json` → `outcome: no_attachments`.

- [ ] **Step 6: Run the existing email-ingest fixtures to confirm no regression**

```bash
cd /home/longboardfella/longboardfella_website
for f in tests/fixtures/email_ingest/job_url_ingest.json tests/fixtures/email_ingest/job_flex_youtube.json tests/fixtures/email_ingest/job_help.json tests/fixtures/email_ingest/job_noreply_skip.json; do
  echo "── $f ──"
  php scripts/replay_email_ingest.php job "$f" --dry-run --suppress-replies | head -8
done
```

Expected: outcomes match what was there before our changes (`dry_run`, `dry_run`, `help`, `skipped` respectively). No errors.

- [ ] **Step 7: Real-DB test — queue a single textify job and verify rows + file**

```bash
cd /home/longboardfella/longboardfella_website
php scripts/replay_email_ingest.php job tests/fixtures/email_ingest/job_textify.json --suppress-replies
sqlite3 site/admin/queue.db "SELECT id, type, email_correlation_id, input_filename FROM jobs WHERE type='pdf_textify' ORDER BY id DESC LIMIT 1;"
ls -la site/admin/queue_files/inputs/ | tail -3
```

Expected: a new `pdf_textify` row with a non-empty `email_correlation_id` and `input_filename`; the corresponding `<jobId>_report.pdf` file exists in `site/admin/queue_files/inputs/`.

**Cleanup after this real-DB test** (the test file is fake PDF bytes and shouldn't actually run through the worker):

```bash
sqlite3 site/admin/queue.db "DELETE FROM jobs WHERE type='pdf_textify' AND created_by='paul@longboardfella.com.au' AND input_data LIKE '%fixture-textify-001%' OR (type='pdf_textify' AND email_correlation_id != '' AND status='pending' AND created_at > datetime('now','-1 hour'));"
rm -f site/admin/queue_files/inputs/*_report.pdf
```

(Better: scope the DELETE to a known marker. Adjust the WHERE clause to match the rows you just inserted.)

- [ ] **Step 8: Commit**

```bash
cd /home/longboardfella/longboardfella_website
git add site/admin/email_job_ingest_lib.php tests/fixtures/email_ingest/job_textify.json tests/fixtures/email_ingest/job_markdown_multi.json tests/fixtures/email_ingest/job_textify_filtered.json tests/fixtures/email_ingest/job_textify_none.json
git commit -m "feat(lab-ingest): TEXTIFY/MARKDOWN keyword routing with attachment dispatch"
```

---

## Task 5: Queue notifier — group-aware textify completion email

**Repo:** longboardfella_website

**Files:**
- Modify: `site/admin/queue_notify.php`

This is the largest task — it adds a new branch ahead of the existing notify logic. We can't reasonably TDD PHP cron code without a full unit-test harness, so we verify by seeding rows in the queue DB and running the notifier with stubbed `mail()`.

- [ ] **Step 1: Add a stubbable mail send function**

Open `site/admin/queue_notify.php`. Near the top, after the `$adminEmail`/`$fromEmail`/`$siteUrl` constants (currently around line 25–28), add:

```php
// Allow tests / replays to capture mail without sending. Set
// QUEUE_NOTIFY_MAIL_SINK=path  to log every send to that file.
function _qn_send_mail(string $to, string $subject, string $body, string $headers): bool {
    $sink = getenv('QUEUE_NOTIFY_MAIL_SINK');
    if ($sink) {
        $entry = "── MAIL ──\nTo: {$to}\nSubject: {$subject}\nHeaders: {$headers}\n\n{$body}\n";
        file_put_contents($sink, $entry, FILE_APPEND | LOCK_EX);
        return true;
    }
    return @mail($to, $subject, $body, $headers);
}
```

Then **find every `@mail(` call in this file** and replace each with `_qn_send_mail(`. There are roughly 4–6 of them (success / failure notifications + the `send_to` forward). Leave the argument lists exactly as-is. Verify with:

```bash
grep -n "@mail(" site/admin/queue_notify.php
```

Expected after the edits: zero matches.

- [ ] **Step 2: Add the textify group-completion branch**

In `site/admin/queue_notify.php`, locate `// ── Process completed jobs needing notification ──` (currently around line 61). **Immediately before** that section's `$stmt = $db->prepare(...)`, insert this entire block:

```php
// ── Textify group completion (one combined email per email_correlation_id) ──
$INLINE_BYTES_CAP   = 20000;
$ATTACH_TOTAL_CAP   = 20 * 1024 * 1024;

$gstmt = $db->prepare("
    SELECT email_correlation_id
    FROM jobs
    WHERE type='pdf_textify'
      AND email_correlation_id != ''
      AND notify_status='pending'
    GROUP BY email_correlation_id
");
$gres = $gstmt->execute();

$readyGroups = [];
while ($row = $gres->fetchArray(SQLITE3_ASSOC)) {
    $cid = (string)$row['email_correlation_id'];
    if ($cid === '') continue;

    $check = $db->prepare("
        SELECT
          SUM(CASE WHEN status IN ('completed','failed','cancelled') THEN 1 ELSE 0 END) AS done,
          COUNT(*) AS total
        FROM jobs WHERE email_correlation_id = :cid
    ");
    $check->bindValue(':cid', $cid, SQLITE3_TEXT);
    $r = $check->execute()->fetchArray(SQLITE3_ASSOC);
    if ((int)$r['done'] === (int)$r['total'] && (int)$r['total'] > 0) {
        $readyGroups[] = $cid;
    }
}

foreach ($readyGroups as $cid) {
    $sib = $db->prepare("
        SELECT id, status, input_data, output_filename, output_data, error_message, submitter_email, completed_at
        FROM jobs
        WHERE email_correlation_id = :cid
        ORDER BY id ASC
    ");
    $sib->bindValue(':cid', $cid, SQLITE3_TEXT);
    $sibRes = $sib->execute();

    $siblings = [];
    $email = '';
    $sendTo = '';
    $mode = 'markdown';
    while ($srow = $sibRes->fetchArray(SQLITE3_ASSOC)) {
        $inp = json_decode($srow['input_data'] ?: '{}', true) ?: [];
        $out = json_decode($srow['output_data'] ?: '{}', true) ?: [];
        if (!$email && !empty($srow['submitter_email'])) $email = (string)$srow['submitter_email'];
        if (!$sendTo && !empty($inp['send_to'])) $sendTo = (string)$inp['send_to'];
        if (!empty($inp['email_textify_mode'])) $mode = (string)$inp['email_textify_mode'];
        $siblings[] = [
            'id'                => (int)$srow['id'],
            'status'            => (string)$srow['status'],
            'original_filename' => (string)($inp['original_filename'] ?? ('job_' . $srow['id'])),
            'output_filename'   => (string)$srow['output_filename'],
            'alt_output_file'   => (string)($out['alt_output_file'] ?? ''),
            'error_message'     => (string)$srow['error_message'],
            'completed_at'      => (string)$srow['completed_at'],
        ];
    }

    if (!$email) {
        // Nothing we can do — mark sent to avoid looping
        $upd = $db->prepare("UPDATE jobs SET notify_status='sent' WHERE email_correlation_id = :cid");
        $upd->bindValue(':cid', $cid, SQLITE3_TEXT);
        $upd->execute();
        continue;
    }

    // Build attachment file list (only for completed siblings)
    $attachExt = $mode === 'text' ? '.txt' : '.md';
    $attachContent = [];   // [{name, bytes}, ...]
    $totalBytes = 0;
    $failed = [];
    foreach ($siblings as $s) {
        if ($s['status'] !== 'completed' || !$s['output_filename']) {
            $failed[] = $s;
            continue;
        }
        $primaryName = $s['output_filename'];
        $primaryPath = $QUEUE_FILES_DIR . '/outputs/' . $primaryName;
        $useThis = $primaryPath;
        // If the primary doesn't match the requested extension, fall back to alt
        if (substr($primaryName, -strlen($attachExt)) !== $attachExt && $s['alt_output_file']) {
            $altPath = (string)$s['alt_output_file'];
            // alt_output_file from the handler is an absolute path inside queue_files/inputs/
            // but we copied .md/.txt into queue_files/outputs/. Look there too.
            $altBase = basename($altPath);
            if (file_exists($QUEUE_FILES_DIR . '/outputs/' . $altBase)) {
                $useThis = $QUEUE_FILES_DIR . '/outputs/' . $altBase;
            } elseif (file_exists($altPath)) {
                $useThis = $altPath;
            }
        }
        if (!file_exists($useThis) || filesize($useThis) === 0) {
            $failed[] = $s;
            continue;
        }
        $bytes = file_get_contents($useThis);
        $stem = pathinfo($s['original_filename'], PATHINFO_FILENAME);
        $attachName = $stem . $attachExt;
        $attachContent[] = ['name' => $attachName, 'bytes' => $bytes];
        $totalBytes += strlen($bytes);
    }

    $modeLabel = $mode === 'text' ? 'plain text' : 'markdown';
    $count = count($attachContent);

    // Compose email
    $subject = "Your TEXTIFY result — {$count} document(s) (" . $modeLabel . ")";
    if ($count === 1 && $totalBytes <= $INLINE_BYTES_CAP) {
        $bodyText = "Your textify result is below.\n\n"
            . "Source: {$siblings[0]['original_filename']}\n"
            . "Mode:   {$modeLabel}\n"
            . str_repeat('─', 60) . "\n\n"
            . $attachContent[0]['bytes']
            . "\n\n" . str_repeat('─', 60) . "\n";
        if (!empty($failed)) {
            $bodyText .= "\nCould not convert:\n";
            foreach ($failed as $f) {
                $bodyText .= "  • {$f['original_filename']}: " . ($f['error_message'] ?: $f['status']) . "\n";
            }
        }
        $bodyText .= "\nThe Lab · Longboardfella\n";
        $headers = "From: The Lab <{$fromEmail}>\r\nReply-To: {$fromEmail}\r\nContent-Type: text/plain; charset=UTF-8\r\n";
        _qn_send_mail($email, $subject, $bodyText, $headers);
        if ($sendTo && filter_var($sendTo, FILTER_VALIDATE_EMAIL)) {
            _qn_send_mail($sendTo, $subject, $bodyText, $headers);
        }
    } else {
        // MIME multipart with one attachment per sibling
        $boundary = 'qn_' . bin2hex(random_bytes(8));
        $intro = "Your TEXTIFY result is attached ({$count} file" . ($count !== 1 ? 's' : '') . ", mode: {$modeLabel}).\n\n";
        if (!empty($failed)) {
            $intro .= "Could not convert:\n";
            foreach ($failed as $f) {
                $intro .= "  • {$f['original_filename']}: " . ($f['error_message'] ?: $f['status']) . "\n";
            }
            $intro .= "\n";
        }
        $intro .= "The Lab · Longboardfella\n";

        $body = "--{$boundary}\r\n"
              . "Content-Type: text/plain; charset=UTF-8\r\n"
              . "Content-Transfer-Encoding: 8bit\r\n\r\n"
              . $intro . "\r\n";

        foreach ($attachContent as $att) {
            if ($totalBytes > $ATTACH_TOTAL_CAP) {
                // Replace remaining attachments with link-style placeholders
                $body .= "--{$boundary}\r\n"
                      . "Content-Type: text/plain; charset=UTF-8\r\n\r\n"
                      . "Attachment {$att['name']} omitted — total reply size would exceed 20 MB. "
                      . "Use the queue page to download: https://longboardfella.com.au/lab/queue.php\r\n\r\n";
                continue;
            }
            $b64 = chunk_split(base64_encode($att['bytes']));
            $mimeType = $mode === 'text' ? 'text/plain' : 'text/markdown';
            $body .= "--{$boundary}\r\n"
                  . "Content-Type: {$mimeType}; charset=UTF-8; name=\"{$att['name']}\"\r\n"
                  . "Content-Disposition: attachment; filename=\"{$att['name']}\"\r\n"
                  . "Content-Transfer-Encoding: base64\r\n\r\n"
                  . $b64 . "\r\n";
        }
        $body .= "--{$boundary}--\r\n";

        $headers = "From: The Lab <{$fromEmail}>\r\n"
                 . "Reply-To: {$fromEmail}\r\n"
                 . "MIME-Version: 1.0\r\n"
                 . "Content-Type: multipart/mixed; boundary=\"{$boundary}\"\r\n";

        _qn_send_mail($email, $subject, $body, $headers);
        if ($sendTo && filter_var($sendTo, FILTER_VALIDATE_EMAIL)) {
            _qn_send_mail($sendTo, $subject, $body, $headers);
        }
    }

    // Mark all siblings sent
    $upd = $db->prepare("UPDATE jobs SET notify_status='sent' WHERE email_correlation_id = :cid");
    $upd->bindValue(':cid', $cid, SQLITE3_TEXT);
    $upd->execute();

    if ($isCliMode) echo "Sent textify completion email for group {$cid} ({$count} doc/s) to {$email}\n";
}
// ── End textify group completion ──
```

- [ ] **Step 3: Stub the worker output for an end-to-end seed test**

We'll seed a single completed `pdf_textify` row with a synthetic `.txt` output file and run the notifier with the mail sink to capture the result.

```bash
cd /home/longboardfella/longboardfella_website
mkdir -p site/admin/queue_files/outputs
echo -e "Heading\n\nThis is a tiny converted document for testing." > site/admin/queue_files/outputs/__qn_test_report.txt

CID=$(php -r 'echo bin2hex(random_bytes(8));')
sqlite3 site/admin/queue.db <<SQL
INSERT INTO jobs (type, status, input_data, output_data, output_filename, created_at, completed_at, submitter_email, created_by, schedule_type, notify_status, email_correlation_id, trace_id)
VALUES (
  'pdf_textify',
  'completed',
  json_object(
    'original_filename','report.pdf',
    'email_textify_mode','text',
    'email_correlation_id','${CID}',
    'email_correlation_count',1,
    'source_system','email',
    'send_to',''
  ),
  '{}',
  '__qn_test_report.txt',
  datetime('now','-1 hour'),
  datetime('now','-5 minutes'),
  'paul@longboardfella.com.au',
  'paul@longboardfella.com.au',
  'anytime',
  'pending',
  '${CID}',
  'qn-test-trace'
);
SQL

SINK=$(mktemp)
QUEUE_NOTIFY_MAIL_SINK=$SINK php site/admin/queue_notify.php
echo "── sink ──"
cat "$SINK"
```

Expected:
- The sink file shows a single `── MAIL ──` entry to `paul@longboardfella.com.au`.
- Subject contains `Your TEXTIFY result — 1 document(s) (plain text)`.
- The body includes the test text inline (since size < 20 KB and count = 1).
- No MIME boundary (inline mode).

- [ ] **Step 4: Verify the seeded job is now marked sent and clean up**

```bash
sqlite3 site/admin/queue.db "SELECT id, notify_status, email_correlation_id FROM jobs WHERE trace_id='qn-test-trace';"
sqlite3 site/admin/queue.db "DELETE FROM jobs WHERE trace_id='qn-test-trace';"
rm -f site/admin/queue_files/outputs/__qn_test_report.txt
rm -f "$SINK"
```

Expected: `notify_status` is `sent`. After deletion, the row is gone.

- [ ] **Step 5: Seed a multi-document group and verify multipart**

```bash
cd /home/longboardfella/longboardfella_website
mkdir -p site/admin/queue_files/outputs
for i in 1 2 3; do
  printf '# Doc %d\n\nBody for document %d.\n' "$i" "$i" > "site/admin/queue_files/outputs/__qn_test_doc${i}.md"
done

CID=$(php -r 'echo bin2hex(random_bytes(8));')
for i in 1 2 3; do
  sqlite3 site/admin/queue.db "
    INSERT INTO jobs (type, status, input_data, output_data, output_filename, created_at, completed_at, submitter_email, created_by, schedule_type, notify_status, email_correlation_id, trace_id)
    VALUES (
      'pdf_textify',
      'completed',
      json_object('original_filename','doc${i}.pdf','email_textify_mode','markdown','email_correlation_id','${CID}','email_correlation_count',3,'source_system','email','send_to',''),
      '{}',
      '__qn_test_doc${i}.md',
      datetime('now','-1 hour'),
      datetime('now','-5 minutes'),
      'paul@longboardfella.com.au',
      'paul@longboardfella.com.au',
      'anytime',
      'pending',
      '${CID}',
      'qn-test-multi-${i}'
    );
  "
done

SINK=$(mktemp)
QUEUE_NOTIFY_MAIL_SINK=$SINK php site/admin/queue_notify.php
echo "── sink ──"
cat "$SINK"
```

Expected:
- One mail entry, subject `Your TEXTIFY result — 3 document(s) (markdown)`.
- Headers contain `Content-Type: multipart/mixed; boundary="qn_..."`.
- Body has three base64-encoded `text/markdown` parts named `doc1.md`, `doc2.md`, `doc3.md`.

Cleanup:

```bash
sqlite3 site/admin/queue.db "DELETE FROM jobs WHERE trace_id LIKE 'qn-test-multi-%';"
rm -f site/admin/queue_files/outputs/__qn_test_doc*.md
rm -f "$SINK"
```

- [ ] **Step 6: Seed a partial-failure group and verify the failed list**

```bash
cd /home/longboardfella/longboardfella_website
mkdir -p site/admin/queue_files/outputs
echo -e "ok content" > site/admin/queue_files/outputs/__qn_test_ok.txt

CID=$(php -r 'echo bin2hex(random_bytes(8));')
sqlite3 site/admin/queue.db "
  INSERT INTO jobs (type, status, input_data, output_data, output_filename, created_at, completed_at, submitter_email, created_by, schedule_type, notify_status, email_correlation_id, trace_id)
  VALUES (
    'pdf_textify','completed',
    json_object('original_filename','ok.pdf','email_textify_mode','text','email_correlation_id','${CID}','email_correlation_count',2,'source_system','email','send_to',''),
    '{}','__qn_test_ok.txt',datetime('now','-1 hour'),datetime('now','-5 minutes'),
    'paul@longboardfella.com.au','paul@longboardfella.com.au','anytime','pending','${CID}','qn-test-partial-ok'
  );
  INSERT INTO jobs (type, status, input_data, output_data, output_filename, created_at, completed_at, error_message, submitter_email, created_by, schedule_type, notify_status, email_correlation_id, trace_id)
  VALUES (
    'pdf_textify','failed',
    json_object('original_filename','bad.pdf','email_textify_mode','text','email_correlation_id','${CID}','email_correlation_count',2,'source_system','email','send_to',''),
    '{}','',datetime('now','-1 hour'),datetime('now','-5 minutes'),'simulated parse failure',
    'paul@longboardfella.com.au','paul@longboardfella.com.au','anytime','pending','${CID}','qn-test-partial-fail'
  );
"

SINK=$(mktemp)
QUEUE_NOTIFY_MAIL_SINK=$SINK php site/admin/queue_notify.php
echo "── sink ──"
cat "$SINK"
sqlite3 site/admin/queue.db "DELETE FROM jobs WHERE trace_id IN ('qn-test-partial-ok','qn-test-partial-fail');"
rm -f site/admin/queue_files/outputs/__qn_test_ok.txt
rm -f "$SINK"
```

Expected: one mail with the inline body (since 1 successful doc, ≤20 KB) **plus** a "Could not convert: bad.pdf: simulated parse failure" line.

- [ ] **Step 7: Verify in-flight groups are NOT notified**

```bash
cd /home/longboardfella/longboardfella_website
CID=$(php -r 'echo bin2hex(random_bytes(8));')
sqlite3 site/admin/queue.db "
  INSERT INTO jobs (type, status, input_data, output_data, created_at, submitter_email, created_by, schedule_type, notify_status, email_correlation_id, trace_id)
  VALUES ('pdf_textify','completed',json_object('original_filename','a.pdf','email_textify_mode','text','email_correlation_id','${CID}'),'{}',datetime('now'),'paul@longboardfella.com.au','paul@longboardfella.com.au','anytime','pending','${CID}','qn-test-inflight-1');
  INSERT INTO jobs (type, status, input_data, output_data, created_at, submitter_email, created_by, schedule_type, notify_status, email_correlation_id, trace_id)
  VALUES ('pdf_textify','processing',json_object('original_filename','b.pdf','email_textify_mode','text','email_correlation_id','${CID}'),'{}',datetime('now'),'paul@longboardfella.com.au','paul@longboardfella.com.au','anytime','pending','${CID}','qn-test-inflight-2');
"

SINK=$(mktemp)
QUEUE_NOTIFY_MAIL_SINK=$SINK php site/admin/queue_notify.php
echo "── sink (should be empty) ──"
cat "$SINK"
sqlite3 site/admin/queue.db "SELECT id, status, notify_status FROM jobs WHERE trace_id LIKE 'qn-test-inflight-%';"

sqlite3 site/admin/queue.db "DELETE FROM jobs WHERE trace_id LIKE 'qn-test-inflight-%';"
rm -f "$SINK"
```

Expected: sink is empty (no mail sent). Both rows still have `notify_status='pending'`. We're waiting for the second sibling to leave `processing`.

- [ ] **Step 8: Commit**

```bash
cd /home/longboardfella/longboardfella_website
git add site/admin/queue_notify.php
git commit -m "feat(queue-notify): grouped textify completion email with inline/MIME delivery"
```

---

## Task 6: End-to-end smoke test (manual)

**Repos:** both — but no code changes here.

This is a real smoke test: a live Lab member sends a real email and we watch it through.

- [ ] **Step 1: Enable the feature flag in the worker**

```bash
cd /home/longboardfella/cortex_suite
grep -q '^LAB_FETCH_ATTACHMENTS=' worker/config.env && \
  sed -i 's/^LAB_FETCH_ATTACHMENTS=.*/LAB_FETCH_ATTACHMENTS=1/' worker/config.env || \
  echo 'LAB_FETCH_ATTACHMENTS=1' >> worker/config.env
grep '^LAB_FETCH_ATTACHMENTS=' worker/config.env
```

Expected: `LAB_FETCH_ATTACHMENTS=1`.

- [ ] **Step 2: Restart the lab mailbox worker**

```bash
pkill -f lab_mailbox_worker.py || true
cd /home/longboardfella/cortex_suite
source venv/bin/activate
nohup python worker/lab_mailbox_worker.py > /tmp/lab_mailbox_worker.log 2>&1 &
sleep 3
tail -n 20 /tmp/lab_mailbox_worker.log
```

Expected: log shows `Lab mailbox worker started: mailbox=lab@longboardfella.com.au` with no errors.

- [ ] **Step 3: Send a test email**

From your normal inbox, send to `lab@longboardfella.com.au`:
- Subject: `TEXTIFY`
- Body: empty (or a SENDTO line if you want to verify forwarding)
- Attachments: one PDF (~1–5 pages), one Excel file, one tiny inline signature/logo

- [ ] **Step 4: Wait for the worker to pick it up and confirm**

```bash
tail -f /tmp/lab_mailbox_worker.log
# Watch for: "Processed Lab email subject=TEXTIFY outcome=accepted job_ids=[NNN]"
```

Then in the website repo:

```bash
cd /home/longboardfella/longboardfella_website
sqlite3 site/admin/queue.db "SELECT id, type, status, email_correlation_id, input_filename FROM jobs WHERE type='pdf_textify' ORDER BY id DESC LIMIT 5;"
```

Expected: a new `pdf_textify` row in `pending` (or `claimed`/`processing`) with the correlation id set and an input file present in `queue_files/inputs/`.

- [ ] **Step 5: Wait for the worker to complete the conversion**

The cortex worker (running separately on its usual schedule) will pick up the pending job. Watch its logs and the queue:

```bash
sqlite3 site/admin/queue.db "SELECT id, status, output_filename, completed_at FROM jobs WHERE type='pdf_textify' ORDER BY id DESC LIMIT 1;"
```

Expected: status flips to `completed`, `output_filename` populated.

- [ ] **Step 6: Trigger queue_notify and confirm the reply lands in your inbox**

```bash
cd /home/longboardfella/longboardfella_website
php site/admin/queue_notify.php
```

Expected: an email arrives at your address from `noreply@longboardfella.com.au` containing the converted text. For a small PDF, it should be inline; for a long PDF, an attachment. Excel and the small icon are not mentioned.

- [ ] **Step 7: Smoke-test the multi-document path**

Send another email with subject `MARKDOWN` and 2–3 different document attachments. Confirm:
- All are queued as a group.
- A single MIME multipart email arrives once all jobs complete.
- Each `.md` attachment opens cleanly.

- [ ] **Step 8: Smoke-test the rejection path**

Send an email with subject `TEXTIFY` and **only** an Excel attachment. Confirm:
- The reply explains "No convertible documents were found" with the supported-formats list.
- No queue rows are created.

- [ ] **Step 9: Commit any stray config tweaks**

If `worker/config.env` was modified locally (e.g. activating the flag) and it's checked into git in this repo, commit that. Otherwise nothing to commit at this step.

```bash
cd /home/longboardfella/cortex_suite
git status
# If config.env is tracked and modified, commit it. If it's gitignored (typical), no action.
```

---

## Self-review

Done after writing the plan, before handing off:

**1. Spec coverage:**

| Spec section | Implemented in |
|---|---|
| End-to-end flow (worker → webhook → handler → notifier) | Tasks 2, 3, 4, 5 |
| Files changed table | File map at top + per-task headers |
| Subject parsing (TEXTIFY / MARKDOWN, MARKDOWN wins) | Task 4 step 4 (`_jilib_detectTextifyMode`) |
| Attachment filter (PDF/DOCX/PPTX kept, ≥50 KB images, ignore xlsx/inline/small) | Task 4 step 4 (`_jilib_filterTextifyAttachments`) |
| Reject email when zero kept | Task 4 step 4 (no_attachments branch) |
| Job submission + correlation id + input_data shape | Task 4 step 4 (`_jilib_handleTextify`) |
| Confirmation reply | Task 4 step 4 (reply construction) |
| Worker handler emits .md and .txt, picks primary by mode | Task 2 |
| Completion delivery: group, wait for all, inline-or-multipart | Task 5 step 2 |
| `send_to` forwarding | Task 5 step 2 (forward to sendTo) |
| 20 MB total reply cap with link fallback | Task 5 step 2 (`$ATTACH_TOTAL_CAP` branch) |
| Partial failure listing | Task 5 step 2 (`$failed` block) |
| Reply transport (PHP `mail()` from noreply) | Task 5 step 1 (`_qn_send_mail`) |
| Open all approved members, no capability flag | (Inherent — no new check added; existing approved-sender gate applies) |
| Rollout flag `LAB_FETCH_ATTACHMENTS` | Task 3 step 1 + step 4 |

All spec items mapped.

**2. Placeholder scan:** No "TODO", "TBD", or "implement appropriate handling" sentences in the plan. Every code step has actual code.

**3. Type consistency:**
- `email_textify_mode` values are `'text'` and `'markdown'` everywhere (handler, ingest lib, notifier).
- `email_correlation_id` is the column name and JSON key everywhere.
- `_jilib_detectTextifyMode`, `_jilib_filterTextifyAttachments`, `_jilib_writeTextifyInput`, `_jilib_handleTextify`, `_qn_send_mail` — names used consistently.
- `LAB_FETCH_ATTACHMENTS` — env var, dataclass field `fetch_attachments`, function arg `fetch=` — consistent.
