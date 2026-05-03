# TEXTIFY / MARKDOWN email handler — design

**Date:** 2026-05-03
**Status:** Approved (design phase)
**Spans repos:** `cortex_suite` (Python worker) + `longboardfella_website` (PHP webhook & notifier)

## Goal

Let approved Lab members email `lab@longboardfella.com.au` with `TEXTIFY` or
`MARKDOWN` in the subject and one or more PDF / DOCX / PPTX / image
attachments, and receive the converted text back. `TEXTIFY` returns plain
text, `MARKDOWN` returns Markdown. Excel attachments and small inline icons
are silently ignored.

## Why

The Cortex Suite already has a robust document-to-Markdown pipeline
(`cortex_engine.textifier.DocumentTextifier`) with a recent plain-text
formatter (`markdown_to_plaintext`). The Lab mailbox is the natural inbound
channel for ad-hoc conversions when away from the Streamlit UI. Today the
inbound pipeline drops attachments on the floor — closing that gap unlocks
this whole class of one-off conversions over email.

## Non-goals

- OCR tuning beyond what the existing textifier already does.
- New file-type support (anything the textifier already handles is in scope;
  we don't add new formats).
- Threaded replies from `lab@`; the reply comes from `noreply@` like every
  other Lab job notification (see "Reply transport" below).
- Persistent storage / retention beyond what the queue already does.

## End-to-end flow

```
User emails lab@longboardfella.com.au
  • Subject contains TEXTIFY or MARKDOWN
  • One or more attachments
  ↓
[cortex_suite/worker/lab_mailbox_worker.py]
  Polls Graph, normalises message.
  NEW: fetches /messages/{id}/attachments, base64-encodes each one,
       includes them in the webhook payload (currently always []).
  ↓
[longboardfella_website/site/admin/email_job_webhook.php
   → email_job_ingest_lib.php]
  • Detects TEXTIFY / MARKDOWN bare keyword in subject (flex parser).
  • Filters attachments (see §"Attachment filter").
  • For each kept attachment: writes to queue_files/inputs/<job_id>_<safe>,
    inserts one `pdf_textify` queue job with a shared
    email_correlation_id.
  • Sends an immediate confirmation reply listing queued files & job ids.
  ↓
[cortex_suite/worker/worker.py + handlers/pdf_textify.py]
  Textifies each input.
  UPDATED: handler also writes a sibling .txt file (via
           DocumentTextifier.markdown_to_plaintext) so either format is
           ready at notification time.
  ↓
[longboardfella_website/site/admin/queue_notify.php  (cron, every 5 min)]
  NEW BRANCH: handles pdf_textify jobs that carry an email_correlation_id.
  • Groups siblings by email_correlation_id; waits until all are
    completed/failed.
  • Picks .md or .txt outputs based on the requested mode.
  • Inline-vs-attachment rule: single doc AND total output ≤ 20 KB →
    inline body; otherwise MIME multipart, one attachment per source.
  • Marks all siblings notify_status='sent'.
  • Forwards to send_to address if set (mirrors RIP/STRIP).
```

## Files changed

| Path | Repo | Change |
|---|---|---|
| `worker/lab_mailbox_worker.py` | cortex_suite | Fetch + filter + base64-encode attachments via Graph |
| `worker/handlers/pdf_textify.py` | cortex_suite | Also emit `.txt` alongside `.md` |
| `site/admin/email_job_ingest_lib.php` | longboardfella_website | TEXTIFY / MARKDOWN keyword routing; attachment-to-job dispatch with correlation id |
| `site/admin/queue_notify.php` | longboardfella_website | Group-aware textify completion email with MIME attachments |
| `site/admin/queue_api_shared.php` *(possibly)* | longboardfella_website | Schema migration: add `email_correlation_id` column to `jobs` if simpler than embedding in `input_data` |

## Subject parsing

- Match `\bTEXTIFY\b` or `\bMARKDOWN\b` case-insensitively in the cleaned
  subject (after `Re:` / `Fwd:` stripping). The `JOB:` prefix is optional
  to mirror the existing flex parser.
- If both keywords appear, **`MARKDOWN` wins**.
- The explicit-keyword match runs *before* the existing
  `_jilib_parseJobSubject` / `_jilib_flexParseSubject` so it short-circuits
  the URL/YouTube path.
- **Implicit fallback:** if no other keyword (TEXTIFY, MARKDOWN, JOB,
  CANCEL, DISPATCH, APPEND, HELP, URL, YOUTUBE, RIP, SCRAPE, STRIP) matches
  the subject **and** the body has no URLs **and** the email carries at
  least one qualifying attachment, route to TEXTIFY in `text` mode. This
  makes "send a PDF to lab@ with any subject" the zero-friction default.
  When the user *did* express clear intent (e.g. `JOB: URL INGEST` with a
  PDF attached), URL ingest still wins and the attachment is ignored.
- Outcome strings used by the ingest log:
  `accepted_textify` (jobs queued), `no_attachments` (rejected), and the
  existing `rejected_sender` / `rejected_capability`-style codes if needed.

## Attachment filter

| Decision | Rule |
|---|---|
| Keep & process | extension in `{pdf, docx, pptx}` (no size floor); OR extension in `{png, jpg, jpeg, webp, tif, tiff}` AND `is_inline=false` AND `size_bytes ≥ 50 000` |
| Skip silently | extension in `{xlsx, xls, csv}`; OR `is_inline=true`; OR image attachments under 50 KB |
| Reject the email | zero attachments survive the filter |

The 50 KB image floor is a pragmatic "small icon" cutoff — it covers footer
logos, signature badges, and tracking pixels.

When the email is rejected for zero qualifying attachments, the reply is:
> No convertible documents were found.
>
> Supported: PDF, DOCX, PPTX, and images ≥ 50 KB.
> Excel files and small inline icons are ignored.
>
> No job submitted.

## Job submission & correlation

Each kept attachment becomes one `pdf_textify` queue job. All siblings from
the same email share an `email_correlation_id` (UUID v4 generated at webhook
time).

`input_data` JSON for each sibling job:

```json
{
  "original_filename": "report.pdf",
  "textify_options": {"use_vision": true, "add_metadata_preface": true},
  "textify_advanced": {"pdf_strategy": "hybrid"},
  "email_textify_mode": "text",
  "email_correlation_id": "f3b2…",
  "email_correlation_count": 3,
  "source_system": "email",
  "send_to": ""
}
```

`email_textify_mode` is `"text"` for `TEXTIFY` and `"markdown"` for
`MARKDOWN`. `send_to` is populated from the body's `SENDTO:` line if
present (existing convention).

`email_correlation_id` is duplicated as a top-level column on `jobs` if the
queue notifier needs to group/index by it efficiently; otherwise it's
extracted from `input_data` JSON. **Decision deferred to the implementation
plan**, but the spec assumes both options work.

The webhook returns this confirmation reply (sent synchronously via Graph
by the lab mailbox worker, like all other webhook replies):

> ✅ TEXTIFY job(s) submitted.
>
> Files: report.pdf, plans.docx, deck.pptx
> Job IDs: #842, #843, #844
> Mode: plain text
>
> You'll receive a single reply with the converted result(s) when all
> jobs are complete. To cancel: reply with subject CANCEL #842 (or any
> sibling id).

## Worker handler change

`handlers/pdf_textify.py` currently writes only `<stem>.md`. Extend it to
also write `<stem>.txt` using
`DocumentTextifier.markdown_to_plaintext(markdown_text, width=80)`.
The handler returns the file matching `email_textify_mode` as
`output_file` (so the queue's `output_filename` column points to the
requested format), and includes the path of the alternate format in
`output_data["alt_output_file"]`.

When `email_textify_mode` is absent (i.e. invocations from
`tools/convert.php`), behaviour is unchanged: only `.md` is written.

## Completion delivery

`queue_notify.php` gets a new branch executed *before* its existing
`type NOT IN (…)` exclusion list:

1. `SELECT … FROM jobs WHERE type='pdf_textify' AND notify_status='pending'
   AND status IN ('completed','failed')`
   filtered to those whose `input_data` contains
   `email_correlation_id`.
2. Group by `email_correlation_id`. For each group, count siblings; if any
   sibling in the group is still `pending` / `claimed` / `processing`,
   skip the group this run.
3. For ready groups: requested mode is taken from any sibling's
   `email_textify_mode`. Pick the matching file
   (`.txt` for text, `.md` for markdown) from `output_filename` (or
   `alt_output_file`).
4. Apply delivery rule:
   - **Single sibling AND total chosen-format size ≤ 20 KB** → inline
     plain-text email body. Wrap with the standard "Lab" header/footer.
   - **Otherwise** → MIME multipart `mixed`, one attachment per sibling
     named `<original-stem>.txt` or `<original-stem>.md`.
5. Send to `submitter_email` via `mail()`. Mark every sibling in the group
   `notify_status='sent'`.
6. If `send_to` is set on any sibling, forward the same email there too.
7. Per-sibling failure: include a "Could not convert" section in the email
   listing each failed file plus its `error_message`. Successful siblings
   still deliver as attachments.

The 20 MB total reply cap (per §"Edge cases") triggers a fallback: replace
oversized attachments with a "View Result" link to
`/lab/result.php?id=<job_id>`.

## Reply transport

All replies (the immediate webhook confirmation **and** the deferred
completion delivery) follow the existing patterns:

- Webhook confirmation reply: returned in the webhook response, sent by
  `lab_mailbox_worker.py` via Graph from `lab@`. Already wired up.
- Completion delivery: PHP `mail()` from `noreply@longboardfella.com.au`,
  matching how every other Lab job notification works today (FULL/PLAIN/TEXT
  mode for RIP/STRIP, batch ingest, etc.).

A threaded `Re:` reply from `lab@` for the completion delivery was
considered and rejected: it would require a second deferred-reply path in
the Python worker, splitting completion delivery between PHP and Python.
Not worth the threading benefit.

## Edge cases

| Case | Behaviour |
|---|---|
| No qualifying attachment | Webhook replies with the §"Attachment filter" rejection text. Outcome `no_attachments`. |
| All siblings fail | Send a single failure email listing each filename + reason. |
| Partial failure | Deliver successes as attachments + a "could not convert" list at the bottom. |
| Reply > 20 MB | Replace oversized attachments with a `/lab/result.php?id=…` link per sibling. |
| Sender not a Lab member | Existing `rejected_sender` path. No attachment processing, no files written. |
| Duplicate `message_id` | Existing dedupe (`job_ingest_log.message_id`) handles this. |
| Cancellation | `CANCEL #<id>` works on any sibling. Cancelled siblings appear in the "could not convert" list when the rest of the group finishes. |
| Mailbox blob too large | Graph attachment fetch is capped at the 25 MB Graph item limit; oversized attachments yield a per-attachment skip with reason. |
| Mixed XLSX + PDF email | XLSX is silently filtered out; PDF is processed. No warning to the sender. |
| Subject has both TEXTIFY and MARKDOWN | MARKDOWN wins. |

## Access control

Open to all approved Lab members — **no new capability flag**. Existing
`lab_members.status='approved'` check applies.

## Migration / rollout

1. Ship worker attachment-fetch behind a feature flag
   (`LAB_FETCH_ATTACHMENTS=1` in `worker/config.env`) so it can be enabled
   without forcing the new path until the PHP side is also live.
2. Ship PHP changes (subject routing + queue_notify branch) in a single
   deploy. Without the flag, existing emails are unaffected because the
   webhook only reacts to TEXTIFY/MARKDOWN keywords that don't exist
   today.
3. Verify end-to-end with one PDF, one DOCX, and a mixed-attachment email
   (PDF + Excel + small icon).

## Open questions

1. Store `email_correlation_id` as a dedicated `jobs` column or extract
   from `input_data` JSON in queue_notify? Decide during plan-writing
   based on how often we'll need to query by it.
2. Should the immediate webhook confirmation list the *ignored* (filtered)
   attachments by name so users know what got dropped? Default: no — keep
   the reply terse — but easy to flip.
