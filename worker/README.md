# Local Queue Worker Harness

This folder contains the local/offline queue worker for the website work queue API.

## Files
- `worker.py`: Main polling worker loop.
- `intel_mailbox_worker.py`: Direct IMAP poller for Cortex-owned intelligence inboxes.
- `notes_mailbox_worker.py`: Dedicated Microsoft Graph worker for the separate `notes@longboardfella.com.au` note-stash mailbox.
- `config.env`: Worker configuration for queue API and polling.
- `handlers/__init__.py`: Job-type handler registry.
- `handlers/pdf_anonymise.py`: `pdf_anonymise` handler (reuses `cortex_engine.anonymizer`).
- `handlers/pdf_textify.py`: `pdf_textify` handler (reuses `cortex_engine.textifier`).
- `handlers/included_study_extract.py`: `included_study_extract` handler (systematic review PDF -> grouped included-study table artifacts).
- `handlers/url_ingest.py`: `url_ingest` handler (open-access PDF discovery + optional textify).
- `handlers/research_resolve.py`: `research_resolve` handler (citation resolution via CrossRef/Unpaywall/SJR).
- `handlers/org_profile_refresh.py`: `org_profile_refresh` handler (official-source organisation refresh with structured profile proposals).
- `handlers/youtube_summarise.py`: `youtube_summarise` handler (video summary workflows).
- `handlers/cortex_sync.py`: `cortex_sync` handler (website knowledge files -> Cortex ingestion pipeline).
- `handlers/signal_episode.py`: `signal_episode` handler (Signal Studio audio generation).

## Setup
1. Create config from template:
```bash
cp worker/config.env.example worker/config.env
```
2. Edit `worker/config.env` with your server URL and queue secret.
3. Install dependencies in your venv:
```bash
pip install requests pymupdf
```

## Run
```bash
venv/bin/python worker/worker.py
```

Direct Cortex-owned intel mailbox intake:
```bash
venv/bin/python worker/intel_mailbox_worker.py
```

Notes mailbox worker:
```bash
venv/bin/python worker/notes_mailbox_worker.py
```

## How It Works
1. Polls `queue_worker_api.php?action=poll&types=...`
2. Downloads input file (`action=download_input`)
3. Starts heartbeat thread (`action=heartbeat` every `HEARTBEAT_INTERVAL`)
4. Runs handler mapped by job type
5. Uploads completion (`action=complete`) with `output_data` + optional file
6. On error, posts failure (`action=fail`)
7. Cleans temporary job directory

## Current Supported Job Types
- `pdf_anonymise`
- `pdf_textify`
- `included_study_extract`
- `url_ingest`
- `research_resolve`
- `org_profile_refresh`
- `youtube_summarise`
- `signal_episode`
- `cortex_sync`

## Notes
- Worker telemetry is written to `worker/tmp/queue_monitor_state.json` by default.
- Queue monitor UI is available in Streamlit at `pages/16_Queue_Monitor.py`.
- `cortex_sync` path remap:
  - If website submits cPanel-style paths like `/home/<user>/public_html/...`, handler auto-remaps to local mirror root.
  - Override local mirror root with `CORTEX_SYNC_SITE_ROOT` (default: `~/longboardfella_website/site`).
- The `pdf_anonymise` worker handler intentionally calls the existing Cortex engine anonymizer:
  - `cortex_engine.anonymizer.DocumentAnonymizer`
- This avoids duplicate anonymization logic between the admin queue worker path and Document Extract UI.
- `intel_mailbox_worker.py` reads IMAP settings from `worker/config.env`:
  - `INTEL_IMAP_HOST`, `INTEL_IMAP_PORT`, `INTEL_IMAP_USERNAME`, `INTEL_IMAP_PASSWORD`
  - `INTEL_IMAP_FOLDER`, `INTEL_IMAP_ORG_NAME`, `INTEL_ALLOWED_SENDERS`
  - optional website callback: `INTEL_RESULTS_POST_URL`, `INTEL_RESULTS_POST_SECRET`
  - provider hostnames are now explicit config, not implicit Gmail defaults
- Intel mailbox routing syntax:
  - plain market-intel note: subject can be plain text or `Note: <headline>`
  - scoped note/document: `entity: <subscriber org> | <headline>`
  - optional modifiers: `depth:detailed`, `force:yes`
  - `INTEL: ...` triggers the direct `intel_extract` path rather than the structured note ingestion path
- Trusted self-relay:
  - messages sent from `intel@longboardfella.com.au` to itself are treated as authorized
  - those messages are attributed as submitted by `paul@longboardfella.com.au` for now
  - the legacy `intel.longboardfella@gmail.com` alias is still accepted during the migration window
- Mailbox scope:
  - `intel@longboardfella.com.au` is the Market Radar / website-intel mailbox
  - `notes@longboardfella.com.au` should be treated as a separate note-stash mailbox rather than routed through the intel webhook path
  - note-shaped subjects accidentally sent to `intel@...` are now suppressed as notes-mailbox traffic instead of being mixed into Market Radar delivery
- `notes_mailbox_worker.py` provides:
  - the dedicated config contract for `notes@...`
  - note route classification (`public_stash`, `private_vault`, `unsupported_market_intel`)
  - Microsoft Graph polling via app-only client credentials when `NOTES_TRANSPORT_MODE=graph`
  - local JSON outbox persistence for audit/debug
  - direct Obsidian markdown writes to `NOTES_PUBLIC_VAULT_DIR` and `NOTES_PRIVATE_VAULT_DIR` when `NOTES_WRITE_VAULT_MARKDOWN=1`
  - Graph access should be mailbox-scoped in Exchange Online so the app can read only `notes@longboardfella.com.au`
  - with least-privilege `Mail.Read`, Graph may reject mark-as-read calls; the worker keeps a local processed-message state file to avoid duplicate local processing
- If no callback URL is configured, processed mailbox extraction results are written to the external DB path under `intel_mailbox/outbox/`.
