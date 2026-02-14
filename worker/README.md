# Local Queue Worker Harness

This folder contains the local/offline queue worker for the website work queue API.

## Files
- `worker.py`: Main polling worker loop.
- `config.env.example`: Template configuration.
- `handlers/__init__.py`: Job-type handler registry.
- `handlers/pdf_anonymise.py`: `pdf_anonymise` handler (reuses `cortex_engine.anonymizer`).

## Setup
1. Copy config template:
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

## Notes
- The `pdf_anonymise` worker handler intentionally calls the existing Cortex engine anonymizer:
  - `cortex_engine.anonymizer.DocumentAnonymizer`
- This avoids duplicate anonymization logic between the admin queue worker path and Document Extract UI.

