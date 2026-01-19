Next Session TODOs
===================

High‑Impact
- Finalization watchdog: if staging exists for N minutes and auto‑finalize didn’t run, show a banner and a persistent Retry button (host + Docker); record reason when finalize fails.
- Normalize document readers in Docker: ensure `.docx` and related readers consistently call `load_data(file=...)` to remove `file_path` warnings; add tests on two sample files.
- Collection sync UX: add a collection selector in Collections to choose an active Chroma collection and sync its IDs into `working_collections.json` (host + Docker).
- Docker run UX: update run scripts to detect stale images and prompt for `docker compose build --no-cache` when repo changed.

Stability & Clarity
- Make Clean Start fully idempotent and add a “force delete” fallback with permission error handling (host + Docker).
- Unify Knowledge Search fallback so Docker and host share the same core implementation (avoid drift).
- Add a CLI smoke script to verify: staging path, staged doc count, finalize success, vector doc count, and default collection sync.
- Add end‑to‑end log breadcrumbs: UI captures the last 20 lines of ingestion.log to surface common faults.

Small UX Enhancements
- Show both Windows and resolved/container paths in more places (e.g., Ingest sidebars) to avoid path confusion.
- Add “Open logs directory” helper in Maintenance (opens container dir or host dir depending on env).
- Success toast after auto‑finalize with: persisted doc count, target collection, and link to Collections.

Documentation
- Expand Docker “Troubleshooting” with common symptoms, staging examples, and exact shell commands (`docker exec`, `tail` paths).
- Add a “What to do after Clean Start” quick checklist (set DB path, run small ingest, verify vector doc count, search works).

