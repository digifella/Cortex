# Cortex Progress Update (2026-02-16)

## Completed

1. Docling pipeline stabilized in the main Cortex runtime.
2. Textifier quality improved for markdown flow/cleanup and image marker handling.
3. PDF processing strategy set to Docling-first hybrid with fallback behavior.
4. Timeout controls added:
   - Docling stage timeout
   - Per-image vision timeout
   - Total image enrichment budget
5. Worker handoff improvements:
   - `pdf_textify` end-to-end website -> cortex -> website working
   - `url_ingest` handler added and working
6. Queue visibility and control:
   - New `Queue Monitor` page (`pages/16_Queue_Monitor.py`)
   - Worker telemetry store (`cortex_engine/queue_monitor.py`)
   - Cancel/clear controls and event history
7. Ingestion consistency:
   - `DocumentTextifier.from_options(...)` shared across UI/worker/URL ingestor
   - Knowledge Ingest default backend moved to Docling
8. URL ingest option forwarding:
   - Advanced textify options now pass through from `input_data` to `DocumentTextifier`

## Current Status

- Website batch queue processing tested and confirmed working.
- URL ingest queue path now supports advanced textify controls from website payload.
- Queue monitor is available for live operations and cancellation workflows.

## Active Contract Notes

- Worker types should include:
  - `pdf_anonymise`
  - `pdf_textify`
  - `url_ingest`
- `worker/config.env` should contain:
  - `SUPPORTED_TYPES=pdf_anonymise,pdf_textify,url_ingest`

## Recommended Next Focus

1. Production validation runbook for mixed batch jobs (pdf_textify + url_ingest).
2. Error-handling hardening for remote queue cancel semantics (explicit cancel action if website adds it).
3. Continue roadmap items after queue pipeline sign-off:
   - Program Transparency Portal implementation phases
   - Further anonymizer granularity/format-preservation enhancements
   - Optional local model strategy tuning (LM Studio/Ollama profiles)
