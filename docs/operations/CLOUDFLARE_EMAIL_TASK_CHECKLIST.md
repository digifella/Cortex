# Cloudflare Email Migration Task Checklist

## Purpose

This checklist is the execution-oriented companion to `CLOUDFLARE_EMAIL_MIGRATION_PLAN.md`. It is intentionally file-specific so another agent can pick up a workstream without rediscovering the codebase.

## Status Legend

- `pending`
- `in_progress`
- `done`
- `blocked`

## Pre-Migration Discovery

- `pending` Record current nameservers for `longboardfella.com.au`
- `pending` Export current VentraIP DNS records
- `pending` Record MX/SPF/DKIM/DMARC values
- `pending` Confirm where `paul@longboardfella.com.au` is hosted
- `pending` Record current Gmail workflow mailboxes and cron jobs

## Cortex Suite Tasks

### A. Message Processing Refactor

Files:

- [cortex_engine/intel_mailbox.py](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py)
- [worker/intel_mailbox_worker.py](/home/longboardfella/cortex_suite/worker/intel_mailbox_worker.py)
- [tests/unit/test_intel_mailbox.py](/home/longboardfella/cortex_suite/tests/unit/test_intel_mailbox.py)

Tasks:

- `pending` Introduce a transport-neutral message schema for mailbox processing
- `pending` Add a primary entrypoint like `process_message_payload(...)`
- `pending` Make `poll_once()` call the transport-neutral path rather than owning business logic directly
- `pending` Keep IMAP support as a compatibility adapter during transition
- `pending` Add tests for payload-driven processing without IMAP

### B. Reply Sending Abstraction

Files:

- [cortex_engine/intel_mailbox.py](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py)
- [tests/unit/test_intel_mailbox.py](/home/longboardfella/cortex_suite/tests/unit/test_intel_mailbox.py)

Tasks:

- `pending` Abstract outbound reply sending behind an interface
- `pending` Keep current SMTP client as one implementation
- `pending` Add a placeholder or implementation for Cloudflare sending
- `pending` Add tests for reply dispatch selection

### C. Operational Config Cleanup

Files:

- [worker/README.md](/home/longboardfella/cortex_suite/worker/README.md)
- [worker/config.env.example](/home/longboardfella/cortex_suite/worker/config.env.example)
- [docs/MARKET_RADAR_ARCHITECTURE.md](/home/longboardfella/cortex_suite/docs/MARKET_RADAR_ARCHITECTURE.md)
- [docs/MAILBOX_ENTITY_ROUTING_SPEC.md](/home/longboardfella/cortex_suite/docs/MAILBOX_ENTITY_ROUTING_SPEC.md)

Tasks:

- `pending` Document the future webhook transport path alongside IMAP compatibility
- `pending` Mark Gmail-specific config as transitional
- `pending` Add Cloudflare transport notes and required secrets

## Website Repo Tasks

Primary files in `/home/longboardfella/longboardfella_website`:

- `site/market_radar_intel_ingest.php`
- `site/job_ingest.php`
- `site/admin/lab_ingest_config.example.php`
- `site/lab/market-radar.php`
- `MARKET_RADAR_PRD.md`

### D. Intel Ingest Refactor

Files:

- `site/market_radar_intel_ingest.php`
- new file: `site/admin/email_intel_ingest_lib.php`

Tasks:

- `pending` Extract transport-neutral intel parsing and ingest logic into a shared library
- `pending` Leave `market_radar_intel_ingest.php` as an IMAP adapter only
- `pending` Add a webhook endpoint for normalized inbound email
- `pending` Add replay/test support for stored `.eml` or JSON payloads

### E. Lab Job Ingest Refactor

Files:

- `site/job_ingest.php`
- new file: `site/admin/email_job_ingest_lib.php`

Tasks:

- `pending` Extract transport-neutral job parsing and queue-creation logic into a shared library
- `pending` Leave `job_ingest.php` as an IMAP adapter only
- `pending` Add a webhook endpoint for normalized inbound email
- `pending` Add replay/test support for representative lab emails

### F. UI And Docs Cleanup

Files:

- `site/lab/market-radar.php`
- `MARKET_RADAR_PRD.md`
- relevant docs/specs under `docs/`

Tasks:

- `pending` Replace visible Gmail workflow addresses with domain addresses
- `pending` Update operator help text from “checked every 5 minutes” to the new architecture after cutover
- `pending` Mark Gmail docs/config as legacy transitional material

## Cloudflare Infrastructure Tasks

- `pending` Add the zone to Cloudflare
- `pending` import or recreate DNS records
- `pending` validate parity before nameserver switch
- `pending` configure Email Routing for `intel@` and `lab@`
- `pending` configure Email Service sending for `noreply@`
- `pending` deploy thin inbound Worker
- `pending` test end-to-end POST into website/Cortex endpoints

## Parallel-Run Checklist

- `pending` Send test mail to `intel@...`
- `pending` Send test mail to `lab@...`
- `pending` Verify intel note ingest
- `pending` Verify digest extraction
- `pending` Verify `JOB:` command ingest
- `pending` Verify CSV profile import
- `pending` Verify attachment handling
- `pending` Verify automated replies

## Decommission Checklist

- `pending` disable Gmail poll cron for website intel ingest
- `pending` disable Gmail poll cron for website job ingest
- `pending` disable Cortex Gmail IMAP worker if no longer needed
- `pending` remove Gmail app-password secrets
- `pending` archive or replace Gmail-specific setup docs

## Human Inputs Required

- current VentraIP DNS export or screenshots
- current nameservers
- current MX/mail-hosting arrangement for `paul@longboardfella.com.au`
- confirmation of whether `paul@...` remains on the current host during this migration

## Notes

- Keep Gmail alive until domain-based mail flow is stable.
- Do not cut over visible/public addresses until webhook-based processing is proven.
- Do not move the human mailbox and workflow transport in the same risky step unless there is a strong reason.
