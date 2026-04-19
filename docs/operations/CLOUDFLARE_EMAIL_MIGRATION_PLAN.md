# Cloudflare Email Migration Plan

## Purpose

Move Cortex and the connected Longboardfella website from Gmail-based workflow inboxes to a domain-based Cloudflare email architecture while keeping VentraIP as the website host unless changed later.

## Target Architecture

- Cloudflare becomes the authoritative DNS provider for `longboardfella.com.au`.
- VentraIP remains the website/app origin host.
- `paul@longboardfella.com.au` remains the protected human mailbox.
- `intel@longboardfella.com.au` replaces `intel.longboardfella@gmail.com`.
- `lab@longboardfella.com.au` replaces `lab.longboardfella@gmail.com`.
- Inbound workflow email is routed through Cloudflare Email Routing / Workers.
- Outbound automated replies and notifications use domain-based sending rather than Gmail SMTP.
- Gmail IMAP polling remains only during the transition window, then is retired.

## Scope

This plan covers:

- DNS and nameserver migration to Cloudflare
- workflow mailbox migration
- transport-neutral Cortex intake refactor
- website transport-neutral intake refactor
- Cloudflare Worker bridging
- cutover, rollback, and decommissioning

This plan does not require:

- moving website hosting away from VentraIP
- immediately moving the human mailbox away from its current provider
- rewriting Cortex extraction logic

## Current State

Confirmed current Cortex-side dependencies:

- IMAP poller loop: [worker/intel_mailbox_worker.py](/home/longboardfella/cortex_suite/worker/intel_mailbox_worker.py:32)
- IMAP/SMTP config and delivery logic: [cortex_engine/intel_mailbox.py](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py:1097)
- mailbox routing and reply flow documented in [docs/MARKET_RADAR_ARCHITECTURE.md](/home/longboardfella/cortex_suite/docs/MARKET_RADAR_ARCHITECTURE.md:137)

## Migration Principles

- Separate mail transport from mail-processing business logic.
- Keep Cortex parsing and extraction logic stable during transport migration.
- Keep Gmail live until the domain-based flow is proven.
- Protect `paul@longboardfella.com.au` from early-risk changes.
- Prefer reversible stages over a big-bang cutover.

## Final Mailbox Map

- `paul@longboardfella.com.au`: human mailbox
- `intel@longboardfella.com.au`: Market Radar / Cortex intel intake
- `lab@longboardfella.com.au`: Lab jobs / dispatch / CSV import intake
- `noreply@longboardfella.com.au`: automated outbound
- Optional future addresses:
  - `signals@longboardfella.com.au`
  - `watch@longboardfella.com.au`
  - `jobs@longboardfella.com.au`

## Workstreams

### 1. DNS And Mail Discovery

Goals:

- export current VentraIP DNS state
- identify current MX/mail hosting arrangement
- preserve website and mailbox continuity during nameserver cutover

Actions:

1. Export all current DNS records from VentraIP.
2. Record nameservers currently in use.
3. Record current MX, SPF, DKIM, DMARC values.
4. Confirm where `paul@longboardfella.com.au` is actually hosted today.
5. Record any email forwarders and cPanel mailbox dependencies.

Deliverables:

- `email-migration-inventory.md`
- DNS parity checklist
- mailbox dependency inventory

### 2. Transport-Neutral Cortex Intake

Goals:

- preserve current Cortex mailbox behavior
- allow the same processing flow to run from IMAP or webhook-delivered messages

Required code shape:

- IMAP fetch remains an adapter
- transport-neutral message processor becomes primary
- SMTP reply sending becomes swappable

Required message contract:

- `message_id`
- `from_email`
- `from_name`
- `to_email`
- `subject`
- `received_at`
- `text_body`
- `html_body`
- `attachments`
- optional raw MIME metadata

### 3. Transport-Neutral Website Intake

Goals:

- make website intel and job ingest callable by Cloudflare Worker
- keep existing parsing behavior intact

Required code shape:

- IMAP poll scripts become adapters only
- reusable ingestion libraries own parsing and job creation
- authenticated webhook endpoints accept normalized email payloads

### 4. Cloudflare DNS Setup

Goals:

- import and validate the zone before nameserver cutover
- ensure web origin still points to VentraIP
- preserve current human mailbox continuity

Actions:

1. Add `longboardfella.com.au` to Cloudflare.
2. Import or recreate all DNS records.
3. Verify website records, API records, and mail records.
4. Lower TTLs before nameserver switch.
5. Do not change nameservers until parity is confirmed.

### 5. Cloudflare Email Setup

Goals:

- create domain-based workflow addresses
- route inbound workflow mail to Workers
- support outbound app sending

Actions:

1. Configure Cloudflare Email Routing for `intel@` and `lab@`.
2. Configure Email Service sending for `noreply@` and later reply-capable addresses.
3. Add required MX/TXT/DKIM records.
4. Deploy a thin inbound Worker that forwards normalized messages to existing application endpoints.

### 6. Parallel Validation

Goals:

- prove the new transport without losing current functionality

Test cases:

1. basic intel note
2. forwarded HTML digest
3. `JOB:` submission
4. CSV profile import
5. attachment-heavy mail
6. `entity:` scoped note
7. `depth:detailed`
8. reply generation path

### 7. Cutover

Cutover sequence:

1. Finish DNS parity validation.
2. Switch nameservers to Cloudflare.
3. Re-verify website functionality.
4. Verify `paul@longboardfella.com.au`.
5. Verify `intel@` and `lab@`.
6. Start production use of the new domain addresses.
7. Keep Gmail as fallback during a short overlap window.

### 8. Gmail Retirement

Only after successful parallel validation:

1. disable Gmail IMAP poll cron jobs
2. disable Cortex Gmail IMAP worker
3. remove Gmail app-password secrets
4. update documentation and UI references
5. set temporary Gmail forwarding or autoresponder if useful

## Rollback Strategy

If nameserver cutover fails:

1. revert nameservers to VentraIP
2. restore prior DNS authority
3. continue Gmail-based intake
4. pause Worker delivery until parity issues are fixed

If email routing works but application processing fails:

1. keep Cloudflare DNS in place
2. re-route workflow inboxes temporarily to human mailbox or holding path
3. continue Gmail-based pollers until webhook handlers are fixed

## Definition Of Done

The migration is complete when:

- mail to `intel@longboardfella.com.au` enters Market Radar/Cortex correctly
- mail to `lab@longboardfella.com.au` creates the expected jobs/imports correctly
- `paul@longboardfella.com.au` remains functional
- VentraIP continues serving the website origin without regression
- Gmail IMAP is no longer required for core workflows
- Gmail SMTP is no longer required for Cortex automated replies
- visible Gmail addresses are removed from UI/help/docs

## Immediate Next Actions

1. Gather current VentraIP DNS and mailbox details manually.
2. Refactor website ingest into transport-neutral libraries and webhook endpoints.
3. Refactor Cortex intake to use a transport-neutral message processor.
4. Prepare the Cloudflare zone only after those refactors are planned and scoped.
