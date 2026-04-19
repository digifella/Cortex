# Website Email Migration Spec

Purpose: handoff spec for the website agent to migrate the Longboardfella website email workflows from the two Gmail workaround inboxes to the new Microsoft 365 shared mailboxes:

- `intel@longboardfella.com.au`
- `lab@longboardfella.com.au`

This spec is intentionally website-focused. Cortex-side changes are being handled separately in `cortex_suite`.

## Scope

In scope:

- website code and docs in `/home/longboardfella/longboardfella_website`
- Market Radar intel ingest flow
- Lab job ingest flow
- operator-facing UI/help text
- config/examples/runbooks
- preparatory refactor so mailbox processing is transport-neutral

Out of scope for this handoff:

- Cloudflare Email Routing for the root domain
- Microsoft 365 tenant administration
- Cortex repo changes
- final Microsoft Graph/OAuth transport implementation unless the website agent is also explicitly assigned that work

## Architecture Decision

Do not perform a naive provider swap from Gmail IMAP to Microsoft 365 IMAP.

Reason:

- current website ingest scripts are tightly coupled to `imap_open(...)` and Gmail-style app-password assumptions
- the new inboxes are Microsoft 365 shared mailboxes
- long-term inbound transport should not depend on basic-auth IMAP assumptions

Required design direction:

1. separate mailbox transport from business processing
2. keep the current IMAP pollers only as temporary adapters
3. make business logic callable from:
   - IMAP poller
   - future webhook/Microsoft Graph adapter
   - local replay test harness

## Canonical Mailboxes

New canonical addresses:

- `intel@longboardfella.com.au`
- `lab@longboardfella.com.au`

Legacy addresses to retire after validation:

- `intel.longboardfella@gmail.com`
- `lab.longboardfella@gmail.com`

## Files To Change

### Primary live code

- `site/market_radar_intel_ingest.php`
- `site/job_ingest.php`
- `site/lab/market-radar.php`
- `site/admin/lab_ingest_config.example.php`

### Supporting docs/specs that should be updated in the same pass

- `MARKET_RADAR_PRD.md`
- `docs/superpowers/specs/2026-03-19-cortex-intel-notes-design.md`
- `docs/superpowers/specs/2026-03-24-subscriptions-email-dispatch-design.md`
- `docs/superpowers/plans/2026-03-24-email-to-dispatch.md`

### Scripts/examples to review

- `docs/operations/scripts/nemoclaw-lab-poller.py`
- `docs/operations/scripts/nemoclaw-voice-processor.py`
- `docs/operations/scripts/nemoclaw-recover-emails.py`

## Required Refactor

### 1. Extract transport-neutral intel ingest logic

Current problem:

- `site/market_radar_intel_ingest.php` mixes IMAP polling, MIME parsing, filtering, persistence, and reply generation in one file

Required outcome:

- introduce a shared library, recommended path:
  - `site/admin/email_intel_ingest_lib.php`
- expose a transport-neutral entrypoint:
  - `processIntelEmail(array $message, array $options = []): array`

Required normalized input shape:

- `message_id`
- `from_email`
- `from_name`
- `to_email`
- `subject`
- `received_at`
- `text_body`
- `html_body`
- `attachments`
- `raw_headers` optional
- `transport_meta` optional

`market_radar_intel_ingest.php` should become a thin IMAP adapter only:

- fetch message from mailbox
- normalize to the shared structure
- call `processIntelEmail(...)`
- log result

### 2. Extract transport-neutral lab ingest logic

Current problem:

- `site/job_ingest.php` mixes IMAP polling, message parsing, command detection, dispatch creation, and reply generation in one file

Required outcome:

- introduce a shared library, recommended path:
  - `site/admin/email_job_ingest_lib.php`
- expose:
  - `processJobEmail(array $message, array $options = []): array`

`job_ingest.php` should become a thin IMAP adapter only.

### 3. Add webhook-compatible entrypoints

Add two authenticated endpoints:

- `site/admin/email_intel_webhook.php`
- `site/admin/email_job_webhook.php`

Requirements:

- accept JSON payloads in the normalized message shape
- validate a shared secret header or query secret
- call the shared processing functions
- return structured JSON

This is the prerequisite for a future Microsoft Graph or Worker-based transport.

### 4. Preserve current behavior during transition

The refactor must preserve:

- current message classification behavior
- current note ingestion behavior
- current job command behavior
- current CSV/profile import behavior
- current reply suppression for no-reply senders
- current logging style as much as practical

## Mailbox Identity Changes

Replace hardcoded mailbox identity references:

- `intel.longboardfella@gmail.com` -> `intel@longboardfella.com.au`
- `lab.longboardfella@gmail.com` -> `lab@longboardfella.com.au`

Important:

- do not hardcode Microsoft server hostnames as the long-term answer
- centralize mailbox identity config in one place
- if temporary IMAP remains, keep provider host/user/pass in config, not in logic

## UI / Operator Text Changes

Update public/operator-facing references:

- `site/lab/market-radar.php`
- `MARKET_RADAR_PRD.md`
- related specs/help text

New visible addresses should be:

- `intel@longboardfella.com.au`
- `lab@longboardfella.com.au`

## Config Expectations

The website agent should convert config/examples away from Gmail-specific guidance.

At minimum:

- remove Gmail app-password instructions from examples
- stop presenting Gmail as the canonical mailbox transport
- if a temporary IMAP example remains, label it as temporary transport configuration rather than architecture

## Testing Requirements

The website agent should provide:

1. a replayable local test path for normalized email payloads
2. a regression test or fixture for:
   - simple intel note
   - routed intel note with `entity:`
   - `JOB:` command email
   - CSV profile import email
3. a manual validation checklist covering:
   - reply-to behavior
   - no-reply suppression
   - attachment handling
   - duplicate protection

## Delivery Order

The website agent should implement in this order:

1. extract shared intel ingest library
2. extract shared lab ingest library
3. add webhook endpoints
4. update UI/docs/help text to the new addresses
5. update config/examples
6. keep IMAP adapters working temporarily
7. stop using Gmail only after validated domain-mail transport is ready

## Definition Of Done

The website handoff is complete when:

- no live website flow requires the Gmail mailbox identities
- business processing can run without direct IMAP coupling
- webhook endpoints exist for both intel and lab ingestion
- visible UI/docs point to the `longboardfella.com.au` addresses
- Gmail-specific operator instructions are removed from active docs/examples
