# Website Lab Mailbox Cutover Spec

Purpose: move the website lab administration and lab job ingest flow from the
legacy Gmail inbox:

- `lab.longboardfella@gmail.com`

to the canonical Microsoft 365 mailbox:

- `lab@longboardfella.com.au`

This spec is lab-only. It does **not** cover the intel/Market Radar mailbox.
The intel intake path is separate and must not be pulled into the lab cutover
work.

## Scope

In scope:

- `longboardfella_website` lab job ingest
- website-facing lab help text and operator copy
- lab mailbox config examples
- transitional IMAP wiring for the lab mailbox
- website docs that still present Gmail as the primary lab address

Out of scope:

- `intel.longboardfella@gmail.com`
- `intel@longboardfella.com.au`
- NemoClaw voice/poller routing for intel intake
- Market Radar business logic beyond references that mention the lab mailbox

## Required End State

The website should present:

- `lab@longboardfella.com.au` as the primary lab submission address
- `lab.longboardfella@gmail.com` only as a temporary fallback during cutover

The website should not continue to describe Gmail as the canonical lab mailbox
once the cutover is complete.

## Required Changes

### 1. Website lab ingest config

Update:

- `site/admin/lab_ingest_config.example.php`

Requirements:

- make `lab@longboardfella.com.au` the example/default mailbox identity
- label the Gmail inbox as transitional only
- remove any wording that suggests Gmail is the long-term target
- keep provider-specific IMAP details isolated from business logic

### 2. Website lab ingest UI/help text

Update:

- `site/lab/market-radar.php`
- any other visible lab help text or admin guidance

Requirements:

- show `lab@longboardfella.com.au` as the address users should send to
- avoid exposing the legacy Gmail inbox as the default option
- preserve the current job-ingest behavior while only changing the mailbox
  identity and operator guidance

### 3. Website lab ingest processing

Keep:

- `site/job_ingest.php`
- `site/admin/email_job_ingest_lib.php`

Requirements:

- preserve the current transport-neutral processing model
- keep IMAP as a temporary adapter only
- ensure replies and logged mailbox identity use `lab@longboardfella.com.au`
  as the canonical address

### 4. Website docs and PRD

Update any active website docs that still present the Gmail inbox as canonical,
including:

- `MARKET_RADAR_PRD.md`
- migration/checklist/runbook docs that mention the lab mailbox

Requirements:

- mention `lab.longboardfella@gmail.com` only as legacy or transitional
- keep the cutover plan explicit so operators know when the Gmail inbox can be
  retired

## Coordination With NemoClaw

During the transition, NemoClaw may poll both:

- `lab.longboardfella@gmail.com`
- `lab@longboardfella.com.au`

That is a temporary bridge only. The website cutover should make
`lab@longboardfella.com.au` the normal target so the legacy Gmail inbox can be
turned off later.

## Acceptance Criteria

- website lab help text points to `lab@longboardfella.com.au`
- website lab ingest configs/examples use `lab@longboardfella.com.au`
- legacy Gmail references are clearly transitional
- no website lab docs describe Gmail as the canonical mailbox
- intel mailbox references remain untouched by this lab cutover

