# Website Agent Checklist

Repository: `/home/longboardfella/longboardfella_website`

Objective: migrate website-owned email workflows from the Gmail workaround inboxes to the domain mailboxes and prepare for non-IMAP transport.

## Phase 1: Audit And Prep

- confirm all live references to `intel.longboardfella@gmail.com`
- confirm all live references to `lab.longboardfella@gmail.com`
- identify current config include files for mailbox credentials
- identify cron jobs invoking `market_radar_intel_ingest.php`
- identify cron jobs invoking `job_ingest.php`

## Phase 2: Intel Refactor

- extract processing logic from `site/market_radar_intel_ingest.php`
- create `site/admin/email_intel_ingest_lib.php`
- add `processIntelEmail(array $message, array $options = []): array`
- keep `site/market_radar_intel_ingest.php` as thin IMAP adapter
- remove hardcoded Gmail mailbox identity from replies/help text

## Phase 3: Lab Refactor

- extract processing logic from `site/job_ingest.php`
- create `site/admin/email_job_ingest_lib.php`
- add `processJobEmail(array $message, array $options = []): array`
- keep `site/job_ingest.php` as thin IMAP adapter
- remove hardcoded Gmail mailbox identity from replies/help text

## Phase 4: Webhook Entry Points

- add `site/admin/email_intel_webhook.php`
- add `site/admin/email_job_webhook.php`
- implement shared-secret validation
- accept normalized JSON email payloads
- return structured JSON status responses

## Phase 5: UI And Docs

- update `site/lab/market-radar.php`
- update `MARKET_RADAR_PRD.md`
- update any current operator help text
- replace Gmail addresses with:
  - `intel@longboardfella.com.au`
  - `lab@longboardfella.com.au`

## Phase 6: Config Cleanup

- update `site/admin/lab_ingest_config.example.php`
- remove Gmail app-password instructions from active config examples
- centralize mailbox identity config where practical
- ensure temporary provider-specific IMAP settings are clearly marked as transitional

## Phase 7: Validation

- test intel note ingest end-to-end
- test `entity:`-routed intel note end-to-end
- test `JOB:` email end-to-end
- test CSV import email end-to-end
- test reply suppression for no-reply senders
- test attachment handling
- test duplicate protection

## Handoff Notes

- do not block the refactor on Microsoft Graph implementation
- do not rewrite business rules unless required by the transport split
- preserve current behavior first, then improve transport
