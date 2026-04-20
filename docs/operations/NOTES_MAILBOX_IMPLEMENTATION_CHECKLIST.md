# Notes Mailbox Implementation Checklist

Objective: introduce `notes@longboardfella.com.au` as a dedicated Cortex note mailbox and remove note-stash behavior from the Market Radar intel mailbox.

Operational runbook:

- `docs/operations/NOTES_MAILBOX_RUNBOOK.md`

## Phase 1: Mailbox Split Policy

- confirm `notes@longboardfella.com.au` exists in Microsoft 365
- confirm Paul has `Read and manage`
- confirm Paul has `Send as`
- confirm sent-items copy is enabled
- remove any forwarding/rules that send note traffic into `intel.longboardfella@gmail.com`

## Phase 2: Intel Mailbox Cleanup

- stop treating `intel@...` as a mixed stash mailbox
- remove or disable Nemoclaw/private-note routing in the intel mailbox path
- keep `intel@...` dedicated to website/Market Radar delivery
- preserve current Market Radar routing behavior

## Phase 3: Notes Mailbox Processor

- create a dedicated Cortex note mailbox processor
- support default `public_note` routing
- support `private_note` routing from subject markers
- keep Market Radar-specific logic out of the notes flow

Recommended files:

- `cortex_engine/notes_mailbox.py`
- `worker/notes_mailbox_worker.py`

## Phase 4: Transport

- choose M365 transport approach for `notes@...`
- do not rely on an implicit Gmail/basic-auth swap
- preferred: Microsoft Graph polling or subscription/webhook delivery
- current implementation supports Graph polling with:
  - `NOTES_TRANSPORT_MODE=graph`
  - `NOTES_GRAPH_TENANT_ID`
  - `NOTES_GRAPH_CLIENT_ID`
  - `NOTES_GRAPH_CLIENT_SECRET`
  - `NOTES_GRAPH_MAILBOX=notes@longboardfella.com.au`
- the Microsoft app must be scoped in Exchange Online to only `notes@longboardfella.com.au`
- keep the Graph app on `Mail.Read` unless server-side mailbox mutation is required
- if Microsoft Graph rejects mark-as-read with 403, the worker still de-duplicates locally using `tmp/notes_mailbox_state.json`
- notes are persisted as markdown when `NOTES_WRITE_VAULT_MARKDOWN=1`:
  - public notes -> `NOTES_PUBLIC_VAULT_DIR`
  - private notes -> `NOTES_PRIVATE_VAULT_DIR`
  - JSON outbox files remain as an audit/debug trail

## Phase 5: Validation

- test plain note to `notes@...` -> public stash
- test `PRIVATE:` note to `notes@...` -> private vault
- test Market Radar-style intel sent to `notes@...` -> rejected/quarantined
- test `intel@...` no longer produces note-stash routing

## Phase 6: Docs

- update worker/operator docs to describe the mailbox split
- update any remaining references that imply `intel@...` is also a note-stash mailbox
- document subject conventions for `notes@...`
